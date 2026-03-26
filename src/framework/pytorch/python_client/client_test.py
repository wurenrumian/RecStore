import torch
from client import RecstoreClient
import numpy as np
import sys
import os

_server_runner = None


def start_server_if_needed():
    """Start ps_server if needed before running tests."""
    global _server_runner
    
    test_scripts_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '../../../test/scripts')
    )
    if test_scripts_path not in sys.path:
        sys.path.insert(0, test_scripts_path)
    
    from ps_server_helpers import should_skip_server_start, get_server_config
    
    skip_server, reason = should_skip_server_start()
    if skip_server:
        print(f"\n[{reason}] Running tests without starting ps_server\n")
        return
    
    config = get_server_config()
    
    print(f"\n{'='*70}")
    print("Starting PS Server for Client Tests")
    print(f"Server path: {config['server_path']}")
    print(f"{'='*70}\n")
    
    from ps_server_runner import PSServerRunner
    _server_runner = PSServerRunner(
        server_path=config['server_path'],
        config_path=config['config_path'],
        log_dir=config['log_dir'],
        timeout=config['timeout'],
        num_shards=config['num_shards'],
        verbose=True
    )
    
    if not _server_runner.start():
        raise RuntimeError("Failed to start PS Server")


def stop_server():
    """Stop ps_server after tests complete."""
    global _server_runner
    
    if _server_runner is not None:
        print(f"\n{'='*70}")
        print("Stopping PS Server")
        print(f"{'='*70}\n")
        _server_runner.stop()
        _server_runner = None


if __name__ == "__main__":
    start_server_if_needed()
    
    try:
        if len(sys.argv) < 2:
            raise RuntimeError("This script requires the path to the compiled C++ library as an argument.")
        library_path = sys.argv[1]
        print(f"--- Loading C++ library from: {library_path} ---")

        client = RecstoreClient(library_path=library_path)
        embedding_dim = 128

        print("\n--- Test 0: Init Embedding Table ---")
        ok = client.init_embedding_table("default", 10000, embedding_dim)
        assert ok, "init_embedding_table returned False"
        print("Init embedding table succeeded.")

        print("\n--- Test 1: Write and Read Verification ---")
        keys_to_write = torch.tensor([1001, 1002, 1003], dtype=torch.int64)
        values_to_write = torch.randn(3, embedding_dim, dtype=torch.float32)

        print(f"Writing embeddings for keys: {keys_to_write.tolist()}")
        client.emb_write(keys_to_write, values_to_write)
        print("Write call successful.")

        print(f"Reading embeddings for keys: {keys_to_write.tolist()}")
        read_values = client.emb_read(keys_to_write, embedding_dim)

        assert read_values.shape == values_to_write.shape, "Shape mismatch after read"
        assert torch.allclose(read_values, values_to_write), "Value mismatch after read"
        print("Read successful. Written values verified.")

        print("\n--- Test 2: Async Prefetch Read ---")
        prefetch_keys = torch.tensor([2001, 2002, 2003, 2004], dtype=torch.int64)
        prefetch_vals = torch.randn(4, embedding_dim, dtype=torch.float32)
        client.emb_write(prefetch_keys, prefetch_vals)

        pid = client.emb_prefetch(prefetch_keys)
        print(f"Issued prefetch id: {pid}")
        prefetched = client.emb_wait_result(pid, embedding_dim)

        print(f"Prefetched embeddings for keys: {prefetch_keys.tolist()}")
        print(f"Prefetch shape: {prefetched.shape}, expected shape: {(4, embedding_dim)}")
        print(f"Prefetched values(first 3): {prefetched.tolist()[:3]}")
        print(f"Expected values(first 3): {prefetch_vals.tolist()[:3]}")
        assert prefetched.shape == (4, embedding_dim), "Prefetch result shape mismatch"
        assert torch.allclose(prefetched, prefetch_vals), "Prefetch values mismatch"
        print("Async prefetch successful and values verified.")

        print("\n--- Test 3: Table-aware Update (smoke) ---")
        update_keys = torch.tensor([1001, 1002], dtype=torch.int64)
        grads = torch.ones(2, embedding_dim, dtype=torch.float32)

        print("Reading values before update...")
        values_before_update = client.emb_read(update_keys, embedding_dim)
        print(f"Values before update (first 5 dims): {values_before_update[:, :5].tolist()}")
        print(f"Expected initial values (first 5 dims): {values_to_write[:2, :5].tolist()}")

        if not torch.allclose(values_before_update, values_to_write[:2]):
            print("WARNING: Values before update don't match values_to_write from Test 1!")
            print(f"  Max diff: {(values_before_update - values_to_write[:2]).abs().max()}")

        print(f"Updating embeddings for keys: {update_keys.tolist()}")
        client.emb_update_table("default", update_keys, grads)
        print("emb_update_table call succeeded.")

        lr = 0.01
        print(f"Reading updated values (backend lr={lr})...")
        updated_values = client.emb_read(update_keys, embedding_dim)
        expected_updated = values_to_write[:2] - (lr * grads)

        print(f"Updated values (first 5 dims): {updated_values[:, :5].tolist()}")
        print(f"Expected updated (first 5 dims): {expected_updated[:, :5].tolist()}")
        print(f"Difference (first 5 dims): {(updated_values - expected_updated)[:, :5].tolist()}")
        print(f"Max absolute difference: {(updated_values - expected_updated).abs().max():.6f}")
        print(f"Max relative difference: {((updated_values - expected_updated).abs() / (expected_updated.abs() + 1e-8)).max():.6f}")

        if torch.allclose(updated_values, expected_updated, rtol=1e-4, atol=1e-6):
            print("✓ Table-aware update verified successfully (with relaxed tolerance).")
        elif torch.allclose(updated_values, expected_updated):
            print("✓ Table-aware update verified successfully.")
        else:
            print("✗ Table-aware update values mismatch!")
            print(f"  Expected: param = param - {lr} * grad")
            print(f"  Are all gradients 1.0? {torch.allclose(grads, torch.ones_like(grads))}")
            raise AssertionError("Table-aware update values mismatch")

        print("\n✓ All tests passed!")

    finally:
        stop_server()
    
    sys.exit(0)
