import argparse
import os
import sys
import torch
from torchrec import EmbeddingBagCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.modules.embedding_configs import EmbeddingBagConfig

RECSTORE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
if RECSTORE_PATH not in sys.path:
    sys.path.insert(0, RECSTORE_PATH)

from python.pytorch.torchrec.EmbeddingBag import RecStoreEmbeddingBagCollection
from python.pytorch.recstore.KVClient import get_kv_client
from python.pytorch.recstore.optimizer import SparseSGD

# --- Constants ---
LEARNING_RATE = 0.01
NUM_TEST_ROUNDS = 10


def get_eb_configs(
    num_embeddings: int,
    embedding_dim: int,
) -> list:
    """
    Generates a list of EmbeddingBagConfig objects for a single table.
    """
    return [
        EmbeddingBagConfig(
            # Use a single default table to align with server expectations.
            name="default",
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
            feature_names=["feature_0"],
        )
    ]

def compare_tensors(tensor1: torch.Tensor, tensor2: torch.Tensor, label: str, atol=1e-6) -> bool:
    """
    Compares two tensors for near-equality and prints a detailed result.
    """
    print(f"\n----- Comparing '{label}' -----")
    t1, t2 = tensor1.detach(), tensor2.detach()
    
    # Ensure both tensors are on the same device (CPU for comparison)
    if t1.device.type != 'cpu':
        t1 = t1.cpu()
    if t2.device.type != 'cpu':
        t2 = t2.cpu()

    if t1.shape != t2.shape:
        print(f"❌ FAILURE: {label} outputs have MISMATCHED SHAPES.")
        print(f"  - Shape of Tensor 1 (Expected): {t1.shape}")
        print(f"  - Shape of Tensor 2 (Actual):   {t2.shape}")
        return False

    are_close = torch.allclose(t1, t2, atol=atol)
    if are_close:
        print(f"   - Sliced Tensor 1 (Expected): \n{t1.flatten()[:8]}")
        print(f"   - Sliced Tensor 2 (Actual):   \n{t2.flatten()[:8]}")
        print(f"✅ SUCCESS: {label} outputs are numerically aligned.")
    else:
        print(f"❌ FAILURE: {label} outputs are NOT aligned.")
        max_diff = (t1 - t2).abs().max().item()
        print(f"   - Max absolute difference: {max_diff:.8f}")
        if max_diff > 1e-5:
            print(f"   - Sliced Tensor 1 (Expected): \n{t1.flatten()[:8]}")
            print(f"   - Sliced Tensor 2 (Actual):   \n{t2.flatten()[:8]}")
    return are_close

def generate_random_batch(num_embeddings, batch_size, device):
    """
    Generates a random KeyedJaggedTensor batch.
    """
    avg_len = max(1, (num_embeddings // batch_size) // 2)
    lengths = torch.randint(1, avg_len * 2, (batch_size,), device=device, dtype=torch.int32)
    values = torch.randint(0, num_embeddings, (lengths.sum().item(),), device=device, dtype=torch.int64)
    return KeyedJaggedTensor.from_lengths_sync(
        keys=["feature_0"],
        values=values,
        lengths=lengths,
    )


def main(args):
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"Running SINGLE TABLE precision test on device: {device} for {NUM_TEST_ROUNDS} rounds.")
    print(f"Testing network-based update mechanism with KVClient.update()")

    eb_configs = get_eb_configs(
        num_embeddings=args.num_embeddings,
        embedding_dim=args.embedding_dim,
    )
    
    recstore_eb_configs_dict = [
        {"name": c.name, "embedding_dim": c.embedding_dim, "num_embeddings": c.num_embeddings, "feature_names": c.feature_names}
        for c in eb_configs
    ]

    print("\nInstantiating standard torchrec.EmbeddingBagCollection (Ground Truth)...")
    standard_ebc = EmbeddingBagCollection(tables=eb_configs, device=device)
    
    print("\n--- Initializing Backend and Synchronizing Weights ---")
    kv_client = get_kv_client()
    config = eb_configs[0]
    
    # Get initial weights from standard_ebc FIRST
    with torch.no_grad():
        initial_weights = standard_ebc.state_dict()[f"embedding_bags.{config.name}.weight"]
        # Ensure CPU, contiguous, and create independent copy
        if initial_weights.device.type != 'cpu':
            initial_weights = initial_weights.cpu()
        initial_weights = initial_weights.contiguous().clone()
        print(f"Standard EBC initial weights (first 5 dims of row 0): {initial_weights[0, :5]}")
    
    # Manually initialize backend table with correct dimensions
    print(f"\nManually initializing backend table '{config.name}'...")
    all_keys = torch.arange(config.num_embeddings, dtype=torch.int64)
    success = kv_client.ops.init_embedding_table(config.name, int(config.num_embeddings), int(config.embedding_dim))
    if not success:
        raise RuntimeError(f"Failed to initialize embedding table '{config.name}'")
    print(f"✓ Backend table '{config.name}' initialized")
    
    # Write initial weights ONCE
    print(f"Writing initial weights to backend...")
    kv_client.ops.emb_write(all_keys, initial_weights)
    print(f"✓ Initial weights written")
    
    # Now create RecStoreEmbeddingBagCollection WITHOUT calling init_data
    # We need to manually set up metadata to skip init_data
    print("\nInstantiating custom RecStoreEmbeddingBagCollection...")
    print("IMPORTANT: Disabling fusion and manual initialization to avoid double-init")
    # Temporarily mark table as already initialized
    kv_client._tensor_meta[config.name] = {'shape': (config.num_embeddings, config.embedding_dim), 'dtype': torch.float32}
    kv_client._full_data_shape[config.name] = (config.num_embeddings, config.embedding_dim)
    kv_client._data_name_list.add(config.name)
    kv_client._gdata_name_list.add(config.name)
    
    recstore_ebc = RecStoreEmbeddingBagCollection(embedding_bag_configs=recstore_eb_configs_dict, lr=LEARNING_RATE, enable_fusion=False).to(device)
    print(f"✓ RecStoreEmbeddingBagCollection initialized")
    
    # Verify the push worked by pulling back and comparing
    print("\n--- Verifying Initial Weight Synchronization ---")
    with torch.no_grad():
        # Now we can use kv_client.pull since we registered the metadata
        pulled_weights = kv_client.pull(name=config.name, ids=all_keys)
        print(f"Pulled weights back (first 5 dims of row 0): {pulled_weights[0, :5]}")
        init_sync_ok = compare_tensors(
            initial_weights,
            pulled_weights,
            "Initial Weight Synchronization"
        )
        if not init_sync_ok:
            print("\n🔥🔥🔥 CRITICAL: Initial weight synchronization failed!")
            print("The push/pull mechanism is not working correctly.")
            print("Expected (row 0, first 10): ", initial_weights[0, :10])
            print("Got      (row 0, first 10): ", pulled_weights[0, :10])
            print("\nThis indicates emb_write/emb_read are not functioning correctly.")
            print("Note: emb_write/emb_read operate on a global default table (not table-aware).")
            return
    
    print("✓ Initial weight synchronization verified!")
    
    print(f"\nSetting up optimizers with LR = {LEARNING_RATE}.")
    standard_optimizer = torch.optim.SGD(standard_ebc.parameters(), lr=LEARNING_RATE)
    sparse_optimizer = SparseSGD([recstore_ebc], lr=LEARNING_RATE)

    all_rounds_ok = True
    
    for i in range(NUM_TEST_ROUNDS):
        print("\n" + "#"*50)
        print(f"### Starting Test Round {i + 1} of {NUM_TEST_ROUNDS} ###")
        print("#"*50)

        batch = generate_random_batch(args.num_embeddings, args.batch_size, device)
        print(f"Generated a new random batch with {batch.values().numel()} values.")

        # Forward pass
        standard_output_kt = standard_ebc(batch)
        recstore_output_kt = recstore_ebc(batch)
        
        forward_pass_ok = compare_tensors(
            standard_output_kt.values(), 
            recstore_output_kt.values(), 
            f"Round {i+1} Forward Pass"
        )
        if not forward_pass_ok:
            print(f"🔥🔥🔥 Halting test: Forward pass failed in round {i+1}.")
            all_rounds_ok = False
            break

        # Verify trace was populated (for RecStore)
        if len(recstore_ebc._trace) == 0:
            print(f"⚠️  WARNING: RecStore EBC trace is empty after forward pass!")
        else:
            print(f"✓ RecStore EBC trace populated with {len(recstore_ebc._trace)} entries")

        dummy_loss_standard = standard_output_kt.values().sum()
        dummy_loss_recstore = recstore_output_kt.values().sum()
        
        standard_optimizer.zero_grad()
        sparse_optimizer.zero_grad()

        dummy_loss_standard.backward()
        dummy_loss_recstore.backward()

        # Verify gradients were traced (for RecStore)
        if len(recstore_ebc._trace) == 0:
            print(f"🔥🔥🔥 CRITICAL: RecStore EBC trace is empty after backward pass!")
            print(f"    This means gradients were not properly traced.")
            all_rounds_ok = False
            break
        else:
            print(f"✓ RecStore EBC traced {len(recstore_ebc._trace)} gradient entries")
            # Print sample trace entry for debugging
            if len(recstore_ebc._trace) > 0:
                sample_trace = recstore_ebc._trace[0]
                print(f"  Sample trace: table={sample_trace[0]}, ids shape={sample_trace[1].shape}, grads shape={sample_trace[2].shape}")

        # Perform update step (this uses kv_client.update() via network)
        print(f"Performing optimization step via network-based update...")
        standard_optimizer.step()
        sparse_optimizer.step()
        print(f"✓ Optimization step completed (SparseSGD used kv_client.update())")

        with torch.no_grad():
            config = eb_configs[0]
            updated_standard_weights = standard_ebc.state_dict()[f"embedding_bags.{config.name}.weight"]
            all_keys = torch.arange(config.num_embeddings, dtype=torch.int64)
            # Use kv_client.pull which will call ops.emb_read internally
            updated_recstore_weights = kv_client.pull(name=config.name, ids=all_keys)
            
            weights_ok = compare_tensors(
                updated_standard_weights, 
                updated_recstore_weights, 
                f"Round {i+1} Updated Weights"
            )
            if not weights_ok:
                print(f"🔥🔥🔥 Halting test: Weight update check failed in round {i+1}.")
                print(f"    This indicates the network-based update may not be working correctly.")
                all_rounds_ok = False
                break
        
        print(f"--- Round {i+1} completed successfully ---")

    print("\n" + "="*30)
    print("### FINAL TEST SUMMARY ###")
    print("="*30)
    if all_rounds_ok:
        print(f"🎉🎉🎉 All {NUM_TEST_ROUNDS} precision test rounds passed!")
        print(f"✓ Forward pass produces correct results")
        print(f"✓ Gradient tracing and aggregation works correctly")
        print(f"✓ Network-based update mechanism (emb_update_table) functions properly")
        print(f"✓ RecStore maintains numerical precision with PyTorch baseline")
    else:
        print("🔥🔥🔥 One or more precision test rounds failed. Please review the logs above.")
        print("\nCommon issues to check:")
        print("  1. Verify emb_update_table() is properly implemented in the backend")
        print("  2. Check that gradient tracing in backward pass is working")
        print("  3. Ensure learning rate is correctly applied in SparseSGD.step()")
        print("  4. Confirm gradient aggregation for duplicate IDs is correct")
        print("  5. Verify emb_write/emb_read operations are correctly synchronized with table operations")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-round precision test for RecStore EBC.")
    parser.add_argument("--num-embeddings", type=int, default=1000, help="Number of embeddings per table.")
    parser.add_argument("--embedding-dim", type=int, default=128, help="Dimension of embeddings (backend fixed to 128).")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for generated data.")
    parser.add_argument("--seed", type=int, default=int(torch.rand(1)[0]), help="Random seed for reproducibility.")
    parser.add_argument("--cpu", action="store_true", help="Force test to run on CPU.")
    
    args = parser.parse_args()
    main(args)
