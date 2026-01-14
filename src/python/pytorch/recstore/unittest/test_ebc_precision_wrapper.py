import unittest
import os
import sys
import argparse
import torch
import importlib.util

RECSTORE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
if RECSTORE_PATH not in sys.path:
    sys.path.insert(0, RECSTORE_PATH)

TEST_SCRIPTS_PATH = os.path.join(RECSTORE_PATH, 'test/scripts')
if TEST_SCRIPTS_PATH not in sys.path:
    sys.path.insert(0, TEST_SCRIPTS_PATH)

from ps_server_runner import ps_server_context
from ps_server_helpers import should_skip_server_start, get_server_config

TEST_MODULE_PATH = os.path.join(os.path.dirname(__file__), 'test_ebc_precision.py')
spec = importlib.util.spec_from_file_location("test_ebc_precision_module", TEST_MODULE_PATH)
test_ebc_precision = importlib.util.module_from_spec(spec)
spec.loader.exec_module(test_ebc_precision)

_server_runner = None


def setUpModule():
    global _server_runner
    
    skip_server, reason = should_skip_server_start()
    if skip_server:
        print(f"\n[{reason}] Running tests without starting ps_server (assuming already running)\n")
        return
    
    config = get_server_config()
    
    print(f"\n{'='*70}")
    print("Starting PS Server for EBC Precision Tests")
    print(f"Server path: {config['server_path']}")
    print(f"Config: {config['config_path'] or 'default'}")
    print(f"Log dir: {config['log_dir']}")
    print(f"Timeout: {config['timeout']}s")
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


def tearDownModule():
    global _server_runner
    
    if _server_runner is not None:
        print(f"\n{'='*70}")
        print("Stopping PS Server")
        print(f"{'='*70}\n")
        _server_runner.stop()
        _server_runner = None


class TestEBCPrecision(unittest.TestCase):
    def test_basic_precision_cpu(self):
        print("\n" + "="*70)
        print("Running Basic EBC Precision Test (CPU)")
        print("="*70)
        
        args = argparse.Namespace(
            num_embeddings=1000,
            embedding_dim=128,  # Backend fixed to 128
            batch_size=64,
            seed=42,
            cpu=True
        )
        
        try:
            # Call the standalone test main function
            test_ebc_precision.main(args)
            print("\n✅ Basic precision test completed successfully")
        except AssertionError as e:
            self.fail(f"Basic precision test failed: {e}")
        except Exception as e:
            self.fail(f"Basic precision test raised unexpected exception: {e}")
    
    def test_small_batch_precision(self):
        print("\n" + "="*70)
        print("Running Small Batch EBC Precision Test (CPU)")
        print("="*70)
        
        args = argparse.Namespace(
            num_embeddings=500,
            embedding_dim=128,
            batch_size=16,
            seed=42,
            cpu=True
        )
        
        try:
            test_ebc_precision.main(args)
            print("\n✅ Small batch precision test completed successfully")
        except AssertionError as e:
            self.fail(f"Small batch precision test failed: {e}")
        except Exception as e:
            self.fail(f"Small batch precision test raised unexpected exception: {e}")
    
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_cuda_precision(self):
        print("\n" + "="*70)
        print("Running CUDA EBC Precision Test")
        print("="*70)
        
        args = argparse.Namespace(
            num_embeddings=1000,
            embedding_dim=128,
            batch_size=64,
            seed=42,
            cpu=False
        )
        
        try:
            test_ebc_precision.main(args)
            print("\n✅ CUDA precision test completed successfully")
        except ImportError as e:
            self.skipTest(f"CUDA test skipped due to import error (likely FBGEMM): {e}")
        except AssertionError as e:
            self.fail(f"CUDA precision test failed: {e}")
        except Exception as e:
            self.fail(f"CUDA precision test raised unexpected exception: {e}")


if __name__ == "__main__":
    import argparse
    
    is_ci = os.environ.get('CI') == 'true' or os.environ.get('GITHUB_ACTIONS') == 'true'
    
    parser = argparse.ArgumentParser(description='EBC Precision Test with PS Server')
    parser.add_argument('--no-server', action='store_true',
                        default=is_ci,
                        help='Skip starting ps_server (assume already running, default in CI)')
    parser.add_argument('--server-path', 
                        default=os.environ.get('PS_SERVER_PATH', './build/bin/ps_server'),
                        help='Path to ps_server binary (default: ./build/bin/ps_server)')
    parser.add_argument('--config', 
                        default=os.environ.get('RECSTORE_CONFIG'),
                        help='Path to recstore config file')
    parser.add_argument('--log-dir', 
                        default=os.environ.get('PS_LOG_DIR', './logs'),
                        help='Directory for server logs (default: ./logs)')
    parser.add_argument('--timeout',
                        type=int,
                        default=int(os.environ.get('PS_TIMEOUT', '60')),
                        help='Server startup timeout in seconds (default: 60)')
    parser.add_argument('unittest_args', nargs='*')
    
    args = parser.parse_args()
    
    sys.argv[1:] = args.unittest_args
    
    if args.no_server:
        env_type = "CI" if is_ci else "manual"
        print(f"[{env_type}] Running tests without starting ps_server (assuming already running)")
        unittest.main(verbosity=2)
    else:
        print(f"\n{'='*70}")
        print("Starting PS Server for EBC Precision Tests")
        print(f"Server path: {args.server_path}")
        print(f"Config: {args.config or 'default'}")
        print(f"Log dir: {args.log_dir}")
        print(f"Timeout: {args.timeout}s")
        print(f"{'='*70}\n")
        
        with ps_server_context(
            server_path=args.server_path,
            config_path=args.config,
            log_dir=args.log_dir,
            timeout=args.timeout,
            verbose=True
        ):
            unittest.main(verbosity=2)
