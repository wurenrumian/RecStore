import unittest
import os
import sys
import argparse
import torch
import importlib.util

RECSTORE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
if RECSTORE_PATH not in sys.path:
    sys.path.insert(0, RECSTORE_PATH)

TEST_MODULE_PATH = os.path.join(os.path.dirname(__file__), 'test_ebc_precision.py')
spec = importlib.util.spec_from_file_location("test_ebc_precision_module", TEST_MODULE_PATH)
test_ebc_precision = importlib.util.module_from_spec(spec)
spec.loader.exec_module(test_ebc_precision)


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
    # Run with verbose output
    unittest.main(verbosity=2)
