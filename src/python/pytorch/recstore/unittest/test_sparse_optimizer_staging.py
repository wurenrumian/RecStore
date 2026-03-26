import unittest

import torch

from ..optimizer import SparseSGD
from ._sparse_test_utils import KVClientIsolationMixin, build_features


class TestSparseOptimizerStaging(KVClientIsolationMixin, unittest.TestCase):
    def test_ebc_update_submits_unscaled_grads_for_backend_optimizer(self):
        ebc, fake = self.build_module()
        optimizer = SparseSGD([ebc], lr=0.1)
        features = build_features()

        optimizer.zero_grad()
        result = ebc(features)
        loss = result.values().sum()
        loss.backward()

        self.assertEqual(fake.update_calls, [])

        optimizer.step()
        optimizer.flush()

        self.assertEqual(len(fake.update_calls), 1)
        table_name, keys, grads = fake.update_calls[0]
        self.assertEqual(table_name, "t0")
        self.assertTrue(torch.equal(keys, torch.tensor([1, 3], dtype=torch.int64)))
        expected_grads = torch.tensor(
            [
                [2.0, 2.0, 2.0, 2.0],
                [1.0, 1.0, 1.0, 1.0],
            ],
            dtype=torch.float32,
        )
        self.assertTrue(torch.allclose(grads, expected_grads))

    def test_sparse_update_not_visible_before_flush(self):
        ebc, fake = self.build_module()
        optimizer = SparseSGD([ebc], lr=0.1)
        features = build_features()

        baseline = fake.read_rows(torch.tensor([1, 3], dtype=torch.int64))

        optimizer.zero_grad()
        result = ebc(features)
        loss = result.values().sum()
        loss.backward()

        optimizer.step()

        self.assertEqual(len(fake.update_calls), 0)
        self.assertEqual(fake.update_apply_count, 0)
        current = fake.read_rows(torch.tensor([1, 3], dtype=torch.int64))
        self.assertTrue(torch.allclose(current, baseline))

    def test_sparse_update_becomes_visible_after_flush(self):
        ebc, fake = self.build_module()
        optimizer = SparseSGD([ebc], lr=0.1)
        features = build_features()

        baseline = fake.read_rows(torch.tensor([1, 3], dtype=torch.int64))

        optimizer.zero_grad()
        result = ebc(features)
        loss = result.values().sum()
        loss.backward()

        optimizer.step()
        optimizer.flush()

        self.assertEqual(fake.update_apply_count, 1)
        self.assertEqual(len(fake.update_calls), 1)
        self.assertTrue(torch.equal(fake.last_applied_keys, torch.tensor([1, 3], dtype=torch.int64)))
        current = fake.read_rows(torch.tensor([1, 3], dtype=torch.int64))
        self.assertFalse(torch.allclose(current, baseline))


if __name__ == "__main__":
    unittest.main()
