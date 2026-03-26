import unittest

import torch

from ..optimizer import SparseSGD
from ._sparse_test_utils import KVClientIsolationMixin, build_features


class TestStepConsistency(KVClientIsolationMixin, unittest.TestCase):
    def _run_sparse_step(self, optimizer: SparseSGD, ebc, features):
        optimizer.zero_grad()
        result = ebc(features)
        loss = result.values().sum()
        loss.backward()
        optimizer.step()

    def test_next_forward_after_flush_reads_updated_embeddings(self):
        ebc, fake = self.build_module()
        optimizer = SparseSGD([ebc], lr=0.1)
        features = build_features()

        baseline_forward = ebc(features).values().detach().clone()
        self._run_sparse_step(optimizer, ebc, features)

        next_step_before_flush = ebc(features).values().detach().clone()
        self.assertTrue(torch.allclose(next_step_before_flush, baseline_forward))

        optimizer.flush()

        next_step_after_flush = ebc(features).values().detach().clone()
        self.assertFalse(torch.allclose(next_step_after_flush, baseline_forward))
        self.assertTrue(torch.equal(fake.last_applied_keys, torch.tensor([1, 3], dtype=torch.int64)))


if __name__ == "__main__":
    unittest.main()
