import importlib
import sys
import types
import unittest

import torch

from ..Dataset import RecStoreDataset


class _NoPrefetchClient:
    def __init__(self):
        self.prefetch_calls = 0

    def prefetch(self, ids: torch.Tensor) -> int:
        self.prefetch_calls += 1
        raise AssertionError("prefetch() should be disabled in strict-step mode")


class _FakeKJT:
    def __init__(self, values: torch.Tensor):
        self._values = values

    def values(self) -> torch.Tensor:
        return self._values


class _DummyEBC:
    pass


class _FailingLoader:
    def __iter__(self):
        return self

    def __next__(self):
        raise ValueError("boom")


class _StuckThread:
    def is_alive(self):
        return True

    def join(self, timeout=None):
        return None


def _install_torchrec_stub():
    original = {name: sys.modules.get(name) for name in (
        "torchrec",
        "torchrec.sparse",
        "torchrec.sparse.jagged_tensor",
    )}
    torchrec_mod = types.ModuleType("torchrec")
    sparse_mod = types.ModuleType("torchrec.sparse")
    jagged_mod = types.ModuleType("torchrec.sparse.jagged_tensor")
    jagged_mod.KeyedJaggedTensor = object
    sys.modules["torchrec"] = torchrec_mod
    sys.modules["torchrec.sparse"] = sparse_mod
    sys.modules["torchrec.sparse.jagged_tensor"] = jagged_mod
    return original


def _restore_modules(original):
    for name, module in original.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def _import_prefetching_iterator():
    original_modules = _install_torchrec_stub()
    sys.modules.pop("model_zoo.torchrec_dlrm.tests.prefetcher", None)
    module = importlib.import_module("model_zoo.torchrec_dlrm.tests.prefetcher")
    return module.PrefetchingIterator, original_modules


class TestPrefetchStrictMode(unittest.TestCase):
    def test_recstore_dataset_returns_empty_handles_without_prefetch(self):
        client = _NoPrefetchClient()
        batch = (
            torch.zeros(1),
            {"f1": _FakeKJT(torch.tensor([1, 2], dtype=torch.int64))},
            torch.zeros(1),
        )
        dataset = RecStoreDataset(
            [batch],
            client,
            key_extractor=lambda item: {"f1": item[1]["f1"].values()},
            prefetch_count=1,
        )
        try:
            _, handles = next(dataset)
            self.assertEqual(handles, {})
            self.assertEqual(client.prefetch_calls, 0)
            with self.assertRaises(StopIteration):
                next(dataset)
            with self.assertRaises(StopIteration):
                next(dataset)
        finally:
            dataset.stop(join=True)

    def test_recstore_dataset_surfaces_producer_errors(self):
        dataset = RecStoreDataset(
            _FailingLoader(),
            _NoPrefetchClient(),
            key_extractor=lambda item: {},
            prefetch_count=1,
        )
        try:
            with self.assertRaisesRegex(RuntimeError, "producer failed"):
                next(dataset)
        finally:
            try:
                dataset.stop(join=True)
            except RuntimeError:
                pass

    def test_recstore_dataset_restart_fails_if_old_thread_is_stuck(self):
        dataset = RecStoreDataset([], _NoPrefetchClient(), key_extractor=lambda item: {}, prefetch_count=1)
        try:
            dataset._thread = _StuckThread()
            with self.assertRaisesRegex(RuntimeError, "did not stop cleanly"):
                dataset.restart()
        finally:
            dataset._thread = None

    def test_prefetching_iterator_returns_empty_handles_without_prefetch(self):
        PrefetchingIterator, original_modules = _import_prefetching_iterator()
        try:
            client = _NoPrefetchClient()
            batch = (
                torch.zeros(1),
                {"f1": _FakeKJT(torch.tensor([1, 2], dtype=torch.int64))},
                torch.zeros(1),
            )
            iterator = PrefetchingIterator(
                [batch],
                _DummyEBC(),
                prefetch_count=1,
            )
            try:
                dense, sparse, labels, handles = next(iterator)
                self.assertTrue(torch.equal(dense, torch.zeros(1)))
                self.assertTrue(torch.equal(labels, torch.zeros(1)))
                self.assertEqual(handles, {})
                self.assertEqual(client.prefetch_calls, 0)
                self.assertEqual(sparse["f1"].values().tolist(), [1, 2])
                with self.assertRaises(StopIteration):
                    next(iterator)
                with self.assertRaises(StopIteration):
                    next(iterator)
            finally:
                iterator.stop(join=True)
        finally:
            _restore_modules(original_modules)
            sys.modules.pop("model_zoo.torchrec_dlrm.tests.prefetcher", None)

    def test_prefetching_iterator_surfaces_producer_errors(self):
        PrefetchingIterator, original_modules = _import_prefetching_iterator()
        try:
            iterator = PrefetchingIterator(_FailingLoader(), _DummyEBC(), prefetch_count=1)
            try:
                with self.assertRaisesRegex(RuntimeError, "producer failed"):
                    next(iterator)
            finally:
                try:
                    iterator.stop(join=True)
                except RuntimeError:
                    pass
        finally:
            _restore_modules(original_modules)
            sys.modules.pop("model_zoo.torchrec_dlrm.tests.prefetcher", None)

    def test_prefetching_iterator_restart_fails_if_old_thread_is_stuck(self):
        PrefetchingIterator, original_modules = _import_prefetching_iterator()
        try:
            iterator = PrefetchingIterator([], _DummyEBC(), prefetch_count=1)
            try:
                iterator._thread = _StuckThread()
                with self.assertRaisesRegex(RuntimeError, "did not stop cleanly"):
                    iterator.restart()
            finally:
                iterator._thread = None
        finally:
            _restore_modules(original_modules)
            sys.modules.pop("model_zoo.torchrec_dlrm.tests.prefetcher", None)
