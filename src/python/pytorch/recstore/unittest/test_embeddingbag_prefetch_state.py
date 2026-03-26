import importlib
import sys
import types
import unittest


class _FakeKeyedJaggedTensor:
    pass


class _FakeKeyedTensor:
    pass


class _FakeEmbeddingBagConfig:
    def __init__(self, **kwargs):
        self.name = kwargs["name"]
        self.embedding_dim = kwargs["embedding_dim"]
        self.num_embeddings = kwargs["num_embeddings"]
        self.feature_names = kwargs["feature_names"]


class _FakeKVClient:
    def init_data(self, **kwargs):
        return None


def _install_torchrec_stub():
    saved = {name: sys.modules.get(name) for name in [
        "torchrec",
        "torchrec.sparse",
        "torchrec.sparse.jagged_tensor",
        "torchrec.modules",
        "torchrec.modules.embedding_configs",
    ]}

    torchrec_mod = types.ModuleType("torchrec")
    sparse_mod = types.ModuleType("torchrec.sparse")
    jagged_mod = types.ModuleType("torchrec.sparse.jagged_tensor")
    modules_mod = types.ModuleType("torchrec.modules")
    embedding_configs_mod = types.ModuleType("torchrec.modules.embedding_configs")

    jagged_mod.KeyedJaggedTensor = _FakeKeyedJaggedTensor
    jagged_mod.KeyedTensor = _FakeKeyedTensor
    embedding_configs_mod.EmbeddingBagConfig = _FakeEmbeddingBagConfig

    sys.modules["torchrec"] = torchrec_mod
    sys.modules["torchrec.sparse"] = sparse_mod
    sys.modules["torchrec.sparse.jagged_tensor"] = jagged_mod
    sys.modules["torchrec.modules"] = modules_mod
    sys.modules["torchrec.modules.embedding_configs"] = embedding_configs_mod
    return saved


def _restore_modules(saved):
    for name, module in saved.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


class TestEmbeddingBagPrefetchState(unittest.TestCase):
    def setUp(self):
        self._saved_modules = _install_torchrec_stub()
        self.embeddingbag_module = importlib.import_module(
            "src.python.pytorch.torchrec_kv.EmbeddingBag"
        )
        self._original_get_kv_client = self.embeddingbag_module.get_kv_client
        self.embeddingbag_module.get_kv_client = lambda: _FakeKVClient()

    def tearDown(self):
        self.embeddingbag_module.get_kv_client = self._original_get_kv_client
        _restore_modules(self._saved_modules)

    def _build_ebc(self):
        return self.embeddingbag_module.RecStoreEmbeddingBagCollection(
            [
                {
                    "name": "t0",
                    "embedding_dim": 4,
                    "num_embeddings": 8,
                    "feature_names": ["f1"],
                }
            ],
            enable_fusion=True,
        )

    def test_empty_prefetch_handles_clear_stale_fused_state(self):
        ebc = self._build_ebc()

        ebc.set_fused_prefetch_handle(17, num_ids=3)
        self.assertEqual(ebc._fused_prefetch_handle, 17)

        ebc.set_prefetch_handles({})

        self.assertEqual(ebc._prefetch_handles, {})
        self.assertEqual(ebc._fused_prefetch_slots, [])
        self.assertIsNone(ebc._fused_prefetch_handle)
        self.assertIsNone(ebc._fused_ids_cpu)
        self.assertIsNone(ebc._fused_inverse)

    def test_fused_handle_replaces_per_feature_prefetch_state(self):
        ebc = self._build_ebc()

        ebc.set_prefetch_handles({"f1": 11})
        self.assertEqual(ebc._prefetch_handles, {"f1": 11})

        ebc.set_fused_prefetch_handle(23, num_ids=2)

        self.assertEqual(ebc._prefetch_handles, {})
        self.assertEqual(len(ebc._fused_prefetch_slots), 1)
        self.assertEqual(ebc._fused_prefetch_handle, 23)


if __name__ == "__main__":
    unittest.main()
