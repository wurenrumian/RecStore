import unittest

import torch

from python.pytorch.torchrec_kv.EmbeddingBag import RecStoreEmbeddingBagCollection


class _RecordingClient:
    """Minimal kv_client stub: records how each table gets pre-created."""

    def __init__(self) -> None:
        self.registered: list[tuple[str, int, int]] = []
        self.init_data_calls: list[tuple[str, int, int]] = []
        self.pushes: list[tuple[str, int, int, int]] = []

    def register_tensor_meta(self, name, shape, dtype, base_offset=0, initialize_backend=False):
        self.registered.append((name, int(shape[0]), int(base_offset)))
        return None

    def init_data(self, name, shape, dtype, base_offset=0, initialize_values=True):
        self.init_data_calls.append((name, int(shape[0]), int(base_offset)))
        return None

    def push(self, name, ids, data):
        keys = ids if isinstance(ids, torch.Tensor) else torch.as_tensor(ids)
        self.pushes.append((name, int(keys.numel()), int(keys.min()), int(keys.max())))
        return None


def _collection(enable_fusion: bool, client) -> RecStoreEmbeddingBagCollection:
    return RecStoreEmbeddingBagCollection(
        [
            {
                "name": f"t_f{i}",
                "num_embeddings": size,
                "embedding_dim": 4,
                "feature_names": [f"f{i}"],
            }
            for i, size in enumerate((1000, 40, 7))
        ],
        enable_fusion=enable_fusion,
        fusion_k=30,
        kv_client=client,
    )


class FusedPreloadTest(unittest.TestCase):
    def test_fused_preload_writes_every_feature_into_the_master_key_space(self) -> None:
        client = _RecordingClient()
        _collection(enable_fusion=True, client=client)

        # The first table is pre-created through the ordinary path, the rest are
        # pushed into the master table at their fused offsets.
        self.assertEqual([("t_f0", 1000, 0)], client.init_data_calls)
        self.assertEqual(
            [("t_f1", 40, 1 << 30), ("t_f2", 7, 2 << 30)],
            client.registered,
        )
        self.assertEqual(
            [
                ("t_f0", 40, 1 << 30, (1 << 30) + 39),
                ("t_f0", 7, 2 << 30, (2 << 30) + 6),
            ],
            client.pushes,
        )

    def test_without_fusion_each_table_is_initialised_in_place(self) -> None:
        client = _RecordingClient()
        _collection(enable_fusion=False, client=client)

        self.assertEqual(
            [("t_f0", 1000, 0), ("t_f1", 40, 0), ("t_f2", 7, 0)],
            client.init_data_calls,
        )
        self.assertEqual([], client.pushes)


if __name__ == "__main__":
    unittest.main()
