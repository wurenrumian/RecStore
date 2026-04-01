from __future__ import annotations

import sys
from pathlib import Path

import torch


def inject_project_paths(repo_root: Path) -> None:
    recstore_src = str(repo_root / "src")
    dlrm_root = str(repo_root / "model_zoo/torchrec_dlrm")
    py_client = str(repo_root / "src/framework/pytorch/python_client")
    for p in (recstore_src, dlrm_root, py_client):
        if p not in sys.path:
            sys.path.insert(0, p)


def build_kjt_batch_from_dense_sparse_labels(
    dense_batch: torch.Tensor,
    sparse_batch: torch.Tensor,
    labels_batch: torch.Tensor,
):
    del labels_batch
    from torchrec.datasets.criteo import DEFAULT_CAT_NAMES
    from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

    sparse_mat = sparse_batch.to(torch.long)
    batch_size = sparse_mat.shape[0]
    values_list = [sparse_mat[:, i] for i in range(26)]
    values = torch.cat(values_list, dim=0)
    one_lengths = torch.ones(batch_size, dtype=torch.int32, device=values.device)
    lengths = torch.cat([one_lengths for _ in range(26)], dim=0)

    kjt = KeyedJaggedTensor.from_lengths_sync(
        keys=DEFAULT_CAT_NAMES,
        values=values,
        lengths=lengths,
    )
    return dense_batch, kjt


def build_batch_ids_from_kjt(sparse_features) -> torch.Tensor:
    ids_chunks = []
    for key in sparse_features.keys():
        ids_chunks.append(sparse_features[key].values().to(torch.int64))
    if not ids_chunks:
        return torch.empty((0,), dtype=torch.int64)
    return torch.cat(ids_chunks, dim=0).cpu().contiguous()


def build_table_offsets_from_eb_configs(eb_configs: list[dict], fusion_k: int) -> dict[str, int]:
    offsets: dict[str, int] = {}
    for idx, cfg in enumerate(eb_configs):
        offsets[cfg["feature_names"][0]] = idx << fusion_k
    return offsets


def convert_kjt_ids_to_fused_ids(sparse_features, table_offsets: dict[str, int]) -> torch.Tensor:
    ids_chunks = []
    for key in sparse_features.keys():
        vals = sparse_features[key].values().to(torch.int64).cpu()
        ids_chunks.append(vals + table_offsets[key])
    if not ids_chunks:
        return torch.empty((0,), dtype=torch.int64)
    return torch.cat(ids_chunks, dim=0).contiguous()


def build_train_dataloader(
    repo_root: Path,
    data_dir_rel: str,
    train_ratio: float,
    num_embeddings: int,
    batch_size: int,
):
    from data.custom_dataloader import CustomCriteoDataset  # type: ignore

    data_dir = (repo_root / data_dir_rel).resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"dataset dir not found: {data_dir}")

    nep = [int(num_embeddings)] * 26
    dataset = CustomCriteoDataset(
        data_dir=str(data_dir),
        stage="train",
        train_ratio=train_ratio,
        num_embeddings_per_feature=nep,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=False,
        num_workers=0,
    )
    return dataset, dataloader

