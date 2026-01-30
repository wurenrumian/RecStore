"""
DLRM (Deep Learning Recommendation Model) implementation using TorchRec.
This module now mirrors the implementation in `dlrm.py` (RecStore) to ensure
identical Neural Network computation characteristics for benchmarking.
"""

import torch
import torch.nn as nn
import time
from typing import List, Optional, Dict
from torchrec import EmbeddingBagCollection, KeyedJaggedTensor
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.mlp import MLP
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor, KeyedTensor

def choose(n: int, k: int) -> int:
    """
    Simple implementation of math.comb for Python 3.7 compatibility.
    """
    if 0 <= k <= n:
        ntok = 1
        ktok = 1
        for t in range(1, min(k, n - k) + 1):
            ntok *= n
            ktok *= t
            n -= 1
        return ntok // ktok
    else:
        return 0


class SparseArch(nn.Module):
    """
    Processes the sparse features of DLRM. Does embedding lookups for all EmbeddingBag
    and embedding features of each collection.
    """
    def __init__(self, embedding_bag_collection: EmbeddingBagCollection) -> None:
        super().__init__()
        self.embedding_bag_collection: EmbeddingBagCollection = embedding_bag_collection
        assert (
            self.embedding_bag_collection.embedding_bag_configs
        ), "Embedding bag collection cannot be empty!"
        self.D: int = self.embedding_bag_collection.embedding_bag_configs()[
            0
        ].embedding_dim
        self._sparse_feature_names: List[str] = [
            name
            for conf in embedding_bag_collection.embedding_bag_configs()
            for name in conf.feature_names
        ]

        self.F: int = len(self._sparse_feature_names)

    def forward(
        self,
        features: KeyedJaggedTensor,
    ) -> torch.Tensor:
        """
        Args:
            features (KeyedJaggedTensor): an input tensor of sparse features.

        Returns:
            torch.Tensor: tensor of shape B X F X D.
        """
        # This mirrors RecStore's implementation
        sparse_features: KeyedTensor = self.embedding_bag_collection(features)

        sparse: Dict[str, torch.Tensor] = sparse_features.to_dict()
        sparse_values: List[torch.Tensor] = []
        for name in self.sparse_feature_names:
            sparse_values.append(sparse[name])

        return torch.cat(sparse_values, dim=1).reshape(-1, self.F, self.D)

    @property
    def sparse_feature_names(self) -> List[str]:
        return self._sparse_feature_names


class DenseArch(nn.Module):
    """
    Processes the dense features of DLRM model.
    """
    def __init__(
        self,
        in_features: int,
        layer_sizes: List[int],
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.model: nn.Module = MLP(
            in_features, layer_sizes, bias=True, activation="relu", device=device
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.model(features)


class InteractionArch(nn.Module):
    """
    Processes the output of both `SparseArch` (sparse_features) and `DenseArch`
    (dense_features). Returns the pairwise dot product of each sparse feature pair,
    the dot product of each sparse features with the output of the dense layer,
    and the dense layer itself (all concatenated).
    """
    def __init__(self, num_sparse_features: int) -> None:
        super().__init__()
        self.F: int = num_sparse_features
        self.register_buffer(
            "triu_indices",
            torch.triu_indices(self.F + 1, self.F + 1, offset=1),
            persistent=False,
        )

    def forward(
        self, dense_features: torch.Tensor, sparse_features: torch.Tensor
    ) -> torch.Tensor:
        if self.F <= 0:
            return dense_features
        (B, D) = dense_features.shape

        combined_values = torch.cat(
            (dense_features.unsqueeze(1), sparse_features), dim=1
        )

        # dense/sparse + sparse/sparse interaction
        # size B X (F + F choose 2)
        interactions = torch.bmm(
            combined_values, torch.transpose(combined_values, 1, 2)
        )
        interactions_flat = interactions[:, self.triu_indices[0], self.triu_indices[1]]

        return torch.cat((dense_features, interactions_flat), dim=1)


class OverArch(nn.Module):
    """
    Final Arch of DLRM - simple MLP over OverArch.
    """
    def __init__(
        self,
        in_features: int,
        layer_sizes: List[int],
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        if len(layer_sizes) <= 1:
            raise ValueError("OverArch must have multiple layers.")
        self.model: nn.Module = nn.Sequential(
            MLP(
                in_features,
                layer_sizes[:-1],
                bias=True,
                activation="relu",
                device=device,
            ),
            nn.Linear(layer_sizes[-2], layer_sizes[-1], bias=True, device=device),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.model(features)


class DLRM(nn.Module):
    """
    Deep Learning Recommendation Model (DLRM).
    """
    def __init__(
        self,
        embedding_bag_collection: EmbeddingBagCollection,
        dense_in_features: int,
        dense_arch_layer_sizes: List[int],
        over_arch_layer_sizes: List[int],
        num_sparse_features: int,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        assert (
            len(embedding_bag_collection.embedding_bag_configs()) > 0
        ), "At least one embedding bag is required"
        for i in range(1, len(embedding_bag_collection.embedding_bag_configs())):
            conf_prev = embedding_bag_collection.embedding_bag_configs()[i - 1]
            conf = embedding_bag_collection.embedding_bag_configs()[i]
            assert (
                conf_prev.embedding_dim == conf.embedding_dim
            ), "All EmbeddingBagConfigs must have the same dimension"
        
        embedding_dim: int = embedding_bag_collection.embedding_bag_configs()[
            0
        ].embedding_dim
        
        if dense_arch_layer_sizes[-1] != embedding_dim:
            raise ValueError(
                f"embedding_bag_collection dimension ({embedding_dim}) and final dense "
                "arch layer size ({dense_arch_layer_sizes[-1]}) must match."
            )

        self.sparse_arch = SparseArch(embedding_bag_collection)
        self.num_sparse_features = len(self.sparse_arch.sparse_feature_names)

        self.dense_arch = DenseArch(
            in_features=dense_in_features,
            layer_sizes=dense_arch_layer_sizes,
            device=device,
        )

        self.inter_arch = InteractionArch(
            num_sparse_features=self.num_sparse_features,
        )

        over_in_features: int = (
            embedding_dim + choose(self.num_sparse_features, 2) + self.num_sparse_features
        )

        self.over_arch = OverArch(
            in_features=over_in_features,
            layer_sizes=over_arch_layer_sizes,
            device=device,
        )

    def forward(
        self,
        dense_features: torch.Tensor,
        sparse_features: KeyedJaggedTensor,
    ) -> torch.Tensor:
        """
        Forward pass of DLRM.
        """
        if dense_features.device.type == "cuda":
            # Event-based timing for CUDA
            if not hasattr(self, "events"):
                 self.events = {
                     "emb_start": torch.cuda.Event(enable_timing=True),
                     "emb_end": torch.cuda.Event(enable_timing=True),
                     "dense_start": torch.cuda.Event(enable_timing=True),
                     "dense_end": torch.cuda.Event(enable_timing=True),
                     "dense2_start": torch.cuda.Event(enable_timing=True),
                     "dense2_end": torch.cuda.Event(enable_timing=True)
                 }

            # Dense computation 1 (Bottom MLP)
            if not hasattr(self, "_device_checked"):
                print(f"DEBUG: DLRM Forward Device Check")
                print(f"  dense_features device: {dense_features.device}")
                try:
                    # Access first parameter of dense_arch
                    print(f"  dense_arch.model param device: {next(self.dense_arch.parameters()).device}")
                except:
                    print(f"  dense_arch.model param device: UNKNOWN")
                self._device_checked = True

            self.events["dense_start"].record()
            embedded_dense = self.dense_arch(dense_features)
            self.events["dense_end"].record() 
            
            # Start Sparse Timer (Pull)
            self.events["emb_start"].record()
            embedded_sparse = self.sparse_arch(sparse_features)
            self.events["emb_end"].record()
            
            # Interaction + Top MLP
            self.events["dense2_start"].record()
            concatenated_dense = self.inter_arch(
                dense_features=embedded_dense, sparse_features=embedded_sparse
            )
            logits = self.over_arch(concatenated_dense)
            self.events["dense2_end"].record()
            
            # Store timings for retrieval
            self.timings = self.events
            return logits
            
        else:
            # CPU Timing
            t0 = time.time()
            embedded_dense = self.dense_arch(dense_features)
            t1 = time.time()
            
            embedded_sparse = self.sparse_arch(sparse_features)
            t2 = time.time()
            
            # Need to get sparse_values for InteractionArch in CPU mode as per original logic?
            # Original dlrm.py SparseArch returns (B, F, D).
            # InteractionArch expects (B, F, D) as sparse_features argument.
            
            concatenated_dense = self.inter_arch(
                dense_features=embedded_dense, sparse_features=embedded_sparse
            )
            logits = self.over_arch(concatenated_dense)
            t3 = time.time()
            
            if not hasattr(self, "timings_cpu"):
                self.timings_cpu = {}
            self.timings_cpu = {
                "dense_ms": (t1 - t0 + t3 - t2) * 1000,
                "sparse_ms": (t2 - t1) * 1000
            }
            return logits


def create_dlrm_model(
    num_embeddings_per_feature: List[int],
    embedding_dim: int = 64,
    dense_in_features: int = 13,
    dense_arch_layer_sizes: Optional[List[int]] = None,
    over_arch_layer_sizes: Optional[List[int]] = None,
    device: Optional[torch.device] = None,
) -> DLRM:
    """
    Factory function to create a DLRM model.
    """
    if dense_arch_layer_sizes is None:
        dense_arch_layer_sizes = [512, 256, embedding_dim]
    
    if over_arch_layer_sizes is None:
        over_arch_layer_sizes = [512, 256, 128]
    
    # Create embedding bag configs
    embedding_bag_configs = [
        EmbeddingBagConfig(
            name=f"t_cat_{i}",
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
            feature_names=[f"cat_{i}"],
        )
        for i, num_embeddings in enumerate(num_embeddings_per_feature)
    ]
    
    # Create embedding bag collection
    embedding_bag_collection = EmbeddingBagCollection(
        tables=embedding_bag_configs,
        device=device,
    )
    
    # Create DLRM model
    model = DLRM(
        embedding_bag_collection=embedding_bag_collection,
        dense_in_features=dense_in_features,
        dense_arch_layer_sizes=dense_arch_layer_sizes,
        over_arch_layer_sizes=over_arch_layer_sizes,
        num_sparse_features=len(num_embeddings_per_feature),
        device=device,
    )
    
    return model
