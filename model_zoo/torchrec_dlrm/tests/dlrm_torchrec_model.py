"""
DLRM (Deep Learning Recommendation Model) implementation using TorchRec.

This module defines the DLRM architecture with:
- Embedding tables for sparse features
- MLP for dense features
- Feature interaction layer
- Top MLP for final prediction
"""

import torch
import torch.nn as nn
from typing import List, Optional
from torchrec import EmbeddingBagCollection, KeyedJaggedTensor
from torchrec.modules.embedding_configs import EmbeddingBagConfig


class DenseArch(nn.Module):
    """MLP for processing dense features."""
    
    def __init__(
        self,
        in_features: int,
        layer_sizes: List[int],
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        layers = []
        for layer_size in layer_sizes:
            layers.append(nn.Linear(in_features, layer_size, device=device))
            layers.append(nn.ReLU())
            in_features = layer_size
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class InteractionArch(nn.Module):
    """Feature interaction layer using dot products."""
    
    def __init__(self, num_sparse_features: int):
        super().__init__()
        self.num_sparse_features = num_sparse_features
    
    def forward(
        self,
        dense_features: torch.Tensor,
        sparse_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute interactions between dense and sparse features.
        
        Args:
            dense_features: Output from bottom MLP, shape (batch_size, embedding_dim)
            sparse_features: Concatenated embeddings, shape (batch_size, num_sparse * embedding_dim)
        
        Returns:
            Concatenated interactions, shape (batch_size, num_interactions)
        """
        batch_size = dense_features.shape[0]
        embedding_dim = dense_features.shape[1]
        
        # Reshape sparse features to (batch_size, num_sparse_features, embedding_dim)
        sparse_features = sparse_features.view(batch_size, self.num_sparse_features, embedding_dim)
        
        # Concatenate dense feature as first feature
        # Shape: (batch_size, num_sparse_features + 1, embedding_dim)
        all_features = torch.cat([dense_features.unsqueeze(1), sparse_features], dim=1)
        
        # Compute dot product interactions
        # Shape: (batch_size, num_features, num_features)
        interactions = torch.bmm(all_features, all_features.transpose(1, 2))
        
        # Extract upper triangular part (including diagonal)
        batch_indices = []
        for i in range(batch_size):
            triu_indices = torch.triu_indices(
                interactions.shape[1], 
                interactions.shape[2],
                device=interactions.device
            )
            batch_indices.append(interactions[i][triu_indices[0], triu_indices[1]])
        
        # Stack all batches
        result = torch.stack(batch_indices, dim=0)
        return result


class OverArch(nn.Module):
    """Top MLP for final prediction."""
    
    def __init__(
        self,
        in_features: int,
        layer_sizes: List[int],
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        layers = []
        for layer_size in layer_sizes:
            layers.append(nn.Linear(in_features, layer_size, device=device))
            layers.append(nn.ReLU())
            in_features = layer_size
        # Final layer for binary classification
        layers.append(nn.Linear(in_features, 1, device=device))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class DLRM(nn.Module):
    """
    Deep Learning Recommendation Model (DLRM).
    
    Architecture:
    1. Sparse features -> Embedding tables
    2. Dense features -> Bottom MLP
    3. Feature interactions (dot products)
    4. Interactions -> Top MLP -> Prediction
    """
    
    def __init__(
        self,
        embedding_bag_collection: EmbeddingBagCollection,
        dense_in_features: int,
        dense_arch_layer_sizes: List[int],
        over_arch_layer_sizes: List[int],
        num_sparse_features: int,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize DLRM model.
        
        Args:
            embedding_bag_collection: TorchRec EmbeddingBagCollection for sparse features
            dense_in_features: Number of dense input features
            dense_arch_layer_sizes: Layer sizes for bottom MLP
            over_arch_layer_sizes: Layer sizes for top MLP
            num_sparse_features: Number of sparse features
            device: Device to place the model on
        """
        super().__init__()
        self.embedding_bag_collection = embedding_bag_collection
        self.num_sparse_features = num_sparse_features
        
        # Bottom MLP for dense features
        self.dense_arch = DenseArch(
            in_features=dense_in_features,
            layer_sizes=dense_arch_layer_sizes,
            device=device,
        )
        
        # Feature interaction layer
        self.interaction_arch = InteractionArch(num_sparse_features=num_sparse_features)
        
        # Calculate input size for top MLP
        # Number of interactions = (num_sparse_features + 1) * (num_sparse_features + 2) / 2
        num_interactions = (num_sparse_features + 1) * (num_sparse_features + 2) // 2
        
        # Top MLP for final prediction
        self.over_arch = OverArch(
            in_features=num_interactions,
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
        
        Args:
            dense_features: Dense input features, shape (batch_size, dense_in_features)
            sparse_features: Sparse features as KeyedJaggedTensor
        
        Returns:
            Predictions, shape (batch_size, 1)
        """
        # Process sparse features through embeddings
        embedded_sparse = self.embedding_bag_collection(sparse_features)
        
        # Concatenate all sparse embeddings
        sparse_values = torch.cat([embedded_sparse[key] for key in embedded_sparse.keys()], dim=1)
        
        # Process dense features through bottom MLP
        dense_embedded = self.dense_arch(dense_features)
        
        # Compute feature interactions
        interactions = self.interaction_arch(dense_embedded, sparse_values)
        
        # Final prediction through top MLP
        logits = self.over_arch(interactions)
        
        # Always print for debugging now
        print(f"DEBUG MODEL: DenseEmb Std: {dense_embedded.std().item()} SparseEmb Std: {sparse_values.std().item()} Interaction Std: {interactions.std().item()} Logits Std: {logits.std().item()}")
        print(f"DEBUG MODEL SHAPES: DenseEmb: {dense_embedded.shape} SparseValues: {sparse_values.shape} Interactions: {interactions.shape}")
        
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
    
    Args:
        num_embeddings_per_feature: List of vocabulary sizes for each sparse feature
        embedding_dim: Dimension of embeddings
        dense_in_features: Number of dense features
        dense_arch_layer_sizes: Layer sizes for bottom MLP (default: [512, 256, 64])
        over_arch_layer_sizes: Layer sizes for top MLP (default: [512, 256, 1])
        device: Device to place the model on
    
    Returns:
        DLRM model instance
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
