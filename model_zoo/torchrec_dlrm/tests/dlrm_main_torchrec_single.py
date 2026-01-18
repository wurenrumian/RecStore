#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import itertools
import os
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator, List, Optional

import torch
import torchmetrics as metrics
from torch import distributed as dist
from torch.utils.data import DataLoader
from torchrec import EmbeddingBagCollection
from torchrec.datasets.criteo import DEFAULT_CAT_NAMES, DEFAULT_INT_NAMES
from torchrec.distributed.comm import get_local_size
from torchrec.distributed.model_parallel import (
    DistributedModelParallel,
    get_default_sharders,
)
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from tqdm import tqdm
from torch.profiler import profile, record_function, ProfilerActivity

# Set paths to import custom modules
RECSTORE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src'))
DLRM_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if RECSTORE_PATH not in sys.path:
    sys.path.insert(0, RECSTORE_PATH)
if DLRM_PATH not in sys.path:
    sys.path.insert(0, DLRM_PATH)



from dlrm_torchrec_model import create_dlrm_model, DLRM
import dlrm_torchrec_model
print(f"DEBUG: dlrm_torchrec_model imported from: {dlrm_torchrec_model.__file__}")

try:
    from data.custom_dataloader import get_dataloader
except ImportError:
    try:
        from ..data.custom_dataloader import get_dataloader
    except ImportError:
        print("Warning: Could not import custom dataloader modules")
        def get_dataloader(args, backend, stage):
            raise NotImplementedError("Please ensure custom dataloader modules are available")

def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="torchrec dlrm training with standard backend")
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="number of epochs to train",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="batch size to use for training",
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=None,
        help="batch size to use for validation and testing",
    )
    parser.add_argument(
        "--drop_last_training_batch",
        dest="drop_last_training_batch",
        action="store_true",
        help="Drop last non-full training batch",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=128,
        help="number of embedding dimensions",
    )
    parser.add_argument(
        "--dcn_num_layers",
        type=int,
        default=3,
        help="number of DCN layers (ignored in this impl, for compatibility)",
    )
    parser.add_argument(
        "--dcn_low_rank_dim",
        type=int,
        default=512,
        help="low rank dimension for DCN (ignored)",
    )
    parser.add_argument(
        "--over_arch_layer_sizes",
        type=str,
        default="1024,1024,512,256,1",
        help="comma separated layer sizes for over arch",
    )
    parser.add_argument(
        "--dense_arch_layer_sizes",
        type=str,
        default="512,256,128",
        help="comma separated layer sizes for dense arch",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        help="learning rate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="random seed for reproducibility",
    )
    parser.add_argument(
        "--pin_memory",
        dest="pin_memory",
        action="store_true",
        help="Use pinned memory for faster GPU transfer",
    )
    parser.add_argument(
        "--mmap_mode",
        dest="mmap_mode",
        action="store_true",
        help="Use memory mapping for loading data",
    )
    parser.add_argument(
        "--in_memory_binary_criteo_path",
        type=str,
        default=None,
        help="Path to preprocessed Criteo dataset",
    )
    parser.add_argument(
        "--validation_freq_within_epoch",
        type=int,
        default=None,
        help="Validation frequency within epoch",
    )
    parser.add_argument(
        "--adagrad",
        dest="adagrad",
        action="store_true",
        help="Use Adagrad optimizer",
    )
    parser.add_argument(
        "--single_day_mode",
        dest="single_day_mode",
        action="store_true",
        help="Enable single day data training mode",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Ratio of data to use for training in single day mode",
    )
    # Ignored args to maintain compatibility with run_single_day.sh calling convention
    parser.add_argument("--num_embeddings", type=int, default=None)
    parser.add_argument("--num_embeddings_per_feature", type=str, default=None)
    parser.add_argument("--interaction_type", type=str, default="original")
    
    args = parser.parse_args(argv)

    if args.single_day_mode:
        if args.in_memory_binary_criteo_path is None:
            raise ValueError("--in_memory_binary_criteo_path must be specified for single day mode")
        
        # Hardcoded embedding sizes for Day 0 (Criteo 1TB)
        # Consistent with dlrm_main_single_day.py
        if args.num_embeddings_per_feature is None:
            LIMIT_FEATURE = 100000
            orig_list = [
                40000000, 39060, 17295, 7424, 20265, 3, 7122, 1543, 63,
                40000000, 3067956, 405282, 10, 2209, 11938, 155, 4, 976,
                14, 40000000, 40000000, 40000000, 590152, 12973, 108, 36,
            ]
            clamped = [str(min(v, LIMIT_FEATURE)) for v in orig_list]
            args.num_embeddings_per_feature = ",".join(clamped)
        
        if not args.adagrad:
            args.learning_rate = 0.005
            args.adagrad = True
        
        print(f"Single day mode enabled. Training with day_0 data only.")
    
    return args

def main(argv: List[str]) -> None:
    args = parse_args(argv)
    
    if torch.cuda.is_available():
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
    else:
        dist.init_process_group(backend="gloo")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device("cpu")
    
    print(f"Distributed training initialized:")
    print(f"  Rank: {rank}")
    print(f"  World size: {world_size}")
    print(f"  Device: {device}")
    
    # Load data using existing loader logic
    train_dataloader = get_dataloader(args, "gloo", "train")
    val_dataloader = get_dataloader(args, "gloo", "val")
    
    def custom_collate(batch):
        if not batch:
            return batch
        
        dense_features = []
        sparse_features = []
        labels = []
        
        for dense, sparse, label in batch:
            dense_features.append(torch.as_tensor(dense, dtype=torch.float32))
            # sparse is a list of tensors or tensor of tensors? 
            # In custom_dataloader it seems to be list of tensors (one per feature).
            # But the original code handled it. 
            # In previous steps I saw sparse_features was list of tensors.
            # dlrm_main_single_day.py: sparse_features.append(sparse)
            sparse_features.append(sparse)
            labels.append(torch.as_tensor(label, dtype=torch.float32))
        
        dense_batch = torch.stack(dense_features)
        labels_batch = torch.stack(labels)

        from torchrec import KeyedJaggedTensor
        feature_names = [f"cat_{i}" for i in range(26)]
        
        # sparse_features is list of (26,) tensors (or list of lists).
        # Let's assume sparse_features is a list of `sparse` where `sparse` is (26,) tensor.
        sparse_mat = torch.stack([s.to(torch.long) for s in sparse_features], dim=0) # (B, 26)
        B = sparse_mat.shape[0]
        
        # Flatten values column by column (feature by feature) as KJT expects?
        # data_utils.py: 
        # for sparse_feature_list in sparse_features: (outer loop over features?)
        #   values.extend(...)
        # No, data_utils.py structures sparse_features as list of lists (features -> samples).
        # Here we have list of samples (samples -> features).
        
        # We need values grouped by feature (key).
        values = []
        lengths = []
        
        for i in range(26):
            # Get all values for feature i across batch
            feat_values = sparse_mat[:, i] # (B,)
            values.append(feat_values)
            lengths.extend([1] * B)
            
        values_tensor = torch.cat(values)
        lengths_tensor = torch.tensor(lengths, dtype=torch.int32, device=values_tensor.device)
        
        sparse_kjt = KeyedJaggedTensor(
            keys=feature_names,
            values=values_tensor,
            lengths=lengths_tensor,
        )
        
        return dense_batch, sparse_kjt, labels_batch
    
    train_loader = DataLoader(
        train_dataloader.dataset,
        batch_size=train_dataloader.batch_size,
        shuffle=True,
        drop_last=args.drop_last_training_batch,
        pin_memory=args.pin_memory,
        collate_fn=custom_collate,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataloader.dataset,
        batch_size=val_dataloader.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=args.pin_memory,
        collate_fn=custom_collate,
        num_workers=0
    )

    # Model Setup
    num_embeddings_per_feature = [int(x) for x in args.num_embeddings_per_feature.split(",")]
    
    # Use the implementation from dlrm_torchrec_model.py
    # Note: architecture sizes might need adjustment to match original if needed, 
    # but using defaults/command line args similar to original script.
    
    if args.dense_arch_layer_sizes:
        dense_sizes = [int(x) for x in args.dense_arch_layer_sizes.split(",")]
    else:
        dense_sizes = [512, 256, args.embedding_dim]

    if args.over_arch_layer_sizes:
        over_sizes = [int(x) for x in args.over_arch_layer_sizes.split(",")]
    else:
         over_sizes = [512, 256, 128]

    model = create_dlrm_model(
        num_embeddings_per_feature=num_embeddings_per_feature,
        embedding_dim=args.embedding_dim,
        dense_in_features=13,
        dense_arch_layer_sizes=dense_sizes,
        over_arch_layer_sizes=over_sizes,
        device=device,
    )

    # Wrap with DistributedModelParallel
    if world_size > 1 and torch.cuda.is_available():
        # Create sharding plan
        planner = EmbeddingShardingPlanner(
            topology=Topology(
                world_size=world_size,
                compute_device=device.type,
            )
        )
        
        # Shard the model
        model = DistributedModelParallel(
            module=model,
            device=device,
        )
    else:
        model = model.to(device)

    # Optimizer
    if args.adagrad:
        optimizer = torch.optim.Adagrad(model.parameters(), lr=args.learning_rate)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    
    criterion = torch.nn.BCEWithLogitsLoss()
    auroc = metrics.AUROC(task="binary").to(device)
    
    # Training Loop
    for epoch in range(args.epochs):
        model.train()
        
        train_loss = 0.0
        train_auroc = 0.0
        num_batches = 0
        forward_time_total = 0.0
        backward_time_total = 0.0
        opt_time_total = 0.0
        
        use_cuda_timing = torch.cuda.is_available() and device.type == 'cuda'
        if use_cuda_timing:
            fwd_start = torch.cuda.Event(enable_timing=True)
            fwd_end = torch.cuda.Event(enable_timing=True)
            bwd_start = torch.cuda.Event(enable_timing=True)
            bwd_end = torch.cuda.Event(enable_timing=True)
            opt_start = torch.cuda.Event(enable_timing=True)
            opt_end = torch.cuda.Event(enable_timing=True)
        
        print(f"Epoch {epoch + 1}/{args.epochs}")
        
        for batch_idx, (dense_features, sparse_features, labels) in enumerate(tqdm(train_loader, desc="Training")):
            dense_features = dense_features.to(device)
            sparse_features = sparse_features.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward
            if use_cuda_timing:
                fwd_start.record()
            else:
                t_fwd_start = time.time()
                
            if batch_idx == 0:
                 print(f"DEBUG: Batch {batch_idx} Input Stats:")
                 print(f"  Dense Mean: {dense_features.mean().item()} Std: {dense_features.std().item()}")
                 print(f"  Dense First 5: {dense_features[:5, 0].cpu().numpy()}") # First feature of first 5 samples
                 
                 # Check sparse values if possible, KJT is complex but we can check values()
                 print(f"  Sparse Values Mean: {sparse_features.values().float().mean().item()} Std: {sparse_features.values().float().std().item()}")
            
            outputs = model(dense_features, sparse_features)
            
            if batch_idx == 0:
                 probs = torch.sigmoid(outputs)
                 print(f"DEBUG: Batch {batch_idx} Output Stats (Logits):")
                 print(f"  Mean: {outputs.mean().item()}")
                 print(f"  Std: {outputs.std().item()}")
                 print(f"  Min: {outputs.min().item()}")
                 print(f"  Max: {outputs.max().item()}")
                 print(f"  First 5 probs: {probs[:5].flatten().detach().cpu().numpy()}")
                 print(f"  First 5 labels: {labels[:5].flatten().detach().cpu().numpy()}")
            
            loss = criterion(outputs, labels.float())
            
            if use_cuda_timing:
                fwd_end.record()
            else:
                t_fwd_end = time.time()
            
            # Backward
            if use_cuda_timing:
                bwd_start.record()
            else:
                t_bwd_start = time.time()
                
            loss.backward()

            if use_cuda_timing:
                bwd_end.record()
            else:
                t_bwd_end = time.time()

            # Optimizer Step
            if use_cuda_timing:
                opt_start.record()
            else:
                t_opt_start = time.time()

            optimizer.step()
            
            if use_cuda_timing:
                opt_end.record()
                torch.cuda.synchronize()
                fwd_ms = fwd_start.elapsed_time(fwd_end)
                bwd_ms = bwd_start.elapsed_time(bwd_end)
                opt_ms = opt_start.elapsed_time(opt_end)
            else:
                t_opt_end = time.time()
                fwd_ms = (t_fwd_end - t_fwd_start) * 1000.0
                bwd_ms = (t_bwd_end - t_bwd_start) * 1000.0
                opt_ms = (t_opt_end - t_opt_start) * 1000.0
            
            forward_time_total += fwd_ms
            backward_time_total += bwd_ms
            opt_time_total += opt_ms
            
            train_loss += loss.item()
            auroc_score = auroc(torch.sigmoid(outputs.squeeze()), labels)
            train_auroc += auroc_score.item()
            num_batches += 1
            
            if batch_idx % 1 == 0:
                 print(f"Batch {batch_idx}: Loss={loss.item():.4f} AUROC={auroc_score.item():.4f} FWD(ms)={fwd_ms:.2f} BWD(ms)={bwd_ms:.2f} OPT(ms)={opt_ms:.2f}")

        avg_train_loss = train_loss / num_batches if num_batches else 0.0
        avg_train_auroc = train_auroc / num_batches if num_batches else 0.0
        avg_fwd = forward_time_total / num_batches if num_batches else 0.0
        avg_bwd = backward_time_total / num_batches if num_batches else 0.0
        avg_opt = opt_time_total / num_batches if num_batches else 0.0
        
        print(f"Epoch {epoch + 1} - Training Loss: {avg_train_loss:.4f}, Training AUROC: {avg_train_auroc:.4f}, AvgFWD(ms): {avg_fwd:.2f}, AvgBWD(ms): {avg_bwd:.2f}, AvgOPT(ms): {avg_opt:.2f}")
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_auroc = 0.0
        val_num_batches = 0
        
        with torch.no_grad():
             for batch in tqdm(val_loader, desc="Validation"):
                dense_features = batch[0].to(device)
                sparse_features = batch[1].to(device)
                labels = batch[2].to(device)
                
                outputs = model(dense_features, sparse_features)
                loss = criterion(outputs, labels.float())
                
                val_loss += loss.item()
                auroc_score = auroc(outputs.squeeze(), labels)
                val_auroc += auroc_score.item()
                val_num_batches += 1
        
        avg_val_loss = val_loss / val_num_batches if val_num_batches else 0.0
        avg_val_auroc = val_auroc / val_num_batches if val_num_batches else 0.0
        
        print(f"Epoch {epoch + 1} - Validation Loss: {avg_val_loss:.4f}, Validation AUROC: {avg_val_auroc:.4f}")
        
    print("Training completed!")
    dist.destroy_process_group()

if __name__ == "__main__":
    main(sys.argv[1:])
