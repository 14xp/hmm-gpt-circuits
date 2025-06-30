"""
Template for preparing integer sequence datasets for GPT training.

This template handles data where:
1. Each token is naturally expressed as an integer
2. Each sequence starts with a BOS token followed by (block_size - 1) tokens
3. The vocabulary size may vary across different data sources
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
from datasets import Dataset

from data.tokenizers import IntegerTokenizer
from data.utils import save_dataset


def load_sequences_from_json(file_path: str) -> List[List[int]]:
    """Load integer sequences from JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    sequences = []
    for item in data:
        if isinstance(item, dict) and 'sequence' in item:
            sequences.append(item['sequence'])
        elif isinstance(item, list):
            sequences.append(item)
        else:
            raise ValueError(f"Unexpected data format: {type(item)}")
    
    return sequences


def load_sequences_from_csv(file_path: str) -> List[List[int]]:
    """Load integer sequences from CSV file (one sequence per row)."""
    df = pd.read_csv(file_path)
    sequences = []
    
    for _, row in df.iterrows():
        # Convert row to list, filtering out NaN values
        sequence = [int(x) for x in row.dropna().tolist()]
        sequences.append(sequence)
    
    return sequences


def load_sequences_from_npy(file_path: str) -> List[List[int]]:
    """Load integer sequences from numpy file."""
    data = np.load(file_path)
    if data.ndim == 1:
        # Single sequence
        return [data.tolist()]
    elif data.ndim == 2:
        # Multiple sequences
        return [row.tolist() for row in data]
    else:
        raise ValueError(f"Unsupported array shape: {data.shape}")


def validate_sequences(sequences: List[List[int]], vocab_size: int, block_size: int, bos_token: int = 0) -> List[List[int]]:
    """
    Validate and process sequences according to GPT requirements.
    
    Args:
        sequences: List of integer sequences
        vocab_size: Size of vocabulary
        block_size: Maximum sequence length
        bos_token: Beginning of sequence token (default: 0)
    
    Returns:
        List of validated sequences
    """
    valid_sequences = []
    
    for i, seq in enumerate(sequences):
        # Check if sequence is the right length
        if len(seq) != block_size:
            print(f"Warning: Sequence {i} has length {len(seq)}, expected {block_size}. Skipping.")
            continue
        
        # Check if first token is BOS
        if seq[0] != bos_token:
            print(f"Warning: Sequence {i} does not start with BOS token {bos_token}. Skipping.")
            continue
        
        # Validate all tokens are within vocabulary
        invalid_tokens = [token for token in seq if token < 0 or token >= vocab_size]
        if invalid_tokens:
            print(f"Warning: Sequence {i} contains invalid tokens {invalid_tokens}. Skipping.")
            continue
        
        valid_sequences.append(seq)
    
    print(f"Validated {len(valid_sequences)} out of {len(sequences)} sequences")
    return valid_sequences


def prepare_integer_dataset(
    input_file: str,
    vocab_size: int,
    block_size: int = 128,
    train_split: float = 0.9,
    bos_token: int = 0,
    num_shards: int = 1,
    output_dir: str = None
):
    """
    Prepare integer sequence dataset for GPT training.
    
    Args:
        input_file: Path to input file (JSON, CSV, or NPY)
        vocab_size: Size of vocabulary
        block_size: Maximum sequence length
        train_split: Fraction of data to use for training
        bos_token: Beginning of sequence token
        num_shards: Number of shards to split data into
        output_dir: Output directory (defaults to same directory as input file)
    """
    # Determine output directory
    if output_dir is None:
        output_dir = os.path.dirname(input_file)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load sequences based on file extension
    file_ext = Path(input_file).suffix.lower()
    
    if file_ext == '.json':
        sequences = load_sequences_from_json(input_file)
    elif file_ext == '.csv':
        sequences = load_sequences_from_csv(input_file)
    elif file_ext == '.npy':
        sequences = load_sequences_from_npy(input_file)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")
    
    print(f"Loaded {len(sequences)} sequences from {input_file}")
    
    # Validate sequences
    valid_sequences = validate_sequences(sequences, vocab_size, block_size, bos_token)
    
    if not valid_sequences:
        raise ValueError("No valid sequences found after validation")
    
    # Split into train and validation
    train_size = int(len(valid_sequences) * train_split)
    train_sequences = valid_sequences[:train_size]
    val_sequences = valid_sequences[train_size:]
    
    print(f"Split: {len(train_sequences)} train, {len(val_sequences)} validation")
    
    # Create datasets
    train_dataset = Dataset.from_list([{"ids": seq} for seq in train_sequences])
    val_dataset = Dataset.from_list([{"ids": seq} for seq in val_sequences])
    
    # Save datasets
    save_dataset(train_dataset, output_dir, "train", num_shards=num_shards)
    save_dataset(val_dataset, output_dir, "val", num_shards=1)
    
    print(f"Saved datasets to {output_dir}")
    
    # Create tokenizer info file
    tokenizer_info = {
        "tokenizer_type": "integer",
        "vocab_size": vocab_size,
        "block_size": block_size,
        "bos_token": bos_token
    }
    
    with open(os.path.join(output_dir, "tokenizer_info.json"), 'w') as f:
        json.dump(tokenizer_info, f, indent=2)
    
    print(f"Created tokenizer info file")