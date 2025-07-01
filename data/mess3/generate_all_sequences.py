#!/usr/bin/env python3
"""
Generate all possible length-N sequences from the mess3 process with BOS token.

This script creates a dataset consisting of all possible sequences of format [3, *, *, ..., *]
where * ∈ {0, 1, 2} and 3 is the BOS token. This is independent of mess3 parameters x and a
since it enumerates all possible sequences combinatorially rather than sampling from the HMM.

For a sequence of length N, this generates 3^(N-1) sequences total.

Usage:
    python -m data.mess3.generate_all_sequences --block_size=10 --format=json
    python -m data.mess3.generate_all_sequences --block_size=8 --format=npy
"""

import argparse
import itertools
import json
import numpy as np
import os
from pathlib import Path
from typing import List

from data.integer_data import prepare_integer_dataset


def generate_all_sequences(block_size: int) -> List[List[int]]:
    """
    Generate all possible sequences of length block_size with BOS and EOS tokens.
    Format: [3, *, *, *, ..., 3] where * ∈ {0, 1, 2}
    
    Args:
        block_size: Total length of each sequence including BOS and EOS tokens
        
    Returns:
        List of sequences, each of length block_size
        Total sequences returned: 3^(block_size-2)
    """
    print(f"Generating all possible sequences of length {block_size}...")
    sequences = []
    
    # Generate all combinations of (block_size-2) positions with values {0, 1, 2}
    for combination in itertools.product([0, 1, 2], repeat=block_size-2):
        sequence = [3] + list(combination) + [3]  # BOS token 3, combination, EOS token 3
        sequences.append(sequence)
    
    expected_count = 3 ** (block_size - 2)
    print(f"Generated {len(sequences)} sequences (expected: {expected_count})")
    
    # Validate count
    assert len(sequences) == expected_count, f"Expected {expected_count} sequences, got {len(sequences)}"
    
    return sequences


def save_json_format(sequences: List[List[int]], output_dir: str, filename_base: str) -> str:
    """
    Save sequences in JSON format compatible with existing mess3 pipeline.
    
    Args:
        sequences: List of sequence lists
        output_dir: Directory to save the file
        filename_base: Base filename (without extension)
        
    Returns:
        Path to saved JSON file
    """
    json_data = [{"sequence": seq} for seq in sequences]
    json_file = os.path.join(output_dir, f"{filename_base}_data.json")
    
    with open(json_file, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"Saved JSON format: {json_file}")
    return json_file


def save_npy_format(sequences: List[List[int]], output_dir: str, filename_base: str) -> str:
    """
    Save sequences in numpy format.
    
    Args:
        sequences: List of sequence lists
        output_dir: Directory to save the file
        filename_base: Base filename (without extension)
        
    Returns:
        Path to saved numpy file
    """
    npy_data = np.array(sequences)
    npy_file = os.path.join(output_dir, f"{filename_base}_data.npy")
    
    np.save(npy_file, npy_data)
    print(f"Saved numpy format: {npy_file}")
    return npy_file


def create_tokenizer_info(output_dir: str, vocab_size: int, block_size: int, bos_token: int):
    """Create tokenizer info file for compatibility with existing pipeline."""
    tokenizer_info = {
        "vocab_size": vocab_size,
        "block_size": block_size,
        "bos_token": bos_token,
        "token_mapping": {
            "0": 0,
            "1": 1, 
            "2": 2,
            "BOS": 3
        },
        "description": "All possible sequences enumeration - no actual tokenizer used"
    }
    
    tokenizer_file = os.path.join(output_dir, "tokenizer_info.json")
    with open(tokenizer_file, 'w') as f:
        json.dump(tokenizer_info, f, indent=2)
    
    print(f"Created tokenizer info: {tokenizer_file}")


def create_all_sequences_dataset(block_size: int, format: str = "json") -> tuple[str, str, int, int, int]:
    """
    Create dataset with all possible sequences of given length.
    
    Args:
        block_size: Length of each sequence
        format: Output format ('json' or 'npy')
        
    Returns:
        Tuple of (data_file_path, output_dir, vocab_size, block_size, bos_token)
    """
    # Fixed parameters for mess3 all-sequences
    vocab_size = 4  # Tokens 0, 1, 2, 3 (where 3 is BOS)
    bos_token = 3
    
    # Generate all sequences
    sequences = generate_all_sequences(block_size)
    
    # Create output directory
    base_dir = os.path.dirname(__file__)
    output_name = f"all_sequences_n_{block_size}"
    output_dir = os.path.join(base_dir, f"{output_name}_{format}_output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save in requested format
    if format == "json":
        data_file = save_json_format(sequences, output_dir, output_name)
    elif format == "npy":
        data_file = save_npy_format(sequences, output_dir, output_name)
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'json' or 'npy'.")
    
    # Create tokenizer info for compatibility
    create_tokenizer_info(output_dir, vocab_size, block_size, bos_token)
    
    return data_file, output_dir, vocab_size, block_size, bos_token


def main():
    parser = argparse.ArgumentParser(description="Generate all possible sequences dataset")
    parser.add_argument("--block_size", type=int, required=True,
                       help="Length of each sequence (including BOS token)")
    parser.add_argument("--format", choices=["json", "npy"], default="json",
                       help="Output format (default: json)")
    parser.add_argument("--prepare", action="store_true",
                       help="Also prepare the dataset for training (create train/val splits)")
    parser.add_argument("--train_split", type=float, default=0.8,
                       help="Training split ratio (default: 0.8)")
    
    args = parser.parse_args()
    
    print("=== All Sequences Dataset Generation ===")
    print(f"Block size: {args.block_size}")
    print(f"Format: {args.format}")
    print(f"Expected sequences: {3 ** (args.block_size - 2)}")
    print()
    
    # Create dataset
    data_file, output_dir, vocab_size, block_size, bos_token = create_all_sequences_dataset(
        block_size=args.block_size,
        format=args.format
    )
    
    # Optionally prepare for training
    if args.prepare:
        print(f"\nPreparing dataset for training...")
        prepare_integer_dataset(
            input_file=data_file,
            vocab_size=vocab_size,
            block_size=block_size,
            train_split=args.train_split,
            bos_token=bos_token,
            num_shards=1,
            output_dir=output_dir
        )
        print("Dataset preparation complete!")
    
    print(f"\n=== Dataset Generation Complete ===")
    print(f"Output directory: {output_dir}")
    print(f"Data file: {data_file}")
    print(f"Vocab size: {vocab_size}")
    print(f"Block size: {block_size}")
    print(f"Total sequences: {3 ** (block_size - 1)}")
    
    if args.prepare:
        print(f"\nReady for training with:")
        print(f"  --data_dir={output_dir}")
        print(f"  --vocab_size={vocab_size}")
        print(f"  --block_size={block_size}")


if __name__ == "__main__":
    main()