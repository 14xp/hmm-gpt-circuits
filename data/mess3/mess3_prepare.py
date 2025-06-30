"""
Mess3 script demonstrating how to prepare integer sequence data for GPT training.

This example creates synthetic integer sequences that simulate a simple mathematical pattern
and shows how to use the integer data preparation pipeline.

Run this example:
    python -m data.mess3.mess3_prepare
"""

import json
import numpy as np
import os
from pathlib import Path

from data.integer_data import prepare_integer_dataset
from data.comp_mech import sample_tokens
from data.mess3 import mess3


def create_mess3_datasets(x: float, a: float, block_size: int, num_tokens: int, seed: int = 42, format: str = "json") -> tuple[str, str, int, int, int]:
    """Create Mess3 datasets in different formats.
    
    Args:
        x: Mess3 parameter x
        a: Mess3 parameter a  
        block_size: Length of each sequence
        num_tokens: Total number of tokens to generate
        seed: Random seed for reproducibility
        format: Format to use ('json' or 'npy'), defaults to 'json'
    """
    
    # Calculate derived parameters
    vocab_size = 4 # Mess3 + BOS token
    num_sequences = int(num_tokens // block_size)
    bos_token = vocab_size - 1  # Should be the largest reserved token
    
    # Set up HMM data source
    transition_tensor = mess3(x, a)
    initial_belief = np.array([1/3, 1/3, 1/3]) 

    # Sample sequences from the transition tensor
    sequences = sample_tokens(transition_matrix=transition_tensor, 
                              initial_belief=initial_belief, 
                              n_samples=num_sequences, 
                              n_tokens=block_size, 
                              seed=seed
                              )
    
    # Get output directory
    base_dir = os.path.dirname(__file__)
    x_str = str(x).replace('0.', '')
    a_str = str(a).replace('0.', '')
    output_name = f"mess3_x_{x_str}_a_{a_str}_b_{block_size}"
    
    # Create output directory based on format
    output_dir = os.path.join(base_dir, f"{output_name}_{format}_output")
    os.makedirs(output_dir, exist_ok=True)
    
    if format == "json":
        # Create Mess3 JSON file in the output directory
        json_data = [{"sequence": seq.tolist()} for seq in sequences]
        data_file = os.path.join(output_dir, f"{output_name}_data.json")
        with open(data_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"Created Mess3 JSON file: {data_file}")
    elif format == "npy":
        # Create Mess3 numpy file in the output directory  
        npy_data = np.array(sequences)
        data_file = os.path.join(output_dir, f"{output_name}_data.npy")
        np.save(data_file, npy_data)
        print(f"Created Mess3 NPY file: {data_file}")
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'json' or 'npy'.")
    
    return data_file, output_dir, vocab_size, block_size, bos_token


def run_mess3():
    """Run the complete Mess3 pipeline."""

    # Set parameters
    x = 0.15  
    a = 0.6
    block_size = 12  # Smaller for demonstration
    num_tokens = int(20 * 1e6)
    seed = 42

    x_str = str(x).replace('0.', '')
    a_str = str(a).replace('0.', '')
    output_name = f"mess3_x_{x_str}_a_{a_str}_b_{block_size}"
    
    print("=== Mess3 Integer Data Preparation Example ===")
    print()
    
    # Set format (default to json)
    format = "json"  # Can be changed to "npy" if needed
    
    # Create Mess3 dataset
    data_file, output_dir, vocab_size, block_size, bos_token = create_mess3_datasets(
        x=x, 
        a=a, 
        block_size=block_size, 
        num_tokens=num_tokens, 
        seed=seed,
        format=format
    )
    
    # Prepare the dataset
    print(f"Preparing Mess3 {format.upper()} dataset...")
    prepare_integer_dataset(
        input_file=data_file,
        vocab_size=vocab_size,
        block_size=block_size,
        train_split=0.8,
        bos_token=bos_token,
        num_shards=1,
        output_dir=output_dir
    )
    
    print()
    print("=== Mess3 Example Complete ===")
    print()
    print("The prepared datasets can now be used for training with configurations like:")
    print(f"  - vocab_size: {vocab_size}")
    print(f"  - block_size: {block_size}")
    print(f"  - Model config: integer_{vocab_size}_{block_size//4}x4 (or create custom)")
    print()
    print("To train a model:")
    print(f"  python -m training.gpt --config=<custom_config> --data_dir=data/mess3/{output_name}_json_output")


if __name__ == "__main__":
    run_mess3()