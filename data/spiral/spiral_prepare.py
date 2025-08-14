"""
Spiral script demonstrating how to prepare integer sequence data for GPT training.

This example creates synthetic integer sequences that simulate a simple mathematical pattern
and shows how to use the integer data preparation pipeline.

Run this example:
    python -m data.spiral.spiral_prepare
"""

import json
import numpy as np
import os
from pathlib import Path
from datetime import datetime

from data.integer_data import prepare_integer_dataset
from data.comp_mech import logit_sample_tokens
from data.spiral import spiral


def create_spiral_datasets(scale_array: np.ndarray, angle_array: np.ndarray, x_scale: float, block_size: int, num_tokens: int, seed: int = 42, format: str = "json") -> tuple[str, str, str, int, int, int]:
    """Create Spiral datasets in different formats.
    
    Args:
        scale_array: Scale array
        angle_array: Angle array
        x_scale: X scale
        block_size: Length of each sequence
        num_tokens: Total number of tokens to generate
        seed: Random seed for reproducibility
        format: Format to use ('json' or 'npy'), defaults to 'json'
    """
    
    # Calculate derived parameters
    vocab_size = 3 # Spiral + BOS token
    num_sequences = int(num_tokens // block_size)
    bos_token = vocab_size - 1  # Should be the largest reserved token
    
    # Set up HMM data source
    transition_matrix = spiral(scale_array=scale_array, angle_array=angle_array, x_scale=x_scale)
    initial_belief = np.array([8.0, 8.0]) 
    final_state = np.array([5.0, 5.0])

    # Sample sequences from the transition tensor
    sequences = logit_sample_tokens(transition_matrix=transition_matrix, 
                              initial_belief=initial_belief, 
                              final_state=final_state,
                              n_samples=num_sequences, 
                              n_tokens=block_size, 
                              seed=seed
                              )
    
    # Get output directory
    base_dir = os.path.dirname(__file__)

    # Add datetime tag to output name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"spiral_b_{block_size}_{timestamp}"
    
    # Create output directory based on format
    output_dir = os.path.join(base_dir, f"{output_name}_{format}_output")
    os.makedirs(output_dir, exist_ok=True)
    
    if format == "json":
        # Create Spiral JSON file in the output directory
        json_data = [{"sequence": seq.tolist()} for seq in sequences]
        data_file = os.path.join(output_dir, f"{output_name}_data.json")
        with open(data_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"Created Spiral JSON file: {data_file}")
    elif format == "npy":
        # Create Spiral numpy file in the output directory  
        npy_data = np.array(sequences)
        data_file = os.path.join(output_dir, f"{output_name}_data.npy")
        np.save(data_file, npy_data)
        print(f"Created Spiral NPY file: {data_file}")
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'json' or 'npy'.")
    
    # Create metadata JSON file
    metadata = {
        "scale_array": scale_array.tolist(),
        "angle_array": angle_array.tolist(),
        "x_scale": x_scale,
        "block_size": block_size,
        "num_tokens": num_tokens,
        "num_sequences": num_sequences,
        "vocab_size": vocab_size,
        "bos_token": bos_token,
        "seed": seed,
        "format": format,
        "timestamp": timestamp,
        "transition_matrix_shape": transition_matrix.shape,
        "initial_belief": initial_belief.tolist(),
        "final_state": final_state.tolist()
    }
    
    metadata_file = os.path.join(output_dir, f"{output_name}_metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Created metadata file: {metadata_file}")
    
    return data_file, output_dir, metadata_file, vocab_size, block_size, bos_token


def run_spiral():
    """Run the complete Spiral pipeline."""

    # Set parameters
    scale_array = np.array([1.15, 1.15])
    angle_array = np.array([np.pi/11, -np.pi/11])
    x_scale = 1.075
    block_size = 12  # Smaller for demonstration
    num_tokens = int(20 * 1e6)
    seed = 42
    
    print("=== Spiral Integer Data Preparation Example ===")
    print()
    
    # Set format (default to json)
    format = "json"  # Can be changed to "npy" if needed
    
    # Create Mess3 dataset
    data_file, output_dir, metadata_file, vocab_size, block_size, bos_token = create_spiral_datasets(
        scale_array=scale_array, 
        angle_array=angle_array, 
        x_scale=x_scale, 
        block_size=block_size, 
        num_tokens=num_tokens, 
        seed=seed,
        format=format
    )
    
    # Prepare the dataset
    print(f"Preparing Spiral {format.upper()} dataset...")
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
    print("=== Spiral Example Complete ===")
    print()
    print("The prepared datasets can now be used for training with configurations like:")
    print(f"  - vocab_size: {vocab_size}")
    print(f"  - block_size: {block_size}")
    print(f"  - Model config: integer_{vocab_size}_{block_size//4}x4 (or create custom)")
    print()


if __name__ == "__main__":
    run_spiral()