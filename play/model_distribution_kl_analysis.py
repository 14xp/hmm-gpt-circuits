#!/usr/bin/env python3
"""
Script to compare model probability distributions with theoretical ground truth distributions.
Computes KL divergence between model predictions and spiral theory for next token prediction.
"""

import sys
import os
import json
from typing import List, Dict, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from models.gpt import GPT
from data.comp_mech import belief_to_token_logits, logit_belief_update, softmax
from data.spiral import spiral


def load_validation_data(data_path: str) -> Tuple[List[List[int]], Dict]:
    """
    Load validation sequences and metadata from JSON files.
    
    Args:
        data_path: Path to the directory containing data and metadata JSON files
        
    Returns:
        sequences: List of token sequences
        metadata: Dictionary containing spiral parameters
    """
    print("Loading validation data...")
    
    # Load sequences data
    data_file = None
    metadata_file = None
    
    for file in os.listdir(data_path):
        if file.endswith('_data.json'):
            data_file = os.path.join(data_path, file)
        elif file.endswith('_metadata.json'):
            metadata_file = os.path.join(data_path, file)
    
    if data_file is None or metadata_file is None:
        raise FileNotFoundError("Could not find data.json or metadata.json files")
    
    # Load sequences
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    sequences = [item['sequence'] for item in data]
    
    # Load metadata
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    print(f"Loaded {len(sequences)} validation sequences")
    print(f"Sequence length: {len(sequences[0])}")
    print(f"Vocabulary size: {metadata['vocab_size']}")
    print(f"Example sequence: {sequences[0]}")
    
    return sequences, metadata


def compute_theoretical_distributions(sequences: List[List[int]], metadata: Dict) -> np.ndarray:
    """
    Compute theoretical next-token distributions using spiral dynamics.
    
    Args:
        sequences: List of token sequences
        metadata: Metadata containing spiral parameters
        
    Returns:
        distributions: Array of shape (num_sequences, seq_len-1, vocab_size)
                      containing theoretical next-token distributions
    """
    print("Computing theoretical distributions...")
    
    # Extract spiral parameters from metadata
    scale_array = np.array(metadata['scale_array'])
    angle_array = np.array(metadata['angle_array'])
    x_scale = metadata['x_scale']
    initial_belief = np.array(metadata['initial_belief'])
    final_state = np.array(metadata['final_state'])
    vocab_size = metadata['vocab_size']
    
    # Create spiral transition matrix
    transition_matrix = spiral(scale_array, angle_array, x_scale)
    
    all_distributions = []
    
    for seq_idx, sequence in enumerate(sequences):
        if seq_idx % 1000 == 0:
            print(f"Processing sequence {seq_idx}/{len(sequences)}")
        
        sequence_distributions = []
        current_belief = initial_belief.copy()
        
        # Process each position in the sequence (except the last, which has no next token)
        for pos in range(len(sequence) - 1):
            # For position 0 (after BOS), use initial belief
            # For subsequent positions, update belief with the observed token
            if pos > 0:
                observed_token = sequence[pos]  # Current token (not previous)
                # Only update belief if token is in transition matrix range (0, 1)
                if observed_token < transition_matrix.shape[0]:
                    current_belief = logit_belief_update(transition_matrix, observed_token, current_belief)
            
            # Compute theoretical next-token distribution
            next_token_logits = belief_to_token_logits(transition_matrix, current_belief, final_state)
            next_token_dist = softmax(next_token_logits)
            
            # Pad distribution to match vocabulary size if needed (for BOS token)
            if len(next_token_dist) < vocab_size:
                # Add zero probability for BOS token
                padded_dist = np.zeros(vocab_size)
                padded_dist[:len(next_token_dist)] = next_token_dist
                next_token_dist = padded_dist
            
            sequence_distributions.append(next_token_dist)
        
        all_distributions.append(sequence_distributions)
    
    distributions_array = np.array(all_distributions)
    print(f"Computed theoretical distributions shape: {distributions_array.shape}")
    
    return distributions_array


def compute_model_distributions(model: GPT, sequences: List[List[int]], device: torch.device) -> np.ndarray:
    """
    Compute model probability distributions for next token prediction.
    
    Args:
        model: Trained GPT model
        sequences: List of token sequences
        device: Device to run model on
        
    Returns:
        distributions: Array of shape (num_sequences, seq_len-1, vocab_size)
                      containing model next-token distributions
    """
    print("Computing model distributions...")
    
    model.eval()
    batch_size = 100  # Process in batches to manage memory
    num_batches = (len(sequences) + batch_size - 1) // batch_size
    
    all_distributions = []
    
    with torch.no_grad():
        for batch_idx in range(num_batches):
            if batch_idx % 10 == 0:
                print(f"Processing batch {batch_idx}/{num_batches}")
            
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(sequences))
            batch_sequences = sequences[start_idx:end_idx]
            
            # Convert to tensor
            batch_tensor = torch.tensor(batch_sequences, device=device)  # (batch_size, seq_len)
            
            # Forward pass to get logits
            output = model(batch_tensor)
            logits = output.logits  # (batch_size, seq_len, vocab_size)
            
            # Apply softmax to get distributions
            distributions = F.softmax(logits, dim=-1)  # (batch_size, seq_len, vocab_size)
            
            # Extract next-token distributions (exclude the last position)
            next_token_dists = distributions[:, :-1, :].cpu().numpy()  # (batch_size, seq_len-1, vocab_size)
            
            all_distributions.extend(next_token_dists)
    
    distributions_array = np.array(all_distributions)
    print(f"Computed model distributions shape: {distributions_array.shape}")
    
    return distributions_array


def compute_kl_divergences(theoretical_dists: np.ndarray, model_dists: np.ndarray, 
                          epsilon: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute KL divergences between theoretical and model distributions.
    
    Args:
        theoretical_dists: Theoretical distributions (num_sequences, seq_len-1, vocab_size)
        model_dists: Model distributions (num_sequences, seq_len-1, vocab_size)
        epsilon: Small value to prevent log(0)
        
    Returns:
        position_kl: KL divergence for each position averaged across sequences
        sequence_kl: KL divergence for each sequence averaged across positions
    """
    print("Computing KL divergences...")
    
    # Add epsilon to prevent log(0)
    theoretical_dists = theoretical_dists + epsilon
    model_dists = model_dists + epsilon
    
    # Renormalize after adding epsilon
    theoretical_dists = theoretical_dists / theoretical_dists.sum(axis=-1, keepdims=True)
    model_dists = model_dists / model_dists.sum(axis=-1, keepdims=True)
    
    # Compute KL divergence: KL(P || Q) = sum(P * log(P / Q))
    log_ratio = np.log(theoretical_dists / model_dists)
    kl_pointwise = theoretical_dists * log_ratio  # (num_sequences, seq_len-1, vocab_size)
    kl_per_position_per_sequence = kl_pointwise.sum(axis=-1)  # (num_sequences, seq_len-1)
    
    # Average across sequences for each position
    position_kl = kl_per_position_per_sequence.mean(axis=0)  # (seq_len-1,)
    
    # Average across positions for each sequence
    sequence_kl = kl_per_position_per_sequence.mean(axis=1)  # (num_sequences,)
    
    print(f"Average KL divergence: {position_kl.mean():.4f}")
    print(f"KL divergence std: {position_kl.std():.4f}")
    print(f"Min/Max KL divergence: {position_kl.min():.4f} / {position_kl.max():.4f}")
    
    return position_kl, sequence_kl


def visualize_kl_analysis(position_kl: np.ndarray, sequence_kl: np.ndarray, 
                         theoretical_dists: np.ndarray, model_dists: np.ndarray):
    """
    Create visualizations for KL divergence analysis.
    
    Args:
        position_kl: KL divergence per position
        sequence_kl: KL divergence per sequence
        theoretical_dists: Theoretical distributions for sample comparisons
        model_dists: Model distributions for sample comparisons
    """
    print("Creating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: KL divergence vs position
    ax = axes[0, 0]
    positions = np.arange(1, len(position_kl) + 1)
    ax.plot(positions, position_kl, 'b-o', linewidth=2, markersize=4)
    ax.set_xlabel('Sequence Position')
    ax.set_ylabel('KL Divergence')
    ax.set_title('KL Divergence vs Sequence Position')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Distribution of KL divergences across sequences
    ax = axes[0, 1]
    ax.hist(sequence_kl, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(sequence_kl.mean(), color='red', linestyle='--', 
               label=f'Mean: {sequence_kl.mean():.4f}')
    ax.set_xlabel('KL Divergence')
    ax.set_ylabel('Number of Sequences')
    ax.set_title('Distribution of KL Divergences Across Sequences')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Sample distribution comparison (early position)
    ax = axes[1, 0]
    pos_idx = 2  # Position 3 (0-indexed)
    seq_idx = 0  # First sequence
    if pos_idx < theoretical_dists.shape[1]:
        theoretical_sample = theoretical_dists[seq_idx, pos_idx, :]
        model_sample = model_dists[seq_idx, pos_idx, :]
        
        x = np.arange(len(theoretical_sample))
        width = 0.35
        
        ax.bar(x - width/2, theoretical_sample, width, label='Theoretical', alpha=0.7)
        ax.bar(x + width/2, model_sample, width, label='Model', alpha=0.7)
        ax.set_xlabel('Token')
        ax.set_ylabel('Probability')
        ax.set_title(f'Distribution Comparison (Seq 0, Pos {pos_idx+1})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 4: Sample distribution comparison (late position)
    ax = axes[1, 1]
    pos_idx = min(8, theoretical_dists.shape[1] - 1)  # Position 9 or last available
    seq_idx = 0  # First sequence
    if pos_idx < theoretical_dists.shape[1]:
        theoretical_sample = theoretical_dists[seq_idx, pos_idx, :]
        model_sample = model_dists[seq_idx, pos_idx, :]
        
        x = np.arange(len(theoretical_sample))
        width = 0.35
        
        ax.bar(x - width/2, theoretical_sample, width, label='Theoretical', alpha=0.7)
        ax.bar(x + width/2, model_sample, width, label='Model', alpha=0.7)
        ax.set_xlabel('Token')
        ax.set_ylabel('Probability')
        ax.set_title(f'Distribution Comparison (Seq 0, Pos {pos_idx+1})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs('play/plots', exist_ok=True)
    output_path = 'play/plots/kl_divergence_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {output_path}")


def print_analysis_summary(position_kl: np.ndarray, sequence_kl: np.ndarray):
    """
    Print summary statistics of the KL divergence analysis.
    """
    print("\n" + "="*60)
    print("KL DIVERGENCE ANALYSIS SUMMARY")
    print("="*60)
    
    print(f"Overall Statistics:")
    print(f"  Mean KL divergence: {position_kl.mean():.4f}")
    print(f"  Std KL divergence:  {position_kl.std():.4f}")
    print(f"  Min KL divergence:  {position_kl.min():.4f}")
    print(f"  Max KL divergence:  {position_kl.max():.4f}")
    
    print(f"\nPosition-wise Analysis:")
    print(f"  Positions analyzed: {len(position_kl)}")
    print(f"  Best position (lowest KL):  {np.argmin(position_kl) + 1} (KL = {position_kl.min():.4f})")
    print(f"  Worst position (highest KL): {np.argmax(position_kl) + 1} (KL = {position_kl.max():.4f})")
    
    print(f"\nSequence-wise Analysis:")
    print(f"  Sequences analyzed: {len(sequence_kl)}")
    print(f"  Mean sequence KL: {sequence_kl.mean():.4f}")
    print(f"  Best sequence KL:  {sequence_kl.min():.4f}")
    print(f"  Worst sequence KL: {sequence_kl.max():.4f}")
    
    # Show position-by-position breakdown
    print(f"\nPosition-by-Position KL Divergences:")
    for i, kl_val in enumerate(position_kl):
        print(f"  Position {i+1:2d}: {kl_val:.4f}")


def main():
    print("=== Model vs Ground Truth Distribution KL Analysis ===\n")
    
    # Configuration
    data_path = "data/spiral/spiral_b_12_20250813_171258_json_output"
    model_path = "checkpoints/spiral_12_64x1_untied"
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = GPT.load(model_path, device)
    model.eval()
    print(f"Model loaded successfully!")
    print(f"Model config: {model.config}\n")
    
    # Load validation data
    sequences, metadata = load_validation_data(data_path)
    
    # Limit to a subset for faster computation (adjust as needed)
    max_sequences = 500  # Adjust based on computational resources
    if len(sequences) > max_sequences:
        print(f"Using subset of {max_sequences} sequences for analysis")
        sequences = sequences[:max_sequences]
    
    # Compute theoretical distributions
    theoretical_dists = compute_theoretical_distributions(sequences, metadata)
    
    # Compute model distributions
    model_dists = compute_model_distributions(model, sequences, device)
    
    # Verify shapes match
    assert theoretical_dists.shape == model_dists.shape, \
        f"Shape mismatch: theoretical {theoretical_dists.shape} vs model {model_dists.shape}"
    
    # Compute KL divergences
    position_kl, sequence_kl = compute_kl_divergences(theoretical_dists, model_dists)
    
    # Create visualizations
    visualize_kl_analysis(position_kl, sequence_kl, theoretical_dists, model_dists)
    
    # Print summary
    print_analysis_summary(position_kl, sequence_kl)
    
    print("\n=== Analysis Complete ===")


if __name__ == "__main__":
    main()