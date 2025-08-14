#!/usr/bin/env python3
"""
Script to plot KL divergence across model training checkpoints.
Analyzes how the model's alignment with theoretical spiral dynamics evolves during training.
"""

import sys
import os
import json
from typing import List, Dict, Tuple
import glob

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
    
    print(f"Average KL divergence: {position_kl.mean():.6f}")
    print(f"KL divergence std: {position_kl.std():.6f}")
    print(f"Min/Max KL divergence: {position_kl.min():.6f} / {position_kl.max():.6f}")
    
    return position_kl, sequence_kl


def get_checkpoint_paths(checkpoint_dir: str) -> List[Tuple[int, str]]:
    """
    Get all available checkpoint paths sorted by step number.
    
    Args:
        checkpoint_dir: Base checkpoint directory
        
    Returns:
        List of (step_number, checkpoint_path) tuples
    """
    checkpoint_pattern = os.path.join(checkpoint_dir, "checkpoint_step_*")
    checkpoint_dirs = glob.glob(checkpoint_pattern)
    
    checkpoints = []
    for checkpoint_path in checkpoint_dirs:
        step_dir = os.path.basename(checkpoint_path)
        try:
            step_number = int(step_dir.split('_')[-1])
            checkpoints.append((step_number, checkpoint_path))
        except ValueError:
            print(f"Warning: Could not parse step number from {step_dir}")
    
    # Sort by step number
    checkpoints.sort(key=lambda x: x[0])
    
    print(f"Found {len(checkpoints)} checkpoints:")
    for step, path in checkpoints:
        print(f"  Step {step}: {path}")
    
    return checkpoints


def analyze_checkpoint_kl_evolution(sequences: List[List[int]], metadata: Dict, 
                                   checkpoint_dir: str, device: torch.device) -> Tuple[List[int], List[float]]:
    """
    Analyze KL divergence evolution across training checkpoints.
    
    Args:
        sequences: Validation sequences
        metadata: Spiral metadata
        checkpoint_dir: Directory containing model checkpoints
        device: Device to run models on
        
    Returns:
        steps: List of training steps
        mean_kl_divergences: List of mean KL divergences for each checkpoint
    """
    print("Analyzing KL divergence evolution across checkpoints...")
    
    # Get checkpoint paths
    checkpoints = get_checkpoint_paths(checkpoint_dir)
    if not checkpoints:
        raise RuntimeError(f"No checkpoints found in {checkpoint_dir}")
    
    # Compute theoretical distributions once (same for all checkpoints)
    theoretical_dists = compute_theoretical_distributions(sequences, metadata)
    
    steps = []
    mean_kl_divergences = []
    position_kl_by_step = []
    
    for step, checkpoint_path in checkpoints:
        print(f"\n=== Analyzing checkpoint step {step} ===")
        
        # Load model from checkpoint
        try:
            model = GPT.load(checkpoint_path, device)
            model.eval()
            print(f"Loaded model from {checkpoint_path}")
        except Exception as e:
            print(f"Error loading checkpoint {checkpoint_path}: {e}")
            continue
        
        # Compute model distributions
        model_dists = compute_model_distributions(model, sequences, device)
        
        # Verify shapes match
        if theoretical_dists.shape != model_dists.shape:
            print(f"Warning: Shape mismatch at step {step}: "
                  f"theoretical {theoretical_dists.shape} vs model {model_dists.shape}")
            continue
        
        # Compute KL divergences
        position_kl, sequence_kl = compute_kl_divergences(theoretical_dists, model_dists)
        
        # Store results
        steps.append(step)
        mean_kl_divergences.append(position_kl.mean())
        position_kl_by_step.append(position_kl)
        
        print(f"Step {step}: Mean KL divergence = {position_kl.mean():.6f}")
    
    return steps, mean_kl_divergences, position_kl_by_step


def visualize_checkpoint_kl_evolution(steps: List[int], mean_kl_divergences: List[float], 
                                     position_kl_by_step: List[np.ndarray]):
    """
    Create visualizations for KL divergence evolution across training checkpoints.
    
    Args:
        steps: Training steps
        mean_kl_divergences: Mean KL divergence for each step
        position_kl_by_step: Position-wise KL divergences for each step
    """
    print("Creating checkpoint evolution visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Overall KL divergence evolution
    ax = axes[0, 0]
    ax.plot(steps, mean_kl_divergences, 'b-o', linewidth=2, markersize=6)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Mean KL Divergence')
    ax.set_title('KL Divergence Evolution During Training')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')  # Log scale for better visualization
    
    # Add value annotations
    for step, kl in zip(steps, mean_kl_divergences):
        ax.annotate(f'{kl:.4f}', (step, kl), textcoords="offset points", 
                   xytext=(0,10), ha='center', fontsize=8)
    
    # Plot 2: KL divergence reduction (relative to initial)
    ax = axes[0, 1]
    if len(mean_kl_divergences) > 1:
        initial_kl = mean_kl_divergences[0]
        relative_kl = [kl / initial_kl for kl in mean_kl_divergences]
        ax.plot(steps, relative_kl, 'g-o', linewidth=2, markersize=6)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('KL Divergence (Relative to Step 0)')
        ax.set_title('Relative KL Divergence Improvement')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Initial KL')
        ax.legend()
    
    # Plot 3: Position-wise KL evolution (heatmap)
    ax = axes[1, 0]
    if position_kl_by_step:
        # Create heatmap data
        position_kl_matrix = np.array(position_kl_by_step)  # (num_steps, num_positions)
        
        im = ax.imshow(position_kl_matrix.T, aspect='auto', cmap='viridis', origin='lower')
        ax.set_xlabel('Checkpoint Index')
        ax.set_ylabel('Sequence Position')
        ax.set_title('Position-wise KL Divergence Evolution')
        
        # Set x-axis labels to training steps
        step_indices = range(len(steps))
        ax.set_xticks(step_indices)
        ax.set_xticklabels([str(step) for step in steps])
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='KL Divergence')
    
    # Plot 4: Position-wise KL for first and last checkpoints
    ax = axes[1, 1]
    if len(position_kl_by_step) >= 2:
        positions = np.arange(1, len(position_kl_by_step[0]) + 1)
        
        ax.plot(positions, position_kl_by_step[0], 'r-o', label=f'Step {steps[0]}', 
                linewidth=2, markersize=4)
        ax.plot(positions, position_kl_by_step[-1], 'b-o', label=f'Step {steps[-1]}', 
                linewidth=2, markersize=4)
        
        ax.set_xlabel('Sequence Position')
        ax.set_ylabel('KL Divergence')
        ax.set_title('Position-wise KL: First vs Last Checkpoint')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs('play/plots', exist_ok=True)
    output_path = 'play/plots/checkpoint_kl_evolution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {output_path}")


def print_evolution_summary(steps: List[int], mean_kl_divergences: List[float]):
    """
    Print summary statistics of the KL divergence evolution.
    """
    print("\n" + "="*60)
    print("KL DIVERGENCE EVOLUTION SUMMARY")
    print("="*60)
    
    if len(steps) < 2:
        print("Insufficient checkpoints for evolution analysis")
        return
    
    initial_kl = mean_kl_divergences[0]
    final_kl = mean_kl_divergences[-1]
    improvement = (initial_kl - final_kl) / initial_kl * 100
    
    print(f"Training Steps Analyzed: {len(steps)}")
    print(f"Step Range: {steps[0]} → {steps[-1]}")
    print(f"Initial KL Divergence (Step {steps[0]}): {initial_kl:.6f}")
    print(f"Final KL Divergence (Step {steps[-1]}): {final_kl:.6f}")
    print(f"Improvement: {improvement:.2f}% reduction")
    
    # Find best and worst checkpoints
    best_idx = np.argmin(mean_kl_divergences)
    worst_idx = np.argmax(mean_kl_divergences)
    
    print(f"\nBest Performance:")
    print(f"  Step {steps[best_idx]}: KL = {mean_kl_divergences[best_idx]:.6f}")
    print(f"Worst Performance:")
    print(f"  Step {steps[worst_idx]}: KL = {mean_kl_divergences[worst_idx]:.6f}")
    
    # Show step-by-step progression
    print(f"\nStep-by-Step Progression:")
    for step, kl in zip(steps, mean_kl_divergences):
        rel_improvement = (initial_kl - kl) / initial_kl * 100
        print(f"  Step {step:4d}: {kl:.6f} ({rel_improvement:+6.2f}% vs initial)")


def main():
    print("=== KL Divergence Evolution Across Training Checkpoints ===\n")
    
    # Configuration
    data_path = "data/spiral/spiral_b_12_20250813_171258_json_output"
    checkpoint_dir = "checkpoints/spiral_12_64x1_untied"
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load validation data
    sequences, metadata = load_validation_data(data_path)
    
    # Limit to a subset for faster computation
    max_sequences = 500  # Adjust based on computational resources
    if len(sequences) > max_sequences:
        print(f"Using subset of {max_sequences} sequences for analysis")
        sequences = sequences[:max_sequences]
    
    # Analyze KL divergence evolution
    steps, mean_kl_divergences, position_kl_by_step = analyze_checkpoint_kl_evolution(
        sequences, metadata, checkpoint_dir, device
    )
    
    if not steps:
        print("No valid checkpoints analyzed!")
        return
    
    # Create visualizations
    visualize_checkpoint_kl_evolution(steps, mean_kl_divergences, position_kl_by_step)
    
    # Print summary
    print_evolution_summary(steps, mean_kl_divergences)
    
    print("\n=== Analysis Complete ===")


if __name__ == "__main__":
    main()