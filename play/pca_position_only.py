#!/usr/bin/env python3
"""
Quick script to create just the position-colored PCA plot.
"""

import sys
import os
import itertools
from typing import List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from models.gpt import GPT


def generate_all_sequences(block_size: int) -> List[List[int]]:
    """Generate all possible length-10 sequences with BOS token 3."""
    sequences = []
    for combination in itertools.product([0, 1, 2], repeat=block_size-2):
        sequence = [3] + list(combination) + [3] 
        sequences.append(sequence)
    return sequences


def capture_activations_batch(model: GPT, sequences: List[List[int]], device: torch.device):
    """Capture activations for a batch of sequences."""
    input_ids = torch.tensor(sequences, device=device)
    intermediate_batch = []
    final_batch = []
    
    with torch.no_grad():
        B, T = input_ids.size()
        pos = torch.arange(0, T, dtype=torch.long, device=device)
        pos_emb = model.transformer.wpe(pos)
        tok_emb = model.transformer.wte(input_ids)
        x = tok_emb + pos_emb
        
        for block in model.transformer.h:
            x_after_attn = x + block.attn(block.ln_1(x))
            intermediate_batch.append(x_after_attn.clone())
            x = x_after_attn + block.mlp(block.ln_2(x_after_attn))
        
        final_batch.append(x.clone())
        x_after_ln = model.transformer.ln_f(x)
        post_layernorm_batch = [x_after_ln.clone()]
    
    return intermediate_batch[0], final_batch[0], post_layernorm_batch[0]


def process_activations(activations: torch.Tensor) -> np.ndarray:
    """Process activations: remove BOS token and flatten."""
    activations_reduced = activations[:, 1:, :]
    num_sequences, seq_len_minus_1, n_embd = activations_reduced.shape
    flattened = activations_reduced.reshape(num_sequences * seq_len_minus_1, n_embd)
    return flattened.cpu().numpy()


def create_position_colored_plot(intermediate_data: np.ndarray, final_data: np.ndarray, post_layernorm_data: np.ndarray, sequences: List[List[int]]):
    """Create a plot colored by sequence position."""
    print("Creating position-colored PCA plots...")
    
    # Perform PCA on activation data
    pca_intermediate = PCA(n_components=2)
    pca_final = PCA(n_components=2)
    pca_post_layernorm = PCA(n_components=2)
    
    intermediate_pca = pca_intermediate.fit_transform(intermediate_data)
    final_pca = pca_final.fit_transform(final_data)
    post_layernorm_pca = pca_post_layernorm.fit_transform(post_layernorm_data)
    
    # Create position labels (1 to 11, corresponding to positions after BOS token)
    seq_length_minus_1 = len(sequences[0]) - 1  # 11 positions after BOS
    num_sequences = len(sequences)
    position_labels = np.tile(np.arange(1, seq_length_minus_1 + 1), num_sequences)
    
    # Create three-panel plot
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    # Panel 1: Intermediate activations
    scatter1 = axes[0].scatter(intermediate_pca[:, 0], intermediate_pca[:, 1], 
                              c=position_labels, cmap='viridis', alpha=0.6, s=8, 
                              edgecolors='black', linewidth=0.05)
    axes[0].set_title('Intermediate Activations PCA\n(Colored by Sequence Position)', fontsize=14)
    axes[0].set_xlabel(f'PC1 ({pca_intermediate.explained_variance_ratio_[0]:.3f} var)', fontsize=12)
    axes[0].set_ylabel(f'PC2 ({pca_intermediate.explained_variance_ratio_[1]:.3f} var)', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # Panel 2: Final activations
    scatter2 = axes[1].scatter(final_pca[:, 0], final_pca[:, 1], 
                              c=position_labels, cmap='viridis', alpha=0.6, s=8,
                              edgecolors='black', linewidth=0.05)
    axes[1].set_title('Final Activations PCA\n(Colored by Sequence Position)', fontsize=14)
    axes[1].set_xlabel(f'PC1 ({pca_final.explained_variance_ratio_[0]:.3f} var)', fontsize=12)
    axes[1].set_ylabel(f'PC2 ({pca_final.explained_variance_ratio_[1]:.3f} var)', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    # Panel 3: Post-LayerNorm activations
    scatter3 = axes[2].scatter(post_layernorm_pca[:, 0], post_layernorm_pca[:, 1], 
                              c=position_labels, cmap='viridis', alpha=0.6, s=8,
                              edgecolors='black', linewidth=0.05)
    axes[2].set_title('Post-LayerNorm Activations PCA\n(Colored by Sequence Position)', fontsize=14)
    axes[2].set_xlabel(f'PC1 ({pca_post_layernorm.explained_variance_ratio_[0]:.3f} var)', fontsize=12)
    axes[2].set_ylabel(f'PC2 ({pca_post_layernorm.explained_variance_ratio_[1]:.3f} var)', fontsize=12)
    axes[2].grid(True, alpha=0.3)
    
    # Add overall title
    fig.suptitle('PCA Analysis Colored by Sequence Position', fontsize=16, y=0.95)
    
    # Adjust layout first
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, bottom=0.2)
    
    # Add colorbar after layout adjustments
    cbar = plt.colorbar(scatter3, ax=axes, orientation='horizontal', pad=0.15, shrink=0.6, aspect=30)
    cbar.set_label('Sequence Position (1=after BOS, 11=final token)', fontsize=12)
    cbar.set_ticks(range(1, seq_length_minus_1 + 1))
    
    # Save the plot
    output_path = 'play/plots/pca_position_colored_all_tokens_fixed.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Position-colored plot saved to {output_path}")
    
    plt.show()


def main():
    print("=== Quick Position-Colored PCA Plot ===\n")
    
    block_size = 12
    model_path = "checkpoints/mess3_12_64x1/checkpoint_step_10000"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {model_path}")
    model = GPT.load(model_path, device)
    model.eval()
    
    # Generate sequences (smaller subset for speed)
    print("Generating sequences...")
    sequences = generate_all_sequences(block_size)
    print(f"Generated {len(sequences)} sequences")
    
    # Process in smaller batches
    batch_size = 1000
    num_batches = min(10, (len(sequences) + batch_size - 1) // batch_size)  # Only process first 10 batches
    
    all_intermediate = []
    all_final = []
    all_post_layernorm = []
    
    print(f"Processing {num_batches} batches...")
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(sequences))
        batch_sequences = sequences[start_idx:end_idx]
        
        print(f"Processing batch {i+1}/{num_batches}")
        
        intermediate_batch, final_batch, post_layernorm_batch = capture_activations_batch(model, batch_sequences, device)
        
        all_intermediate.append(intermediate_batch)
        all_final.append(final_batch)
        all_post_layernorm.append(post_layernorm_batch)
    
    # Concatenate batches
    intermediate_activations = torch.cat(all_intermediate, dim=0)
    final_activations = torch.cat(all_final, dim=0)
    post_layernorm_activations = torch.cat(all_post_layernorm, dim=0)
    
    # Process activations
    intermediate_processed = process_activations(intermediate_activations)
    final_processed = process_activations(final_activations)
    post_layernorm_processed = process_activations(post_layernorm_activations)
    
    print(f"Processed shapes: {intermediate_processed.shape}")
    
    # Create plot with subset of sequences
    sequences_subset = sequences[:num_batches * batch_size]
    create_position_colored_plot(intermediate_processed, final_processed, post_layernorm_processed, sequences_subset)
    
    print("=== Complete ===")


if __name__ == "__main__":
    main()