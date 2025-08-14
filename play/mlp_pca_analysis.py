#!/usr/bin/env python3
"""
Script to perform PCA analysis on post-MLP activations from a GPT model.
Generates all possible sequences with vocab_size=3, captures post-MLP activations,
and visualizes the first two principal components.
"""

import sys
import os
import itertools
from typing import List, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from models.gpt import GPT


def generate_all_sequences(vocab_size: int, block_size: int) -> List[List[int]]:
    """
    Generate all possible sequences with given vocab_size and block_size.
    Format: [BOS, *, *, ..., *] where * ∈ {0, 1} and BOS = vocab_size
    """
    print(f"Generating all possible binary sequences with vocab_size={vocab_size}, block_size={block_size}...")
    
    sequences = []
    bos_token = vocab_size - 1  # BOS token is the highest valid token, outside binary range {0, 1}
    
    # Generate all combinations for (block_size-1) positions with binary values {0, 1}
    for combination in itertools.product(range(2), repeat=block_size-1):
        sequence = [bos_token] + list(combination)
        sequences.append(sequence)
    
    print(f"Generated {len(sequences)} binary sequences")
    return sequences


def analyze_token_distribution(sequences: List[List[int]], vocab_size: int):
    """
    Analyze the distribution of tokens in the generated sequences.
    """
    print("=== Token Distribution Analysis ===")
    
    # Basic statistics
    total_sequences = len(sequences)
    seq_length = len(sequences[0])
    print(f"Total sequences: {total_sequences}")
    print(f"Sequence length: {seq_length}")
    print(f"Expected vocab size: {vocab_size}")
    print(f"BOS token: {vocab_size - 1}")
    print()
    
    # Sample sequences for inspection
    print("Sample sequences (first 10):")
    for i in range(min(10, len(sequences))):
        print(f"  {sequences[i]}")
    print()
    
    # Token frequency analysis (excluding BOS)
    token_counts = {}
    position_token_counts = [{} for _ in range(seq_length)]
    total_non_bos_tokens = 0
    
    for seq in sequences:
        for pos, token in enumerate(seq):
            # Overall counts (excluding BOS position)
            if pos > 0:  # Skip BOS token
                token_counts[token] = token_counts.get(token, 0) + 1
                total_non_bos_tokens += 1
            
            # Position-wise counts
            position_token_counts[pos][token] = position_token_counts[pos].get(token, 0) + 1
    
    # Overall token distribution (excluding BOS)
    print("Overall token distribution (excluding BOS position):")
    for token in sorted(token_counts.keys()):
        count = token_counts[token]
        percentage = (count / total_non_bos_tokens) * 100
        print(f"  Token {token}: {count:,} occurrences ({percentage:.2f}%)")
    print(f"  Total non-BOS tokens: {total_non_bos_tokens:,}")
    print()
    
    # Check unique tokens in non-BOS positions
    unique_tokens_non_bos = set(token_counts.keys())
    print(f"Unique tokens in non-BOS positions: {sorted(unique_tokens_non_bos)}")
    print(f"Number of unique non-BOS tokens: {len(unique_tokens_non_bos)}")
    print()
    
    # Position-wise analysis (first few positions)
    print("Position-wise token distribution:")
    print("Position | Token Counts")
    print("---------|-------------")
    for pos in range(min(5, seq_length)):  # Show first 5 positions
        counts_str = " | ".join([f"{token}:{position_token_counts[pos].get(token, 0)}" 
                                for token in sorted(position_token_counts[pos].keys())])
        pos_name = "BOS" if pos == 0 else f"{pos}"
        print(f"    {pos_name:2}   | {counts_str}")
    print()
    
    # Analyze sequences with only 0s and 1s (excluding BOS)
    only_01_sequences = 0
    contains_2_sequences = 0
    
    for seq in sequences:
        non_bos_tokens = set(seq[1:])  # Exclude BOS token
        if non_bos_tokens.issubset({0, 1}):
            only_01_sequences += 1
        if 2 in non_bos_tokens:
            contains_2_sequences += 1
    
    print("Sequence composition analysis:")
    print(f"  Sequences with only 0s and 1s (excluding BOS): {only_01_sequences:,} ({(only_01_sequences/total_sequences)*100:.2f}%)")
    print(f"  Sequences containing token 2 (excluding BOS): {contains_2_sequences:,} ({(contains_2_sequences/total_sequences)*100:.2f}%)")
    print()
    
    # Analyze proportion of 0s distribution for validation
    print("Proportion of 0s analysis (positions 1 to seq_length-2):")
    proportion_counts = {}
    
    for seq in sequences:
        for pos in range(1, len(seq)-1):  # Skip BOS and final position
            subsequence = seq[1:pos+1]
            num_zeros = sum(1 for token in subsequence if token == 0)
            proportion_zeros = num_zeros / len(subsequence)
            # Round to avoid floating point precision issues
            proportion_rounded = round(proportion_zeros, 3)
            proportion_counts[proportion_rounded] = proportion_counts.get(proportion_rounded, 0) + 1
    
    print("Proportion | Count")
    print("-----------|------")
    for prop in sorted(proportion_counts.keys()):
        count = proportion_counts[prop]
        print(f"    {prop:5.3f}  | {count:,}")
    
    total_proportion_points = sum(proportion_counts.values())
    print(f"Total proportion data points: {total_proportion_points:,}")
    print()
    
    return token_counts, unique_tokens_non_bos


def capture_mlp_activations(model: GPT, sequences: List[List[int]], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Capture both pre-layernorm and post-layernorm+MLP activations for a batch of sequences.
    Returns:
        pre_ln_activations: Activations before MLP layernorm (resid_mid)
        post_ln_mlp_activations: Activations after layernorm + MLP (resid_post)
    """
    batch_size = len(sequences)
    seq_len = len(sequences[0])
    
    # Convert sequences to tensor
    input_ids = torch.tensor(sequences, device=device)  # Shape: (batch_size, seq_len)
    
    pre_ln_activations = []
    post_ln_mlp_activations = []
    
    with torch.no_grad():
        # Manual forward pass to capture both pre and post layernorm activations
        B, T = input_ids.size()
        
        # Embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=device)
        pos_emb = model.transformer.wpe(pos)
        tok_emb = model.transformer.wte(input_ids)
        x = tok_emb + pos_emb
        
        # Forward through transformer blocks and capture both activation types
        for block_idx, block in enumerate(model.transformer.h):
            # Apply attention
            resid_mid = x + block.attn(block.ln_1(x))
            
            # Capture pre-layernorm activations (before ln_2 is applied)
            pre_ln_activations.append(resid_mid.clone())
            
            # Apply MLP with layernorm and capture post-layernorm+MLP activations
            resid_post = resid_mid + block.mlp(block.ln_2(resid_mid))
            post_ln_mlp_activations.append(resid_post.clone())
            
            # Continue with residual connection for next block
            x = resid_post
    
    # Use activations from the last transformer block
    final_pre_ln = pre_ln_activations[-1]  # Shape: (batch_size, seq_len, n_embd)
    final_post_ln_mlp = post_ln_mlp_activations[-1]  # Shape: (batch_size, seq_len, n_embd)
    
    return final_pre_ln, final_post_ln_mlp


def process_activations(activations: torch.Tensor, remove_bos: bool = True) -> np.ndarray:
    """
    Process activations: optionally remove BOS token and final position, then flatten over sequence dimension.
    Input shape: (num_sequences, seq_len, n_embd)
    Output shape: (num_sequences * seq_len_processed, n_embd)
    """
    if remove_bos:
        # Remove BOS token (position 0) and final position (position -1)
        activations_processed = activations[:, 1:-1, :]  # Shape: (num_sequences, seq_len-2, n_embd)
    else:
        activations_processed = activations
    
    # Flatten over sequence dimension
    num_sequences, seq_len_processed, n_embd = activations_processed.shape
    flattened = activations_processed.reshape(num_sequences * seq_len_processed, n_embd)
    
    return flattened.cpu().numpy()


def perform_pca_and_plot(activations: np.ndarray, sequences: List[List[int]], vocab_size: int, analysis_type: str = "Post-MLP", suffix: str = ""):
    """
    Perform PCA on activation data and create visualization.
    """
    print("Performing PCA analysis...")
    
    # Perform PCA with 3 components for 3D visualization
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(activations)
    
    print(f"PCA explained variance ratio: PC1={pca.explained_variance_ratio_[0]:.4f}, PC2={pca.explained_variance_ratio_[1]:.4f}, PC3={pca.explained_variance_ratio_[2]:.4f}")
    print(f"Total variance explained by PC1+PC2: {sum(pca.explained_variance_ratio_[:2]):.4f}")
    print(f"Total variance explained by PC1+PC2+PC3: {sum(pca.explained_variance_ratio_[:3]):.4f}")
    
    # Create colors based on proportion of 0's in sequence so far
    num_sequences = len(sequences)
    seq_len_without_bos_and_final = len(sequences[0]) - 2  # Remove BOS token and final position
    
    # Create proportion of 0's coloring and hover info
    zeros_proportion_colors = []
    hover_info = []
    
    for seq_idx, sequence in enumerate(sequences):
        for pos in range(1, len(sequence)-1):  # Skip BOS token and final position
            # Count 0's from position 1 to current position (inclusive)
            subsequence = sequence[1:pos+1]  # From after BOS to current position
            num_zeros = sum(1 for token in subsequence if token == 0)
            proportion_zeros = num_zeros / len(subsequence)
            num_tokens_observed = len(subsequence)
            
            zeros_proportion_colors.append(proportion_zeros)
            hover_info.append(f'Proportion 0s: {proportion_zeros:.3f}<br>Tokens observed: {num_tokens_observed}<br>Position: {pos}<br>Sequence: {seq_idx}')
    
    zeros_proportion_colors = np.array(zeros_proportion_colors)
    
    # Subsample data for interactive plot (plotly can be slow with too many points)
    n_subsample = min(50000, len(pca_result))  # Use max 50k points for interactive plot
    if n_subsample < len(pca_result):
        print(f"Subsampling {n_subsample} points from {len(pca_result)} for interactive 3D plot...")
        indices = np.random.choice(len(pca_result), n_subsample, replace=False)
        pca_result_sub = pca_result[indices]
        zeros_proportion_colors_sub = zeros_proportion_colors[indices]
        hover_info_sub = [hover_info[i] for i in indices]
    else:
        pca_result_sub = pca_result
        zeros_proportion_colors_sub = zeros_proportion_colors
        hover_info_sub = hover_info
    
    # Create single 2D plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Single plot colored by proportion of 0's
    scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], 
                        c=zeros_proportion_colors, cmap='RdYlBu', 
                        alpha=0.6, s=8, edgecolors='black', linewidth=0.1)
    ax.set_title(f'PCA: {analysis_type} Activations (Colored by Proportion of 0\'s)\nPC1: {pca.explained_variance_ratio_[0]:.3f}, PC2: {pca.explained_variance_ratio_[1]:.3f}')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f} variance)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f} variance)')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Proportion of 0\'s in sequence so far')
    
    plt.tight_layout()
    
    # Save the 2D plot
    output_path_2d = f'play/plots/mlp_pca_analysis{suffix}.png'
    plt.savefig(output_path_2d, dpi=300, bbox_inches='tight')
    print(f"2D plot saved to {output_path_2d}")
    
    # Create interactive 3D plot with Plotly
    print("Creating interactive 3D PCA plot...")
    
    # Create single 3D figure
    fig_3d = go.Figure()
    
    # Add trace colored by proportion of 0's
    fig_3d.add_trace(go.Scatter3d(
        x=pca_result_sub[:, 0],
        y=pca_result_sub[:, 1], 
        z=pca_result_sub[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=zeros_proportion_colors_sub,
            colorscale='RdYlBu',
            colorbar=dict(title="Proportion of 0's in sequence so far"),
            opacity=0.6
        ),
        text=hover_info_sub,
        name='Colored by Proportion of 0s'
    ))
    
    fig_3d.update_layout(
        title=f'Interactive 3D PCA: {analysis_type} Activations (Colored by Proportion of 0\'s)<br>PC1: {pca.explained_variance_ratio_[0]:.3f}, PC2: {pca.explained_variance_ratio_[1]:.3f}, PC3: {pca.explained_variance_ratio_[2]:.3f}',
        scene=dict(
            xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]:.3f} variance)',
            yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]:.3f} variance)',
            zaxis_title=f'PC3 ({pca.explained_variance_ratio_[2]:.3f} variance)'
        ),
        width=1000,
        height=700
    )
    
    # Save the interactive 3D plot
    output_path_3d = f'play/plots/mlp_pca_analysis{suffix}_3d.html'
    fig_3d.write_html(output_path_3d)
    print(f"Interactive 3D plot saved to {output_path_3d}")
    
    # Print summary statistics
    print(f"\n=== {analysis_type} PCA Results Summary ===")
    print(f"Input data shape: {activations.shape}")
    print(f"PCA output shape: {pca_result.shape}")
    print(f"PC1 variance explained: {pca.explained_variance_ratio_[0]:.4f}")
    print(f"PC2 variance explained: {pca.explained_variance_ratio_[1]:.4f}")
    print(f"PC3 variance explained: {pca.explained_variance_ratio_[2]:.4f}")
    print(f"Total variance explained by PC1+PC2: {sum(pca.explained_variance_ratio_[:2]):.4f}")
    print(f"Total variance explained by PC1+PC2+PC3: {sum(pca.explained_variance_ratio_[:3]):.4f}")
    print(f"Number of sequences: {num_sequences}")
    print(f"Sequence length (without BOS and final): {seq_len_without_bos_and_final}")
    print(f"Total data points: {len(pca_result)}")


def main():
    print("=== Pre & Post LayerNorm MLP Activation PCA Analysis ===\n")
    
    # Configuration
    vocab_size = 3  # Tokens: 0, 1, 2
    block_size = 12  # Total sequence length including BOS token
    model_path = "checkpoints/spiral_12_64x4_untied"  # Use spiral model for vocab_size=3
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = GPT.load(model_path, device)
    model.eval()
    print(f"Model loaded successfully!")
    print(f"Model config: {model.config}\n")
    
    # Generate all sequences
    sequences = generate_all_sequences(vocab_size, block_size)
    print(f"Total sequences to process: {len(sequences)}")
    print(f"Each sequence length: {len(sequences[0])}")
    print(f"Example sequences:")
    for i in range(min(5, len(sequences))):
        print(f"  {sequences[i]}")
    print()
    
    # Analyze token distribution
    token_counts, unique_tokens = analyze_token_distribution(sequences, vocab_size)
    
    # Process sequences in batches to manage memory
    batch_size = 500  # Adjust based on available memory
    num_batches = (len(sequences) + batch_size - 1) // batch_size
    
    all_pre_ln = []
    all_post_ln_mlp = []
    
    print(f"Processing {num_batches} batches of size {batch_size}...")
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(sequences))
        batch_sequences = sequences[start_idx:end_idx]
        
        print(f"Processing batch {i+1}/{num_batches} (sequences {start_idx}-{end_idx-1})")
        
        # Capture both pre-layernorm and post-layernorm+MLP activations for this batch
        pre_ln_batch, post_ln_mlp_batch = capture_mlp_activations(model, batch_sequences, device)
        all_pre_ln.append(pre_ln_batch)
        all_post_ln_mlp.append(post_ln_mlp_batch)
    
    # Concatenate all batches
    print("Concatenating all batches...")
    pre_ln_activations = torch.cat(all_pre_ln, dim=0)
    post_ln_mlp_activations = torch.cat(all_post_ln_mlp, dim=0)
    print(f"Pre-LayerNorm activations shape: {pre_ln_activations.shape}")
    print(f"Post-LayerNorm+MLP activations shape: {post_ln_mlp_activations.shape}")
    
    # Process both sets of activations (remove BOS token and flatten)
    print("Processing pre-LayerNorm activations...")
    processed_pre_ln = process_activations(pre_ln_activations, remove_bos=True)
    print(f"Processed pre-LayerNorm activations shape: {processed_pre_ln.shape}")
    
    print("Processing post-LayerNorm+MLP activations...")
    processed_post_ln_mlp = process_activations(post_ln_mlp_activations, remove_bos=True)
    print(f"Processed post-LayerNorm+MLP activations shape: {processed_post_ln_mlp.shape}")
    
    # Perform PCA and create plots for pre-LayerNorm activations
    print("\n" + "="*60)
    print("ANALYZING PRE-LAYERNORM ACTIVATIONS")
    print("="*60)
    perform_pca_and_plot(processed_pre_ln, sequences, vocab_size, 
                        analysis_type="Pre-LayerNorm", suffix="_pre_ln")
    
    # Perform PCA and create plots for post-LayerNorm+MLP activations
    print("\n" + "="*60)
    print("ANALYZING POST-LAYERNORM+MLP ACTIVATIONS")
    print("="*60)
    perform_pca_and_plot(processed_post_ln_mlp, sequences, vocab_size, 
                        analysis_type="Post-LayerNorm+MLP", suffix="_post_ln")
    
    print("\n=== Analysis Complete ===")


if __name__ == "__main__":
    main()