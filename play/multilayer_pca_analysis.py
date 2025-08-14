#!/usr/bin/env python3
"""
Script to perform PCA analysis on multi-layer residual stream activations from a GPT model.
Concatenates post-MLP residual stream activations from ALL transformer layers at each position.
Generates all possible sequences with vocab_size=3, captures multi-layer activations,
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
    
    return token_counts, unique_tokens_non_bos


def capture_multilayer_residual_activations(model: GPT, sequences: List[List[int]], device: torch.device) -> torch.Tensor:
    """
    Capture post-MLP residual stream activations from ALL transformer layers.
    For each position, concatenates activations across all layers.
    
    Returns:
        multilayer_activations: Activations with shape (batch_size, seq_len, n_embd * n_layer)
    """
    batch_size = len(sequences)
    seq_len = len(sequences[0])
    n_layer = model.config.n_layer
    n_embd = model.config.n_embd
    
    print(f"Capturing multi-layer residual stream activations...")
    print(f"Model has {n_layer} layers, each with {n_embd} embedding dimensions")
    print(f"Output will have concatenated dimension: {n_embd * n_layer}")
    
    # Convert sequences to tensor
    input_ids = torch.tensor(sequences, device=device)  # Shape: (batch_size, seq_len)
    
    all_layer_activations = []  # Will store activations from each layer
    
    with torch.no_grad():
        # Manual forward pass to capture activations from all layers
        B, T = input_ids.size()
        
        # Embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=device)
        pos_emb = model.transformer.wpe(pos)
        tok_emb = model.transformer.wte(input_ids)
        x = tok_emb + pos_emb
        
        # Forward through transformer blocks and capture post-MLP activations from each layer
        for block_idx, block in enumerate(model.transformer.h):
            # Apply attention
            resid_mid = x + block.attn(block.ln_1(x))
            
            # Apply MLP with layernorm to get post-MLP residual stream
            resid_post = resid_mid + block.mlp(block.ln_2(resid_mid))
            
            # Store post-MLP residual activations for this layer
            all_layer_activations.append(resid_post.clone())  # Shape: (batch_size, seq_len, n_embd)
            
            # Continue with residual connection for next block
            x = resid_post
    
    # Concatenate activations from all layers along the embedding dimension
    # all_layer_activations: List of tensors, each (batch_size, seq_len, n_embd)
    # Result: (batch_size, seq_len, n_embd * n_layer)
    multilayer_activations = torch.cat(all_layer_activations, dim=-1)
    
    print(f"Multi-layer activations shape: {multilayer_activations.shape}")
    return multilayer_activations


def process_activations(activations: torch.Tensor, remove_bos: bool = True) -> np.ndarray:
    """
    Process activations: optionally remove BOS token and final position, then flatten over sequence dimension.
    Input shape: (num_sequences, seq_len, n_embd_total)
    Output shape: (num_sequences * seq_len_processed, n_embd_total)
    """
    if remove_bos:
        # Remove BOS token (position 0) and final position (position -1)
        activations_processed = activations[:, 1:-1, :]  # Shape: (num_sequences, seq_len-2, n_embd_total)
    else:
        activations_processed = activations
    
    # Flatten over sequence dimension
    num_sequences, seq_len_processed, n_embd_total = activations_processed.shape
    flattened = activations_processed.reshape(num_sequences * seq_len_processed, n_embd_total)
    
    return flattened.cpu().numpy()


def perform_pca_and_plot(activations: np.ndarray, sequences: List[List[int]], vocab_size: int, 
                        n_layer: int, n_embd: int, analysis_type: str = "Multi-Layer Residual Stream"):
    """
    Perform PCA on multi-layer activation data and create visualization.
    """
    print("Performing PCA analysis on multi-layer residual stream activations...")
    
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
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Single plot colored by proportion of 0's
    scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], 
                        c=zeros_proportion_colors, cmap='RdYlBu', 
                        alpha=0.6, s=8, edgecolors='black', linewidth=0.1)
    ax.set_title(f'PCA: {analysis_type} ({n_layer} Layers × {n_embd}D = {n_layer*n_embd}D)\nColored by Proportion of 0\'s | PC1: {pca.explained_variance_ratio_[0]:.3f}, PC2: {pca.explained_variance_ratio_[1]:.3f}')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f} variance)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f} variance)')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Proportion of 0\'s in sequence so far')
    
    plt.tight_layout()
    
    # Save the 2D plot
    output_path_2d = f'play/plots/multilayer_pca_analysis.png'
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
        title=f'Interactive 3D PCA: {analysis_type} ({n_layer} Layers × {n_embd}D = {n_layer*n_embd}D)<br>Colored by Proportion of 0\'s | PC1: {pca.explained_variance_ratio_[0]:.3f}, PC2: {pca.explained_variance_ratio_[1]:.3f}, PC3: {pca.explained_variance_ratio_[2]:.3f}',
        scene=dict(
            xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]:.3f} variance)',
            yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]:.3f} variance)',
            zaxis_title=f'PC3 ({pca.explained_variance_ratio_[2]:.3f} variance)'
        ),
        width=1000,
        height=700
    )
    
    # Save the interactive 3D plot
    output_path_3d = f'play/plots/multilayer_pca_analysis_3d.html'
    fig_3d.write_html(output_path_3d)
    print(f"Interactive 3D plot saved to {output_path_3d}")
    
    # Print summary statistics
    print(f"\n=== {analysis_type} PCA Results Summary ===")
    print(f"Model Architecture: {n_layer} layers × {n_embd} dimensions")
    print(f"Concatenated Activation Dimension: {n_layer * n_embd}")
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
    print("=== Multi-Layer Residual Stream PCA Analysis ===\n")
    
    # Configuration
    vocab_size = 3  # Tokens: 0, 1, 2
    block_size = 12  # Total sequence length including BOS token
    model_path = "checkpoints/spiral_12_64x4_untied"  # Use 4-layer spiral model
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = GPT.load(model_path, device)
    model.eval()
    print(f"Model loaded successfully!")
    print(f"Model config: {model.config}")
    print(f"Model has {model.config.n_layer} layers with {model.config.n_embd} embedding dimensions each")
    print(f"Total concatenated activation dimension will be: {model.config.n_layer * model.config.n_embd}")
    print()
    
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
    
    all_multilayer_activations = []
    
    print(f"Processing {num_batches} batches of size {batch_size}...")
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(sequences))
        batch_sequences = sequences[start_idx:end_idx]
        
        print(f"Processing batch {i+1}/{num_batches} (sequences {start_idx}-{end_idx-1})")
        
        # Capture multi-layer residual stream activations for this batch
        multilayer_batch = capture_multilayer_residual_activations(model, batch_sequences, device)
        all_multilayer_activations.append(multilayer_batch)
    
    # Concatenate all batches
    print("Concatenating all batches...")
    multilayer_activations = torch.cat(all_multilayer_activations, dim=0)
    print(f"Multi-layer activations shape: {multilayer_activations.shape}")
    
    # Process activations (remove BOS token and flatten)
    print("Processing multi-layer activations...")
    processed_multilayer = process_activations(multilayer_activations, remove_bos=True)
    print(f"Processed multi-layer activations shape: {processed_multilayer.shape}")
    
    # Perform PCA and create plots
    print("\n" + "="*60)
    print("ANALYZING MULTI-LAYER RESIDUAL STREAM ACTIVATIONS")
    print("="*60)
    perform_pca_and_plot(processed_multilayer, sequences, vocab_size, 
                        model.config.n_layer, model.config.n_embd,
                        analysis_type="Multi-Layer Residual Stream")
    
    print("\n=== Analysis Complete ===")


if __name__ == "__main__":
    main()