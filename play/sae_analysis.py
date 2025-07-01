#!/usr/bin/env python3
"""
Script to analyze SAE feature activations compared to theoretical belief states.
Loads trained SAE models and visualizes how well the 2D latent features capture
the theoretical computational structure of the mess3 process.

Key mapping:
- sae.0 activations correspond to constrained belief states
- sae.1 activations correspond to regular belief states
"""

import sys
import os
import itertools
import json
from typing import List, Tuple, Dict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from safetensors.torch import load_file

from models.gpt import GPT
from models.sae.standard import StandardSAE
from data.comp_mech import belief_update, constrained_belief_update, stationary_distribution
from data.mess3 import mess3
from play.utils import uniform_centered_projection


def load_gpt_model(checkpoint_path: str) -> GPT:
    """
    Load the GPT model from the specified checkpoint directory.
    
    Args:
        checkpoint_path: Path to the GPT model checkpoint directory
        
    Returns:
        Loaded GPT model
    """
    print(f"Loading GPT model from {checkpoint_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the model using the standard GPT.load method
    model = GPT.load(checkpoint_path, device)
    model.eval()
    
    print(f"GPT model loaded successfully!")
    print(f"Config: {model.config}")
    
    return model


def load_sae_models(checkpoint_path: str) -> Tuple[StandardSAE, StandardSAE]:
    """
    Load the SAE models from the specified checkpoint directory.
    
    Args:
        checkpoint_path: Path to the SAE checkpoint directory
        
    Returns:
        Tuple of (sae.0, sae.1) models
    """
    print(f"Loading SAE models from {checkpoint_path}")
    
    # Load SAE configuration
    sae_config_path = os.path.join(checkpoint_path, "sae.json")
    with open(sae_config_path, "r") as f:
        sae_config = json.load(f)
    
    print(f"SAE config: {sae_config}")
    
    # Load SAE model weights
    sae_0_path = os.path.join(checkpoint_path, "sae.0.safetensors")
    sae_1_path = os.path.join(checkpoint_path, "sae.1.safetensors")
    
    sae_0_weights = load_file(sae_0_path)
    sae_1_weights = load_file(sae_1_path)
    
    print(f"SAE.0 weights keys: {list(sae_0_weights.keys())}")
    print(f"SAE.1 weights keys: {list(sae_1_weights.keys())}")
    
    # TODO: Properly instantiate SAE models with correct config
    # For now, we'll work with the raw weights
    return sae_0_weights, sae_1_weights


def generate_sequences(block_size: int) -> List[List[int]]:
    """
    Generate all possible length-block_size sequences with BOS token 3.
    Format: [3, *, *, *, *, *, *, *, *, *, *, 3] where * ∈ {0, 1, 2}
    """
    print("Generating all possible sequences...")
    sequences = []
    
    # Generate all combinations of block_size-2 positions with values {0, 1, 2}
    for combination in itertools.product([0, 1, 2], repeat=block_size-2):
        sequence = [3] + list(combination) + [3] 
        sequences.append(sequence)
    
    print(f"Generated {len(sequences)} sequences")
    return sequences


def capture_sae_activations(gpt_model: GPT, sae_weights: Tuple[Dict, Dict], 
                          sequences: List[List[int]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Capture SAE activations for the given sequences.
    
    Args:
        gpt_model: Loaded GPT model
        sae_weights: Tuple of (sae.0 weights, sae.1 weights)
        sequences: List of input sequences
        
    Returns:
        Tuple of (sae.0 activations, sae.1 activations) as numpy arrays
    """
    print("Capturing SAE activations...")
    
    device = next(gpt_model.parameters()).device
    sae_0_weights, sae_1_weights = sae_weights
    
    # Process sequences in batches
    batch_size = 1000
    num_batches = (len(sequences) + batch_size - 1) // batch_size
    
    all_sae_0_activations = []
    all_sae_1_activations = []
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(sequences))
        batch_sequences = sequences[start_idx:end_idx]
        
        print(f"Processing batch {i+1}/{num_batches} (sequences {start_idx}-{end_idx-1})")
        
        # Convert sequences to tensor
        input_ids = torch.tensor(batch_sequences, device=device)
        
        # FIX THIS -- there is functionality to call SAE encoders directly
        ########################
        with torch.no_grad():
            # Manual forward pass to capture activations at hook points
            B, T = input_ids.size()
            
            # Embeddings
            pos = torch.arange(0, T, dtype=torch.long, device=device)
            pos_emb = gpt_model.transformer.wpe(pos)
            tok_emb = gpt_model.transformer.wte(input_ids)
            x = tok_emb + pos_emb
            
            # Forward through transformer blocks
            for block_idx, block in enumerate(gpt_model.transformer.h):
                # Capture resid_mid (after attention, before MLP)
                x_after_attn = x + block.attn(block.ln_1(x))
                
                if block_idx == 0:  # Assuming we want the first layer's activations
                    resid_mid = x_after_attn
                
                # Continue through MLP
                x = x_after_attn + block.mlp(block.ln_2(x_after_attn))
                
                if block_idx == 0:  # Assuming we want the first layer's activations
                    resid_post = x
        
        # Apply SAE encoders to get 2D latent features
        # Remove BOS tokens and final tokens for processing
        resid_mid_processed = resid_mid[:, 1:-1, :]  # Shape: (batch, seq_len-2, d_model)
        resid_post_processed = resid_post[:, 1:-1, :]  # Shape: (batch, seq_len-2, d_model)
        
        # Flatten over sequence dimension
        resid_mid_flat = resid_mid_processed.reshape(-1, resid_mid_processed.size(-1))
        resid_post_flat = resid_post_processed.reshape(-1, resid_post_processed.size(-1))
        
        # Apply SAE encoders (assuming they have W_enc and b_enc)
        # SAE.0 (constrained belief states)
        if 'W_enc' in sae_0_weights and 'b_enc' in sae_0_weights:
            sae_0_features = (resid_mid_flat - sae_0_weights.get('b_dec', 0)) @ sae_0_weights['W_enc'] + sae_0_weights['b_enc']
        else:
            # Fallback: use decoder weights transposed
            W_enc_0 = sae_0_weights['W_dec'].T
            sae_0_features = resid_mid_flat @ W_enc_0
        
        # SAE.1 (regular belief states)  
        if 'W_enc' in sae_1_weights and 'b_enc' in sae_1_weights:
            sae_1_features = (resid_post_flat - sae_1_weights.get('b_dec', 0)) @ sae_1_weights['W_enc'] + sae_1_weights['b_enc']
        else:
            # Fallback: use decoder weights transposed
            W_enc_1 = sae_1_weights['W_dec'].T
            sae_1_features = resid_post_flat @ W_enc_1
        ########################
        
        all_sae_0_activations.append(sae_0_features.cpu().numpy())
        all_sae_1_activations.append(sae_1_features.cpu().numpy())
    
    # Concatenate all batches
    sae_0_activations = np.concatenate(all_sae_0_activations, axis=0)
    sae_1_activations = np.concatenate(all_sae_1_activations, axis=0)
    
    print(f"SAE.0 activations shape: {sae_0_activations.shape}")
    print(f"SAE.1 activations shape: {sae_1_activations.shape}")
    
    return sae_0_activations, sae_1_activations


def compute_constrained_belief_projections(sequences: List[List[int]]) -> np.ndarray:
    """
    Compute constrained belief state projections for comparison with SAE.0.
    
    Args:
        sequences: List of input sequences
        
    Returns:
        2D projections of constrained belief states
    """
    print("Computing constrained belief state projections...")
    
    # Initialize Mess3 transition matrix
    x = 0.15
    a = 0.6  # Updated parameter
    transition_matrix = mess3(x, a)
    
    # Get initial belief state (stationary distribution)
    initial_belief = stationary_distribution(transition_matrix)
    
    all_belief_states = []
    
    for seq_idx, sequence in enumerate(sequences):
        if seq_idx % 1000 == 0:
            print(f"Processing sequence {seq_idx+1}/{len(sequences)}")
        
        # Start with initial belief
        current_belief = initial_belief.copy()
        sequence_beliefs = []
        
        # Process each observation in the sequence (skip BOS token and final token)
        for pos in range(1, len(sequence) - 1):
            observation = sequence[pos]
            
            # Update belief using constrained belief update
            current_belief = constrained_belief_update(transition_matrix, observation, current_belief, initial_belief)
            sequence_beliefs.append(current_belief.copy())
        
        all_belief_states.extend(sequence_beliefs)
    
    belief_states_array = np.array(all_belief_states)
    
    # Apply uniform-centered projection to get 2D coordinates
    projections = uniform_centered_projection(belief_states_array)
    
    print(f"Constrained belief projections shape: {projections.shape}")
    return projections


def compute_regular_belief_projections(sequences: List[List[int]]) -> np.ndarray:
    """
    Compute regular belief state projections for comparison with SAE.1.
    
    Args:
        sequences: List of input sequences
        
    Returns:
        2D projections of regular belief states
    """
    print("Computing regular belief state projections...")
    
    # Initialize Mess3 transition matrix
    x = 0.15
    a = 0.6  # Updated parameter
    transition_matrix = mess3(x, a)
    
    # Get initial belief state (stationary distribution)
    initial_belief = stationary_distribution(transition_matrix)
    
    all_belief_states = []
    
    for seq_idx, sequence in enumerate(sequences):
        if seq_idx % 1000 == 0:
            print(f"Processing sequence {seq_idx+1}/{len(sequences)}")
        
        # Start with initial belief
        current_belief = initial_belief.copy()
        sequence_beliefs = []
        
        # Process each observation in the sequence (skip BOS token and final token)
        for pos in range(1, len(sequence) - 1):
            observation = sequence[pos]
            
            # Update belief using regular belief update
            current_belief = belief_update(transition_matrix, observation, current_belief)
            sequence_beliefs.append(current_belief.copy())
        
        all_belief_states.extend(sequence_beliefs)
    
    belief_states_array = np.array(all_belief_states)
    
    # Apply uniform-centered projection to get 2D coordinates
    projections = uniform_centered_projection(belief_states_array)
    
    print(f"Regular belief projections shape: {projections.shape}")
    return projections


def find_optimal_transformation(ae_data: np.ndarray, theoretical_data: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray, dict]:
    """
    Find optimal linear transformation (rotation/reflection + scaling + translation) to align AE data with theoretical data.
    Uses orthogonal Procrustes analysis allowing reflections.
    
    Args:
        ae_data: Array of shape (N, 2) containing AE coordinates
        theoretical_data: Array of shape (N, 2) containing theoretical projections
        
    Returns:
        Tuple of (orthogonal_matrix, scale, translation, metrics)
        - orthogonal_matrix: 2x2 orthogonal matrix (rotation or reflection)
        - scale: scalar scaling factor
        - translation: 2D translation vector
        - metrics: dictionary with alignment quality metrics
    """
    # Center both datasets
    ae_centered = ae_data - np.mean(ae_data, axis=0)
    theoretical_centered = theoretical_data - np.mean(theoretical_data, axis=0)
    
    # Compute cross-covariance matrix
    H = ae_centered.T @ theoretical_centered
    
    # SVD decomposition
    U, S, Vt = np.linalg.svd(H)
    
    # Optimal orthogonal matrix (allows reflections)
    Q = U @ Vt
    
    # Apply orthogonal transformation to centered AE data
    ae_transformed = ae_centered @ Q.T
    
    # Find optimal scaling using least squares
    # ||s * ae_transformed - theoretical_centered||^2
    numerator = np.sum(ae_transformed * theoretical_centered)
    denominator = np.sum(ae_transformed * ae_transformed)
    optimal_scale = numerator / denominator if denominator > 0 else 1.0
    
    # Find optimal translation (centroids after scaling and rotation)
    theoretical_centroid = np.mean(theoretical_data, axis=0)
    ae_centroid = np.mean(ae_data, axis=0)
    optimal_translation = theoretical_centroid - optimal_scale * (Q @ ae_centroid)
    
    # Compute alignment metrics
    aligned_ae = optimal_scale * (ae_data @ Q.T) + optimal_translation
    distances = np.linalg.norm(aligned_ae - theoretical_data, axis=1)
    
    metrics = {
        'rmse': np.sqrt(np.mean(distances**2)),
        'mean_distance': np.mean(distances),
        'max_distance': np.max(distances),
        'determinant': np.linalg.det(Q),
        'is_reflection': np.linalg.det(Q) < 0,
        'correlation_x': np.corrcoef(aligned_ae[:, 0], theoretical_data[:, 0])[0, 1],
        'correlation_y': np.corrcoef(aligned_ae[:, 1], theoretical_data[:, 1])[0, 1],
        'original_rmse': np.sqrt(np.mean(np.linalg.norm(ae_data - theoretical_data, axis=1)**2))
    }
    
    return Q, optimal_scale, optimal_translation, metrics


def align_ae_to_theoretical(ae_data: np.ndarray, orthogonal_matrix: np.ndarray, 
                           scale: float, translation: np.ndarray) -> np.ndarray:
    """
    Apply the optimal transformation to align AE data with theoretical predictions.
    
    Args:
        ae_data: Array of shape (N, 2) containing AE coordinates
        orthogonal_matrix: 2x2 orthogonal transformation matrix
        scale: scalar scaling factor
        translation: 2D translation vector
        
    Returns:
        Aligned AE data of shape (N, 2)
    """
    return scale * (ae_data @ orthogonal_matrix.T) + translation


def compute_alignment_metrics(sae_coords: np.ndarray, theoretical_coords: np.ndarray) -> Dict:
    """
    Compute alignment metrics between SAE coordinates and theoretical projections.
    
    Args:
        sae_coords: SAE feature coordinates (N, 2)
        theoretical_coords: Theoretical belief state projections (N, 2)
        
    Returns:
        Dictionary of alignment metrics
    """
    # Compute correlations
    corr_x = np.corrcoef(sae_coords[:, 0], theoretical_coords[:, 0])[0, 1]
    corr_y = np.corrcoef(sae_coords[:, 1], theoretical_coords[:, 1])[0, 1]
    
    # Compute RMSE
    rmse = np.sqrt(np.mean(np.sum((sae_coords - theoretical_coords)**2, axis=1)))
    
    # Compute mean distance
    distances = np.linalg.norm(sae_coords - theoretical_coords, axis=1)
    mean_distance = np.mean(distances)
    max_distance = np.max(distances)
    
    return {
        'correlation_x': corr_x,
        'correlation_y': corr_y,
        'rmse': rmse,
        'mean_distance': mean_distance,
        'max_distance': max_distance
    }


def create_comparison_plots(sae_0_coords: np.ndarray, sae_1_coords: np.ndarray,
                          constrained_proj: np.ndarray, regular_proj: np.ndarray,
                          sequences: List[List[int]]):
    """
    Create comparison plots showing AE activations vs theoretical projections with alignment analysis.
    
    Args:
        sae_0_coords: AE.0 feature coordinates
        sae_1_coords: AE.1 feature coordinates  
        constrained_proj: Constrained belief state projections
        regular_proj: Regular belief state projections
        sequences: Original sequences for belief state colors
    """
    print("Creating comparison plots with alignment analysis...")
    
    # Compute optimal transformations
    print("Computing optimal transformations...")
    Q_0, scale_0, trans_0, metrics_0 = find_optimal_transformation(sae_0_coords, constrained_proj)
    Q_1, scale_1, trans_1, metrics_1 = find_optimal_transformation(sae_1_coords, regular_proj)
    
    # Apply transformations
    sae_0_aligned = align_ae_to_theoretical(sae_0_coords, Q_0, scale_0, trans_0)
    sae_1_aligned = align_ae_to_theoretical(sae_1_coords, Q_1, scale_1, trans_1)
    
    # Compute belief states for coloring
    x = 0.15
    a = 0.6
    transition_matrix = mess3(x, a)
    initial_belief = stationary_distribution(transition_matrix)
    
    # Compute constrained belief states for coloring
    constrained_belief_states = []
    for sequence in sequences:
        current_belief = initial_belief.copy()
        for pos in range(1, len(sequence) - 1):
            observation = sequence[pos]
            current_belief = constrained_belief_update(transition_matrix, observation, current_belief, initial_belief)
            constrained_belief_states.append(current_belief.copy())
    
    # Compute regular belief states for coloring
    regular_belief_states = []
    for sequence in sequences:
        current_belief = initial_belief.copy()
        for pos in range(1, len(sequence) - 1):
            observation = sequence[pos]
            current_belief = belief_update(transition_matrix, observation, current_belief)
            regular_belief_states.append(current_belief.copy())
    
    constrained_colors = np.array(constrained_belief_states)
    regular_colors = np.array(regular_belief_states)
    
    # Create six-panel plot (3 rows, 2 columns)
    fig, axes = plt.subplots(3, 2, figsize=(16, 24))
    
    scatter_params = {'alpha': 0.6, 's': 8, 'edgecolors': 'black', 'linewidth': 0.05}
    
    # Row 1: Original AE activations
    # Panel (0,0): AE.0 activations
    axes[0, 0].scatter(sae_0_coords[:, 0], sae_0_coords[:, 1], 
                      c=constrained_colors, **scatter_params)
    axes[0, 0].set_title('Original AE.0 Latent Activations\n(resid_mid → constrained belief states)', fontsize=12)
    axes[0, 0].set_xlabel('AE.0 Feature 1', fontsize=10)
    axes[0, 0].set_ylabel('AE.0 Feature 2', fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Panel (0,1): AE.1 activations
    axes[0, 1].scatter(sae_1_coords[:, 0], sae_1_coords[:, 1],
                      c=regular_colors, **scatter_params)
    axes[0, 1].set_title('Original AE.1 Latent Activations\n(resid_post → regular belief states)', fontsize=12)
    axes[0, 1].set_xlabel('AE.1 Feature 1', fontsize=10)
    axes[0, 1].set_ylabel('AE.1 Feature 2', fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Row 2: Theoretical projections
    # Panel (1,0): Constrained belief state projections
    axes[1, 0].scatter(constrained_proj[:, 0], constrained_proj[:, 1],
                      c=constrained_colors, **scatter_params)
    axes[1, 0].set_title('Theoretical: Constrained Belief States\n(Uniform-Centered Projection)', fontsize=12)
    axes[1, 0].set_xlabel('First Coordinate (tangent to 2-simplex)', fontsize=10)
    axes[1, 0].set_ylabel('Second Coordinate (tangent to 2-simplex)', fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Panel (1,1): Regular belief state projections
    axes[1, 1].scatter(regular_proj[:, 0], regular_proj[:, 1],
                      c=regular_colors, **scatter_params)
    axes[1, 1].set_title('Theoretical: Regular Belief States\n(Uniform-Centered Projection)', fontsize=12)
    axes[1, 1].set_xlabel('First Coordinate (tangent to 2-simplex)', fontsize=10)
    axes[1, 1].set_ylabel('Second Coordinate (tangent to 2-simplex)', fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Row 3: Aligned AE activations
    # Panel (2,0): Aligned AE.0 activations
    transform_type_0 = "Reflection" if metrics_0['is_reflection'] else "Rotation"
    axes[2, 0].scatter(sae_0_aligned[:, 0], sae_0_aligned[:, 1],
                      c=constrained_colors, **scatter_params)
    axes[2, 0].set_title(f'Aligned AE.0 Latent Activations\n({transform_type_0}, RMSE: {metrics_0["rmse"]:.4f})', fontsize=12)
    axes[2, 0].set_xlabel('Aligned AE.0 Feature 1', fontsize=10)
    axes[2, 0].set_ylabel('Aligned AE.0 Feature 2', fontsize=10)
    axes[2, 0].grid(True, alpha=0.3)
    
    # Panel (2,1): Aligned AE.1 activations
    transform_type_1 = "Reflection" if metrics_1['is_reflection'] else "Rotation"
    axes[2, 1].scatter(sae_1_aligned[:, 0], sae_1_aligned[:, 1],
                      c=regular_colors, **scatter_params)
    axes[2, 1].set_title(f'Aligned AE.1 Latent Activations\n({transform_type_1}, RMSE: {metrics_1["rmse"]:.4f})', fontsize=12)
    axes[2, 1].set_xlabel('Aligned AE.1 Feature 1', fontsize=10)
    axes[2, 1].set_ylabel('Aligned AE.1 Feature 2', fontsize=10)
    axes[2, 1].grid(True, alpha=0.3)
    
    # Add overall title and color explanation
    fig.suptitle('AE Latent Activations vs Theoretical Belief State Projections with Optimal Alignment', fontsize=16, y=0.98)
    fig.text(0.5, 0.02, 'Color represents belief state: Red=State 0, Green=State 1, Blue=State 2', 
             ha='center', fontsize=12, style='italic')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.94, bottom=0.06)
    
    # Print alignment results
    print(f"\n=== Alignment Results ===")
    print(f"AE.0 vs Constrained Belief States:")
    print(f"  Transformation: {transform_type_0} (det = {metrics_0['determinant']:.4f})")
    print(f"  Scale factor: {scale_0:.4f}")
    print(f"  Translation: [{trans_0[0]:.4f}, {trans_0[1]:.4f}]")
    print(f"  RMSE improvement: {metrics_0['original_rmse']:.4f} → {metrics_0['rmse']:.4f}")
    print(f"  Correlations: X={metrics_0['correlation_x']:.4f}, Y={metrics_0['correlation_y']:.4f}")
    
    print(f"\nAE.1 vs Regular Belief States:")
    print(f"  Transformation: {transform_type_1} (det = {metrics_1['determinant']:.4f})")
    print(f"  Scale factor: {scale_1:.4f}")
    print(f"  Translation: [{trans_1[0]:.4f}, {trans_1[1]:.4f}]")
    print(f"  RMSE improvement: {metrics_1['original_rmse']:.4f} → {metrics_1['rmse']:.4f}")
    print(f"  Correlations: X={metrics_1['correlation_x']:.4f}, Y={metrics_1['correlation_y']:.4f}")
    
    # Save the plot
    output_path = 'play/plots/ae_vs_theoretical_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to {output_path}")
    
    plt.show()


def main():
    """Main function to run the SAE analysis."""
    print("=== SAE Feature Activation Analysis ===\n")
    
    # Configuration
    block_size = 12  # Total length including BOS tokens
    gpt_checkpoint = "checkpoints/mess3_12_64x1"
    sae_checkpoint = "checkpoints/jsae_block.mess3_12_64x1_2feat_dense_play"
    
    # Load models
    gpt_model = load_gpt_model(gpt_checkpoint)
    sae_weights = load_sae_models(sae_checkpoint)
    
    # Generate sequences
    sequences = generate_sequences(block_size)
    print(f"Total sequences: {len(sequences)}")
    
    # Capture SAE activations
    sae_0_activations, sae_1_activations = capture_sae_activations(gpt_model, sae_weights, sequences)
    
    # Compute theoretical projections
    constrained_projections = compute_constrained_belief_projections(sequences)
    regular_projections = compute_regular_belief_projections(sequences)
    
    # Compute alignment metrics
    print("\n=== Alignment Metrics ===")
    
    metrics_0 = compute_alignment_metrics(sae_0_activations, constrained_projections)
    print(f"AE.0 vs Constrained Belief States:")
    for key, value in metrics_0.items():
        print(f"  {key}: {value:.4f}")
    
    metrics_1 = compute_alignment_metrics(sae_1_activations, regular_projections)
    print(f"\nAE.1 vs Regular Belief States:")
    for key, value in metrics_1.items():
        print(f"  {key}: {value:.4f}")
    
    # Create comparison plots
    create_comparison_plots(sae_0_activations, sae_1_activations,
                          constrained_projections, regular_projections, sequences)
    
    print("\n=== Analysis Complete ===")


if __name__ == "__main__":
    main()