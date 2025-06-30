#!/usr/bin/env python3
"""
Script to perform PCA analysis on GPT model activations.
Generates all possible length-N sequences with BOS token 3, captures activations,
and performs PCA visualization.
"""

import sys
import os
import itertools
from typing import List, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from models.gpt import GPT
from data.comp_mech import belief_update, constrained_belief_update, stationary_distribution
from data.mess3 import mess3
from play.utils import uniform_centered_projection


def generate_all_sequences(block_size: int) -> List[List[int]]:
    """
    Generate all possible length-10 sequences with BOS token 3.
    Format: [3, *, *, *, *, *, *, *, *, *] where * ∈ {0, 1, 2}
    Returns 3^block_size-1
    """
    print("Generating all possible sequences...")
    sequences = []
    
    # Generate all combinations of 9 positions with values {0, 1, 2}
    for combination in itertools.product([0, 1, 2], repeat=block_size-2):
        sequence = [3] + list(combination) + [3] 
        sequences.append(sequence)
    
    print(f"Generated {len(sequences)} sequences")
    return sequences


class ActivationCapture:
    """Helper class to capture activations during forward pass."""
    
    def __init__(self):
        self.intermediate_activations = []  # After attention, before MLP
        self.final_activations = []         # Before final layer norm
        
    def clear(self):
        self.intermediate_activations.clear()
        self.final_activations.clear()


def capture_activations_batch(model: GPT, sequences: List[List[int]], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Capture activations for a batch of sequences.
    Returns intermediate, final, and post-layernorm activations.
    """
    batch_size = len(sequences)
    seq_len = len(sequences[0])
    
    # Convert sequences to tensor
    input_ids = torch.tensor(sequences, device=device)  # Shape: (batch_size, seq_len)
    
    intermediate_batch = []
    final_batch = []
    
    with torch.no_grad():
        # Manual forward pass to capture activations
        B, T = input_ids.size()
        
        # Embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=device)
        pos_emb = model.transformer.wpe(pos)
        tok_emb = model.transformer.wte(input_ids)
        x = tok_emb + pos_emb
        
        # Forward through transformer blocks
        for block in model.transformer.h:
            # Capture activation after attention but before MLP
            x_after_attn = x + block.attn(block.ln_1(x))
            intermediate_batch.append(x_after_attn.clone())
            
            # Continue through MLP
            x = x_after_attn + block.mlp(block.ln_2(x_after_attn))
        
        # x now contains final activations before layer norm
        final_batch.append(x.clone())
        
        # Apply final layer norm
        x_after_ln = model.transformer.ln_f(x)
        post_layernorm_batch = [x_after_ln.clone()]
    
    # Stack the captured activations
    intermediate_activations = intermediate_batch[0]  # Shape: (batch_size, seq_len, n_embd)
    final_activations = final_batch[0]                # Shape: (batch_size, seq_len, n_embd)
    post_layernorm_activations = post_layernorm_batch[0]  # Shape: (batch_size, seq_len, n_embd)
    
    return intermediate_activations, final_activations, post_layernorm_activations


def process_activations(activations: torch.Tensor) -> np.ndarray:
    """
    Process activations: remove BOS token and flatten over sequence dimension.
    Input shape: (num_sequences, seq_len, n_embd)
    Output shape: (num_sequences * (seq_len-1), n_embd)
    """

    # Remove BOS token and final position (position 0 & final token)
    num_block = activations.shape[1]
    activations_reduced = activations[:, 1:num_block-1, :]  # Shape: (num_sequences, 8, n_embd)

    # Flatten over sequence dimension
    num_sequences, seq_len_minus_2, n_embd = activations_reduced.shape
    flattened = activations_reduced.reshape(num_sequences * seq_len_minus_2, n_embd)
    
    return flattened.cpu().numpy()


def compute_theoretical_belief_states(sequences: List[List[int]], constrained: bool = False) -> np.ndarray:
    """
    Compute theoretical belief states for each sequence position using belief update functions.
    
    Args:
        sequences: List of sequences, each starting with BOS token 3
        constrained: If True, use constrained_belief_update; if False, use belief_update
    
    Returns:
        Array of shape (num_sequences * (seq_len-2), 3) containing belief states
        Excludes BOS token (position 0) and final token positions
    """
    print(f"Computing theoretical belief states (constrained={constrained})...")
    
    # Initialize Mess3 transition matrix with parameters from model training
    # These parameters should match the training setup
    x = 0.15  # Parameter from mess3 process
    a = 0.6  # Parameter from mess3 process
    transition_matrix = mess3(x, a)
    
    # Get initial belief state (stationary distribution)
    initial_belief = stationary_distribution(transition_matrix)
    print(f"Initial belief state: {initial_belief}")
    
    all_belief_states = []
    
    for seq_idx, sequence in enumerate(sequences):
        if seq_idx % 1000 == 0:
            print(f"Processing sequence {seq_idx+1}/{len(sequences)}")
        
        # Start with initial belief
        current_belief = initial_belief.copy()
        sequence_beliefs = []
        
        # Process each observation in the sequence (skip BOS token at position 0 and final token)
        for pos in range(1, len(sequence) - 1):
            observation = sequence[pos]
            
            # Update belief based on observation
            if constrained:
                current_belief = constrained_belief_update(transition_matrix, observation, current_belief, initial_belief)
            else:
                current_belief = belief_update(transition_matrix, observation, current_belief)
            
            sequence_beliefs.append(current_belief.copy())
        
        all_belief_states.extend(sequence_beliefs)
    
    return np.array(all_belief_states)


def find_optimal_transformation(pca_data: np.ndarray, theoretical_data: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray, dict]:
    """
    Find optimal linear transformation (rotation/reflection + scaling + translation) to align PCA data with theoretical data.
    Uses orthogonal Procrustes analysis allowing reflections.
    
    Args:
        pca_data: Array of shape (N, 2) containing PCA coordinates
        theoretical_data: Array of shape (N, 2) containing theoretical projections
        
    Returns:
        Tuple of (orthogonal_matrix, scale, translation, metrics)
        - orthogonal_matrix: 2x2 orthogonal matrix (rotation or reflection)
        - scale: scalar scaling factor
        - translation: 2D translation vector
        - metrics: dictionary with alignment quality metrics
    """
    # Center both datasets
    pca_centered = pca_data - np.mean(pca_data, axis=0)
    theoretical_centered = theoretical_data - np.mean(theoretical_data, axis=0)
    
    # Compute cross-covariance matrix
    H = pca_centered.T @ theoretical_centered
    
    # SVD decomposition
    U, S, Vt = np.linalg.svd(H)
    
    # Optimal orthogonal matrix (allows reflections)
    Q = U @ Vt
    
    # Apply orthogonal transformation to centered PCA data
    pca_transformed = pca_centered @ Q.T
    
    # Find optimal scaling using least squares
    # ||s * pca_transformed - theoretical_centered||^2
    numerator = np.sum(pca_transformed * theoretical_centered)
    denominator = np.sum(pca_transformed * pca_transformed)
    optimal_scale = numerator / denominator if denominator > 0 else 1.0
    
    # Find optimal translation (centroids after scaling and rotation)
    theoretical_centroid = np.mean(theoretical_data, axis=0)
    pca_centroid = np.mean(pca_data, axis=0)
    optimal_translation = theoretical_centroid - optimal_scale * (Q @ pca_centroid)
    
    # Compute alignment metrics
    aligned_pca = optimal_scale * (pca_data @ Q.T) + optimal_translation
    distances = np.linalg.norm(aligned_pca - theoretical_data, axis=1)
    
    metrics = {
        'rmse': np.sqrt(np.mean(distances**2)),
        'mean_distance': np.mean(distances),
        'max_distance': np.max(distances),
        'determinant': np.linalg.det(Q),
        'is_reflection': np.linalg.det(Q) < 0,
        'correlation_x': np.corrcoef(aligned_pca[:, 0], theoretical_data[:, 0])[0, 1],
        'correlation_y': np.corrcoef(aligned_pca[:, 1], theoretical_data[:, 1])[0, 1],
        'original_rmse': np.sqrt(np.mean(np.linalg.norm(pca_data - theoretical_data, axis=1)**2))
    }
    
    return Q, optimal_scale, optimal_translation, metrics


def align_pca_to_theoretical(pca_data: np.ndarray, orthogonal_matrix: np.ndarray, 
                           scale: float, translation: np.ndarray) -> np.ndarray:
    """
    Apply the optimal transformation to align PCA data with theoretical predictions.
    
    Args:
        pca_data: Array of shape (N, 2) containing PCA coordinates
        orthogonal_matrix: 2x2 orthogonal transformation matrix
        scale: scalar scaling factor
        translation: 2D translation vector
        
    Returns:
        Aligned PCA data of shape (N, 2)
    """
    return scale * (pca_data @ orthogonal_matrix.T) + translation


def compute_alignment_summary(transformations: dict) -> str:
    """
    Create a summary string of all transformation results.
    
    Args:
        transformations: Dictionary containing transformation results for each activation type
        
    Returns:
        Formatted summary string
    """
    summary = "=== Alignment Results Summary ===\n"
    
    for name, (Q, scale, translation, metrics) in transformations.items():
        transformation_type = "Reflection" if metrics['is_reflection'] else "Rotation"
        summary += f"\n{name}:\n"
        summary += f"  Transformation: {transformation_type} (det = {metrics['determinant']:.4f})\n"
        summary += f"  Scale factor: {scale:.4f}\n"
        summary += f"  Translation: [{translation[0]:.4f}, {translation[1]:.4f}]\n"
        summary += f"  RMSE improvement: {metrics['original_rmse']:.4f} → {metrics['rmse']:.4f}\n"
        summary += f"  Correlations: X={metrics['correlation_x']:.4f}, Y={metrics['correlation_y']:.4f}\n"
    
    return summary


def perform_pca_and_plot(intermediate_data: np.ndarray, final_data: np.ndarray, post_layernorm_data: np.ndarray, sequences: List[List[int]]):
    """
    Perform PCA on all three datasets and create nine-panel visualization comparing with theoretical predictions and aligned PCA.
    """
    print("Performing PCA analysis and computing theoretical predictions...")
    
    # Compute theoretical belief states
    print("Computing constrained belief states for intermediate activations...")
    constrained_belief_states = compute_theoretical_belief_states(sequences, constrained=True)
    
    print("Computing regular belief states for final/post-layernorm activations...")
    regular_belief_states = compute_theoretical_belief_states(sequences, constrained=False)
    
    # Apply uniform-centered projection to theoretical belief states
    print("Applying uniform-centered projection to theoretical belief states...")
    constrained_projected = uniform_centered_projection(constrained_belief_states)
    regular_projected = uniform_centered_projection(regular_belief_states)
    
    # Perform PCA on activation data
    print("Performing PCA on activation data...")
    pca_intermediate = PCA(n_components=2)
    pca_final = PCA(n_components=2)
    pca_post_layernorm = PCA(n_components=2)
    
    intermediate_pca = pca_intermediate.fit_transform(intermediate_data)
    final_pca = pca_final.fit_transform(final_data)
    post_layernorm_pca = pca_post_layernorm.fit_transform(post_layernorm_data)
    
    # Find optimal transformations for each PCA dataset
    print("Computing optimal transformations...")
    transformations = {}
    
    print("  Aligning intermediate PCA to constrained belief states...")
    Q_int, scale_int, trans_int, metrics_int = find_optimal_transformation(intermediate_pca, constrained_projected)
    transformations['Intermediate Activations'] = (Q_int, scale_int, trans_int, metrics_int)
    
    print("  Aligning final PCA to regular belief states...")
    Q_final, scale_final, trans_final, metrics_final = find_optimal_transformation(final_pca, regular_projected)
    transformations['Final Activations'] = (Q_final, scale_final, trans_final, metrics_final)
    
    print("  Aligning post-layernorm PCA to regular belief states...")
    Q_post, scale_post, trans_post, metrics_post = find_optimal_transformation(post_layernorm_pca, regular_projected)
    transformations['Post-LayerNorm Activations'] = (Q_post, scale_post, trans_post, metrics_post)
    
    # Apply transformations
    intermediate_aligned = align_pca_to_theoretical(intermediate_pca, Q_int, scale_int, trans_int)
    final_aligned = align_pca_to_theoretical(final_pca, Q_final, scale_final, trans_final)
    post_layernorm_aligned = align_pca_to_theoretical(post_layernorm_pca, Q_post, scale_post, trans_post)
    
    # Create nine-panel plot
    fig, axes = plt.subplots(3, 3, figsize=(24, 24))
    
    # Set up RGB coloring based on belief states
    constrained_colors = constrained_belief_states  # Shape: (N, 3) - use as RGB
    regular_colors = regular_belief_states  # Shape: (N, 3) - use as RGB
    
    # Common plot parameters
    scatter_params = {'alpha': 0.6, 's': 8, 'edgecolors': 'black', 'linewidth': 0.05}
    
    # Row 1: Original PCA projections
    # Panel (0,0): Intermediate activations PCA
    axes[0, 0].scatter(intermediate_pca[:, 0], intermediate_pca[:, 1], 
                      c=constrained_colors, **scatter_params)
    axes[0, 0].set_title('Original PCA: Intermediate Activations\n(After Attention, Before MLP)', fontsize=12)
    axes[0, 0].set_xlabel(f'PC1 ({pca_intermediate.explained_variance_ratio_[0]:.3f} var)', fontsize=10)
    axes[0, 0].set_ylabel(f'PC2 ({pca_intermediate.explained_variance_ratio_[1]:.3f} var)', fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Panel (0,1): Final activations PCA
    axes[0, 1].scatter(final_pca[:, 0], final_pca[:, 1],
                      c=regular_colors, **scatter_params)
    axes[0, 1].set_title('Original PCA: Final Activations\n(Before Final Layer Norm)', fontsize=12)
    axes[0, 1].set_xlabel(f'PC1 ({pca_final.explained_variance_ratio_[0]:.3f} var)', fontsize=10)
    axes[0, 1].set_ylabel(f'PC2 ({pca_final.explained_variance_ratio_[1]:.3f} var)', fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Panel (0,2): Post-LayerNorm activations PCA
    axes[0, 2].scatter(post_layernorm_pca[:, 0], post_layernorm_pca[:, 1],
                      c=regular_colors, **scatter_params)
    axes[0, 2].set_title('Original PCA: Post-LayerNorm Activations\n(After Final Layer Norm)', fontsize=12)
    axes[0, 2].set_xlabel(f'PC1 ({pca_post_layernorm.explained_variance_ratio_[0]:.3f} var)', fontsize=10)
    axes[0, 2].set_ylabel(f'PC2 ({pca_post_layernorm.explained_variance_ratio_[1]:.3f} var)', fontsize=10)
    axes[0, 2].grid(True, alpha=0.3)
    
    # Row 2: Theoretical predictions
    # Panel (1,0): Constrained belief states projection
    axes[1, 0].scatter(constrained_projected[:, 0], constrained_projected[:, 1],
                      c=constrained_colors, **scatter_params)
    axes[1, 0].set_title('Theoretical: Constrained Belief States\n(Uniform-Centered Projection)', fontsize=12)
    axes[1, 0].set_xlabel('First Coordinate (tangent to 2-simplex)', fontsize=10)
    axes[1, 0].set_ylabel('Second Coordinate (tangent to 2-simplex)', fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Panel (1,1): Regular belief states projection
    axes[1, 1].scatter(regular_projected[:, 0], regular_projected[:, 1],
                      c=regular_colors, **scatter_params)
    axes[1, 1].set_title('Theoretical: Regular Belief States\n(Uniform-Centered Projection)', fontsize=12)
    axes[1, 1].set_xlabel('First Coordinate (tangent to 2-simplex)', fontsize=10)
    axes[1, 1].set_ylabel('Second Coordinate (tangent to 2-simplex)', fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Panel (1,2): Regular belief states projection (duplicate for consistency)
    axes[1, 2].scatter(regular_projected[:, 0], regular_projected[:, 1],
                      c=regular_colors, **scatter_params)
    axes[1, 2].set_title('Theoretical: Regular Belief States\n(Uniform-Centered Projection)', fontsize=12)
    axes[1, 2].set_xlabel('First Coordinate (tangent to 2-simplex)', fontsize=10)
    axes[1, 2].set_ylabel('Second Coordinate (tangent to 2-simplex)', fontsize=10)
    axes[1, 2].grid(True, alpha=0.3)
    
    # Row 3: Aligned PCA projections
    # Panel (2,0): Aligned intermediate PCA
    transform_type_int = "Reflection" if metrics_int['is_reflection'] else "Rotation"
    axes[2, 0].scatter(intermediate_aligned[:, 0], intermediate_aligned[:, 1],
                      c=constrained_colors, **scatter_params)
    axes[2, 0].set_title(f'Aligned PCA: Intermediate Activations\n({transform_type_int}, RMSE: {metrics_int["rmse"]:.4f})', fontsize=12)
    axes[2, 0].set_xlabel('Aligned Coordinate 1', fontsize=10)
    axes[2, 0].set_ylabel('Aligned Coordinate 2', fontsize=10)
    axes[2, 0].grid(True, alpha=0.3)
    
    # Panel (2,1): Aligned final PCA
    transform_type_final = "Reflection" if metrics_final['is_reflection'] else "Rotation"
    axes[2, 1].scatter(final_aligned[:, 0], final_aligned[:, 1],
                      c=regular_colors, **scatter_params)
    axes[2, 1].set_title(f'Aligned PCA: Final Activations\n({transform_type_final}, RMSE: {metrics_final["rmse"]:.4f})', fontsize=12)
    axes[2, 1].set_xlabel('Aligned Coordinate 1', fontsize=10)
    axes[2, 1].set_ylabel('Aligned Coordinate 2', fontsize=10)
    axes[2, 1].grid(True, alpha=0.3)
    
    # Panel (2,2): Aligned post-layernorm PCA
    transform_type_post = "Reflection" if metrics_post['is_reflection'] else "Rotation"
    axes[2, 2].scatter(post_layernorm_aligned[:, 0], post_layernorm_aligned[:, 1],
                      c=regular_colors, **scatter_params)
    axes[2, 2].set_title(f'Aligned PCA: Post-LayerNorm Activations\n({transform_type_post}, RMSE: {metrics_post["rmse"]:.4f})', fontsize=12)
    axes[2, 2].set_xlabel('Aligned Coordinate 1', fontsize=10)
    axes[2, 2].set_ylabel('Aligned Coordinate 2', fontsize=10)
    axes[2, 2].grid(True, alpha=0.3)
    
    # Add overall title and color explanation
    fig.suptitle('PCA vs Theoretical Predictions with Optimal Alignment', fontsize=16, y=0.98)
    fig.text(0.5, 0.02, 'Color represents belief state: Red=State 0, Green=State 1, Blue=State 2', 
             ha='center', fontsize=12, style='italic')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.94, bottom=0.06)
    
    # Save the plot
    output_path = 'play/plots/pca_vs_theoretical_nine_panel.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Nine-panel plot saved to {output_path}")
    
    plt.show()
    
    # Print comprehensive summary statistics
    print(f"\n=== PCA Results Summary ===")
    print(f"Intermediate Activations (After Attention, Before MLP):")
    print(f"  Total variance explained by PC1+PC2: {sum(pca_intermediate.explained_variance_ratio_[:2]):.4f}")
    print(f"  PC1: {pca_intermediate.explained_variance_ratio_[0]:.4f}")
    print(f"  PC2: {pca_intermediate.explained_variance_ratio_[1]:.4f}")
    
    print(f"\nFinal Activations (Before Final Layer Norm):")
    print(f"  Total variance explained by PC1+PC2: {sum(pca_final.explained_variance_ratio_[:2]):.4f}")
    print(f"  PC1: {pca_final.explained_variance_ratio_[0]:.4f}")
    print(f"  PC2: {pca_final.explained_variance_ratio_[1]:.4f}")
    
    print(f"\nPost-LayerNorm Activations (After Final Layer Norm):")
    print(f"  Total variance explained by PC1+PC2: {sum(pca_post_layernorm.explained_variance_ratio_[:2]):.4f}")
    print(f"  PC1: {pca_post_layernorm.explained_variance_ratio_[0]:.4f}")
    print(f"  PC2: {pca_post_layernorm.explained_variance_ratio_[1]:.4f}")
    
    print(f"\n=== Theoretical Belief States Summary ===")
    print(f"Constrained belief states shape: {constrained_belief_states.shape}")
    print(f"Regular belief states shape: {regular_belief_states.shape}") 
    print(f"Projected constrained states shape: {constrained_projected.shape}")
    print(f"Projected regular states shape: {regular_projected.shape}")
    
    # Print detailed alignment results
    print(compute_alignment_summary(transformations))


def main():
    print("=== GPT Activation PCA Analysis ===\n")
    
    # Fix variables
    block_size = 12  # Total length including BOS token
    model_path = "checkpoints/mess3_12_64x1/checkpoint_step_0"

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {model_path}")
    model = GPT.load(model_path, device)
    model.eval()
    print(f"Model loaded successfully!")
    print(f"Config: {model.config}\n")
    
    # Generate all sequences
    sequences = generate_all_sequences(block_size)
    print(f"Total sequences to process: {len(sequences)}")
    print(f"Each sequence length: {len(sequences[0])}")
    print(f"Example sequence: {sequences[0]}\n")
    
    # Process sequences in batches to manage memory
    batch_size = 1000  # Adjust based on available memory
    num_batches = (len(sequences) + batch_size - 1) // batch_size
    
    all_intermediate = []
    all_final = []
    all_post_layernorm = []
    
    print(f"Processing {num_batches} batches of size {batch_size}...")
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(sequences))
        batch_sequences = sequences[start_idx:end_idx]
        
        print(f"Processing batch {i+1}/{num_batches} (sequences {start_idx}-{end_idx-1})")
        
        # Capture activations for this batch
        intermediate_batch, final_batch, post_layernorm_batch = capture_activations_batch(model, batch_sequences, device)
        
        all_intermediate.append(intermediate_batch)
        all_final.append(final_batch)
        all_post_layernorm.append(post_layernorm_batch)
    
    # Concatenate all batches
    print("Concatenating all batches...")
    intermediate_activations = torch.cat(all_intermediate, dim=0)
    final_activations = torch.cat(all_final, dim=0)
    post_layernorm_activations = torch.cat(all_post_layernorm, dim=0)
    
    print(f"Intermediate activations shape: {intermediate_activations.shape}")
    print(f"Final activations shape: {final_activations.shape}")
    print(f"Post-LayerNorm activations shape: {post_layernorm_activations.shape}")
    
    # Process activations (remove BOS token and flatten)
    print("Processing activations...")
    intermediate_processed = process_activations(intermediate_activations)
    final_processed = process_activations(final_activations)
    post_layernorm_processed = process_activations(post_layernorm_activations)
    
    print(f"Processed intermediate shape: {intermediate_processed.shape}")
    print(f"Processed final shape: {final_processed.shape}")
    print(f"Processed post-LayerNorm shape: {post_layernorm_processed.shape}")
    print(f"Total activation vectors: {intermediate_processed.shape[0]}")
    
    # Perform PCA and create plots
    perform_pca_and_plot(intermediate_processed, final_processed, post_layernorm_processed, sequences)
    
    print("\n=== Analysis Complete ===")


if __name__ == "__main__":
    main()