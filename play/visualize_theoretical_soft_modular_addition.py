#!/usr/bin/env python3
"""
Visualization script for theoretical soft modular addition belief state geometry.
"""

import sys
import os
import itertools

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

from data.comp_mech import belief_update_general, stationary_distribution
from data.modular_addition import soft_modular_addition


# def generate_modular_addition_sequences(p: int):
#     """Generate all valid modular addition sequences [BOS, a, b, c] where a + b ≡ c (mod p)."""
#     sequences = []
#     for a in range(p):
#         for b in range(p):
#             c = (a + b) % p
#             sequence = [p, a, b, c]  # BOS token is p
#             sequences.append(sequence)
#     return sequences

def generate_modular_addition_sequences(p: int):
    """Generate all valid modular addition sequences [BOS, a, b, c] where a + b ≡ c (mod p)."""
    sequences = []
    for a in range(p):
        for b in range(p):
            for c in range(p):
                    sequence = [p, a, b, c]  # BOS token is p
                    sequences.append(sequence)
    return sequences


def compute_belief_projections(sequences, p: int):
    """Compute belief state projections using QR decomposition for change of basis."""
    # Get transition matrix
    transition_matrix = soft_modular_addition(p)
    
    # Set up initial belief and one_vector (fixed patterns for all p)
    num_states = 7  # Always 7 states regardless of p
    # Initial belief: exactly [1,0,0,0,0,0,0] for all p
    initial_belief = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # One vector: exactly [[1], [0], [0], [1], [0], [0], [1]] for all p
    one_vector = np.array([[1], [0], [0], [1], [0], [0], [1]])
    
    print(f"Initial belief state: {initial_belief}")
    print(f"One vector: {one_vector.flatten()}")
    
    all_belief_states = []
    sequence_labels = []  # Track which sequence each belief state came from
    position_labels = []  # Track which position each belief state came from
    
    for seq_idx, sequence in enumerate(sequences):
        current_belief = initial_belief.copy()
        # Update belief on positions 1,2 (a,b) - skip BOS and final result
        # for pos in range(1, len(sequence) - 1):
        # for pos in [len(sequence) - 1]:
        # for pos in [1,2]:
        # for pos in [1,2,3]:
        for pos in [1,2]:
            observation = sequence[pos]
            current_belief = belief_update_general(transition_matrix, observation, current_belief, one_vector)
            # Check for NaN values and handle them
            if np.any(np.isnan(current_belief)) or np.any(np.isinf(current_belief)):
                print(f"Warning: NaN/Inf detected in belief update for sequence {sequence}, pos {pos}, obs {observation}")
                print(f"Current belief before: {current_belief}")
                # Skip this belief state or handle appropriately
                continue
            all_belief_states.append(current_belief.copy())
            sequence_labels.append(seq_idx)
            position_labels.append(pos)
    
    if len(all_belief_states) == 0:
        print("Warning: No valid belief states generated")
        return np.array([]).reshape(0, num_states-1), np.array([]).reshape(0, num_states)
    
    belief_states_array = np.array(all_belief_states)
    
    # Check for NaN values in belief states array
    if np.any(np.isnan(belief_states_array)) or np.any(np.isinf(belief_states_array)):
        print("Warning: NaN/Inf values found in belief states array")
        # Remove rows with NaN/Inf values
        valid_mask = ~(np.isnan(belief_states_array).any(axis=1) | np.isinf(belief_states_array).any(axis=1))
        belief_states_array = belief_states_array[valid_mask]
        print(f"Filtered to {len(belief_states_array)} valid belief states")
    
    if len(belief_states_array) == 0:
        print("Warning: No valid belief states after filtering")
        return np.array([]).reshape(0, num_states-1), np.array([]).reshape(0, num_states)
    
    # Create change of basis matrix with one_vector as first basis vector
    one_vector_normalized = one_vector / np.linalg.norm(one_vector)
    
    # Create matrix with one_vector as first column and identity for remaining columns
    A = np.column_stack([one_vector_normalized.flatten(), np.eye(num_states)])
    
    # Use QR decomposition to get orthonormal basis
    Q, _ = np.linalg.qr(A)
    basis_matrix = Q[:, :num_states]
    
    # Transform belief states to new basis
    belief_states_new_basis = (basis_matrix.T @ belief_states_array.T).T
    
    # Extract all coordinates except the zeroth
    projections = belief_states_new_basis[:, 1:]
    
    return projections, belief_states_array, sequence_labels, position_labels


def visualize_3d(projections, sequence_labels, sequences, p, pca, position_labels):
    """Create 3D visualization using first three PCA components."""
    # Perform PCA to reduce dimensionality to 3D
    pca_3d = PCA(n_components=3)
    projections_3d = pca_3d.fit_transform(projections)
    
    print(f"3D PCA explained variance ratio: {pca_3d.explained_variance_ratio_}")
    print(f"3D Total variance explained: {pca_3d.explained_variance_ratio_.sum():.3f}")
    
    # Create 3D visualization
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Convert to numpy arrays for easier indexing
    projections_3d = np.array(projections_3d)
    position_labels = np.array(position_labels)
    sequence_labels = np.array(sequence_labels)
    
    # Create colors based on sum up to that point or difference for pos3
    colors = []
    for i, (seq_idx, pos) in enumerate(zip(sequence_labels, position_labels)):
        sequence = sequences[seq_idx]
        a = sequence[1]  # First operand
        b = sequence[2]  # Second operand
        c = sequence[3]  # Result (expected to be (a + b) mod p)
        
        if pos == 1:
            # After processing 'a', the sum so far is just 'a'
            color_value = a
        elif pos == 2:
            # After processing 'a' and 'b', the sum so far is 'a + b mod p'
            color_value = (a + b) % p
        elif pos == 3:
            # After processing 'c', color by the value of c itself
            # This shows how the model represents the final result
            color_value = c
        else:
            # Fallback for other positions
            color_value = 0
            
        colors.append(color_value)
    colors = np.array(colors)
    
    # Plot different positions with different markers
    unique_positions = np.unique(position_labels)
    markers = ['o', '^', 's', 'D']  # circle, triangle, square, diamond
    marker_labels = ['Position 1 (after a)', 'Position 2 (after b)', 'Position 3 (after c)', 'Position 4']
    
    scatters = []
    for i, pos in enumerate(unique_positions):
        mask = position_labels == pos
        marker = markers[i % len(markers)]
        label = marker_labels[i % len(marker_labels)]
        
        scatter = ax.scatter(projections_3d[mask, 0], projections_3d[mask, 1], projections_3d[mask, 2],
                            c=colors[mask], cmap='viridis', alpha=0.6, s=30, 
                            marker=marker, edgecolors='black', linewidth=0.3, label=label)
        scatters.append(scatter)
    
    ax.set_xlabel(f'PC1 ({pca_3d.explained_variance_ratio_[0]:.3f} variance)')
    ax.set_ylabel(f'PC2 ({pca_3d.explained_variance_ratio_[1]:.3f} variance)')
    ax.set_zlabel(f'PC3 ({pca_3d.explained_variance_ratio_[2]:.3f} variance)')
    
    ax.set_title(f'3D Soft Modular Addition Belief States (p={p})')
    
    # Add legend for markers
    ax.legend(loc='upper left', bbox_to_anchor=(0.0, 1.0))
    
    # Add colorbar for values
    cbar = plt.colorbar(scatters[0], ax=ax, shrink=0.5, aspect=20)
    cbar.set_label('Color: pos1=a, pos2=a+b mod p, pos3=c')
    
    plt.tight_layout()
    
    # Save 3D plot
    plt.savefig('play/plots/theoretical_soft_modular_addition_3d.png', dpi=300, bbox_inches='tight')
    print("3D plot saved to 'play/plots/theoretical_soft_modular_addition_3d.png'")
    
    # Show interactive plot (allows rotation with mouse)
    print("Showing interactive 3D plot. Use mouse to rotate, zoom, and pan.")
    print("Close the plot window to continue...")
    plt.show()


def visualize_multiple_3d_projections(projections_5d, pca_5, sequence_labels, sequences, p, position_labels):
    """Create multiple 3D projections of the 5D space to explore structure."""
    
    # Define the different 3D projections we want to create
    projections_to_show = [
        ([0, 1, 2], "PC1-PC2-PC3 (Highest Variance)"),
        ([0, 2, 4], "PC1-PC3-PC5 (Skip PC2)"),
        ([1, 2, 3], "PC2-PC3-PC4 (Middle Components)"),
        ([2, 3, 4], "PC3-PC4-PC5 (Lower Variance)")
    ]
    
    # Create a 2x2 subplot layout
    fig = plt.figure(figsize=(16, 12))
    
    # Convert to numpy arrays for easier indexing
    projections_5d = np.array(projections_5d)
    position_labels = np.array(position_labels)
    sequence_labels = np.array(sequence_labels)
    
    # Create colors based on sum up to that point or difference for pos3
    colors = []
    for i, (seq_idx, pos) in enumerate(zip(sequence_labels, position_labels)):
        sequence = sequences[seq_idx]
        a = sequence[1]  # First operand
        b = sequence[2]  # Second operand
        c = sequence[3]  # Result (expected to be (a + b) mod p)
        
        if pos == 1:
            # After processing 'a', the sum so far is just 'a'
            color_value = a
        elif pos == 2:
            # After processing 'a' and 'b', the sum so far is 'a + b mod p'
            color_value = (a + b) % p
        elif pos == 3:
            # After processing 'c', color by the value of c itself
            color_value = c
        else:
            # Fallback for other positions
            color_value = 0
            
        colors.append(color_value)
    colors = np.array(colors)
    
    # Plot different positions with different markers
    unique_positions = np.unique(position_labels)
    markers = ['o', '^', 's', 'D']  # circle, triangle, square, diamond
    marker_labels = ['Position 1 (after a)', 'Position 2 (after b)', 'Position 3 (after c)', 'Position 4']
    
    for i, (pc_indices, title) in enumerate(projections_to_show):
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        
        # Extract the relevant 3D projection
        proj_3d = projections_5d[:, pc_indices]
        
        # Calculate variance explained by this 3D projection
        variance_explained = pca_5.explained_variance_ratio_[pc_indices].sum()
        
        scatters = []
        for j, pos in enumerate(unique_positions):
            mask = position_labels == pos
            marker = markers[j % len(markers)]
            label = marker_labels[j % len(marker_labels)]
            
            scatter = ax.scatter(proj_3d[mask, 0], proj_3d[mask, 1], proj_3d[mask, 2],
                                c=colors[mask], cmap='viridis', alpha=0.6, s=30, 
                                marker=marker, edgecolors='black', linewidth=0.3, label=label)
            scatters.append(scatter)
        
        # Set labels with variance information
        ax.set_xlabel(f'PC{pc_indices[0]+1} ({pca_5.explained_variance_ratio_[pc_indices[0]]:.3f})')
        ax.set_ylabel(f'PC{pc_indices[1]+1} ({pca_5.explained_variance_ratio_[pc_indices[1]]:.3f})')
        ax.set_zlabel(f'PC{pc_indices[2]+1} ({pca_5.explained_variance_ratio_[pc_indices[2]]:.3f})')
        
        ax.set_title(f'{title}\n({variance_explained:.3f} total variance)')
        
        # Add legend only to the first subplot to avoid clutter
        if i == 0:
            ax.legend(loc='upper left', bbox_to_anchor=(0.0, 1.0), fontsize=8)
    
    plt.suptitle(f'Multiple 3D Projections of 5D Soft Modular Addition Space (p={p})', fontsize=16)
    plt.tight_layout()
    
    # Add a single colorbar for the entire figure, positioned to the right
    # Use the last scatter for the colorbar
    cbar = fig.colorbar(scatters[0], ax=fig.get_axes(), shrink=0.8, aspect=40, pad=0.02, fraction=0.02)
    cbar.set_label('Color: pos1=a, pos2=a+b mod p, pos3=c')
    
    # Save the multi-view plot
    plt.savefig('play/plots/theoretical_soft_modular_addition_multi_3d.png', dpi=300, bbox_inches='tight')
    print("Multi-view 3D plot saved to 'play/plots/theoretical_soft_modular_addition_multi_3d.png'")
    
    # Show interactive plot
    print("Showing interactive multi-view 3D plot. Use mouse to rotate, zoom, and pan.")
    print("Close the plot window to continue...")
    plt.show()


def main():
    """Main function to visualize soft modular addition belief states."""
    p = 17
    sequences = generate_modular_addition_sequences(p)
    print(f"Generated {len(sequences)} sequences for p={p}")
    
    # Get projected belief states
    projections, belief_states, sequence_labels, position_labels = compute_belief_projections(sequences, p)
    print(f"Projection shape: {projections.shape}")
    
    if len(projections) == 0:
        print("No valid projections to visualize")
        return
    
    # Perform PCA to reduce dimensionality to 2D
    pca = PCA(n_components=2)
    projections_2d = pca.fit_transform(projections)
    
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total variance explained: {pca.explained_variance_ratio_.sum():.3f}")
    
    # Perform PCA with 4 components to get variance explained by 4 PCs
    pca_4 = PCA(n_components=4)
    projections_4d = pca_4.fit_transform(projections)
    print(f"Total variance explained by 4 PCs: {pca_4.explained_variance_ratio_.sum():.3f}")
    
    # Perform PCA with 5 components to get variance explained by 5 PCs
    pca_5 = PCA(n_components=5)
    projections_5d = pca_5.fit_transform(projections)
    print(f"Total variance explained by 5 PCs: {pca_5.explained_variance_ratio_.sum():.3f}")
    print(f"5D PCA explained variance ratio: {pca_5.explained_variance_ratio_}")
    
    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    scatter_params = {'alpha': 0.7, 's': 50, 'edgecolors': 'black', 'linewidth': 0.5}
    
    # Color points based on the value of the last token (c = result of a + b mod p)
    colors = []
    for seq_idx in sequence_labels:
        sequence = sequences[seq_idx]
        last_token = sequence[-1]  # The result c = (a + b) % p
        colors.append(last_token)
    
    colors = np.array(colors)
    
    scatter = ax.scatter(projections_2d[:, 0], projections_2d[:, 1], 
                        c=colors, cmap='viridis', **scatter_params)
    
    ax.set_xlabel(f'First Principal Component ({pca.explained_variance_ratio_[0]:.3f} variance)')
    ax.set_ylabel(f'Second Principal Component ({pca.explained_variance_ratio_[1]:.3f} variance)')
    ax.set_title(f'Soft Modular Addition Belief States After a+b (p={p})')
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Result Value (c)')
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs('play/plots', exist_ok=True)
    plt.savefig('play/plots/theoretical_soft_modular_addition.png', dpi=300, bbox_inches='tight')
    print("Plot saved to 'play/plots/theoretical_soft_modular_addition.png'")
    plt.close()
    
    # Create 3D visualization
    visualize_3d(projections, sequence_labels, sequences, p, pca, position_labels)
    
    # Create multiple 3D projections of the 5D space
    visualize_multiple_3d_projections(projections_5d, pca_5, sequence_labels, sequences, p, position_labels)


if __name__ == "__main__":
    main()