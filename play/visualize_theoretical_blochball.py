#!/usr/bin/env python3
"""
Minimal script to visualize theoretical blochball belief state geometry.
"""

import sys
import os
import itertools

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from data.comp_mech import belief_update, belief_update_general, stationary_distribution
from data.blochball import bloch


def generate_sequences(block_size: int):
    """Generate all possible sequences with BOS tokens."""
    sequences = []
    for combination in itertools.product([0, 1, 2, 3], repeat=block_size-2):
        sequence = [4] + list(combination) + [4] 
        sequences.append(sequence)
    return sequences


def compute_belief_projections(sequences: list, one_vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute belief state projections."""
    a = 1.0
    b = np.sqrt(51)
    transition_matrix = bloch(a, b)
    initial_belief = stationary_distribution(transition_matrix)
    print(f"Initial belief state: {initial_belief}")
    
    all_belief_states = []
    
    for sequence in sequences:
        current_belief = initial_belief.copy()
        for pos in range(1, len(sequence) - 1):
            observation = sequence[pos]
            current_belief = belief_update_general(transition_matrix, observation, current_belief, one_vector)
            all_belief_states.append(current_belief.copy())
    
    if len(all_belief_states) == 0:
        # Return empty arrays if no belief states were generated
        return np.array([]).reshape(0, 2), np.array([]).reshape(0, 3)
    
    belief_states_array = np.array(all_belief_states)

    # Create change of basis matrix with one_vector as first basis vector
    one_vector_normalized = one_vector / np.linalg.norm(one_vector)
    
    # Create matrix with one_vector as first column and identity for remaining columns
    dim = one_vector.shape[0]
    A = np.column_stack([one_vector_normalized.flatten(), np.eye(dim)])
    
    # Use QR decomposition to get orthonormal basis (Gram-Schmidt)
    Q, _ = np.linalg.qr(A)
    basis_matrix = Q[:, :dim]
    
    # Transform belief states to new basis
    belief_states_new_basis = (basis_matrix.T @ belief_states_array.T).T
    
    # Extract 2D coordinates directly (indices 1 and 2, skipping the one_vector direction)
    projections = belief_states_new_basis[:, [1, 2]]

    return projections, belief_states_array


def main():
    """Main function."""
    sequences = generate_sequences(10)
    one_vector = np.array([[1], [0], [0]])

    belief_proj, belief_states = compute_belief_projections(sequences, one_vector)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    scatter_params = {'alpha': 0.3, 's': 8, 'edgecolors': 'black', 'linewidth': 0.05}
    
    # Create color maps that handle negative values by shifting to [0,1] range
    # Use the first coordinate (x-axis) for coloring
    belief_x = belief_proj[:, 0]
    
    # Handle edge cases where min equals max (constant values)
    belief_range = belief_x.max() - belief_x.min()
    if belief_range == 0:
        belief_colors = np.zeros_like(belief_x)
    else:
        belief_colors = (belief_x - belief_x.min()) / belief_range
    
    # Handle empty arrays
    if len(belief_proj) > 0:
        ax.scatter(belief_proj[:, 0], belief_proj[:, 1], c=belief_colors, **scatter_params)
    ax.set_title('Belief States')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    os.makedirs('play/plots', exist_ok=True)
    plt.savefig('play/plots/theoretical_blochball_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()