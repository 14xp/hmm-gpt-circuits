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

from data.comp_mech import belief_update, constrained_belief_update, stationary_distribution
from data.blochball import bloch


def generate_sequences(block_size: int):
    """Generate all possible sequences with BOS tokens."""
    sequences = []
    for combination in itertools.product([0, 1, 2, 3], repeat=block_size-2):
        sequence = [4] + list(combination) + [4] 
        sequences.append(sequence)
    return sequences


def compute_belief_projections(sequences, use_constrained=True):
    """Compute belief state projections."""
    a = 1.0
    b = np.sqrt(10)
    transition_matrix = bloch(a, b)
    initial_belief = stationary_distribution(transition_matrix)
    print(f"Initial belief state: {initial_belief}")
    
    all_belief_states = []
    
    for sequence in sequences:
        current_belief = initial_belief.copy()
        for pos in range(1, len(sequence) - 1):
            observation = sequence[pos]
            if use_constrained:
                current_belief = constrained_belief_update(transition_matrix, observation, current_belief, initial_belief)
            else:
                current_belief = belief_update(transition_matrix, observation, current_belief)
            all_belief_states.append(current_belief.copy())
    
    if len(all_belief_states) == 0:
        # Return empty arrays if no belief states were generated
        return np.array([]).reshape(0, 2), np.array([]).reshape(0, 3)
    
    belief_states_array = np.array(all_belief_states)
    
    # Use only the second two coordinates (indices 1 and 2) normalized by the first coordinate
    first_coord = belief_states_array[:, 0]
    second_coord = belief_states_array[:, 1]
    third_coord = belief_states_array[:, 2]
    
    # Normalize by first coordinate (avoid division by zero)
    normalized_second = np.divide(second_coord, first_coord, out=np.zeros_like(second_coord), where=first_coord!=0)
    normalized_third = np.divide(third_coord, first_coord, out=np.zeros_like(third_coord), where=first_coord!=0)
    
    projections = np.column_stack([normalized_second, normalized_third])
    
    return projections, belief_states_array


def main():
    """Main function."""
    sequences = generate_sequences(10)
    
    constrained_proj, constrained_states = compute_belief_projections(sequences, use_constrained=True)
    regular_proj, regular_states = compute_belief_projections(sequences, use_constrained=False)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    scatter_params = {'alpha': 0.3, 's': 8, 'edgecolors': 'black', 'linewidth': 0.05}
    
    # Create color maps that handle negative values by shifting to [0,1] range
    # Use the first coordinate (x-axis) for coloring
    constrained_x = constrained_proj[:, 0]
    regular_x = regular_proj[:, 0]
    
    # Handle edge cases where min equals max (constant values)
    constrained_range = constrained_x.max() - constrained_x.min()
    if constrained_range == 0:
        constrained_colors = np.zeros_like(constrained_x)
    else:
        constrained_colors = (constrained_x - constrained_x.min()) / constrained_range
    
    regular_range = regular_x.max() - regular_x.min()
    if regular_range == 0:
        regular_colors = np.zeros_like(regular_x)
    else:
        regular_colors = (regular_x - regular_x.min()) / regular_range
    
    # Handle empty arrays
    if len(constrained_proj) > 0:
        ax1.scatter(constrained_proj[:, 0], constrained_proj[:, 1], c=constrained_colors, **scatter_params)
    ax1.set_title('Constrained Belief States')
    ax1.grid(True, alpha=0.3)
    
    if len(regular_proj) > 0:
        ax2.scatter(regular_proj[:, 0], regular_proj[:, 1], c=regular_colors, **scatter_params)
    ax2.set_title('Regular Belief States')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    os.makedirs('play/plots', exist_ok=True)
    plt.savefig('play/plots/theoretical_blochball_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()