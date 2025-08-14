#!/usr/bin/env python3
"""
Minimal script to visualize theoretical spiral belief state geometry.
"""

import sys
import os
import itertools

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from data.comp_mech import logit_belief_update, stationary_distribution
from data.spiral import spiral
from play.utils import uniform_centered_projection


def generate_sequences(block_size: int):
    """Generate all possible sequences with BOS tokens."""
    sequences = []
    for combination in itertools.product([0, 1], repeat=block_size-2):
        sequence = [2] + list(combination) + [2] 
        sequences.append(sequence)
    return sequences


def compute_belief_projections(sequences):
    """Compute belief state projections."""
    # scale_array = np.array([1.15, 1.15])
    # angle_array = np.array([np.pi/11, -np.pi/11])
    # x_scale = 1.075

    scale_array = np.array([1.15, 1.15])
    angle_array = np.array([np.pi/11, -np.pi/11])
    x_scale = 1
    
    transition_matrix = spiral(scale_array, angle_array, x_scale)
    
    # For spiral, use uniform initial belief since stationary distribution may not exist
    initial_belief = np.array([0, 1.0])
    
    all_belief_states = []
    
    for sequence in sequences:
        current_belief = initial_belief.copy()
        for pos in range(1, len(sequence) - 1):
            observation = sequence[pos]
            current_belief = logit_belief_update(transition_matrix, observation, current_belief)
            all_belief_states.append(current_belief.copy())
    
    belief_states_array = np.array(all_belief_states)
    return belief_states_array


def compute_belief_coordinates(sequences):
    """Compute belief state coordinates (2D for spiral)."""
    belief_states = compute_belief_projections(sequences)
    return belief_states


def main():
    """Main function."""
    block_size = 12
    sequences = generate_sequences(block_size)
    
    belief_coordinates = compute_belief_coordinates(sequences)
    
    # Create colors based on proportion of 0's in sequence so far
    zeros_proportion_colors = []
    
    for seq_idx, sequence in enumerate(sequences):
        for pos in range(1, len(sequence) - 1):  # Skip BOS token and final position
            # Count 0's from position 1 to current position (inclusive)
            subsequence = sequence[1:pos+1]  # From after BOS to current position
            num_zeros = sum(1 for token in subsequence if token == 0)
            proportion_zeros = num_zeros / len(subsequence)
            zeros_proportion_colors.append(proportion_zeros)
    
    zeros_proportion_colors = np.array(zeros_proportion_colors)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    scatter = ax.scatter(belief_coordinates[:, 0], belief_coordinates[:, 1], 
                        c=zeros_proportion_colors, cmap='RdYlBu', 
                        alpha=0.6, s=8, edgecolors='black', linewidth=0.1)
    ax.set_title('Spiral Logit Belief States (Colored by Proportion of 0\'s)')
    ax.set_xlabel('Belief Coordinate 1')
    ax.set_ylabel('Belief Coordinate 2')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Proportion of 0\'s in sequence so far')
    
    plt.tight_layout()
    
    os.makedirs('play/plots', exist_ok=True)
    plt.savefig('play/plots/theoretical_spiral_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()