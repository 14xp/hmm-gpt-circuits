#!/usr/bin/env python3
"""
Minimal script to visualize theoretical mess3 belief state geometry.
"""

import sys
import os
import itertools

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from data.comp_mech import belief_update, logit_belief_update, constrained_belief_update, stationary_distribution
from data.mess3 import mess3
from play.utils import uniform_centered_projection


def rotations(angles: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """Rotations in 2D."""
    return np.array([radii[i]*np.array([[np.cos(angles[i]), -np.sin(angles[i])], [np.sin(angles[i]), np.cos(angles[i])]]) for i in range(len(angles))])

def generate_sequences(block_size: int):
    """Generate all possible sequences with BOS tokens."""
    sequences = []
    for combination in itertools.product([0, 1, 2], repeat=block_size-2):
        sequence = [3] + list(combination) + [3] 
        sequences.append(sequence)
    return sequences

def compute_logit_belief_projections(sequences):
    """Compute belief state projections."""
    angles = np.array([0, np.pi/2, np.pi])
    radii = np.array([1, 0.6, 0.4])
    transition_matrix = rotations(angles, radii)
    initial_belief = np.array([1.0, 1.0])
    
    all_belief_states = []
    
    for sequence in sequences:
        current_belief = initial_belief.copy()
        for pos in range(1, len(sequence) - 1):
            observation = sequence[pos]
            current_belief = logit_belief_update(transition_matrix, observation, current_belief)
            all_belief_states.append(current_belief.copy())
    
    belief_states_array = np.array(all_belief_states)
    return belief_states_array

def main():
    """Main function."""
    block_size = 10
    sequences = generate_sequences(block_size)
    
    logit_states = compute_logit_belief_projections(sequences)
    
    fig, (ax1) = plt.subplots(1, 1, figsize=(12, 5))
    
    scatter_params = {'alpha': 0.6, 's': 8, 'edgecolors': 'black', 'linewidth': 0.05}
    
    ax1.scatter(logit_states[:, 0], logit_states[:, 1], **scatter_params)
    ax1.set_title('Logit Belief States')
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    os.makedirs('play/plots', exist_ok=True)
    plt.savefig('play/plots/logit_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()