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

from data.comp_mech import belief_update, constrained_belief_update, stationary_distribution
from data.mess3 import mess3
from play.utils import uniform_centered_projection


def generate_sequences(block_size: int):
    """Generate all possible sequences with BOS tokens."""
    sequences = []
    for combination in itertools.product([0, 1, 2], repeat=block_size-2):
        sequence = [3] + list(combination) + [3] 
        sequences.append(sequence)
    return sequences


def compute_belief_projections(sequences, use_constrained=True):
    """Compute belief state projections."""
    x = 0.15
    a = 0.6
    transition_matrix = mess3(x, a)
    initial_belief = stationary_distribution(transition_matrix)
    
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
    
    belief_states_array = np.array(all_belief_states)
    projections = uniform_centered_projection(belief_states_array)
    return projections, belief_states_array


def main():
    """Main function."""
    sequences = generate_sequences(10)
    
    constrained_proj, constrained_states = compute_belief_projections(sequences, use_constrained=True)
    regular_proj, regular_states = compute_belief_projections(sequences, use_constrained=False)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    scatter_params = {'alpha': 0.6, 's': 8, 'edgecolors': 'black', 'linewidth': 0.05}
    
    ax1.scatter(constrained_proj[:, 0], constrained_proj[:, 1], c=constrained_states, **scatter_params)
    ax1.set_title('Constrained Belief States')
    ax1.grid(True, alpha=0.3)
    
    ax2.scatter(regular_proj[:, 0], regular_proj[:, 1], c=regular_states, **scatter_params)
    ax2.set_title('Regular Belief States')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    os.makedirs('play/plots', exist_ok=True)
    plt.savefig('play/plots/theoretical_mess3_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()