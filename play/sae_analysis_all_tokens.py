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


def load_gpt_model(checkpoint_path: str = "checkpoints/mess3_12_64x1") -> GPT:
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
        # Use all sequence values
        resid_mid_processed = resid_mid  # Shape: (batch, seq_len, d_model)
        resid_post_processed = resid_post  # Shape: (batch, seq_len, d_model)
        
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
        
        # Process each position in the sequence (including BOS and final tokens)
        for pos in range(len(sequence)):
            observation = sequence[pos]
            
            # For BOS tokens (value 3), don't update belief state, just append current belief
            if observation == 3:
                sequence_beliefs.append(current_belief.copy())
            else:
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
        
        # Process each position in the sequence (including BOS and final tokens)
        for pos in range(len(sequence)):
            observation = sequence[pos]
            
            # For BOS tokens (value 3), don't update belief state, just append current belief
            if observation == 3:
                sequence_beliefs.append(current_belief.copy())
            else:
                # Update belief using regular belief update
                current_belief = belief_update(transition_matrix, observation, current_belief)
                sequence_beliefs.append(current_belief.copy())
        
        all_belief_states.extend(sequence_beliefs)
    
    belief_states_array = np.array(all_belief_states)
    
    # Apply uniform-centered projection to get 2D coordinates
    projections = uniform_centered_projection(belief_states_array)
    
    print(f"Regular belief projections shape: {projections.shape}")
    return projections


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
    Create comparison plots showing SAE activations vs theoretical projections.
    
    Args:
        sae_0_coords: SAE.0 feature coordinates
        sae_1_coords: SAE.1 feature coordinates  
        constrained_proj: Constrained belief state projections
        regular_proj: Regular belief state projections
        sequences: Original sequences for belief state colors
    """
    print("Creating comparison plots...")
    
    # Compute belief states for coloring
    x = 0.15
    a = 0.6
    transition_matrix = mess3(x, a)
    initial_belief = stationary_distribution(transition_matrix)
    
    # Compute constrained belief states for coloring
    constrained_belief_states = []
    for sequence in sequences:
        current_belief = initial_belief.copy()
        for pos in range(len(sequence)):
            observation = sequence[pos]
            # For BOS tokens (value 3), don't update belief state, just append current belief
            if observation == 3:
                constrained_belief_states.append(current_belief.copy())
            else:
                current_belief = constrained_belief_update(transition_matrix, observation, current_belief, initial_belief)
                constrained_belief_states.append(current_belief.copy())
    
    # Compute regular belief states for coloring
    regular_belief_states = []
    for sequence in sequences:
        current_belief = initial_belief.copy()
        for pos in range(len(sequence)):
            observation = sequence[pos]
            # For BOS tokens (value 3), don't update belief state, just append current belief
            if observation == 3:
                regular_belief_states.append(current_belief.copy())
            else:
                current_belief = belief_update(transition_matrix, observation, current_belief)
                regular_belief_states.append(current_belief.copy())
    
    constrained_colors = np.array(constrained_belief_states)
    regular_colors = np.array(regular_belief_states)
    
    # Create four-panel plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    
    scatter_params = {'alpha': 0.6, 's': 8, 'edgecolors': 'black', 'linewidth': 0.05}
    
    # Panel (0,0): SAE.0 activations
    axes[0, 0].scatter(sae_0_coords[:, 0], sae_0_coords[:, 1], 
                      c=constrained_colors, **scatter_params)
    axes[0, 0].set_title('SAE.0 Latent Activations\n(resid_mid → constrained belief states)', fontsize=12)
    axes[0, 0].set_xlabel('SAE.0 Feature 1', fontsize=10)
    axes[0, 0].set_ylabel('SAE.0 Feature 2', fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Panel (0,1): Constrained belief state projections
    axes[0, 1].scatter(constrained_proj[:, 0], constrained_proj[:, 1],
                      c=constrained_colors, **scatter_params)
    axes[0, 1].set_title('Theoretical: Constrained Belief States\n(Uniform-Centered Projection)', fontsize=12)
    axes[0, 1].set_xlabel('First Coordinate (tangent to 2-simplex)', fontsize=10)
    axes[0, 1].set_ylabel('Second Coordinate (tangent to 2-simplex)', fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Panel (1,0): SAE.1 activations
    axes[1, 0].scatter(sae_1_coords[:, 0], sae_1_coords[:, 1],
                      c=regular_colors, **scatter_params)
    axes[1, 0].set_title('SAE.1 Latent Activations\n(resid_post → regular belief states)', fontsize=12)
    axes[1, 0].set_xlabel('SAE.1 Feature 1', fontsize=10)
    axes[1, 0].set_ylabel('SAE.1 Feature 2', fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Panel (1,1): Regular belief state projections
    axes[1, 1].scatter(regular_proj[:, 0], regular_proj[:, 1],
                      c=regular_colors, **scatter_params)
    axes[1, 1].set_title('Theoretical: Regular Belief States\n(Uniform-Centered Projection)', fontsize=12)
    axes[1, 1].set_xlabel('First Coordinate (tangent to 2-simplex)', fontsize=10)
    axes[1, 1].set_ylabel('Second Coordinate (tangent to 2-simplex)', fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add overall title and color explanation
    fig.suptitle('SAE Latent Activations vs Theoretical Belief State Projections', fontsize=16, y=0.98)
    fig.text(0.5, 0.02, 'Color represents belief state: Red=State 0, Green=State 1, Blue=State 2', 
             ha='center', fontsize=12, style='italic')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.94, bottom=0.06)
    
    # Save the plot
    output_path = 'play/plots/sae_vs_theoretical_comparison_all_tokens.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to {output_path}")
    
    plt.show()


def create_position_colored_plots(sae_0_coords: np.ndarray, sae_1_coords: np.ndarray,
                                 constrained_proj: np.ndarray, regular_proj: np.ndarray,
                                 sequences: List[List[int]]):
    """
    Create comparison plots showing SAE activations vs theoretical projections,
    colored by sequence position.
    
    Args:
        sae_0_coords: SAE.0 feature coordinates
        sae_1_coords: SAE.1 feature coordinates  
        constrained_proj: Constrained belief state projections
        regular_proj: Regular belief state projections
        sequences: Original sequences for position mapping
    """
    print("Creating position-colored comparison plots...")
    
    # Create position labels for coloring
    position_labels = []
    for sequence in sequences:
        for pos in range(len(sequence)):
            position_labels.append(pos)
    
    position_colors = np.array(position_labels)
    
    # Create four-panel plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    
    scatter_params = {'alpha': 0.7, 's': 8, 'edgecolors': 'black', 'linewidth': 0.05}
    
    # Panel (0,0): SAE.0 activations
    scatter1 = axes[0, 0].scatter(sae_0_coords[:, 0], sae_0_coords[:, 1], 
                                 c=position_colors, cmap='viridis', **scatter_params)
    axes[0, 0].set_title('SAE.0 Latent Activations\n(resid_mid → constrained belief states)', fontsize=12)
    axes[0, 0].set_xlabel('SAE.0 Feature 1', fontsize=10)
    axes[0, 0].set_ylabel('SAE.0 Feature 2', fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=axes[0, 0], label='Sequence Position')
    
    # Panel (0,1): Constrained belief state projections
    scatter2 = axes[0, 1].scatter(constrained_proj[:, 0], constrained_proj[:, 1],
                                 c=position_colors, cmap='viridis', **scatter_params)
    axes[0, 1].set_title('Theoretical: Constrained Belief States\n(Uniform-Centered Projection)', fontsize=12)
    axes[0, 1].set_xlabel('First Coordinate (tangent to 2-simplex)', fontsize=10)
    axes[0, 1].set_ylabel('Second Coordinate (tangent to 2-simplex)', fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=axes[0, 1], label='Sequence Position')
    
    # Panel (1,0): SAE.1 activations
    scatter3 = axes[1, 0].scatter(sae_1_coords[:, 0], sae_1_coords[:, 1],
                                 c=position_colors, cmap='viridis', **scatter_params)
    axes[1, 0].set_title('SAE.1 Latent Activations\n(resid_post → regular belief states)', fontsize=12)
    axes[1, 0].set_xlabel('SAE.1 Feature 1', fontsize=10)
    axes[1, 0].set_ylabel('SAE.1 Feature 2', fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    plt.colorbar(scatter3, ax=axes[1, 0], label='Sequence Position')
    
    # Panel (1,1): Regular belief state projections
    scatter4 = axes[1, 1].scatter(regular_proj[:, 0], regular_proj[:, 1],
                                 c=position_colors, cmap='viridis', **scatter_params)
    axes[1, 1].set_title('Theoretical: Regular Belief States\n(Uniform-Centered Projection)', fontsize=12)
    axes[1, 1].set_xlabel('First Coordinate (tangent to 2-simplex)', fontsize=10)
    axes[1, 1].set_ylabel('Second Coordinate (tangent to 2-simplex)', fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    plt.colorbar(scatter4, ax=axes[1, 1], label='Sequence Position')
    
    # Add overall title and explanation
    fig.suptitle('SAE Latent Activations vs Theoretical Belief State Projections\n(Colored by Sequence Position)', fontsize=16, y=0.98)
    fig.text(0.5, 0.02, 'Color represents position in sequence: Dark (position 0) to Bright (position 11)', 
             ha='center', fontsize=12, style='italic')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.94, bottom=0.06)
    
    # Save the plot
    output_path = 'play/plots/sae_vs_theoretical_position_colored_all_tokens.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Position-colored comparison plot saved to {output_path}")
    
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
    print(f"SAE.0 vs Constrained Belief States:")
    for key, value in metrics_0.items():
        print(f"  {key}: {value:.4f}")
    
    metrics_1 = compute_alignment_metrics(sae_1_activations, regular_projections)
    print(f"\nSAE.1 vs Regular Belief States:")
    for key, value in metrics_1.items():
        print(f"  {key}: {value:.4f}")
    
    # Create comparison plots
    create_comparison_plots(sae_0_activations, sae_1_activations,
                          constrained_projections, regular_projections, sequences)
    
    # Create position-colored plots
    create_position_colored_plots(sae_0_activations, sae_1_activations,
                                constrained_projections, regular_projections, sequences)
    
    print("\n=== Analysis Complete ===")


if __name__ == "__main__":
    main()