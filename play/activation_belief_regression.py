#!/usr/bin/env python3
"""
Script to perform linear regression from model activations to theoretical belief states.
Compares model representations with theoretical spiral belief states.
"""

import sys
import os
import itertools
from typing import List, Tuple, Dict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score

from models.gpt import GPT
from data.comp_mech import logit_belief_update
from data.spiral import spiral


def generate_sequences(vocab_size: int, block_size: int) -> List[List[int]]:
    """
    Generate all possible binary sequences with given vocab_size and block_size.
    Format: [BOS, *, *, ..., *] where * ∈ {0, 1} and BOS = vocab_size - 1
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


def compute_theoretical_belief_states(sequences: List[List[int]]) -> np.ndarray:
    """
    Compute theoretical belief states using spiral transition matrix.
    Returns structured data maintaining sequence-position correspondence.
    
    Returns:
        np.ndarray: Shape (num_sequences, num_positions, 2)
    """
    print("Computing theoretical belief states...")
    
    # Spiral parameters from visualize_theoretical_spiral.py
    scale_array = np.array([1.15, 1.15])
    angle_array = np.array([np.pi/11, -np.pi/11])
    x_scale = 1.075
    
    transition_matrix = spiral(scale_array, angle_array, x_scale)
    
    # For spiral, use uniform initial belief since stationary distribution may not exist
    initial_belief = np.array([0, 1.0])
    
    all_belief_states = []
    
    for sequence in sequences:
        sequence_beliefs = []
        current_belief = initial_belief.copy()
        for pos in range(1, len(sequence) - 1):  # Skip BOS and final tokens
            observation = sequence[pos]
            current_belief = logit_belief_update(transition_matrix, observation, current_belief)
            sequence_beliefs.append(current_belief.copy())
        all_belief_states.append(sequence_beliefs)
    
    belief_states_structured = np.array(all_belief_states)  # Shape: (num_sequences, num_positions, 2)
    print(f"Computed belief states, shape: {belief_states_structured.shape}")
    
    return belief_states_structured


def capture_model_activations(model: GPT, sequences: List[List[int]], device: torch.device, 
                            activation_type: str = "post_ln_mlp") -> torch.Tensor:
    """
    Capture model activations for given sequences.
    
    Args:
        activation_type: One of "pre_ln", "post_ln_mlp", "logits"
    """
    print(f"Capturing {activation_type} activations...")
    
    batch_size = len(sequences)
    seq_len = len(sequences[0])
    
    # Convert sequences to tensor
    input_ids = torch.tensor(sequences, device=device)  # Shape: (batch_size, seq_len)
    
    if activation_type == "logits":
        # Capture logits (post-unembed, pre-softmax)
        with torch.no_grad():
            output = model(input_ids)
            logits = output.logits  # Shape: (batch_size, seq_len, vocab_size)
        return logits
    
    else:
        # Capture MLP activations (pre or post layernorm)
        activations = []
        
        with torch.no_grad():
            # Manual forward pass to capture activations
            B, T = input_ids.size()
            
            # Embeddings
            pos = torch.arange(0, T, dtype=torch.long, device=device)
            pos_emb = model.transformer.wpe(pos)
            tok_emb = model.transformer.wte(input_ids)
            x = tok_emb + pos_emb
            
            # Forward through transformer blocks
            for block_idx, block in enumerate(model.transformer.h):
                # Apply attention
                resid_mid = x + block.attn(block.ln_1(x))
                
                if activation_type == "pre_ln":
                    # Capture pre-layernorm activations (before ln_2 is applied)
                    activations.append(resid_mid.clone())
                
                # Apply MLP with layernorm
                resid_post = resid_mid + block.mlp(block.ln_2(resid_mid))
                
                if activation_type == "post_ln_mlp":
                    # Capture post-layernorm+MLP activations
                    activations.append(resid_post.clone())
                
                # Continue with residual connection for next block
                x = resid_post
        
        # Use activations from the last transformer block
        final_activations = activations[-1]  # Shape: (batch_size, seq_len, n_embd)
        return final_activations


def process_activations_and_beliefs(activations: torch.Tensor, belief_states_structured: np.ndarray, 
                                  sequences: List[List[int]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Process activations and belief states maintaining sequence-position correspondence.
    
    Args:
        activations: (num_sequences, seq_len, activation_dim)
        belief_states_structured: (num_sequences, num_positions, 2)
        sequences: List of sequences
    
    Returns:
        flattened_activations: (n_samples, activation_dim) - flattened for regression
        flattened_beliefs: (n_samples, 2) - flattened for regression
        flattened_colors: (n_samples,) - flattened for visualization
        structured_activations: (num_sequences, num_positions, activation_dim) - for position analysis
        structured_colors: (num_sequences, num_positions) - for position analysis
    """
    print("Processing activations and belief states with maintained correspondence...")
    
    # Remove BOS token (position 0) and final position (position -1) from activations
    activations_processed = activations[:, 1:-1, :]  # Shape: (num_sequences, num_positions, activation_dim)
    
    # Create structured proportion colors maintaining correspondence
    num_sequences, num_positions, activation_dim = activations_processed.shape
    structured_colors = np.zeros((num_sequences, num_positions))
    
    for seq_idx, sequence in enumerate(sequences):
        for pos_idx in range(num_positions):
            pos = pos_idx + 1  # Actual position in sequence (skip BOS)
            # Count 0's from position 1 to current position (inclusive)
            subsequence = sequence[1:pos+1]  # From after BOS to current position
            num_zeros = sum(1 for token in subsequence if token == 0)
            proportion_zeros = num_zeros / len(subsequence)
            structured_colors[seq_idx, pos_idx] = proportion_zeros
    
    # Flatten all data maintaining correspondence
    # Order: seq0_pos0, seq0_pos1, ..., seq0_pos(n-1), seq1_pos0, seq1_pos1, ...
    flattened_activations = activations_processed.reshape(-1, activation_dim).cpu().numpy()
    flattened_beliefs = belief_states_structured.reshape(-1, 2)
    flattened_colors = structured_colors.reshape(-1)
    
    print(f"Structured activations shape: {activations_processed.shape}")
    print(f"Structured belief states shape: {belief_states_structured.shape}")
    print(f"Structured colors shape: {structured_colors.shape}")
    print(f"Flattened activations shape: {flattened_activations.shape}")
    print(f"Flattened belief states shape: {flattened_beliefs.shape}")
    print(f"Flattened colors shape: {flattened_colors.shape}")
    
    # Verify alignment
    assert flattened_activations.shape[0] == flattened_beliefs.shape[0] == flattened_colors.shape[0], \
        "Flattened data must have same number of samples"
    
    return (flattened_activations, flattened_beliefs, flattened_colors, 
            activations_processed.cpu().numpy(), structured_colors)


def perform_linear_regression(X: np.ndarray, y: np.ndarray, activation_type: str) -> Dict:
    """
    Perform linear regression from activations X to belief states y.
    
    Args:
        X: Activations (n_samples, activation_dim)
        y: Belief states (n_samples, 2)
        activation_type: String identifier for the activation type
        
    Returns:
        Dictionary with regression results
    """
    print(f"Performing linear regression for {activation_type}...")
    
    results = {
        'activation_type': activation_type,
        'n_samples': X.shape[0],
        'activation_dim': X.shape[1],
        'models': {},
        'predictions': {},
        'metrics': {}
    }
    
    # Perform regression for each belief coordinate
    for coord_idx in range(2):
        coord_name = f'belief_coord_{coord_idx}'
        y_coord = y[:, coord_idx]
        
        # Fit linear regression
        model = LinearRegression()
        model.fit(X, y_coord)
        
        # Make predictions
        y_pred = model.predict(X)
        
        # Compute metrics
        r2 = r2_score(y_coord, y_pred)
        mse = mean_squared_error(y_coord, y_pred)
        rmse = np.sqrt(mse)
        
        # Cross-validation for more robust evaluation
        cv_scores = cross_val_score(model, X, y_coord, cv=5, scoring='r2')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        # Store results
        results['models'][coord_name] = model
        results['predictions'][coord_name] = y_pred
        results['metrics'][coord_name] = {
            'r2': r2,
            'mse': mse,
            'rmse': rmse,
            'cv_r2_mean': cv_mean,
            'cv_r2_std': cv_std
        }
        
        print(f"  {coord_name}: R² = {r2:.4f}, RMSE = {rmse:.4f}, CV R² = {cv_mean:.4f} ± {cv_std:.4f}")
    
    return results


def visualize_regression_results(results: Dict, belief_states: np.ndarray, 
                               proportion_colors: np.ndarray, suffix: str):
    """
    Create visualizations for regression results.
    """
    activation_type = results['activation_type']
    print(f"Creating visualizations for {activation_type}...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Predicted vs Actual for Belief Coordinate 0
    ax = axes[0, 0]
    y_true_0 = belief_states[:, 0]
    y_pred_0 = results['predictions']['belief_coord_0']
    r2_0 = results['metrics']['belief_coord_0']['r2']
    
    scatter = ax.scatter(y_true_0, y_pred_0, c=proportion_colors, cmap='RdYlBu', alpha=0.6, s=8)
    ax.plot([y_true_0.min(), y_true_0.max()], [y_true_0.min(), y_true_0.max()], 'r--', lw=2)
    ax.set_xlabel('True Belief Coordinate 0')
    ax.set_ylabel('Predicted Belief Coordinate 0')
    ax.set_title(f'Belief Coord 0: R² = {r2_0:.4f}')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Proportion of 0\'s')
    
    # Plot 2: Predicted vs Actual for Belief Coordinate 1
    ax = axes[0, 1]
    y_true_1 = belief_states[:, 1]
    y_pred_1 = results['predictions']['belief_coord_1']
    r2_1 = results['metrics']['belief_coord_1']['r2']
    
    scatter = ax.scatter(y_true_1, y_pred_1, c=proportion_colors, cmap='RdYlBu', alpha=0.6, s=8)
    ax.plot([y_true_1.min(), y_true_1.max()], [y_true_1.min(), y_true_1.max()], 'r--', lw=2)
    ax.set_xlabel('True Belief Coordinate 1')
    ax.set_ylabel('Predicted Belief Coordinate 1')
    ax.set_title(f'Belief Coord 1: R² = {r2_1:.4f}')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Proportion of 0\'s')
    
    # Plot 3: Residuals for Belief Coordinate 0
    ax = axes[1, 0]
    residuals_0 = y_true_0 - y_pred_0
    scatter = ax.scatter(y_pred_0, residuals_0, c=proportion_colors, cmap='RdYlBu', alpha=0.6, s=8)
    ax.axhline(y=0, color='r', linestyle='--')
    ax.set_xlabel('Predicted Belief Coordinate 0')
    ax.set_ylabel('Residuals')
    ax.set_title('Residuals: Belief Coordinate 0')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Proportion of 0\'s')
    
    # Plot 4: Residuals for Belief Coordinate 1
    ax = axes[1, 1]
    residuals_1 = y_true_1 - y_pred_1
    scatter = ax.scatter(y_pred_1, residuals_1, c=proportion_colors, cmap='RdYlBu', alpha=0.6, s=8)
    ax.axhline(y=0, color='r', linestyle='--')
    ax.set_xlabel('Predicted Belief Coordinate 1')
    ax.set_ylabel('Residuals')
    ax.set_title('Residuals: Belief Coordinate 1')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Proportion of 0\'s')
    
    plt.suptitle(f'Linear Regression: {activation_type} → Belief States\n'
                f'Overall R² = {(r2_0 + r2_1) / 2:.4f}', fontsize=14)
    plt.tight_layout()
    
    # Save plot
    os.makedirs('play/plots', exist_ok=True)
    output_path = f'play/plots/belief_regression{suffix}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {output_path}")


def visualize_by_position_comparison(results: Dict, structured_activations: np.ndarray, 
                                   structured_beliefs: np.ndarray, structured_colors: np.ndarray, 
                                   activation_type: str, suffix: str):
    """
    Create comparison of predicted vs ground truth belief states with all positions combined.
    
    Args:
        results: Regression results with trained models
        structured_activations: (num_sequences, num_positions, activation_dim)
        structured_beliefs: (num_sequences, num_positions, 2)
        structured_colors: (num_sequences, num_positions)
        activation_type: String identifier for the activation type
        suffix: Suffix for the output filename
    """
    print(f"Creating combined visualization for {activation_type}...")
    
    num_sequences, num_positions, _ = structured_activations.shape
    
    # Collect all predictions and ground truth across all positions
    all_pred_coord_0 = []
    all_pred_coord_1 = []
    all_true_coord_0 = []
    all_true_coord_1 = []
    all_colors = []
    
    for pos in range(num_positions):
        # Extract data for this position across all sequences
        pos_activations = structured_activations[:, pos, :]  # (num_sequences, activation_dim)
        pos_true_beliefs = structured_beliefs[:, pos, :]  # (num_sequences, 2) 
        pos_colors = structured_colors[:, pos]  # (num_sequences,)
        
        # Predict for this position using the single regression model
        pos_pred_coord_0 = results['models']['belief_coord_0'].predict(pos_activations)
        pos_pred_coord_1 = results['models']['belief_coord_1'].predict(pos_activations)
        
        # Accumulate all data
        all_pred_coord_0.extend(pos_pred_coord_0)
        all_pred_coord_1.extend(pos_pred_coord_1)
        all_true_coord_0.extend(pos_true_beliefs[:, 0])
        all_true_coord_1.extend(pos_true_beliefs[:, 1])
        all_colors.extend(pos_colors)
    
    # Convert to numpy arrays
    all_pred_coord_0 = np.array(all_pred_coord_0)
    all_pred_coord_1 = np.array(all_pred_coord_1)
    all_true_coord_0 = np.array(all_true_coord_0)
    all_true_coord_1 = np.array(all_true_coord_1)
    all_colors = np.array(all_colors)
    
    # Create figure with 1x2 layout
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left subplot: All predicted belief states
    ax_left = axes[0]
    scatter = ax_left.scatter(all_pred_coord_0, all_pred_coord_1, c=all_colors, 
                            cmap='RdYlBu', alpha=0.6, s=8, edgecolors='black', linewidth=0.1)
    ax_left.set_title('All Positions: Predicted Beliefs')
    ax_left.set_xlabel('Predicted Belief Coord 0')
    ax_left.set_ylabel('Predicted Belief Coord 1')
    ax_left.grid(True, alpha=0.3)
    
    # Right subplot: All ground truth belief states
    ax_right = axes[1]
    scatter = ax_right.scatter(all_true_coord_0, all_true_coord_1, c=all_colors, 
                             cmap='RdYlBu', alpha=0.6, s=8, edgecolors='black', linewidth=0.1)
    ax_right.set_title('All Positions: Ground Truth Beliefs')
    ax_right.set_xlabel('True Belief Coord 0')
    ax_right.set_ylabel('True Belief Coord 1')
    ax_right.grid(True, alpha=0.3)
    
    # Make axes limits consistent for easier comparison
    all_x = np.concatenate([all_pred_coord_0, all_true_coord_0])
    all_y = np.concatenate([all_pred_coord_1, all_true_coord_1])
    x_lim = [all_x.min() - 0.1, all_x.max() + 0.1]
    y_lim = [all_y.min() - 0.1, all_y.max() + 0.1]
    
    ax_left.set_xlim(x_lim)
    ax_left.set_ylim(y_lim)
    ax_right.set_xlim(x_lim)
    ax_right.set_ylim(y_lim)
    
    # Add colorbar
    plt.colorbar(scatter, ax=ax_right, label='Proportion of 0\'s')
    
    # Overall title
    avg_r2 = (results['metrics']['belief_coord_0']['r2'] + results['metrics']['belief_coord_1']['r2']) / 2
    plt.suptitle(f'Combined Positions: {activation_type} → Belief States (Avg R² = {avg_r2:.3f})', 
                fontsize=14)
    plt.tight_layout()
    
    # Save plot
    output_path = f'play/plots/belief_regression_by_position{suffix}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Combined visualization saved to {output_path}")


def print_regression_summary(results_list: List[Dict]):
    """
    Print a summary of all regression results.
    """
    print("\n" + "="*80)
    print("REGRESSION SUMMARY")
    print("="*80)
    
    for results in results_list:
        activation_type = results['activation_type']
        print(f"\n{activation_type.upper()}:")
        print(f"  Input dimension: {results['activation_dim']}")
        print(f"  Number of samples: {results['n_samples']}")
        
        for coord_idx in range(2):
            coord_name = f'belief_coord_{coord_idx}'
            metrics = results['metrics'][coord_name]
            print(f"  Belief Coordinate {coord_idx}:")
            print(f"    R² Score: {metrics['r2']:.4f}")
            print(f"    RMSE: {metrics['rmse']:.4f}")
            print(f"    CV R² (5-fold): {metrics['cv_r2_mean']:.4f} ± {metrics['cv_r2_std']:.4f}")
        
        # Overall average R²
        avg_r2 = (results['metrics']['belief_coord_0']['r2'] + 
                 results['metrics']['belief_coord_1']['r2']) / 2
        print(f"  Average R²: {avg_r2:.4f}")


def main():
    print("=== Activation to Belief State Linear Regression Analysis ===\n")
    
    # Configuration
    vocab_size = 3  # Tokens: 0, 1, 2
    block_size = 12  # Total sequence length including BOS token
    model_path = "checkpoints/spiral_12_64x1_untied"  # Use spiral model for vocab_size=3
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = GPT.load(model_path, device)
    model.eval()
    print(f"Model loaded successfully!")
    print(f"Model config: {model.config}\n")
    
    # Generate sequences
    sequences = generate_sequences(vocab_size, block_size)
    print(f"Total sequences to process: {len(sequences)}")
    print(f"Each sequence length: {len(sequences[0])}")
    print(f"Example sequences:")
    for i in range(min(5, len(sequences))):
        print(f"  {sequences[i]}")
    print()
    
    # Compute theoretical belief states (structured)
    belief_states_structured = compute_theoretical_belief_states(sequences)
    
    # Define activation types to analyze
    activation_types = ["pre_ln", "post_ln_mlp", "logits"]
    suffixes = ["_pre_ln", "_post_ln", "_logits"]
    
    all_results = []
    
    # Process sequences in batches to manage memory
    batch_size = 500
    num_batches = (len(sequences) + batch_size - 1) // batch_size
    
    for act_type, suffix in zip(activation_types, suffixes):
        print(f"\n" + "="*60)
        print(f"ANALYZING {act_type.upper()} ACTIVATIONS")
        print("="*60)
        
        all_activations = []
        
        print(f"Processing {num_batches} batches of size {batch_size}...")
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(sequences))
            batch_sequences = sequences[start_idx:end_idx]
            
            print(f"Processing batch {i+1}/{num_batches} (sequences {start_idx}-{end_idx-1})")
            
            # Capture activations for this batch
            activations_batch = capture_model_activations(model, batch_sequences, device, act_type)
            all_activations.append(activations_batch)
        
        # Concatenate all batches
        print("Concatenating all batches...")
        activations = torch.cat(all_activations, dim=0)
        print(f"Activations shape: {activations.shape}")
        
        # Process activations and belief states maintaining correspondence
        (flattened_activations, flattened_beliefs, flattened_colors, 
         structured_activations, structured_colors) = process_activations_and_beliefs(
            activations, belief_states_structured, sequences)
        
        # Perform linear regression on flattened data
        results = perform_linear_regression(flattened_activations, flattened_beliefs, act_type)
        all_results.append(results)
        
        # Create standard visualizations
        visualize_regression_results(results, flattened_beliefs, flattened_colors, suffix)
        
        # Create position-by-position comparison
        visualize_by_position_comparison(results, structured_activations, belief_states_structured, 
                                       structured_colors, act_type, suffix)
    
    # Print overall summary
    print_regression_summary(all_results)
    
    print("\n=== Analysis Complete ===")


if __name__ == "__main__":
    main()