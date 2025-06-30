#!/usr/bin/env python3
"""
Script to identify sequence classes from PCA analysis.
Maps activation vectors back to sequences and analyzes patterns in each class.
"""

import sys
import os
import itertools
from typing import List, Tuple, Dict
from collections import Counter, defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from models.gpt import GPT


def generate_all_sequences() -> List[List[int]]:
    """Generate all possible length-10 sequences with BOS token 3."""
    sequences = []
    for combination in itertools.product([0, 1, 2], repeat=9):
        sequence = [3] + list(combination)
        sequences.append(sequence)
    return sequences


def capture_activations_batch(model: GPT, sequences: List[List[int]], device: torch.device) -> torch.Tensor:
    """Capture final activations (before layer norm) for a batch of sequences."""
    input_ids = torch.tensor(sequences, device=device)
    
    with torch.no_grad():
        B, T = input_ids.size()
        
        # Embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=device)
        pos_emb = model.transformer.wpe(pos)
        tok_emb = model.transformer.wte(input_ids)
        x = tok_emb + pos_emb
        
        # Forward through transformer blocks
        for block in model.transformer.h:
            x_after_attn = x + block.attn(block.ln_1(x))
            x = x_after_attn + block.mlp(block.ln_2(x_after_attn))
        
        # x contains final activations before layer norm
        return x


def create_activation_mapping(sequences: List[List[int]], activations: torch.Tensor) -> List[Tuple[int, int, List[int]]]:
    """
    Create mapping from activation vectors to (sequence_idx, position, sequence).
    Returns list of (seq_idx, pos, sequence) for each activation vector (excluding BOS).
    """
    mapping = []
    for seq_idx, sequence in enumerate(sequences):
        for pos in range(1, len(sequence)):  # Skip BOS token at position 0
            mapping.append((seq_idx, pos, sequence))
    return mapping


def classify_activations(activations_pca: np.ndarray, method='kmeans') -> Tuple[np.ndarray, float]:
    """
    Classify activations into two groups based on PC1.
    Returns labels and threshold.
    """
    pc1_values = activations_pca[:, 0]
    
    if method == 'kmeans':
        # Use K-means with k=2
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = kmeans.fit_predict(pc1_values.reshape(-1, 1))
        
        # Ensure class 0 has lower PC1 values
        if np.mean(pc1_values[labels == 0]) > np.mean(pc1_values[labels == 1]):
            labels = 1 - labels
        
        threshold = np.mean(kmeans.cluster_centers_)
        
    elif method == 'median':
        # Use median as threshold
        threshold = np.median(pc1_values)
        labels = (pc1_values > threshold).astype(int)
    
    return labels, threshold


def analyze_class_patterns(sequences: List[List[int]], mapping: List[Tuple[int, int, List[int]]], 
                          labels: np.ndarray, class_id: int) -> Dict:
    """Analyze patterns for a specific class."""
    class_mask = (labels == class_id)
    class_mapping = [mapping[i] for i in range(len(mapping)) if class_mask[i]]
    
    # Extract sequences and positions for this class
    class_sequences = [item[2] for item in class_mapping]
    class_positions = [item[1] for item in class_mapping]
    
    # Token frequency analysis
    token_counts = Counter()
    position_token_counts = defaultdict(Counter)
    
    for (seq_idx, pos, sequence), position in zip(class_mapping, class_positions):
        token = sequence[position]
        token_counts[token] += 1
        position_token_counts[position][token] += 1
    
    # N-gram analysis (bigrams)
    bigram_counts = Counter()
    for seq_idx, pos, sequence in class_mapping:
        if pos > 1:  # Can create bigram
            bigram = (sequence[pos-1], sequence[pos])
            bigram_counts[bigram] += 1
    
    # Find most representative sequences
    unique_sequences = list(set(tuple(seq) for seq in class_sequences))
    sequence_counts = Counter(tuple(seq) for seq in class_sequences)
    most_common_sequences = sequence_counts.most_common(10)
    
    return {
        'count': len(class_mapping),
        'token_counts': token_counts,
        'position_token_counts': dict(position_token_counts),
        'bigram_counts': bigram_counts,
        'most_common_sequences': most_common_sequences,
        'unique_sequences_count': len(unique_sequences)
    }


def visualize_classes(activations_pca: np.ndarray, labels: np.ndarray, threshold: float):
    """Create visualizations of the two classes."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # PCA scatter plot with classes
    class_0_mask = (labels == 0)
    class_1_mask = (labels == 1)
    
    ax1.scatter(activations_pca[class_0_mask, 0], activations_pca[class_0_mask, 1], 
               alpha=0.6, s=1, label=f'Class 0 (n={np.sum(class_0_mask)})', color='blue')
    ax1.scatter(activations_pca[class_1_mask, 0], activations_pca[class_1_mask, 1], 
               alpha=0.6, s=1, label=f'Class 1 (n={np.sum(class_1_mask)})', color='red')
    ax1.axvline(x=threshold, color='black', linestyle='--', alpha=0.7, label=f'Threshold: {threshold:.3f}')
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.set_title('PCA with Class Labels')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # PC1 histogram
    ax2.hist(activations_pca[class_0_mask, 0], bins=50, alpha=0.7, label='Class 0', color='blue', density=True)
    ax2.hist(activations_pca[class_1_mask, 0], bins=50, alpha=0.7, label='Class 1', color='red', density=True)
    ax2.axvline(x=threshold, color='black', linestyle='--', alpha=0.7, label=f'Threshold')
    ax2.set_xlabel('PC1 Value')
    ax2.set_ylabel('Density')
    ax2.set_title('PC1 Distribution by Class')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # PC2 histogram
    ax3.hist(activations_pca[class_0_mask, 1], bins=50, alpha=0.7, label='Class 0', color='blue', density=True)
    ax3.hist(activations_pca[class_1_mask, 1], bins=50, alpha=0.7, label='Class 1', color='red', density=True)
    ax3.set_xlabel('PC2 Value')
    ax3.set_ylabel('Density')
    ax3.set_title('PC2 Distribution by Class')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Position distribution
    positions_class_0 = []
    positions_class_1 = []
    
    for i in range(len(labels)):
        pos = (i % 9) + 1  # Position in sequence (1-9, excluding BOS)
        if labels[i] == 0:
            positions_class_0.append(pos)
        else:
            positions_class_1.append(pos)
    
    pos_counts_0 = Counter(positions_class_0)
    pos_counts_1 = Counter(positions_class_1)
    
    positions = list(range(1, 10))
    counts_0 = [pos_counts_0[pos] for pos in positions]
    counts_1 = [pos_counts_1[pos] for pos in positions]
    
    x = np.arange(len(positions))
    width = 0.35
    
    ax4.bar(x - width/2, counts_0, width, label='Class 0', alpha=0.7, color='blue')
    ax4.bar(x + width/2, counts_1, width, label='Class 1', alpha=0.7, color='red')
    ax4.set_xlabel('Position in Sequence')
    ax4.set_ylabel('Count')
    ax4.set_title('Class Distribution by Position')
    ax4.set_xticks(x)
    ax4.set_xticklabels(positions)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('play/plots/sequence_classes.png', dpi=300, bbox_inches='tight')
    plt.show()


def print_class_analysis(class_0_analysis: Dict, class_1_analysis: Dict):
    """Print detailed analysis of both classes."""
    print("=" * 60)
    print("CLASS ANALYSIS RESULTS")
    print("=" * 60)
    
    for class_id, analysis in [(0, class_0_analysis), (1, class_1_analysis)]:
        print(f"\n🔍 CLASS {class_id} ANALYSIS")
        print("-" * 40)
        print(f"Total activations: {analysis['count']}")
        print(f"Unique sequences represented: {analysis['unique_sequences_count']}")
        
        print(f"\n📊 Token Frequency:")
        total_tokens = sum(analysis['token_counts'].values())
        for token, count in sorted(analysis['token_counts'].items()):
            percentage = (count / total_tokens) * 100
            print(f"  Token {token}: {count:,} ({percentage:.1f}%)")
        
        print(f"\n📍 Most Common Bigrams:")
        for bigram, count in analysis['bigram_counts'].most_common(10):
            print(f"  {bigram}: {count}")
        
        print(f"\n🔥 Most Representative Sequences:")
        for i, (seq_tuple, count) in enumerate(analysis['most_common_sequences'][:5]):
            seq_str = ' '.join(map(str, seq_tuple))
            print(f"  {i+1}. [{seq_str}] (appears {count} times)")
        
        print(f"\n📋 Position-wise Token Distribution:")
        for pos in range(1, 10):
            if pos in analysis['position_token_counts']:
                pos_counts = analysis['position_token_counts'][pos]
                total_pos = sum(pos_counts.values())
                token_probs = {token: (count/total_pos)*100 for token, count in pos_counts.items()}
                prob_str = ', '.join([f"{token}: {prob:.1f}%" for token, prob in sorted(token_probs.items())])
                print(f"    Pos {pos}: {prob_str}")


def main():
    print("=== Sequence Class Identification ===\n")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model_path = "checkpoints/mess3_64x1"
    model = GPT.load(model_path, device)
    model.eval()
    
    # Generate sequences
    print("Generating sequences...")
    sequences = generate_all_sequences()
    print(f"Generated {len(sequences)} sequences")
    
    # Capture activations
    print("Capturing activations...")
    batch_size = 1000
    num_batches = (len(sequences) + batch_size - 1) // batch_size
    
    all_activations = []
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(sequences))
        batch_sequences = sequences[start_idx:end_idx]
        
        batch_activations = capture_activations_batch(model, batch_sequences, device)
        all_activations.append(batch_activations)
        
        if (i + 1) % 5 == 0:
            print(f"  Processed {i + 1}/{num_batches} batches")
    
    # Concatenate and process
    final_activations = torch.cat(all_activations, dim=0)
    
    # Remove BOS token and flatten
    activations_no_bos = final_activations[:, 1:, :]  # Shape: (19683, 9, 64)
    activations_flat = activations_no_bos.reshape(-1, 64).cpu().numpy()  # Shape: (177147, 64)
    
    print(f"Final activation shape: {activations_flat.shape}")
    
    # Create mapping
    print("Creating activation mapping...")
    mapping = create_activation_mapping(sequences, final_activations)
    
    # Perform PCA
    print("Performing PCA...")
    pca = PCA(n_components=2)
    activations_pca = pca.fit_transform(activations_flat)
    
    print(f"PCA explained variance: PC1={pca.explained_variance_ratio_[0]:.4f}, PC2={pca.explained_variance_ratio_[1]:.4f}")
    
    # Classify activations
    print("Classifying activations...")
    labels, threshold = classify_activations(activations_pca, method='kmeans')
    
    print(f"Classification threshold: {threshold:.4f}")
    print(f"Class 0 count: {np.sum(labels == 0)}")
    print(f"Class 1 count: {np.sum(labels == 1)}")
    
    # Analyze patterns
    print("Analyzing class patterns...")
    class_0_analysis = analyze_class_patterns(sequences, mapping, labels, 0)
    class_1_analysis = analyze_class_patterns(sequences, mapping, labels, 1)
    
    # Print results
    print_class_analysis(class_0_analysis, class_1_analysis)
    
    # Create visualizations
    print("\nCreating visualizations...")
    visualize_classes(activations_pca, labels, threshold)
    print("Visualization saved to play/plots/sequence_classes.png")
    
    print("\n=== Analysis Complete ===")


if __name__ == "__main__":
    main()