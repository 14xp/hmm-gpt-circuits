#!/usr/bin/env python3
"""
Script to load the GPT model from checkpoints/mess3_64x1 and demonstrate inference.
Produces example output, logits, and internal activations.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from models.gpt import GPT
from config.gpt.models import GPTConfig


def main():
    print("=== GPT Model Inference Demo ===\n")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the model
    print("Loading model from checkpoints/mess3_64x1...")
    model_path = "checkpoints/mess3_64x1"
    model = GPT.load(model_path, device)
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Config: {model.config}")
    print(f"Tokenizer vocab size: {model.config.tokenizer.vocab_size}")
    print()
    
    # Create sample input (vocab_size=4, so tokens are 0, 1, 2, 3)
    print("=== Creating Sample Input ===")
    sample_tokens = [3, 1, 2, 0, 0]  # Simple sequence within vocab bounds
    input_ids = torch.tensor([sample_tokens], device=device)  # Shape: (1, 5)
    print(f"Input tokens: {sample_tokens}")
    print(f"Input shape: {input_ids.shape}")
    print(f"Decoded input: {model.config.tokenizer.decode_sequence(sample_tokens)}")
    print()
    
    # Run inference with activations capture
    print("=== Running Inference ===")
    with torch.no_grad():
        # Forward pass to get logits
        logits, _ = model(input_ids)
        
        # Manual forward pass to capture activations before final layer norm
        B, T = input_ids.size()
        pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device)
        pos_emb = model.transformer.wpe(pos)
        tok_emb = model.transformer.wte(input_ids)
        x = tok_emb + pos_emb
        
        # Forward through transformer blocks
        for block in model.transformer.h:
            x = block(x)
        
        # x now contains activations before final layer norm
        activations_before_ln = x.clone()
        
        # Apply final layer norm and get logits
        x = model.transformer.ln_f(x)
        final_logits = model.lm_head(x)
    
    print("Forward pass completed!")
    print()
    
    # Display results
    print("=== Model Outputs ===")
    
    # 1. Logits
    print("1. Raw Logits:")
    print(f"Shape: {logits.shape}")
    for t in range(logits.shape[1]):
        print(f"  Position {t}: {logits[0, t].cpu().numpy()}")
    print()
    
    # 2. Activations before final layer norm
    print("2. Activations before final layer norm:")
    print(f"Shape: {activations_before_ln.shape}")
    for t in range(min(3, activations_before_ln.shape[1])):  # Show first 3 positions
        activation = activations_before_ln[0, t].cpu().numpy()
        print(f"  Position {t}: mean={activation.mean():.4f}, std={activation.std():.4f}")
        print(f"    First 10 values: {activation[:10]}")
    print()
    
    # 3. Probabilities
    print("3. Token Probabilities (softmax of logits):")
    probs = F.softmax(logits, dim=-1)
    for t in range(logits.shape[1]):
        pos_probs = probs[0, t].cpu().numpy()
        print(f"  Position {t}: {pos_probs}")
        # Show most likely token
        top_token = torch.argmax(probs[0, t]).item()
        top_prob = probs[0, t, top_token].item()
        print(f"    Most likely: token {top_token} (prob={top_prob:.4f})")
    print()
    
    # 4. Generate next token predictions
    print("4. Next Token Predictions:")
    print("Input sequence:", model.config.tokenizer.decode_sequence(sample_tokens))
    
    # Get predictions for each position
    for t in range(logits.shape[1]):
        predicted_token = torch.argmax(logits[0, t]).item()
        confidence = F.softmax(logits[0, t], dim=-1)[predicted_token].item()
        decoded_token = model.config.tokenizer.decode_token(predicted_token)
        print(f"  After position {t}: predict token {predicted_token} '{decoded_token}' (confidence: {confidence:.4f})")
    
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()