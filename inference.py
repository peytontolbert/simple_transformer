import torch
from transformer.model import SimpleTransformer
from transformer import _model_stack
import numpy as np

def load_model(model_path, vocab_size, d_model=256, num_heads=8, num_layers=3):
    """Load a trained transformer model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint['epoch']} with validation loss: {checkpoint['val_loss']:.4f}")
    return model, device

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering."""
    top_k = min(top_k, logits.size(-1))
    return _model_stack.runtime_filter_logits(
        logits,
        top_k=top_k,
        top_p=top_p,
        filter_value=filter_value,
    )

def generate_sequence(model, start_sequence, max_length=100, temperature=0.7, top_k=50, top_p=0.9, end_token=None):
    """Generate a sequence using the trained model with better sampling strategies"""
    device = next(model.parameters()).device
    model.eval()
    
    with torch.no_grad():
        current_sequence = start_sequence.to(device)
        
        for _ in range(max_length - current_sequence.size(1)):
            # Get model predictions
            output = model(current_sequence)
            next_token_logits = output[:, -1, :]

            next_token = _model_stack.sample_next_token(
                next_token_logits,
                current_sequence,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            
            # Append to sequence
            current_sequence = torch.cat([current_sequence, next_token], dim=1)
            
            # Stop if we predict the end token
            if end_token is not None and next_token.item() == end_token:
                break
    
    return current_sequence

def main():
    # Example usage
    vocab_size = 1000  # Should match training
    model_path = 'transformer_model_best.pth'  # Using best model
    
    # Load model
    model, device = load_model(model_path, vocab_size)
    
    # Create a sample start sequence
    start_tokens = torch.tensor([[1, 2, 3]]).to(device)  # Replace with actual start tokens
    
    # Generate sequence with different parameters
    print("\nGenerating with high temperature (more random):")
    generated_high_temp = generate_sequence(model, start_tokens, temperature=1.0)
    print(generated_high_temp.cpu().numpy())
    
    print("\nGenerating with low temperature (more focused):")
    generated_low_temp = generate_sequence(model, start_tokens, temperature=0.5)
    print(generated_low_temp.cpu().numpy())
    
    print("\nGenerating with nucleus sampling:")
    generated_nucleus = generate_sequence(model, start_tokens, top_p=0.9, top_k=0)
    print(generated_nucleus.cpu().numpy())

if __name__ == '__main__':
    main() 
