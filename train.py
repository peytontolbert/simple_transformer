import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformer.model import SimpleTransformer
from tqdm import tqdm
import numpy as np

def create_sample_data(num_samples=1000, seq_length=20, vocab_size=1000):
    """Create sample data for demonstration"""
    x = torch.randint(1, vocab_size, (num_samples, seq_length))
    y = torch.randint(1, vocab_size, (num_samples, seq_length))
    return x, y

def train(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    for batch_idx, (data, target) in enumerate(progress_bar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        
        loss = criterion(output.view(-1, output.size(-1)), target.view(-1))
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': total_loss / (batch_idx + 1)})
    
    return total_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc='Validation')
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output.view(-1, output.size(-1)), target.view(-1))
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': total_loss / (batch_idx + 1)})
    
    return total_loss / len(val_loader)

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def main():
    # Hyperparameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab_size = 1000
    d_model = 256
    num_heads = 8
    num_layers = 3
    batch_size = 32
    num_epochs = 50  # Increased epochs since we have early stopping
    learning_rate = 0.001  # Slightly higher initial learning rate
    val_split = 0.1
    patience = 7  # Early stopping patience
    
    # Create model
    model = SimpleTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers
    ).to(device)
    
    # Create sample data
    x_data, y_data = create_sample_data(vocab_size=vocab_size)
    dataset = TensorDataset(x_data, y_data)
    
    # Split into train and validation
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    early_stopping = EarlyStopping(patience=patience)
    
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(1, num_epochs + 1):
        train_loss = train(model, train_loader, criterion, optimizer, device, epoch)
        val_loss = validate(model, val_loader, criterion, device)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch}/{num_epochs}:')
        print(f'  Train Loss: {train_loss:.4f}')
        print(f'  Val Loss: {val_loss:.4f}')
        print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, 'transformer_model_best.pth')
        
        # Save latest model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, 'transformer_model_latest.pth')
        
        # Early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
    
    print("Training completed and models saved!")

if __name__ == '__main__':
    main() 