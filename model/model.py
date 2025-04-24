import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import os
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from utilities import prepare_data_for_training, create_dataloaders

class RunoffLSTM(nn.Module):
    def __init__(self, input_size=12, hidden_size=64, num_layers=2):
        super(RunoffLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        # x shape: [batch, seq_len, features]
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, hidden_size]
        
        # Optional attention mechanism
        attention_weights = self.attention(lstm_out)  # [batch, seq_len, 1]
        context = torch.sum(attention_weights * lstm_out, dim=1)  # [batch, hidden_size]
        
        # For sequence-to-sequence prediction (return predictions for each time step)
        output = self.fc(lstm_out)  # [batch, seq_len, 1]
        return output

def custom_loss(pred, target, high_flow_weight=0.5):
    """
    Custom loss function that penalizes errors on high flow events more.
    Args:
        pred: Model predictions
        target: Ground truth values
        high_flow_weight: Weight for high flow errors (0.5 = 50% extra weight)
    """
    # Standard MSE
    mse = torch.mean((pred - target) ** 2)
    
    # Identify high flow events (top 20%)
    threshold = torch.quantile(target, 0.8)
    high_flow_mask = target > threshold
    
    # If we have high flow events, add additional penalty
    if torch.any(high_flow_mask):
        high_flow_error = torch.mean((pred[high_flow_mask] - target[high_flow_mask]) ** 2)
        return mse + high_flow_weight * high_flow_error
    return mse

def train_one_epoch(model, train_loader, optimizer, device='cpu'):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = custom_loss(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def validate(model, val_loader, device='cpu'):
    """Validate the model."""
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += custom_loss(output, target).item()
    
    return val_loss / len(val_loader)

def train_model(folder_path, epochs=100, batch_size=32, learning_rate=0.001, window_size=8, step=2):
    """Train the runoff prediction model."""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directories
    os.makedirs('runs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Prepare data
    X_tensor, y_tensor = prepare_data_for_training(folder_path, window_size=window_size, step=step)
    if X_tensor is None or y_tensor is None:
        print("Failed to prepare data. Exiting.")
        return None
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(X_tensor, y_tensor, batch_size=batch_size)
    
    # Initialize model, optimizer, and scheduler
    input_size = X_tensor.shape[2]  # Number of features
    model = RunoffLSTM(input_size=input_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # TensorBoard writer
    writer = SummaryWriter(f'runs/runoff_experiment_{time.strftime("%Y%m%d-%H%M%S")}')
    
    # Early stopping parameters
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    best_model_path = 'models/best_runoff_model.pth'
    
    # Training loop
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate(model, val_loader, device)
        val_losses.append(val_loss)
        
        # Log to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/validation', val_loss, epoch)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
        
        # Print progress
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        
        # Log histograms of model parameters
        for name, param in model.named_parameters():
            writer.add_histogram(f'Parameters/{name}', param, epoch)
    
    # Load best model for evaluation
    model.load_state_dict(torch.load(best_model_path))
    
    # Plot training history
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 1, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.tight_layout()
    plt.savefig('models/training_history.png')
    
    return model, (train_loader, val_loader, test_loader)

def evaluate_model(model, test_loader, device='cpu'):
    """Evaluate the model on test data."""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # Collect all predictions and targets
            all_preds.append(output.cpu().numpy())
            all_targets.append(target.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Calculate metrics
    mse = mean_squared_error(all_targets.flatten(), all_preds.flatten())
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_targets.flatten(), all_preds.flatten())
    r2 = r2_score(all_targets.flatten(), all_preds.flatten())
    
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Test R² Score: {r2:.4f}")
    
    # Plot sample predictions
    plt.figure(figsize=(15, 6))
    
    # Plot a sample sequence
    sample_idx = np.random.randint(0, len(all_preds))
    sample_pred = all_preds[sample_idx, :, 0]
    sample_target = all_targets[sample_idx, :, 0]
    
    plt.subplot(1, 2, 1)
    plt.plot(sample_target, 'b-', label='Observed')
    plt.plot(sample_pred, 'r-', label='Predicted')
    plt.title(f'Sample Sequence Prediction (idx={sample_idx})')
    plt.legend()
    
    # Plot actual vs predicted scatter
    plt.subplot(1, 2, 2)
    plt.scatter(all_targets.flatten(), all_preds.flatten(), alpha=0.3)
    plt.plot([min(all_targets.flatten()), max(all_targets.flatten())], 
             [min(all_targets.flatten()), max(all_targets.flatten())], 'r--')
    plt.xlabel('Observed Flow')
    plt.ylabel('Predicted Flow')
    plt.title('Actual vs Predicted')
    
    plt.tight_layout()
    plt.savefig('models/model_evaluation.png')
    plt.show()
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

if __name__ == "__main__":
    folder_path = "./data/20380357"  # Adjust as needed
    
    # Train model
    model, (train_loader, val_loader, test_loader) = train_model(
        folder_path, 
        epochs=100,
        batch_size=32,
        learning_rate=0.001,
        window_size=8,
        step=2
    )
    
    # Evaluate model
    if model is not None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        metrics = evaluate_model(model, test_loader, device)
        print(metrics)