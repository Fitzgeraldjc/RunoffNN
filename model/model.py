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
from torch.utils.data import TensorDataset, DataLoader
from utilities import combine_data_from_folders
import pandas as pd
from datetime import datetime

# First, simplify the model architecture for stability
class RunoffLSTM(nn.Module):
    def __init__(self, input_size=12, hidden_size=32, num_layers=1):
        super(RunoffLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0  # Disable dropout initially
        )
        # Simpler architecture - no attention for now
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
    def forward(self, x):
        # x shape: [batch, seq_len, features]
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, hidden_size]
        output = self.fc(lstm_out)  # [batch, seq_len, 1]
        return output

# Use standard MSE loss initially until training stabilizes
def custom_loss(pred, target, high_flow_weight=0.0):  # Setting weight to 0 = standard MSE
    return torch.mean((pred - target) ** 2)

# Train one epoch with gradient clipping
# and NaN loss handling
def train_one_epoch(model, train_loader, optimizer, device='cpu', clip_value=0.25):
    """Train the model for one epoch with gradient clipping to prevent explosions."""
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = custom_loss(output, target)
        
        # Check for NaN loss
        if torch.isnan(loss).item():
            print(f"Warning: NaN loss detected in batch {batch_idx}. Skipping.")
            continue
            
        loss.backward()
        
        # Add gradient clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / max(1, len(train_loader))  # Avoid division by zero

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

# In the train_model function, add data preprocessing:
def train_model(folder_paths, epochs=100, batch_size=32, learning_rate=0.0001, window_size=8, step=2):
    """
    Train the runoff prediction model on combined data from multiple folders.
    Train on data outside Oct 2022-Apr 2023, test on Oct 2022-Apr 2023.
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directories
    os.makedirs('runs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Prepare data from multiple folders with time-based split
    X_train, y_train, X_test, y_test = combine_data_from_folders(
        folder_paths, window_size=window_size, step=step)
    
    # Check input data for problems
    if X_train is not None:
        print(f"X_train shape: {X_train.shape}, contains NaN: {torch.isnan(X_train).any()}")
        print(f"X_train min: {X_train.min()}, max: {X_train.max()}")
        print(f"X_train mean: {X_train.mean()}")
        
        # Check for infinite values
        print(f"X_train contains inf: {torch.isinf(X_train).any()}")
        
        # Sample some values to inspect
        print(f"First sequence first timestep: {X_train[0, 0]}")
        
    if y_train is not None:
        print(f"y_train shape: {y_train.shape}, contains NaN: {torch.isnan(y_train).any()}")
        print(f"y_train min: {y_train.min()}, max: {y_train.max()}")
        print(f"y_train mean: {y_train.mean()}")
    
    if X_train is None or y_train is None:
        print("Failed to prepare data. Exiting.")
        return None
    
    # Custom normalization function
    def normalize_data(tensor):
        # Handle NaN values first
        tensor = torch.nan_to_num(tensor, nan=0.0)
        # Apply log1p to handle extreme values but keep signs
        sign = torch.sign(tensor)
        log_tensor = torch.log1p(torch.abs(tensor))
        return sign * log_tensor
    
    # Apply to both datasets
    X_train_norm = normalize_data(X_train.clone())
    y_train_norm = normalize_data(y_train.clone())
    
    if X_test is not None:
        X_test_norm = normalize_data(X_test.clone())
        y_test_norm = normalize_data(y_test.clone())
        print(f"After normalization - X_test min: {X_test_norm.min()}, max: {X_test_norm.max()}")
        print(f"After normalization - y_test min: {y_test_norm.min()}, max: {y_test_norm.max()}")
    else:
        X_test_norm, y_test_norm = None, None
    
    print(f"After normalization - X_train min: {X_train_norm.min()}, max: {X_train_norm.max()}")
    print(f"After normalization - y_train min: {y_train_norm.min()}, max: {y_train_norm.max()}")
    
    # Use normalized data for creating datasets
    train_dataset = TensorDataset(X_train_norm, y_train_norm)
    total_size = len(train_dataset)
    train_size = int(0.9 * total_size)  # 90% for training
    val_size = total_size - train_size   # 10% for validation
    
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size)
    
    # Test loader from test data
    test_loader = None
    if X_test is not None and y_test is not None:
        test_dataset = TensorDataset(X_test_norm, y_test_norm)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        print(f"Test set: {len(test_dataset)} sequences from Oct 2022-Apr 2023")
    
    # Initialize model, optimizer, and scheduler
    input_size = X_train.shape[2]  # Number of features

    # Data normalization - check for extreme values
    print(f"X_train min: {X_train.min()}, max: {X_train.max()}")
    print(f"y_train min: {y_train.min()}, max: {y_train.max()}")
    
    # Initialize model with xavier initialization for better stability
    model = RunoffLSTM(input_size=input_size).to(device)
    
    # Use a more stable initialization
    for name, param in model.named_parameters():
        if 'weight' in name:
            nn.init.xavier_normal_(param)
        elif 'bias' in name:
            nn.init.constant_(param, 0.0)
            
    # Use a much lower learning rate
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, eps=1e-8)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # TensorBoard writer
    writer = SummaryWriter(f'runs/runoff_experiment_{time.strftime("%Y%m%d-%H%M%S")}')
    
    # Early stopping parameters
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    best_model_path = 'models/best_runoff_model.pth'
    
    prev_lr = learning_rate  # Initialize previous learning rate
    
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

        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']
    
        if epoch > 0 and current_lr != prev_lr:
            print(f"Learning rate adjusted to {current_lr}")
    
        prev_lr = current_lr
        
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
        
        # Log histograms of model parameters - skip NaN values
        for name, param in model.named_parameters():
            if not torch.isnan(param).any():
                try:
                    writer.add_histogram(f'Parameters/{name}', param, epoch)
                except ValueError as e:
                    print(f"Warning: Could not add histogram for {name}: {e}")
    
    # Load best model for evaluation
    if np.isfinite(best_val_loss):  # Only try to load if we actually saved a model
        try:
            model.load_state_dict(torch.load(best_model_path))
        except FileNotFoundError:
            print("Warning: Could not load best model. Using final model instead.")
    else:
        print("Warning: No finite validation loss encountered. Using final model.")
    
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

def visualize_leadtime_boxplots(model, test_loader, device='cpu', denormalize_func=None):
    """
    Creates boxplots showing observed runoff, NWM forecast, and LSTM-corrected forecast by lead time.
    
    Args:
        model: Trained LSTM model
        test_loader: DataLoader containing test data
        device: Device to run inference on
        denormalize_func: Optional function to convert normalized values back to original scale
    """
    model.eval()
    
    # Lists to store data
    all_lead_times = []
    all_streamIDs = []
    observed_values = []
    nwm_forecasts = []
    lstm_forecasts = []
    
    # Collect predictions, actual values, and original NWM forecasts
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            # Get LSTM predictions
            output = model(data)
            
            # Extract NWM forecasts from input features (index 2 contains streamflow_value)
            nwm_forecast = data[:, :, 2]
            
            # Extract lead time (hours_from_init is at index 11)
            lead_times = data[:, :, 11].cpu().numpy()
            
            # Extract streamID (at index 4)
            stream_ids = data[:, :, 4].cpu().numpy()
            
            # Process batch
            for batch_idx in range(data.shape[0]):
                for seq_idx in range(data.shape[1]):
                    # Store data
                    all_lead_times.append(lead_times[batch_idx, seq_idx])
                    all_streamIDs.append(int(stream_ids[batch_idx, seq_idx]))
                    observed_values.append(target[batch_idx, seq_idx, 0].cpu().item())
                    nwm_forecasts.append(nwm_forecast[batch_idx, seq_idx].cpu().item())
                    lstm_forecasts.append(output[batch_idx, seq_idx, 0].cpu().item())
    
    # Convert to numpy arrays
    all_lead_times = np.array(all_lead_times)
    all_streamIDs = np.array(all_streamIDs)
    observed_values = np.array(observed_values)
    nwm_forecasts = np.array(nwm_forecasts)
    lstm_forecasts = np.array(lstm_forecasts)
    
    # Denormalize if function provided
    if denormalize_func:
        observed_values = denormalize_func(observed_values)
        nwm_forecasts = denormalize_func(nwm_forecasts)
        lstm_forecasts = denormalize_func(lstm_forecasts)
    
    # Create DataFrame for easier grouping and analysis
    df = pd.DataFrame({
        'LeadTime': all_lead_times,
        'StreamID': all_streamIDs,
        'Observed': observed_values,
        'NWM': nwm_forecasts,
        'LSTM': lstm_forecasts
    })
    
    # Define lead time bins (e.g., 0-6h, 6-12h, 12-24h, 24-48h, 48-72h)
    lead_time_bins = [0, 6, 12, 24, 48, 72, np.inf]
    lead_time_labels = ['0-6h', '6-12h', '12-24h', '24-48h', '48-72h', '72+h']
    
    # Add lead time bin column
    df['LeadTimeBin'] = pd.cut(df['LeadTime'], bins=lead_time_bins, labels=lead_time_labels)
    
    # Create boxplots by lead time
    plt.figure(figsize=(15, 10))
    
    # Plot all data together
    plt.subplot(2, 1, 1)
    
    # Get actual lead time bins present in the data
    present_bins = [bin for bin in lead_time_labels if bin in df['LeadTimeBin'].values]
    
    # Create positions for grouped boxplots
    positions = np.arange(len(present_bins)) * 4
    width = 1.0
    
    # Plot observed, NWM, and LSTM boxplots side by side
    bp1 = plt.boxplot([df[df['LeadTimeBin'] == bin]['Observed'].values for bin in present_bins], 
                     positions=positions-width, patch_artist=True, widths=width)
    bp2 = plt.boxplot([df[df['LeadTimeBin'] == bin]['NWM'].values for bin in present_bins], 
                     positions=positions, patch_artist=True, widths=width)
    bp3 = plt.boxplot([df[df['LeadTimeBin'] == bin]['LSTM'].values for bin in present_bins], 
                     positions=positions+width, patch_artist=True, widths=width)
    
    # Set colors
    for box in bp1['boxes']:
        box.set(facecolor='lightblue')
    for box in bp2['boxes']:
        box.set(facecolor='lightgreen')
    for box in bp3['boxes']:
        box.set(facecolor='salmon')
    
    plt.xticks(positions, present_bins)
    plt.ylabel('Flow Value')
    plt.title('Lead Time Comparison of Flow Values')
    plt.legend([bp1["boxes"][0], bp2["boxes"][0], bp3["boxes"][0]], 
               ['Observed', 'NWM Forecast', 'LSTM Corrected'], loc='upper right')
    
    # Create percent improvement plot
    plt.subplot(2, 1, 2)
    
    # Calculate error metrics by lead time bin
    leadtime_metrics = []
    for bin in present_bins:
        bin_data = df[df['LeadTimeBin'] == bin]
        
        # NWM errors
        nwm_mae = np.mean(np.abs(bin_data['NWM'] - bin_data['Observed']))
        
        # LSTM errors
        lstm_mae = np.mean(np.abs(bin_data['LSTM'] - bin_data['Observed']))
        
        # Percent improvement
        improvement = ((nwm_mae - lstm_mae) / nwm_mae) * 100
        
        leadtime_metrics.append({
            'LeadTimeBin': bin,
            'NWM_MAE': nwm_mae,
            'LSTM_MAE': lstm_mae,
            'Improvement': improvement
        })
    
    df_metrics = pd.DataFrame(leadtime_metrics)
    
    # Create bar chart of improvement percentages
    plt.bar(present_bins, df_metrics['Improvement'], color='purple')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.ylabel('% Improvement in MAE')
    plt.title('LSTM Model Improvement over NWM by Lead Time')
    plt.tight_layout()
    plt.savefig('models/leadtime_improvement.png')
    plt.show()
    
    # Add value labels on top of bars
    for i, val in enumerate(df_metrics['Improvement']):
        plt.text(i, val + (1 if val > 0 else -5), f"{val:.1f}%", ha='center')
    
    plt.tight_layout()
    plt.savefig('models/leadtime_performance.png')
    
    # Create separate plots by stream gauge
    plt.figure(figsize=(15, 10))
    for stream_id in [0, 1]:
        stream_name = '20380357' if stream_id == 0 else '21609641'
        stream_data = df[df['StreamID'] == stream_id]
        if stream_data.empty:
            continue
            
        plt.subplot(2, 1, stream_id + 1)
        
        # Get bins present in this stream's data
        stream_bins = [bin for bin in present_bins if bin in stream_data['LeadTimeBin'].values]
        
        # Skip if no data for this stream
        if not stream_bins:
            continue
            
        # Create positions for grouped boxplots
        positions = np.arange(len(stream_bins)) * 4
        
        # Plot observed, NWM, and LSTM boxplots side by side
        bp1 = plt.boxplot([stream_data[stream_data['LeadTimeBin'] == bin]['Observed'].values for bin in stream_bins], 
                         positions=positions-width, patch_artist=True, widths=width)
        bp2 = plt.boxplot([stream_data[stream_data['LeadTimeBin'] == bin]['NWM'].values for bin in stream_bins], 
                         positions=positions, patch_artist=True, widths=width)
        bp3 = plt.boxplot([stream_data[stream_data['LeadTimeBin'] == bin]['LSTM'].values for bin in stream_bins], 
                         positions=positions+width, patch_artist=True, widths=width)
        
        # Set colors
        for box in bp1['boxes']:
            box.set(facecolor='lightblue')
        for box in bp2['boxes']:
            box.set(facecolor='lightgreen')
        for box in bp3['boxes']:
            box.set(facecolor='salmon')
        
        plt.xticks(positions, stream_bins)
        plt.ylabel('Flow Value')
        plt.title(f'Lead Time Comparison for Stream Gauge {stream_name}')
        plt.legend([bp1["boxes"][0], bp2["boxes"][0], bp3["boxes"][0]], 
                  ['Observed', 'NWM Forecast', 'LSTM Corrected'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig('models/leadtime_performance_by_gauge.png')
    plt.show()
    
    # Return a summary of improvements by lead time
    return df_metrics

if __name__ == "__main__":
    # Update to use both folders
    folder_paths = ["./data/20380357", "./data/21609641"]
    
    # Train model on combined data with much lower learning rate
    model, (train_loader, val_loader, test_loader) = train_model(
        folder_paths, 
        epochs=100,
        batch_size=32,
        learning_rate=0.0001,  # Much lower learning rate
        window_size=8,
        step=2
    )
    
    # Evaluate model
    if model is not None and test_loader is not None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Standard evaluation
        metrics = evaluate_model(model, test_loader, device)
        print(metrics)
        
        # Define inverse normalization function to match the normalization used in train_model
        def denormalize(normalized_values):
            # Inverse of sign(x) * log(1 + |x|)
            return np.sign(normalized_values) * (np.exp(np.abs(normalized_values)) - 1)
        
        # Generate lead time visualizations
        print("\nGenerating lead time performance visualizations...")
        leadtime_metrics = visualize_leadtime_boxplots(model, test_loader, device, denormalize)
        print("\nLead time performance metrics:")
        print(leadtime_metrics)