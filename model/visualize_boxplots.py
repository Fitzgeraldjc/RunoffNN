import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import sys

# Add the model directory to path for imports
sys.path.append('/Users/jacobfitzgerald/Documents/GitHub/RunoffNN/model')
from utilities import load_nwm_predictions, load_usgs_observations
from model import RunoffLSTM

def generate_lead_time_boxplots(folder_paths, model_path, output_file='visualizations/leadtime_boxplots.png'):
    """Generate side-by-side boxplots with normalized scales for better visualization."""
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    try:
        model = RunoffLSTM(input_size=12)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Process data but only keeping the key parts needed for the boxplot visualization
    
    stream_id = 0
    folder_path = folder_paths[0]
    gauge_name = '20380357'
    
    # Prepare data collection
    lead_time_data = {}
    for hour in range(1, 19):
        if hour == 9:  # Skip problematic hour 9
            continue
        lead_time_data[hour] = {
            'observed': [],
            'nwm': [],
            'lstm': []
        }
    
    try:
        # Load and prepare data
        predictions_df = load_nwm_predictions(folder_path, stream_id)
        observations_df = load_usgs_observations(folder_path)
        
        # Make sure all datetimes are properly formatted
        predictions_df['timestamp'] = pd.to_datetime(predictions_df['timestamp']).dt.tz_localize(None)
        observations_df['timestamp'] = pd.to_datetime(observations_df['timestamp']).dt.tz_localize(None)
        
        # Merge predictions with observations
        merged_df = pd.merge(predictions_df, observations_df, on='timestamp', how='inner')
        print(f"Found {len(merged_df)} matching records after merge")
        
        # Process data in exactly the same way as during training
        merged_df['input_time_numeric'] = merged_df['model_initialization_time'].apply(lambda x: x.timestamp())
        merged_df['output_time_numeric'] = merged_df['timestamp'].apply(lambda x: x.timestamp())
        
        # Cyclical encoding of time features
        merged_df['hour_sin'] = np.sin(2 * np.pi * merged_df['timestamp'].dt.hour / 24)
        merged_df['hour_cos'] = np.cos(2 * np.pi * merged_df['timestamp'].dt.hour / 24)
        merged_df['day_sin'] = np.sin(2 * np.pi * merged_df['timestamp'].dt.day / 31)
        merged_df['day_cos'] = np.cos(2 * np.pi * merged_df['timestamp'].dt.day / 31)
        merged_df['month_sin'] = np.sin(2 * np.pi * merged_df['timestamp'].dt.month / 12)
        merged_df['month_cos'] = np.cos(2 * np.pi * merged_df['timestamp'].dt.month / 12)
        
        # Time delta from initialization (in hours)
        merged_df['hours_from_init'] = (merged_df['timestamp'] - 
                                     merged_df['model_initialization_time']).dt.total_seconds() / 3600
        
        # Normalize streamflow
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        streamflow_values = merged_df['streamflow_value'].values.reshape(-1, 1)
        merged_df['streamflow_norm'] = scaler.fit_transform(streamflow_values).flatten()
        
        # Group data by lead time (rounded to nearest hour)
        merged_df['lead_time_hour'] = merged_df['hours_from_init'].round().astype(int)
        
        # Process each lead time group
        for lead_hour in range(1, 19):
            if lead_hour == 9:  # Skip problematic lead time
                continue
                
            # Get data for this lead time
            lead_data = merged_df[merged_df['lead_time_hour'] == lead_hour].copy()
            
            if lead_data.empty:
                continue
                
            # Get observations and NWM forecasts for this lead time
            valid_indices = ~lead_data['USGSFlowValue'].isna()
            
            if valid_indices.sum() == 0:
                continue
                
            obs_values = lead_data.loc[valid_indices, 'USGSFlowValue'].values
            nwm_values = lead_data.loc[valid_indices, 'streamflow_value'].values
            
            # Get LSTM predictions
            features = np.column_stack((
                lead_data['input_time_numeric'].values.astype(np.float32),
                lead_data['output_time_numeric'].values.astype(np.float32),
                lead_data['streamflow_value'].values.astype(np.float32),
                lead_data['streamflow_norm'].values.astype(np.float32),
                lead_data['streamID'].values.astype(np.float32),
                lead_data['hour_sin'].values.astype(np.float32),
                lead_data['hour_cos'].values.astype(np.float32),
                lead_data['day_sin'].values.astype(np.float32),
                lead_data['day_cos'].values.astype(np.float32),
                lead_data['month_sin'].values.astype(np.float32),
                lead_data['month_cos'].values.astype(np.float32),
                lead_data['hours_from_init'].values.astype(np.float32)
            ))
            
            # Normalize features
            def normalize_data(tensor_array):
                tensor_array = np.nan_to_num(tensor_array, nan=0.0)
                sign = np.sign(tensor_array)
                log_tensor = np.log1p(np.abs(tensor_array))
                return sign * log_tensor
            
            norm_features = normalize_data(features)
            
            # Process through model
            with torch.no_grad():
                inputs = torch.from_numpy(norm_features).float().to(device)
                
                # Process in batches
                batch_size = 128
                all_outputs = []
                
                for i in range(0, len(inputs), batch_size):
                    batch = inputs[i:i+batch_size]
                    if len(batch.shape) == 2:  # [samples, features]
                        batch = batch.unsqueeze(1)  # -> [samples, 1, features]
                    
                    batch_output = model(batch)
                    
                    if len(batch_output.shape) == 3:  # [samples, 1, 1]
                        batch_output = batch_output.squeeze(-1).squeeze(-1)
                    elif len(batch_output.shape) == 2:  # [samples, 1]
                        batch_output = batch_output.squeeze(-1)
                        
                    all_outputs.append(batch_output.cpu().numpy())
                
                outputs = np.concatenate(all_outputs)
            
            # Denormalize outputs
            def denormalize(normalized_values):
                return np.sign(normalized_values) * (np.exp(np.abs(normalized_values)) - 1)
                
            lstm_values = denormalize(outputs)
            
            # Store values only for valid indices
            lead_time_data[lead_hour]['observed'].extend(obs_values.tolist())
            lead_time_data[lead_hour]['nwm'].extend(nwm_values.tolist())
            lead_time_data[lead_hour]['lstm'].extend(lstm_values[valid_indices].tolist())
            
    except Exception as e:
        print(f"Error processing data: {e}")
        import traceback
        traceback.print_exc()
    
    # Create the boxplot visualization
    valid_lead_times = []
    for lt in range(1, 19):
        if lt in lead_time_data and len(lead_time_data[lt]['observed']) > 0:
            valid_lead_times.append(lt)
    
    print(f"Valid lead times: {valid_lead_times}")
    
    if not valid_lead_times:
        print("No valid lead times found")
        return
    
    # Create normalized versions of the data for better visualization
    # Use ratio to observed as the normalized metric
    normalized_data = {}
    for lt in valid_lead_times:
        normalized_data[lt] = {
            'observed': [1.0] * len(lead_time_data[lt]['observed']),  # Always 1.0
            'nwm': [],
            'lstm': []
        }
        
        for i in range(len(lead_time_data[lt]['observed'])):
            obs = lead_time_data[lt]['observed'][i]
            if obs > 0:
                normalized_data[lt]['nwm'].append(lead_time_data[lt]['nwm'][i] / obs)
                normalized_data[lt]['lstm'].append(lead_time_data[lt]['lstm'][i] / obs)
            else:
                # Handle division by zero
                normalized_data[lt]['nwm'].append(np.nan)
                normalized_data[lt]['lstm'].append(np.nan)
    
    # Create the multi-panel figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 15))
    
    # -----------------------------------
    # Raw Values Boxplot (Log Scale)
    # -----------------------------------
    width = 0.25
    positions = np.arange(len(valid_lead_times))
    
    # Compute positions for each group
    pos_obs = positions - width
    pos_nwm = positions
    pos_lstm = positions + width
    
    # Create boxplots
    bp1 = ax1.boxplot([lead_time_data[lt]['observed'] for lt in valid_lead_times], 
                      positions=pos_obs, patch_artist=True, widths=width*0.9, showfliers=False)
    
    bp2 = ax1.boxplot([lead_time_data[lt]['nwm'] for lt in valid_lead_times], 
                      positions=pos_nwm, patch_artist=True, widths=width*0.9, showfliers=False)
    
    bp3 = ax1.boxplot([lead_time_data[lt]['lstm'] for lt in valid_lead_times], 
                      positions=pos_lstm, patch_artist=True, widths=width*0.9, showfliers=False)
    
    # Set colors
    for box in bp1['boxes']: box.set(facecolor='lightblue')
    for box in bp2['boxes']: box.set(facecolor='lightgreen')
    for box in bp3['boxes']: box.set(facecolor='salmon')
    
    # Add legend and labels
    ax1.legend([bp1["boxes"][0], bp2["boxes"][0], bp3["boxes"][0]], 
              ['Observed', 'NWM Forecast', 'LSTM Corrected'], loc='upper right')
    
    # Set y-scale to logarithmic
    ax1.set_yscale('log')
    ax1.set_ylabel('Flow Value (cfs)', fontsize=12)
    ax1.set_title('Raw Flow Values by Lead Time (Log Scale)', fontsize=14)
    
    # Set x-ticks to lead time labels
    ax1.set_xticks(positions)
    ax1.set_xticklabels([f"{lt}h" for lt in valid_lead_times])
    
    # -----------------------------------
    # Normalized Values Boxplot (Linear Scale)
    # -----------------------------------
    
    # Create boxplots for normalized data
    bp1 = ax2.boxplot([normalized_data[lt]['observed'] for lt in valid_lead_times], 
                      positions=pos_obs, patch_artist=True, widths=width*0.9, showfliers=False)
    
    bp2 = ax2.boxplot([normalized_data[lt]['nwm'] for lt in valid_lead_times], 
                      positions=pos_nwm, patch_artist=True, widths=width*0.9, showfliers=False)
    
    bp3 = ax2.boxplot([normalized_data[lt]['lstm'] for lt in valid_lead_times], 
                      positions=pos_lstm, patch_artist=True, widths=width*0.9, showfliers=False)
    
    # Set colors
    for box in bp1['boxes']: box.set(facecolor='lightblue')
    for box in bp2['boxes']: box.set(facecolor='lightgreen')
    for box in bp3['boxes']: box.set(facecolor='salmon')
    
    # Add legend and labels
    ax2.legend([bp1["boxes"][0], bp2["boxes"][0], bp3["boxes"][0]], 
              ['Observed (Always 1.0)', 'NWM/Observed Ratio', 'LSTM/Observed Ratio'], loc='upper right')
    
    # Add horizontal line at y=1 (perfect match)
    ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
    
    # Set y-scale to linear but with reasonable limits
    ax2.set_ylim(0, 50)  # Cap at 50x the observed values for visibility
    ax2.set_ylabel('Ratio to Observed Value', fontsize=12)
    ax2.set_title('Normalized Flow Values by Lead Time (Linear Scale)', fontsize=14)
    ax2.set_xlabel('Lead Time (hours)', fontsize=12)
    
    # Set x-ticks to lead time labels
    ax2.set_xticks(positions)
    ax2.set_xticklabels([f"{lt}h" for lt in valid_lead_times])
    
    # Add overall title
    fig.suptitle('Comparison of Flow Values by Lead Time - Gauge 20380357', fontsize=16)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    # Make sure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save figure
    plt.savefig(output_file, dpi=300)
    plt.show()
    
    # Also save a close-up version focusing on the LSTM vs Observed comparison
    plt.figure(figsize=(18, 8))
    
    # Positions for side-by-side boxes
    width = 0.4
    pos_obs = positions - width/2
    pos_lstm = positions + width/2
    
    # Create boxplots - just observed and LSTM
    bp1 = plt.boxplot([lead_time_data[lt]['observed'] for lt in valid_lead_times], 
                     positions=pos_obs, patch_artist=True, widths=width*0.9, showfliers=False)
    
    bp3 = plt.boxplot([lead_time_data[lt]['lstm'] for lt in valid_lead_times], 
                     positions=pos_lstm, patch_artist=True, widths=width*0.9, showfliers=False)
    
    # Set colors
    for box in bp1['boxes']: box.set(facecolor='lightblue')
    for box in bp3['boxes']: box.set(facecolor='salmon')
    
    # Add legend and labels
    plt.legend([bp1["boxes"][0], bp3["boxes"][0]], 
              ['Observed', 'LSTM Corrected'], loc='upper right')
    
    plt.ylabel('Flow Value (cfs)', fontsize=12)
    plt.title('Observed vs LSTM Corrected Flow Values by Lead Time', fontsize=14)
    plt.xlabel('Lead Time (hours)', fontsize=12)
    
    # Set x-ticks to lead time labels
    plt.xticks(positions, [f"{lt}h" for lt in valid_lead_times])
    
    plt.tight_layout()
    
    # Save figure
    closeup_file = os.path.splitext(output_file)[0] + "_closeup.png"
    plt.savefig(closeup_file, dpi=300)
    plt.show()
    
    return lead_time_data

if __name__ == "__main__":
    folder_paths = ["./data/20380357", "./data/21609641"]
    model_path = 'models/best_runoff_model.pth'
    
    generate_lead_time_boxplots(folder_paths, model_path)