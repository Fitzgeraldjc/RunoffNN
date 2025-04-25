# RunoffNN: Neural Network for Streamflow Prediction

## Overview
RunoffNN is a deep learning model for predicting streamflow levels in Watauga County using LSTM neural networks. The model takes National Water Model (NWM) predictions as input and adjusts them using historical USGS observations to create more accurate streamflow forecasts.

## Features
- LSTM-based neural network for sequence prediction
- Handles data from multiple stream gauges
- Custom feature engineering for temporal data
- Time-based train/test split (training on data outside Oct 2022-Apr 2023, testing on Oct 2022-Apr 2023)
- Advanced data normalization to handle extreme flow values
- Early stopping and learning rate scheduling
- TensorBoard integration for monitoring training

## Project Structure
- `model/`
  - `model.py` - LSTM model definition and training pipeline
  - `utilities.py` - Data loading, preprocessing, and feature engineering

## Data Format
The model works with two types of data:
1. **NWM Predictions**: CSV files starting with "streamflow_" containing columns:
   - model_initialization_time
   - model_output_valid_time
   - streamflow_value

2. **USGS Observations**: CSV files containing "Strt" with columns:
   - DateTime
   - USGSFlowValue

## Usage

### Training the Model
```python
from model.model import train_model, evaluate_model

# Define paths to data folders for different stream gauges
folder_paths = ["./data/20380357", "./data/21609641"]

# Train the model
model, loaders = train_model(
    folder_paths,
    epochs=100,
    batch_size=32,
    learning_rate=0.0001,
    window_size=8,
    step=2
)

# Evaluate the trained model
metrics = evaluate_model(model, loaders[2])  # test_loader is at index 2
print(metrics)
```
### Model Hyperparameters
- window_size: Length of input sequences (default: 8)
- step: Step size for sliding window (default: 2)
- learning_rate: Initial learning rate (default: 0.0001)
- batch_size: Mini-batch size (default: 32)
- hidden_size: LSTM hidden dimension (default: 32)

## Performance Metrics

The model is evaluated using:
- Root Mean Square Error (RMSE)
- Mean Absolute Error (MAE)
- R^2 Score

## Visualizations

The training process generates:
- Loss curves for training and validation
- Prediction vs. observation scatter plots
- Sample sequence predictions
C:\Users\speci\Documents\GitHub\RunoffNN\models\training_history.png
C:\Users\speci\Documents\GitHub\RunoffNN\models\model_evaluation.png
Restults are saved in the models directory

## Requirements

- Python 3.8+
- PyTorch 1.8+
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- TensorBoard
