import os
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, DataLoader
import time # For epoch timing

# Assuming BrainWaveNet.py is in the same directory or accessible in PYTHONPATH
from BrainWaveNet import BrainWaveNet 

def load_config(config_path='config.yaml'):
    """Loads the YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_preprocessed_data(config):
    """
    Loads preprocessed EEG data from the .npz file.
    Args:
        config (dict): The loaded configuration dictionary.
    Returns:
        tuple: (X_data, y_labels, participant_ids, sampling_rate)
               Returns (None, None, None, None) if file not found.
    """
    base_dir = config['data_acquisition']['experiment_base_dir']
    preprocessed_dir = os.path.join(base_dir, "preprocessed")
    data_path = os.path.join(preprocessed_dir, "preprocessed_eeg_data.npz")

    try:
        data = np.load(data_path, allow_pickle=True)
        X_data = data['X_data']
        y_labels = data['y_labels']
        participant_ids = data['participant_ids']
        sampling_rate = data['sampling_rate']
        print(f"Loaded preprocessed data from: {data_path}")
        return X_data, y_labels, participant_ids, sampling_rate
    except FileNotFoundError:
        print(f"Error: Preprocessed data file not found at {data_path}")
        return None, None, None, None

def build_model(config, num_classes):
    """
    Instantiates the BrainWaveNet model.
    Args:
        config (dict): The loaded configuration dictionary (model_training section).
        num_classes (int): The number of unique classes for the output layer.
    Returns:
        BrainWaveNet: The instantiated model.
    """
    mt_config = config['model_training']
    model = BrainWaveNet(
        input_channels=mt_config['input_channels'],
        num_filters=mt_config['num_filters'],
        lstm_hidden_size=mt_config['lstm_hidden_size'],
        num_classes=num_classes,
        dropout_p=mt_config['dropout_p']
    )
    return model

def main():
    config = load_config()
    if not config:
        print("Configuration could not be loaded. Exiting.")
        return

    # Load preprocessed data
    X_data, y_labels, participant_ids, sampling_rate = load_preprocessed_data(config)
    if X_data is None:
        print("Failed to load data. Exiting.")
        return

    print(f"Original X_data shape: {X_data.shape}, y_labels length: {len(y_labels)}")

    # Create checkpoints directory
    checkpoints_dir = os.path.join(config['data_acquisition']['experiment_base_dir'], "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    print(f"Checkpoints will be saved in: {checkpoints_dir}")

    # Label Encoding
    label_encoder = LabelEncoder()
    y_labels_numerical = label_encoder.fit_transform(y_labels)
    num_classes = len(label_encoder.classes_)
    print(f"Number of unique classes: {num_classes}")
    
    # Save LabelEncoder classes
    label_encoder_path = os.path.join(checkpoints_dir, "label_encoder_classes.npy")
    np.save(label_encoder_path, label_encoder.classes_)
    print(f"LabelEncoder classes saved to: {label_encoder_path}")


    # Data Splitting
    val_split = config['model_training']['validation_split']
    random_seed = config['general']['random_seed']
    
    if y_labels_numerical.ndim > 1:
         y_labels_for_split = y_labels_numerical.squeeze()
    else:
         y_labels_for_split = y_labels_numerical

    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X_data, y_labels_numerical, 
            test_size=val_split, 
            random_state=random_seed,
            stratify=y_labels_for_split
        )
    except ValueError as e:
        print(f"Error during train_test_split: {e}. Trying without stratification.")
        X_train, X_val, y_train, y_val = train_test_split(
            X_data, y_labels_numerical, 
            test_size=val_split, 
            random_state=random_seed
        )

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")

    # PyTorch DataLoaders
    batch_size = config['model_training']['batch_size']
    X_train_tensor = torch.from_numpy(X_train).float()
    y_train_tensor = torch.from_numpy(y_train).long()
    X_val_tensor = torch.from_numpy(X_val).float()
    y_val_tensor = torch.from_numpy(y_val).long()

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")

    # Model, Optimizer, Loss
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = build_model(config, num_classes)
    model.to(device)
    print("\nModel Architecture:")
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=config['model_training']['learning_rate'])
    criterion = nn.CrossEntropyLoss() # Renamed from loss_fn for clarity
    print(f"\nOptimizer: {optimizer}")
    print(f"Loss function: {criterion}")

    # Training Loop
    num_epochs = config['model_training']['num_epochs']
    best_val_acc = 0.0
    
    print(f"\nStarting training for {num_epochs} epochs...\n")

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += targets.size(0)
            correct_predictions += (predicted == targets).sum().item()

        epoch_loss = running_loss / total_samples
        epoch_acc = correct_predictions / total_samples
        
        # Validation Loop
        model.eval()
        val_running_loss = 0.0
        val_correct_predictions = 0
        val_total_samples = 0
        
        with torch.no_grad():
            for inputs_val, targets_val in val_loader:
                inputs_val, targets_val = inputs_val.to(device), targets_val.to(device)
                outputs_val = model(inputs_val)
                val_loss = criterion(outputs_val, targets_val)
                
                val_running_loss += val_loss.item() * inputs_val.size(0)
                _, predicted_val = torch.max(outputs_val.data, 1)
                val_total_samples += targets_val.size(0)
                val_correct_predictions += (predicted_val == targets_val).sum().item()
        
        epoch_val_loss = val_running_loss / val_total_samples
        epoch_val_acc = val_correct_predictions / val_total_samples
        epoch_duration = time.time() - epoch_start_time

        print(f"Epoch [{epoch+1}/{num_epochs}] - {epoch_duration:.2f}s - "
              f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f} - "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")

        # Model Checkpointing (Best Model)
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_model_path = os.path.join(checkpoints_dir, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"  New best model saved with Val Acc: {best_val_acc:.4f} to {best_model_path}")

    # Save Final Model
    final_model_path = os.path.join(checkpoints_dir, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"\nTraining complete. Final model saved to {final_model_path}")
    print(f"Best validation accuracy achieved: {best_val_acc:.4f}")

if __name__ == "__main__":
    main()
