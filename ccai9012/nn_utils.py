"""
Neural Network Utilities Module
===============================

This module provides a comprehensive set of utilities for building, training, and evaluating 
neural network models using PyTorch. It includes functions for data preprocessing, model training,
and performance evaluation for both regression and classification tasks.

Main components:
- Data preparation: Functions to prepare DataLoaders with proper train/val/test splits
- Device management: Automatic detection of optimal compute device (CPU/CUDA/MPS)
- Model training: Streamlined training loop with validation and progress tracking
- Model evaluation: Comprehensive metrics and visualization for regression and classification tasks

This module aims to simplify common PyTorch workflows and provide consistent interfaces for
neural network experimentation and evaluation.

Usage:
    ### Prepare data
    train_loader, val_loader, test_loader, scaler = prepare_dataloaders(X, y)
    
    ### Create and train model
    model = YourNeuralNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    train_losses, val_losses = train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs=100)
    
    ### Evaluate model
    metrics = evaluate_regression_model(model, test_loader)
"""


import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.metrics import classification_report


def prepare_dataloaders(
    X,
    y,
    batch_size=64,
    train_ratio=0.7,
    val_ratio=0.15
):
    """
    Prepare train/validation/test DataLoaders from raw features and labels.

    This function standardizes features, creates PyTorch tensors, splits the dataset
    into training, validation, and test sets, and returns DataLoader objects for each set.
    It handles the entire data preparation pipeline for neural network training.

    Parameters:
        X (numpy.ndarray): Input features matrix with shape (n_samples, n_features).
        y (numpy.ndarray): Target labels with shape (n_samples,) or (n_samples, n_targets).
        batch_size (int, optional): Number of samples per batch. Defaults to 64.
        train_ratio (float, optional): Proportion of data to use for training. Defaults to 0.7.
        val_ratio (float, optional): Proportion of data to use for validation. Defaults to 0.15.
                                     The remaining proportion (1 - train_ratio - val_ratio)
                                     will be used for testing.

    Returns:
        tuple: A tuple containing:
            - train_loader (DataLoader): DataLoader for the training set.
            - val_loader (DataLoader): DataLoader for the validation set.
            - test_loader (DataLoader): DataLoader for the test set.
            - scaler (StandardScaler): Fitted scaler used to standardize the features.

    Example:
        >>> train_loader, val_loader, test_loader, scaler = prepare_dataloaders(X, y)
        >>> # Use the loaders for model training and evaluation
        >>> for features, targets in train_loader:
        >>>     # Training loop code
    """

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Convert to tensors
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    # Create full dataset
    full_dataset = TensorDataset(X_tensor, y_tensor)

    # Calculate split sizes
    total_len = len(full_dataset)
    train_size = int(train_ratio * total_len)
    val_size = int(val_ratio * total_len)
    test_size = total_len - train_size - val_size

    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    # If custom dataset_class provided, usually transform applies inside dataset, no need to wrap again.
    # But if you want to wrap splits again with transform, you can do it here.

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader, scaler

def get_best_device():
    """
    Determines the best available computational device for PyTorch operations.

    This function checks for the availability of CUDA (NVIDIA GPUs), MPS (Apple Silicon),
    or falls back to CPU. Using the appropriate device can significantly speed up
    neural network training and inference.

    Returns:
        torch.device: The best available device in the following priority order:
                     1. CUDA (if NVIDIA GPU is available)
                     2. MPS (if Apple Silicon GPU is available)
                     3. CPU (as fallback)

    Example:
        device = get_best_device()
        model.to(device)  # Move model to optimal device
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs, device=None, verbose=True):
    """
    Train a PyTorch neural network model with validation.

    This function implements a complete training loop for neural networks, including:
    - Moving the model and data to the appropriate device (GPU/CPU)
    - Forward and backward passes
    - Optimization steps
    - Loss tracking for both training and validation sets
    - Progress reporting

    Parameters:
        model (torch.nn.Module): The PyTorch model to train.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        val_loader (torch.utils.data.DataLoader): DataLoader for validation data.
        optimizer (torch.optim.Optimizer): Optimizer for updating model weights.
        criterion (callable): Loss function to minimize.
        num_epochs (int): Number of complete passes through the training dataset.
        device (torch.device, optional): Device to run the training on. If None,
                                        the best available device is automatically selected.
        verbose (bool, optional): Whether to print progress updates. Defaults to True.

    Returns:
        tuple: A tuple containing:
            - train_losses (list): Average training loss for each epoch.
            - val_losses (list): Average validation loss for each epoch.

    Example:
        >>> model = MyNeuralNetwork()
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        >>> criterion = torch.nn.MSELoss()
        >>> train_losses, val_losses = train_model(
        >>>     model, train_loader, val_loader,
        >>>     optimizer, criterion, num_epochs=50
        >>> )
        >>> # Plot learning curves
        >>> plt.plot(train_losses, label='Training Loss')
        >>> plt.plot(val_losses, label='Validation Loss')
    """
    if device is None:
        device = get_best_device()
    model.to(device)

    train_losses = []
    val_losses = []

    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * features.size(0)

        train_loss = running_loss / train_size
        train_losses.append(train_loss)

        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for val_features, val_targets in val_loader:
                val_features = val_features.to(device)
                val_targets = val_targets.to(device)
                val_outputs = model(val_features)
                val_loss = criterion(val_outputs, val_targets)
                val_running_loss += val_loss.item() * val_features.size(0)

        val_loss_epoch = val_running_loss / val_size
        val_losses.append(val_loss_epoch)

        if verbose:
            print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss_epoch:.4f}")

    return train_losses, val_losses

def mean_absolute_percentage_error(y_true, y_pred):
    """
    Calculate Mean Absolute Percentage Error (MAPE) while avoiding division by zero.

    MAPE measures the size of error in percentage terms, providing an intuitive measure
    of prediction error for regression tasks. This implementation handles zero values
    in the true labels by excluding them from the calculation.

    Parameters:
        y_true (numpy.ndarray or list): True target values.
        y_pred (numpy.ndarray or list): Predicted target values.

    Returns:
        float: MAPE value as a percentage (not as a fraction). Lower values indicate
              better model performance.

    Example:
        >>> y_true = [10, 20, 30, 0, 40]  # Note the zero value
        >>> y_pred = [11, 18, 33, 1, 38]
        >>> error = mean_absolute_percentage_error(y_true, y_pred)
        >>> print(f"MAPE: {error:.2f}%")
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_idx = y_true != 0
    return np.mean(np.abs((y_true[non_zero_idx] - y_pred[non_zero_idx]) / y_true[non_zero_idx])) * 100

def evaluate_regression_model(
    model,
    loader,
    device=None,
    show_examples=True,
    metrics=['rmse', 'mse', 'mae', 'mape'],
    title="Predicted vs Actual on Test Set",
    xlabel="Actual Values",
    ylabel="Predicted Values"
):
    """
    Evaluate a regression model on a dataset loader and compute selected metrics.

    This function performs a comprehensive evaluation of regression models by:
    - Computing standard regression metrics (MSE, RMSE, MAE, MAPE, R²)
    - Printing performance statistics
    - Optionally visualizing predictions against actual values

    Parameters:
        model (torch.nn.Module): The PyTorch model to evaluate.
        loader (torch.utils.data.DataLoader): DataLoader providing (features, labels).
        device (torch.device, optional): Device to run evaluation on. If None,
                                       the best available device is automatically selected.
        show_examples (bool, optional): Whether to plot a scatter plot of predictions
                                      vs actual values. Defaults to True.
        metrics (list, optional): List of metrics to compute. Options include 'mse', 'rmse',
                                'mae', and 'mape'. Defaults to all of them.
        title (str, optional): Title for the scatter plot. Defaults to "Predicted vs Actual on Test Set".
        xlabel (str, optional): X-axis label for the scatter plot. Defaults to "Actual Values".
        ylabel (str, optional): Y-axis label for the scatter plot. Defaults to "Predicted Values".

    Returns:
        dict: A dictionary containing computed metrics. Always includes 'r2' (R² score),
              and includes other metrics as specified in the 'metrics' parameter.

    Example:
        >>> model = MyRegressionModel()
        >>> results = evaluate_regression_model(model, test_loader)
        >>> print(f"R² Score: {results['r2']:.4f}")
        >>> print(f"RMSE: {results['rmse']:.4f}")
    """

    model.eval()
    if device is None:
        device = get_best_device()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            outputs = model(features)
            all_preds.append(outputs.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    results = {}

    if 'mse' in metrics:
        results['mse'] = mean_squared_error(all_labels, all_preds)
        print(f"MSE: {results['mse']:.4f}")
    if 'rmse' in metrics:
        results['rmse'] = root_mean_squared_error(all_labels, all_preds)
        print(f"RMSE: {results['rmse']:.4f}")
    if 'mae' in metrics:
        results['mae'] = mean_absolute_error(all_labels, all_preds)
        print(f"MAE: {results['mae']:.4f}")
    if 'mape' in metrics:
        results['mape'] = mean_absolute_percentage_error(all_labels, all_preds)
        print(f"MAPE: {results['mape']:.2f}%")

    r2 = r2_score(all_labels, all_preds)
    results['r2'] = r2
    print(f"R^2 Score: {r2:.4f}")

    if show_examples:
        plt.figure(figsize=(6,6))
        plt.scatter(all_labels, all_preds, alpha=0.5)
        min_val, max_val = all_labels.min(), all_labels.max()
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.show()

    return results

def evaluate_classification_model(
    model,
    loader,
    device=None,
    show_examples=True,
    class_names=None,
    num_examples=10,
    example_plot_title="Prediction Examples",
    cmap="gray"
):
    """
    Evaluate a classification model on a dataset and visualize predictions.

    This function performs a comprehensive evaluation of classification models by:
    - Computing accuracy and generating a detailed classification report
    - Printing performance statistics including precision, recall, and f1-score
    - Optionally visualizing example predictions with their true labels

    Parameters:
        model (torch.nn.Module): The PyTorch model to evaluate.
        loader (torch.utils.data.DataLoader): DataLoader providing (inputs, labels).
        device (torch.device, optional): Device to run evaluation on. If None,
                                        the best available device is automatically selected.
        show_examples (bool, optional): Whether to display example predictions. Defaults to True.
        class_names (list, optional): List of class names for display purposes. Required if
                                     show_examples=True.
        num_examples (int, optional): Number of example predictions to show. Defaults to 10.
        example_plot_title (str, optional): Title for the examples plot. Defaults to "Prediction Examples".
        cmap (str, optional): Colormap for image display. Defaults to "gray".

    Returns:
        dict: A dictionary containing:
            - 'accuracy': Classification accuracy as a percentage.
            - 'classification_report': Detailed classification report as a string,
                                      including precision, recall, and f1-score.

    Example:
        >>> model = MyClassificationModel()
        >>> class_names = ['cat', 'dog', 'bird']
        >>> results = evaluate_classification_model(model, test_loader, class_names=class_names)
        >>> print(f"Accuracy: {results['accuracy']:.2f}%")
    """
    if device is None:
        device = get_best_device()

    model.eval()
    all_preds = []
    all_labels = []
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy:.2f}%")
    report = classification_report(all_labels, all_preds)
    print("\nClassification Report:\n", report)

    results = {
        'accuracy': accuracy,
        'classification_report': report
    }

    if show_examples and class_names is not None:
        import math
        import matplotlib.pyplot as plt
        num_cols = min(num_examples, 5)
        num_rows = math.ceil(num_examples / num_cols)
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(3*num_cols, 3*num_rows))
        axs = axs.flatten() if num_examples > 1 else [axs]

        data_iter = iter(loader)
        inputs, labels = next(data_iter)
        outputs = model(inputs.to(device))
        _, preds = torch.max(outputs, 1)

        for i in range(num_examples):
            img = inputs[i].cpu()
            label = labels[i].item()
            pred = preds[i].cpu().item()

            ax = axs[i]
            # If image has channel dim at front, e.g. (C,H,W), convert to (H,W,C)
            if img.ndim == 3 and img.shape[0] in [1,3]:
                img_disp = img.permute(1, 2, 0)
                # If grayscale (1 channel), squeeze last dim
                if img_disp.shape[2] == 1:
                    img_disp = img_disp.squeeze(2)
            else:
                img_disp = img

            ax.imshow(img_disp, cmap=cmap)
            ax.set_title(f"P: {class_names[pred]}\nT: {class_names[label]}")
            ax.axis('off')

        for j in range(num_examples, len(axs)):
            axs[j].axis('off')

        plt.suptitle(example_plot_title, fontsize=16)
        plt.tight_layout()
        plt.show()

    return results
