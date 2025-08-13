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
    Prepare train/val/test DataLoaders from raw features and labels.

    Parameters:
    - X: numpy array, input features
    - y: numpy array, target labels
    - batch_size: int, batch size for DataLoader
    - train_ratio, val_ratio, test_ratio: float, must sum to 1, dataset split ratios
    - transform: callable or None, transformation to apply to dataset samples (optional)
    - dataset_class: custom Dataset class that wraps raw data and transform (optional)
                     If provided, it will be used to wrap subsets instead of TensorDataset.

    Returns:
    - train_loader, val_loader, test_loader: torch DataLoader objects
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
        Choose proper device to train and do inference
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs, device=None, verbose=True):
    """
    Train a PyTorch model with validation.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        optimizer (torch.optim.Optimizer): Optimizer.
        criterion (loss function): Loss function.
        num_epochs (int): Number of training epochs.
        train_size (int): Number of training samples (used for loss averaging).
        val_size (int): Number of validation samples.
        device (torch.device or None): Device to run training on (e.g., 'cuda' or 'cpu'). If None, use CPU.
        verbose (bool): Whether to print progress.

    Returns:
        train_losses (list): List of average training losses per epoch.
        val_losses (list): List of average validation losses per epoch.
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
    """Calculate MAPE, avoid division by zero."""
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
    Evaluate a regression model on a dataset loader, compute selected metrics.

    Parameters:
    - model: PyTorch model to evaluate
    - loader: DataLoader providing (features, labels)
    - device: torch.device, device to run model on (default: cpu if None)
    - show_examples: bool, whether to plot predicted vs actual scatter plot
    - metrics: list of str, which metrics to compute ('mse', 'rmse', 'mae', 'mape')
    - example_plot_title: str, title for the scatter plot
    - xlabel, ylabel: str, axis labels for the scatter plot

    Returns:
    - dict of computed metrics
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
    Evaluate a classification model on a dataset loader, print accuracy and classification report,
    optionally display example predictions.

    Args:
        model: PyTorch model to evaluate.
        loader: DataLoader providing (inputs, labels).
        device: torch.device, device to run model on (default: get_best_device() if None).
        show_examples: bool, whether to plot example predictions.
        class_names: list of class names (optional, for display).
        num_examples: int, number of examples to show.
        example_plot_title: str, title for example prediction plot.
        cmap: str, matplotlib colormap for images (default: 'gray').

    Returns:
        dict: {'accuracy': float, 'classification_report': str}
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
