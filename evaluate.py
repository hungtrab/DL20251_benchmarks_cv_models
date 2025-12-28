import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix, classification_report
import time
import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats


# ===================== Section 6.1: Robustness Benchmark =====================

def add_gaussian_noise(images: torch.Tensor, sigma: float = 0.15) -> torch.Tensor:
    """Add Gaussian noise to images."""
    noise = torch.randn_like(images) * sigma
    return torch.clamp(images + noise, 0, 1)


def add_salt_pepper_noise(images: torch.Tensor, density: float = 0.05) -> torch.Tensor:
    """Add salt and pepper noise to images."""
    noisy = images.clone()
    # Salt (white pixels)
    salt_mask = torch.rand_like(images) < (density / 2)
    noisy[salt_mask] = 1.0
    # Pepper (black pixels)
    pepper_mask = torch.rand_like(images) < (density / 2)
    noisy[pepper_mask] = 0.0
    return noisy


def add_gaussian_blur(images: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    """Apply Gaussian blur to images."""
    # Create Gaussian kernel
    sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
    x = torch.arange(kernel_size).float() - kernel_size // 2
    gauss = torch.exp(-x.pow(2) / (2 * sigma ** 2))
    kernel_1d = gauss / gauss.sum()
    
    # Create 2D kernel
    kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
    kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)
    
    # Apply to each channel
    batch_size, channels, h, w = images.shape
    blurred = []
    padding = kernel_size // 2
    
    for c in range(channels):
        channel = images[:, c:c+1, :, :]
        kernel = kernel_2d.to(images.device)
        blurred_channel = F.conv2d(channel, kernel, padding=padding)
        blurred.append(blurred_channel)
    
    return torch.cat(blurred, dim=1)


def evaluate_robustness(
    model: nn.Module,
    test_dataloader: DataLoader,
    device: torch.device,
    num_classes: int = 10
) -> Dict[str, Any]:
    """
    Evaluate model robustness against various noise types.
    
    As per plan.md Section 6.1:
    - Gaussian Noise (σ=0.15)
    - Salt & Pepper Noise (Density=0.05)
    - Gaussian Blur (Kernel=5)
    
    Returns:
        Dictionary with clean accuracy and accuracy under each noise type,
        plus delta accuracy (robustness measure).
    """
    model.eval()
    
    noise_types = {
        'clean': lambda x: x,
        'gaussian_noise': lambda x: add_gaussian_noise(x, sigma=0.15),
        'salt_pepper': lambda x: add_salt_pepper_noise(x, density=0.05),
        'gaussian_blur': lambda x: add_gaussian_blur(x, kernel_size=5),
    }
    
    results = {}
    
    for noise_name, noise_fn in noise_types.items():
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(test_dataloader, desc=f"Evaluating {noise_name}"):
                images = images.to(device)
                labels = labels.to(device)
                
                # Apply noise
                noisy_images = noise_fn(images)
                
                outputs = model(noisy_images)
                _, preds = torch.max(outputs, 1)
                
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        
        accuracy = 100 * correct / total
        results[f'{noise_name}_accuracy'] = accuracy
    
    # Calculate delta accuracy (robustness measure)
    clean_acc = results['clean_accuracy']
    for noise_name in ['gaussian_noise', 'salt_pepper', 'gaussian_blur']:
        delta = clean_acc - results[f'{noise_name}_accuracy']
        results[f'{noise_name}_delta'] = delta
    
    # Average robustness (lower is better)
    avg_delta = np.mean([
        results['gaussian_noise_delta'],
        results['salt_pepper_delta'],
        results['gaussian_blur_delta']
    ])
    results['avg_robustness_delta'] = avg_delta
    
    return results


# ===================== Section 6.2: Calibration Evaluation (ECE) =====================

def compute_ece(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    n_bins: int = 15
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute Expected Calibration Error (ECE).
    
    As per plan.md Section 6.2:
    - Divide confidence (0-1) into 15 bins
    - Compare average accuracy and average confidence in each bin
    
    Args:
        model: PyTorch model
        dataloader: Test dataloader
        device: Device to run on
        n_bins: Number of calibration bins (default: 15)
    
    Returns:
        Tuple of (ECE value, detailed bin statistics)
    """
    model.eval()
    
    all_confidences = []
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Computing ECE"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probabilities = F.softmax(outputs, dim=1)
            
            confidences, predictions = torch.max(probabilities, dim=1)
            
            all_confidences.extend(confidences.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_confidences = np.array(all_confidences)
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Compute ECE
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    bin_stats = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Samples in this bin
        in_bin = (all_confidences > bin_lower) & (all_confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = (all_predictions[in_bin] == all_labels[in_bin]).mean()
            avg_confidence_in_bin = all_confidences[in_bin].mean()
            
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            bin_stats.append({
                'bin_range': (bin_lower, bin_upper),
                'proportion': prop_in_bin,
                'accuracy': accuracy_in_bin,
                'confidence': avg_confidence_in_bin,
                'gap': avg_confidence_in_bin - accuracy_in_bin
            })
        else:
            bin_stats.append({
                'bin_range': (bin_lower, bin_upper),
                'proportion': 0,
                'accuracy': 0,
                'confidence': 0,
                'gap': 0
            })
    
    return ece, {'bin_stats': bin_stats, 'n_bins': n_bins}


def plot_reliability_diagram(
    bin_stats: List[Dict],
    save_path: str,
    ece_value: float
):
    """Plot reliability diagram showing calibration."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    accuracies = [b['accuracy'] for b in bin_stats if b['proportion'] > 0]
    confidences = [b['confidence'] for b in bin_stats if b['proportion'] > 0]
    proportions = [b['proportion'] for b in bin_stats if b['proportion'] > 0]
    
    # Plot bars
    bin_centers = [(b['bin_range'][0] + b['bin_range'][1]) / 2 for b in bin_stats if b['proportion'] > 0]
    width = 1.0 / len(bin_stats) * 0.8
    
    ax.bar(bin_centers, accuracies, width=width, alpha=0.8, color='steelblue', 
           edgecolor='black', label='Accuracy')
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    
    ax.set_xlabel('Confidence', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(f'Reliability Diagram (ECE = {ece_value:.4f})', fontsize=14)
    ax.legend(loc='upper left')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'reliability_diagram.png'), dpi=150)
    plt.close()


# ===================== Section 6.3: Statistical Significance Testing =====================

def bootstrap_confidence_interval(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95
) -> Dict[str, float]:
    """
    Compute bootstrap confidence interval for accuracy.
    
    As per plan.md Section 6.3:
    - Use Bootstrap Sampling (1000 times) to compute 95% CI for Accuracy
    
    Args:
        model: PyTorch model
        dataloader: Test dataloader
        device: Device to run on
        n_bootstrap: Number of bootstrap samples (default: 1000)
        confidence_level: Confidence level for interval (default: 0.95)
    
    Returns:
        Dictionary with mean, lower bound, upper bound of confidence interval
    """
    model.eval()
    
    # Collect all predictions
    all_correct = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Collecting predictions for bootstrap"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            correct = (preds == labels).cpu().numpy()
            all_correct.extend(correct)
    
    all_correct = np.array(all_correct)
    n_samples = len(all_correct)
    
    # Bootstrap sampling
    bootstrap_accuracies = []
    for _ in tqdm(range(n_bootstrap), desc="Bootstrap sampling"):
        # Sample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        bootstrap_acc = all_correct[indices].mean() * 100
        bootstrap_accuracies.append(bootstrap_acc)
    
    bootstrap_accuracies = np.array(bootstrap_accuracies)
    
    # Compute confidence interval
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower_bound = np.percentile(bootstrap_accuracies, lower_percentile)
    upper_bound = np.percentile(bootstrap_accuracies, upper_percentile)
    mean_accuracy = bootstrap_accuracies.mean()
    std_accuracy = bootstrap_accuracies.std()
    
    return {
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'confidence_level': confidence_level
    }


def mcnemar_test(
    model_a: nn.Module,
    model_b: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Dict[str, Any]:
    """
    Perform McNemar's test for pairwise model comparison.
    
    As per plan.md Section 6.3:
    - Compare pairs directly based on samples they predict correctly/incorrectly
    - Accept model is better if p-value < 0.05
    
    Args:
        model_a: First model
        model_b: Second model
        dataloader: Test dataloader
        device: Device to run on
    
    Returns:
        Dictionary with test statistic, p-value, and conclusion
    """
    model_a.eval()
    model_b.eval()
    
    # Collect predictions
    a_correct = []
    b_correct = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Collecting predictions for McNemar"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs_a = model_a(images)
            outputs_b = model_b(images)
            
            _, preds_a = torch.max(outputs_a, 1)
            _, preds_b = torch.max(outputs_b, 1)
            
            a_correct.extend((preds_a == labels).cpu().numpy())
            b_correct.extend((preds_b == labels).cpu().numpy())
    
    a_correct = np.array(a_correct)
    b_correct = np.array(b_correct)
    
    return _mcnemar_test_from_predictions(a_correct, b_correct)


def _mcnemar_test_from_predictions(
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    labels: np.ndarray
) -> Dict[str, Any]:
    """
    Internal helper: Perform McNemar's test from prediction arrays.
    
    Args:
        preds_a: Predictions from model A
        preds_b: Predictions from model B
        labels: Ground truth labels
    
    Returns:
        Dictionary with test statistic, p-value, significance, and effect size
    """
    # Determine correct/incorrect for each model
    a_correct = (preds_a == labels)
    b_correct = (preds_b == labels)
    
    # Build contingency table
    # n_01: A wrong, B correct
    # n_10: A correct, B wrong
    n_01 = np.sum((~a_correct) & b_correct)
    n_10 = np.sum(a_correct & (~b_correct))
    
    # McNemar's test (with continuity correction)
    if n_01 + n_10 == 0:
        return {
            'chi_squared': 0.0,
            'p_value': 1.0,
            'n_01': int(n_01),
            'n_10': int(n_10),
            'is_significant': False,
            'effect_size': 0.0
        }
    
    chi_squared = ((abs(n_01 - n_10) - 1) ** 2) / (n_01 + n_10)
    p_value = 1 - stats.chi2.cdf(chi_squared, df=1)
    
    # Effect size (Cohen's g)
    effect_size = (n_01 - n_10) / np.sqrt(n_01 + n_10)
    
    return {
        'chi_squared': float(chi_squared),
        'p_value': float(p_value),
        'n_01': int(n_01),
        'n_10': int(n_10),
        'is_significant': p_value < 0.05,
        'effect_size': float(effect_size)
    }


# ===================== Main Evaluation Function =====================

def evaluate_model(model, test_dataloader, num_class=10, save_path='confusion_matrix.png',
                   compute_robustness: bool = False,
                   compute_calibration: bool = False,
                   compute_bootstrap_ci: bool = False,
                   n_bootstrap: int = 1000,
                   n_calibration_bins: int = 15):
    """
    Evaluate a model on test data with various metrics including top-1 and top-5 error rates.
    
    Optionally includes advanced evaluation metrics as per plan.md Section 6:
    - Robustness testing (noise injection)
    - Calibration (ECE)
    - Statistical significance (Bootstrap CI)
    
    Args:
        model: The PyTorch model to evaluate
        test_dataloader: DataLoader for the test dataset
        num_class: Number of classes in the dataset
        save_path: Path to save evaluation results
        compute_robustness: Whether to compute robustness metrics (Section 6.1)
        compute_calibration: Whether to compute ECE and reliability diagram (Section 6.2)
        compute_bootstrap_ci: Whether to compute bootstrap confidence interval (Section 6.3)
        n_bootstrap: Number of bootstrap samples (default: 1000)
        n_calibration_bins: Number of bins for ECE calculation (default: 15)
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    model.eval()
    
    correct = 0
    total = 0
    top5_correct = 0
    all_preds = []
    all_labels = []
    
    start_time = time.time()
    with torch.no_grad():
        for images, labels in tqdm(test_dataloader, desc="Evaluating", unit="batch"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            
            # Top-1 accuracy calculation
            _, preds = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            
            # Top-5 accuracy calculation (only if num_class > 5)
            if num_class > 5:
                _, top5_preds = outputs.topk(5, 1, largest=True, sorted=True)
                top5_preds = top5_preds.t()
                top5_correct_batch = top5_preds.eq(labels.view(1, -1).expand_as(top5_preds))
                top5_correct += top5_correct_batch.sum().item()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    eval_time = time.time() - start_time
    
    # Calculate metrics
    top1_accuracy = 100 * correct / total
    top1_error = 100 - top1_accuracy
    
    if num_class > 5:
        top5_accuracy = 100 * top5_correct / total
        top5_error = 100 - top5_accuracy
    else:
        top5_accuracy = None
        top5_error = None
    
    print(f'\n{"="*60}')
    print(f'EVALUATION RESULTS')
    print(f'{"="*60}')
    print(f'Top-1 Accuracy: {top1_accuracy:.2f}%')
    print(f'Top-1 Error: {top1_error:.2f}%')
    
    if num_class > 5:
        print(f'Top-5 Accuracy: {top5_accuracy:.2f}%')
        print(f'Top-5 Error: {top5_error:.2f}%')
    
    print(f'Time taken for evaluation: {eval_time:.2f} seconds')
    
    # Confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    # Classification report
    class_report = classification_report(all_labels, all_preds, target_names=[str(i) for i in range(num_class)])
    
    print(class_report)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[str(i) for i in range(num_class)], 
                yticklabels=[str(i) for i in range(num_class)])
    
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'confusion_matrix.png'))
    plt.close()
    
    # Compile results
    results = {
        'top1_accuracy': top1_accuracy,
        'top1_error': top1_error,
        'top5_accuracy': top5_accuracy if num_class > 5 else None,
        'top5_error': top5_error if num_class > 5 else None,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report,
        'eval_time': eval_time
    }
    
    # ========== Section 6.1: Robustness Benchmark ==========
    if compute_robustness:
        print(f'\n{"="*60}')
        print(f'ROBUSTNESS EVALUATION (Section 6.1)')
        print(f'{"="*60}')
        
        robustness_results = evaluate_robustness(model, test_dataloader, device, num_class)
        results['robustness'] = robustness_results
        
        print(f"Clean Accuracy: {robustness_results['clean_accuracy']:.2f}%")
        print(f"Gaussian Noise Accuracy: {robustness_results['gaussian_noise_accuracy']:.2f}% (Δ = {robustness_results['gaussian_noise_delta']:.2f}%)")
        print(f"Salt & Pepper Accuracy: {robustness_results['salt_pepper_accuracy']:.2f}% (Δ = {robustness_results['salt_pepper_delta']:.2f}%)")
        print(f"Gaussian Blur Accuracy: {robustness_results['gaussian_blur_accuracy']:.2f}% (Δ = {robustness_results['gaussian_blur_delta']:.2f}%)")
        print(f"Average Robustness Delta: {robustness_results['avg_robustness_delta']:.2f}% (lower is better)")
    
    # ========== Section 6.2: Calibration Evaluation ==========
    if compute_calibration:
        print(f'\n{"="*60}')
        print(f'CALIBRATION EVALUATION (Section 6.2)')
        print(f'{"="*60}')
        
        ece_value, calibration_details = compute_ece(model, test_dataloader, device, n_calibration_bins)
        results['ece'] = ece_value
        results['calibration_details'] = calibration_details
        
        print(f"Expected Calibration Error (ECE): {ece_value:.4f}")
        print(f"Number of bins: {n_calibration_bins}")
        
        # Plot reliability diagram
        plot_reliability_diagram(calibration_details['bin_stats'], save_path, ece_value)
        print(f"Reliability diagram saved to: {os.path.join(save_path, 'reliability_diagram.png')}")
    
    # ========== Section 6.3: Statistical Significance ==========
    if compute_bootstrap_ci:
        print(f'\n{"="*60}')
        print(f'STATISTICAL SIGNIFICANCE (Section 6.3)')
        print(f'{"="*60}')
        
        bootstrap_results = bootstrap_confidence_interval(
            model, test_dataloader, device, 
            n_bootstrap=n_bootstrap, 
            confidence_level=0.95
        )
        results['bootstrap_ci'] = bootstrap_results
        
        print(f"Bootstrap Mean Accuracy: {bootstrap_results['mean_accuracy']:.2f}%")
        print(f"Standard Deviation: {bootstrap_results['std_accuracy']:.2f}%")
        print(f"95% Confidence Interval: [{bootstrap_results['lower_bound']:.2f}%, {bootstrap_results['upper_bound']:.2f}%]")
    
    print(f'\n{"="*60}')
    print(f'Evaluation complete!')
    print(f'{"="*60}\n')
    
    return results


def full_benchmark_evaluation(
    model: nn.Module,
    test_dataloader: DataLoader,
    num_class: int = 10,
    save_path: str = 'results',
    n_bootstrap: int = 1000,
    n_calibration_bins: int = 15
) -> Dict[str, Any]:
    """
    Run full benchmark evaluation including all metrics from plan.md Section 6.
    
    This is a convenience function that enables all evaluation features.
    
    Args:
        model: PyTorch model to evaluate
        test_dataloader: Test dataloader
        num_class: Number of classes
        save_path: Path to save results
        n_bootstrap: Number of bootstrap samples
        n_calibration_bins: Number of calibration bins
    
    Returns:
        Complete evaluation results dictionary
    """
    return evaluate_model(
        model=model,
        test_dataloader=test_dataloader,
        num_class=num_class,
        save_path=save_path,
        compute_robustness=True,
        compute_calibration=True,
        compute_bootstrap_ci=True,
        n_bootstrap=n_bootstrap,
        n_calibration_bins=n_calibration_bins
    )


def pairwise_model_comparison(
    models: Dict[str, nn.Module],
    test_dataloader: DataLoader,
    save_path: str = 'results',
    significance_level: float = 0.05
) -> pd.DataFrame:
    """
    Perform pairwise statistical comparison of multiple models using McNemar's test.
    
    As per plan.md Section 6.3 - Statistical Significance.
    
    Args:
        models: Dictionary of {model_name: model_instance}
        test_dataloader: Test dataloader
        save_path: Path to save comparison results
        significance_level: Alpha level for statistical significance (default: 0.05)
    
    Returns:
        DataFrame with pairwise comparison results
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_names = list(models.keys())
    n_models = len(model_names)
    
    if n_models < 2:
        raise ValueError("Need at least 2 models for pairwise comparison")
    
    print(f'\n{"="*60}')
    print(f'PAIRWISE MODEL COMPARISON (McNemar\'s Test)')
    print(f'{"="*60}')
    print(f"Comparing {n_models} models: {model_names}")
    print(f"Significance level: α = {significance_level}")
    
    # Collect predictions from all models
    model_predictions = {}
    
    for name, model in models.items():
        model = model.to(device)
        model.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(test_dataloader, desc=f"Evaluating {name}", unit="batch"):
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                _, preds = torch.max(outputs.data, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        model_predictions[name] = {
            'predictions': np.array(all_preds),
            'labels': np.array(all_labels),
            'accuracy': 100 * np.mean(np.array(all_preds) == np.array(all_labels))
        }
        print(f"  {name}: {model_predictions[name]['accuracy']:.2f}% accuracy")
    
    # Perform pairwise McNemar's test
    comparison_results = []
    
    for i in range(n_models):
        for j in range(i + 1, n_models):
            name_a = model_names[i]
            name_b = model_names[j]
            
            preds_a = model_predictions[name_a]['predictions']
            preds_b = model_predictions[name_b]['predictions']
            labels = model_predictions[name_a]['labels']
            
            result = _mcnemar_test_from_predictions(preds_a, preds_b, labels)
            
            acc_diff = model_predictions[name_a]['accuracy'] - model_predictions[name_b]['accuracy']
            winner = name_a if acc_diff > 0 else (name_b if acc_diff < 0 else "Tie")
            
            comparison_results.append({
                'Model A': name_a,
                'Model B': name_b,
                'Accuracy A (%)': model_predictions[name_a]['accuracy'],
                'Accuracy B (%)': model_predictions[name_b]['accuracy'],
                'Accuracy Diff (%)': abs(acc_diff),
                'Higher Performer': winner,
                'Chi-squared': result['chi_squared'],
                'p-value': result['p_value'],
                'Significant': result['is_significant'],
                'Effect Size': result['effect_size']
            })
    
    # Create DataFrame
    df = pd.DataFrame(comparison_results)
    
    # Print results
    print(f'\n{"="*60}')
    print("Pairwise Comparison Results:")
    print(f'{"="*60}')
    
    for _, row in df.iterrows():
        sig_marker = "***" if row['Significant'] else ""
        print(f"\n{row['Model A']} vs {row['Model B']}:")
        print(f"  Accuracy: {row['Accuracy A (%)']:.2f}% vs {row['Accuracy B (%)']:.2f}%")
        print(f"  Difference: {row['Accuracy Diff (%)']:.2f}% (in favor of {row['Higher Performer']})")
        print(f"  McNemar χ²: {row['Chi-squared']:.4f}, p-value: {row['p-value']:.4f} {sig_marker}")
        print(f"  Effect size (Cohen's g): {row['Effect Size']:.4f}")
        if row['Significant']:
            print(f"  → STATISTICALLY SIGNIFICANT (p < {significance_level})")
    
    # Save results
    csv_path = os.path.join(save_path, 'pairwise_comparison.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    return df


# ===================== Command Line Interface =====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate model with comprehensive metrics')
    parser.add_argument('--model', type=str, required=True, help='Model name or path to checkpoint')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--save_path', type=str, default='results', help='Path to save results')
    parser.add_argument('--robustness', action='store_true', help='Compute robustness metrics')
    parser.add_argument('--calibration', action='store_true', help='Compute calibration metrics')
    parser.add_argument('--bootstrap', action='store_true', help='Compute bootstrap CI')
    parser.add_argument('--full', action='store_true', help='Run full benchmark evaluation')
    
    args = parser.parse_args()
    
    print(f"Evaluation script initialized with args: {args}")
    print("Use the evaluate_model() or full_benchmark_evaluation() functions in Python code.")