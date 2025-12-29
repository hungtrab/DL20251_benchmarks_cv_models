#!/usr/bin/env python3
"""
Efficiency Benchmarking Module

Implements Section 2 of plan.md:
- Throughput (images/sec)
- Latency (inference time)
- Peak VRAM usage
- Model size metrics

Usage:
    from benchmark_efficiency import benchmark_efficiency
    results = benchmark_efficiency(model, test_loader, device='cuda')
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Any, Optional
from tqdm import tqdm


def get_model_size(model: nn.Module) -> Dict[str, Any]:
    """
    Calculate model size metrics.
    
    Returns:
        Dictionary with num_params, size_mb
    """
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate size in MB (assuming float32 = 4 bytes)
    size_bytes = num_params * 4
    size_mb = size_bytes / (1024 ** 2)
    
    return {
        'num_params': num_params,
        'num_trainable_params': num_trainable,
        'size_mb': size_mb,
        'size_bytes': size_bytes
    }


def measure_throughput(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    batch_size: int = 32,
    num_batches: int = 100
) -> Dict[str, float]:
    """
    Measure throughput (images/second) on GPU.
    
    As per plan.md Section 2.2:
    - Test with batch_size=32/64
    - Report images/sec
    
    Args:
        model: PyTorch model
        dataloader: Test dataloader
        device: Device to run on
        batch_size: Batch size for throughput test
        num_batches: Number of batches to measure
    
    Returns:
        Dictionary with throughput metrics
    """
    model = model.to(device)
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            if i >= 10:  # 10 warmup batches
                break
            images = images.to(device)
            _ = model(images)
    
    # Synchronize before measurement
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Measure throughput
    total_images = 0
    start_time = time.time()
    
    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            if i >= num_batches:
                break
            
            images = images.to(device)
            _ = model(images)
            
            total_images += images.size(0)
    
    # Synchronize after measurement
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    elapsed_time = time.time() - start_time
    throughput = total_images / elapsed_time
    
    return {
        'throughput_imgs_per_sec': throughput,
        'batch_size_used': batch_size,
        'total_images_processed': total_images,
        'elapsed_time_sec': elapsed_time
    }


def measure_latency(
    model: nn.Module,
    input_size: tuple,
    device: torch.device,
    num_iterations: int = 1000
) -> Dict[str, float]:
    """
    Measure inference latency (single image).
    
    As per plan.md Section 2.2:
    - Test with batch_size=1 (simulate real-world deployment)
    - Report average latency in milliseconds
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (C, H, W)
        device: Device to run on
        num_iterations: Number of iterations for measurement
    
    Returns:
        Dictionary with latency metrics
    """
    model = model.to(device)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, *input_size).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Synchronize before measurement
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Measure latency
    latencies = []
    
    with torch.no_grad():
        for _ in range(num_iterations):
            start_time = time.time()
            _ = model(dummy_input)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            latency = (time.time() - start_time) * 1000  # Convert to ms
            latencies.append(latency)
    
    latencies = np.array(latencies)
    
    return {
        'mean_latency_ms': float(np.mean(latencies)),
        'median_latency_ms': float(np.median(latencies)),
        'std_latency_ms': float(np.std(latencies)),
        'min_latency_ms': float(np.min(latencies)),
        'max_latency_ms': float(np.max(latencies)),
        'p95_latency_ms': float(np.percentile(latencies, 95)),
        'p99_latency_ms': float(np.percentile(latencies, 99)),
    }


def measure_peak_vram(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_batches: int = 10
) -> Dict[str, float]:
    """
    Measure peak VRAM usage during inference.
    
    As per plan.md Section 2.3:
    - Track peak GPU memory usage
    - Report in MB
    
    Args:
        model: PyTorch model
        dataloader: Test dataloader
        device: Device to run on
        num_batches: Number of batches to measure
    
    Returns:
        Dictionary with VRAM metrics
    """
    if device.type != 'cuda':
        return {
            'peak_vram_mb': 0.0,
            'note': 'VRAM measurement only available on CUDA devices'
        }
    
    model = model.to(device)
    model.eval()
    
    # Reset peak memory stats
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    
    # Measure peak memory during inference
    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            if i >= num_batches:
                break
            
            images = images.to(device)
            _ = model(images)
    
    # Get peak memory
    peak_memory_bytes = torch.cuda.max_memory_allocated(device)
    peak_memory_mb = peak_memory_bytes / (1024 ** 2)
    
    # Also get current allocated and reserved memory
    current_allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
    current_reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
    
    return {
        'peak_vram_mb': peak_memory_mb,
        'current_allocated_mb': current_allocated,
        'current_reserved_mb': current_reserved,
    }


def benchmark_efficiency(
    model: nn.Module,
    test_dataloader: DataLoader,
    device: torch.device,
    input_size: tuple = (3, 224, 224),
    batch_size: int = 32,
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run comprehensive efficiency benchmark as per plan.md Section 2.
    
    Measures:
    1. Throughput (images/sec)
    2. Latency (ms per image)
    3. Peak VRAM (MB)
    4. Model size (MB, num params)
    
    Args:
        model: PyTorch model to benchmark
        test_dataloader: Test dataloader
        device: Device to run benchmarks on
        input_size: Input tensor size (C, H, W)
        batch_size: Batch size for throughput test
        save_path: Optional path to save results
    
    Returns:
        Dictionary with all efficiency metrics
    """
    print(f'\n{"="*60}')
    print(f'EFFICIENCY BENCHMARK (Section 2)')
    print(f'{"="*60}')
    print(f'Device: {device}')
    print(f'Input size: {input_size}')
    print(f'Batch size: {batch_size}')
    
    results = {}
    
    # 1. Model Size
    print(f'\n{"-"*60}')
    print('1. Model Size Metrics')
    print(f'{"-"*60}')
    size_metrics = get_model_size(model)
    results['model_size'] = size_metrics
    print(f"  Parameters: {size_metrics['num_params']:,}")
    print(f"  Trainable Parameters: {size_metrics['num_trainable_params']:,}")
    print(f"  Model Size: {size_metrics['size_mb']:.2f} MB")
    
    # 2. Throughput
    print(f'\n{"-"*60}')
    print('2. Throughput Measurement')
    print(f'{"-"*60}')
    throughput_metrics = measure_throughput(model, test_dataloader, device, batch_size)
    results['throughput'] = throughput_metrics
    print(f"  Throughput: {throughput_metrics['throughput_imgs_per_sec']:.2f} images/sec")
    print(f"  Batch size: {throughput_metrics['batch_size_used']}")
    
    # 3. Latency (single image)
    print(f'\n{"-"*60}')
    print('3. Latency Measurement (Single Image)')
    print(f'{"-"*60}')
    latency_metrics = measure_latency(model, input_size, device)
    results['latency'] = latency_metrics
    print(f"  Mean Latency: {latency_metrics['mean_latency_ms']:.3f} ms")
    print(f"  Median Latency: {latency_metrics['median_latency_ms']:.3f} ms")
    print(f"  P95 Latency: {latency_metrics['p95_latency_ms']:.3f} ms")
    print(f"  P99 Latency: {latency_metrics['p99_latency_ms']:.3f} ms")
    
    # 4. Peak VRAM (GPU only)
    if device.type == 'cuda':
        print(f'\n{"-"*60}')
        print('4. Peak VRAM Usage')
        print(f'{"-"*60}')
        vram_metrics = measure_peak_vram(model, test_dataloader, device)
        results['vram'] = vram_metrics
        print(f"  Peak VRAM: {vram_metrics['peak_vram_mb']:.2f} MB")
        print(f"  Current Allocated: {vram_metrics['current_allocated_mb']:.2f} MB")
    else:
        results['vram'] = {'note': 'VRAM measurement only available on CUDA'}
        print('\n  Note: VRAM measurement skipped (CPU mode)')
    
    # Summary
    print(f'\n{"="*60}')
    print('EFFICIENCY SUMMARY')
    print(f'{"="*60}')
    print(f"Model Size: {size_metrics['size_mb']:.2f} MB ({size_metrics['num_params']:,} params)")
    print(f"Throughput: {throughput_metrics['throughput_imgs_per_sec']:.2f} images/sec")
    print(f"Latency: {latency_metrics['mean_latency_ms']:.3f} ms/image")
    if device.type == 'cuda':
        print(f"Peak VRAM: {vram_metrics['peak_vram_mb']:.2f} MB")
    
    # Save to file if path provided
    if save_path:
        import json
        results_path = os.path.join(save_path, 'efficiency_benchmark.json')
        os.makedirs(save_path, exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'\nResults saved to: {results_path}')
    
    print(f'{"="*60}\n')
    
    return results


# ===================== Command Line Interface =====================

if __name__ == "__main__":
    import argparse
    from model import create_model
    from torch.utils.data import DataLoader, TensorDataset
    
    parser = argparse.ArgumentParser(description='Benchmark model efficiency')
    parser.add_argument('--model', type=str, required=True, help='Model name or checkpoint path')
    parser.add_argument('--num_classes', type=int, default=100, help='Number of classes')
    parser.add_argument('--input_size', type=int, default=224, help='Input image size')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for throughput test')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to run on')
    parser.add_argument('--save_path', type=str, default='results', help='Path to save results')
    
    args = parser.parse_args()
    
    # Create model
    if os.path.exists(args.model):
        print(f"Loading model from checkpoint: {args.model}")
        checkpoint = torch.load(args.model, map_location='cpu')
        # Extract model from checkpoint if needed
        if 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
        else:
            model_state = checkpoint
        # This is simplified - you'd need to know the model architecture
        model = create_model(args.model, args.num_classes)
        model.load_state_dict(model_state)
    else:
        print(f"Creating model: {args.model}")
        model = create_model(args.model, args.num_classes)
    
    # Create dummy dataloader for testing
    dummy_data = torch.randn(1000, 3, args.input_size, args.input_size)
    dummy_labels = torch.randint(0, args.num_classes, (1000,))
    dummy_dataset = TensorDataset(dummy_data, dummy_labels)
    test_loader = DataLoader(dummy_dataset, batch_size=args.batch_size, shuffle=False)
    
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    
    # Run benchmark
    results = benchmark_efficiency(
        model=model,
        test_dataloader=test_loader,
        device=device,
        input_size=(3, args.input_size, args.input_size),
        batch_size=args.batch_size,
        save_path=args.save_path
    )
