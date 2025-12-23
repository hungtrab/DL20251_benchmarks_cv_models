import os
import re
import csv
import argparse
from pathlib import Path
from datetime import datetime


def parse_evaluation_file(txt_path):
    """
    Parse evaluation_results.txt file to extract metrics.
    
    Returns:
        dict: Dictionary containing extracted metrics or None if parsing fails
    """
    try:
        with open(txt_path, 'r') as f:
            content = f.read()
        
        # Extract configuration
        dataset_match = re.search(r'Dataset:\s+(\S+)', content)
        model_match = re.search(r'Model:\s+(\S+)', content)
        timestamp_match = re.search(r'Timestamp:\s+(\S+)', content)
        
        if not all([dataset_match, model_match, timestamp_match]):
            print(f"⚠️  Could not parse configuration from {txt_path}")
            return None
        
        dataset = dataset_match.group(1)
        model_name = model_match.group(1)
        timestamp = timestamp_match.group(1)
        
        # Extract Top-1 and Top-5 accuracy
        top1_acc_match = re.search(r'Top-1 Accuracy:\s+([\d.]+)%', content)
        top5_acc_match = re.search(r'Top-5 Accuracy:\s+([\d.]+)%', content)
        
        if not top1_acc_match:
            print(f"⚠️  Could not parse Top-1 accuracy from {txt_path}")
            return None
        
        top1_accuracy = float(top1_acc_match.group(1))
        top5_accuracy = float(top5_acc_match.group(1)) if top5_acc_match else None
        
        # Extract weighted average metrics from classification report
        # Look for the weighted avg line
        weighted_avg_pattern = r'weighted avg\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+[\d]+'
        weighted_match = re.search(weighted_avg_pattern, content)
        
        if not weighted_match:
            print(f"⚠️  Could not parse weighted avg metrics from {txt_path}")
            return None
        
        precision = float(weighted_match.group(1))
        recall = float(weighted_match.group(2))
        f1_score = float(weighted_match.group(3))
        
        return {
            'dataset': dataset,
            'model_name': model_name,
            'timestamp': timestamp,
            'f1_score': f1_score,
            'precision': precision,
            'recall': recall,
            'top1_accuracy': top1_accuracy,
            'top5_accuracy': top5_accuracy
        }
        
    except Exception as e:
        print(f"❌ Error parsing {txt_path}: {str(e)}")
        return None


def collect_statistics(results_dir):
    """
    Collect statistics from all evaluation_results.txt files in results directory.
    
    Returns:
        list: List of dictionaries containing metrics from each evaluation
    """
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return []
    
    statistics = []
    processed = 0
    skipped = 0
    
    # Iterate through all subdirectories
    for run_dir in sorted(results_path.iterdir()):
        if not run_dir.is_dir():
            continue
        
        # Look for evaluation_results.txt
        txt_file = run_dir / 'evaluation_results.txt'
        if not txt_file.exists():
            skipped += 1
            continue
        
        print(f"📄 Processing: {run_dir.name}/evaluation_results.txt")
        metrics = parse_evaluation_file(txt_file)
        
        if metrics is not None:
            statistics.append(metrics)
            processed += 1
            print(f"   ✅ Extracted metrics for {metrics['dataset']}_{metrics['model_name']}")
        else:
            skipped += 1
    
    print(f"\n📊 Statistics Collection Summary:")
    print(f"   Processed: {processed}")
    print(f"   Skipped: {skipped}")
    
    return statistics


def save_to_csv(statistics, output_path):
    """
    Save collected statistics to CSV file.
    
    Args:
        statistics: List of dictionaries containing metrics
        output_path: Path to output CSV file
    """
    if not statistics:
        print("⚠️  No statistics to save")
        return
    
    # Define CSV columns
    fieldnames = [
        'dataset',
        'model_name',
        'timestamp',
        'f1_score',
        'precision',
        'recall',
        'top1_accuracy',
        'top5_accuracy'
    ]
    
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for stat in statistics:
            writer.writerow(stat)
    
    print(f"\n✅ Statistics saved to: {output_path}")
    print(f"   Total entries: {len(statistics)}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect statistics from evaluation results and generate CSV report"
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        default='results',
        help='Path to results directory (default: results)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV file path (default: results/statistics_TIMESTAMP.csv)'
    )
    args = parser.parse_args()
    
    print(f"🔍 Collecting statistics from: {args.results_dir}\n")
    
    # Collect statistics
    statistics = collect_statistics(args.results_dir)
    
    if not statistics:
        print("❌ No statistics collected. Exiting.")
        return
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(args.results_dir, f'statistics_{timestamp}.csv')
    
    # Save to CSV
    save_to_csv(statistics, output_path)
    
    # Print summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    
    # Group by dataset
    datasets = {}
    for stat in statistics:
        dataset = stat['dataset']
        if dataset not in datasets:
            datasets[dataset] = []
        datasets[dataset].append(stat)
    
    for dataset, stats in sorted(datasets.items()):
        print(f"\n{dataset.upper()}:")
        print(f"  Models evaluated: {len(stats)}")
        avg_top1 = sum(s['top1_accuracy'] for s in stats) / len(stats)
        print(f"  Average Top-1 Accuracy: {avg_top1:.2f}%")
        if stats[0]['top5_accuracy'] is not None:
            avg_top5 = sum(s['top5_accuracy'] for s in stats if s['top5_accuracy'] is not None) / len([s for s in stats if s['top5_accuracy'] is not None])
            print(f"  Average Top-5 Accuracy: {avg_top5:.2f}%")
        avg_f1 = sum(s['f1_score'] for s in stats) / len(stats)
        print(f"  Average F1-Score: {avg_f1:.4f}")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
