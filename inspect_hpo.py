#!/usr/bin/env python3
"""
Inspect and analyze Optuna HPO results from SQLite database.

Usage:
    python inspect_hpo.py --storage hpo_results.db
    python inspect_hpo.py --storage hpo_results.db --study hpo_study_legacy
    python inspect_hpo.py --storage hpo_results.db --top 10
"""

import argparse
import sys
from pathlib import Path

try:
    import optuna
    import pandas as pd
except ImportError:
    print("Error: Required packages not installed.")
    print("Run: pip install optuna pandas")
    sys.exit(1)


def list_studies(storage_url: str):
    """List all studies in the database."""
    print(f"\n{'='*60}")
    print(f"Studies in database: {storage_url}")
    print(f"{'='*60}")
    
    study_summaries = optuna.get_all_study_summaries(storage=storage_url)
    
    if not study_summaries:
        print("No studies found in database.")
        return
    
    for i, summary in enumerate(study_summaries, 1):
        print(f"\n{i}. Study: {summary.study_name}")
        print(f"   Number of trials: {summary.n_trials}")
        print(f"   Best value: {summary.best_trial.value if summary.best_trial else 'N/A'}")
        print(f"   Direction: {summary.direction}")
        if summary.datetime_start:
            print(f"   Started: {summary.datetime_start}")


def inspect_study(storage_url: str, study_name: str = None, top_n: int = 10):
    """Inspect a specific study and show top trials."""
    
    # Load study
    if study_name:
        study = optuna.load_study(study_name=study_name, storage=storage_url)
    else:
        # Load the first study if no name specified
        summaries = optuna.get_all_study_summaries(storage=storage_url)
        if not summaries:
            print("No studies found in database.")
            return
        study = optuna.load_study(study_name=summaries[0].study_name, storage=storage_url)
    
    print(f"\n{'='*60}")
    print(f"Study: {study.study_name}")
    print(f"{'='*60}")
    print(f"Number of trials: {len(study.trials)}")
    print(f"Best value (validation accuracy): {study.best_value:.4f}")
    print(f"Best trial number: {study.best_trial.number}")
    
    # Show best parameters
    print(f"\n{'='*60}")
    print("Best Parameters:")
    print(f"{'='*60}")
    for param, value in study.best_params.items():
        print(f"  {param}: {value}")
    
    # Get all completed trials
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    failed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
    
    print(f"\n{'='*60}")
    print("Trial Statistics:")
    print(f"{'='*60}")
    print(f"  Completed: {len(completed_trials)}")
    print(f"  Pruned: {len(pruned_trials)}")
    print(f"  Failed: {len(failed_trials)}")
    
    # Show top N trials
    if completed_trials:
        print(f"\n{'='*60}")
        print(f"Top {min(top_n, len(completed_trials))} Trials (by validation accuracy):")
        print(f"{'='*60}")
        
        # Sort by value (higher is better for accuracy)
        sorted_trials = sorted(completed_trials, key=lambda t: t.value, reverse=True)[:top_n]
        
        trial_data = []
        for trial in sorted_trials:
            trial_data.append({
                'Trial': trial.number,
                'Val_Acc': f"{trial.value:.4f}",
                'LR': f"{trial.params.get('learning_rate', 'N/A'):.6f}" if isinstance(trial.params.get('learning_rate'), float) else trial.params.get('learning_rate', 'N/A'),
                'WD': f"{trial.params.get('weight_decay', 'N/A'):.6f}" if isinstance(trial.params.get('weight_decay'), float) else trial.params.get('weight_decay', 'N/A'),
                'Dropout': f"{trial.params.get('dropout_rate', 'N/A'):.3f}" if isinstance(trial.params.get('dropout_rate'), float) else trial.params.get('dropout_rate', 'N/A'),
                'Optimizer': trial.params.get('optimizer', 'N/A'),
                'Scheduler': trial.params.get('scheduler', 'N/A'),
                'Label_Smooth': f"{trial.params.get('label_smoothing', 'N/A'):.3f}" if isinstance(trial.params.get('label_smoothing'), float) else trial.params.get('label_smoothing', 'N/A'),
            })
        
        df = pd.DataFrame(trial_data)
        print(df.to_string(index=False))
        
        # Save to CSV
        csv_path = f"{study.study_name}_top_trials.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nTop trials saved to: {csv_path}")


def export_all_trials(storage_url: str, study_name: str = None, output_file: str = None):
    """Export all trial data to CSV."""
    
    # Load study
    if study_name:
        study = optuna.load_study(study_name=study_name, storage=storage_url)
    else:
        summaries = optuna.get_all_study_summaries(storage=storage_url)
        if not summaries:
            print("No studies found in database.")
            return
        study = optuna.load_study(study_name=summaries[0].study_name, storage=storage_url)
    
    # Export trials to DataFrame
    df = study.trials_dataframe()
    
    # Set output filename
    if output_file is None:
        output_file = f"{study.study_name}_all_trials.csv"
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"\nAll trials exported to: {output_file}")
    print(f"Total trials: {len(df)}")
    
    return df


def compare_studies(storage_url: str):
    """Compare multiple studies in the database."""
    summaries = optuna.get_all_study_summaries(storage=storage_url)
    
    if len(summaries) < 2:
        print("Need at least 2 studies for comparison.")
        return
    
    print(f"\n{'='*60}")
    print("Study Comparison:")
    print(f"{'='*60}")
    
    comparison_data = []
    for summary in summaries:
        comparison_data.append({
            'Study': summary.study_name,
            'Trials': summary.n_trials,
            'Best_Val_Acc': f"{summary.best_trial.value:.4f}" if summary.best_trial else "N/A",
            'Started': summary.datetime_start.strftime("%Y-%m-%d %H:%M") if summary.datetime_start else "N/A"
        })
    
    df = pd.DataFrame(comparison_data)
    df = df.sort_values('Best_Val_Acc', ascending=False)
    print(df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description='Inspect Optuna HPO results')
    parser.add_argument('--storage', type=str, required=True,
                        help='Path to SQLite database (e.g., hpo_results.db or sqlite:///hpo_results.db)')
    parser.add_argument('--study', type=str, default=None,
                        help='Specific study name to inspect (default: first study)')
    parser.add_argument('--list', action='store_true',
                        help='List all studies in database')
    parser.add_argument('--top', type=int, default=10,
                        help='Number of top trials to show (default: 10)')
    parser.add_argument('--export', action='store_true',
                        help='Export all trials to CSV')
    parser.add_argument('--compare', action='store_true',
                        help='Compare all studies')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV filename for export')
    
    args = parser.parse_args()
    
    # Handle storage URL format
    if args.storage.startswith('sqlite:///'):
        storage_url = args.storage
    else:
        storage_url = f"sqlite:///{args.storage}"
    
    # Check if database exists
    db_path = storage_url.replace('sqlite:///', '')
    if not Path(db_path).exists():
        print(f"Error: Database file not found: {db_path}")
        sys.exit(1)
    
    # Execute requested action
    if args.list:
        list_studies(storage_url)
    elif args.compare:
        compare_studies(storage_url)
    elif args.export:
        export_all_trials(storage_url, args.study, args.output)
    else:
        # Default: inspect study and show top trials
        inspect_study(storage_url, args.study, args.top)


if __name__ == "__main__":
    main()
