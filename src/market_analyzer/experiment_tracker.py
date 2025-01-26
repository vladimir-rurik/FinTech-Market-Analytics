"""
Experiment tracking and metrics storage.
"""

import pandas as pd
import json
import os
from datetime import datetime
from typing import Dict, List, Optional

class ExperimentTracker:
    """Track and store experiment results."""
    
    def __init__(self, results_dir: str = 'results'):
        """Initialize experiment tracker."""
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # Load existing results if any
        self.results_file = os.path.join(results_dir, 'experiment_results.json')
        self.results = self._load_results()
    
    def _load_results(self) -> Dict:
        """Load existing results from file."""
        if os.path.exists(self.results_file):
            with open(self.results_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_experiment(self, 
                       experiment_name: str,
                       model_name: str,
                       train_metrics: Dict,
                       test_metrics: Dict,
                       validation_metrics: Dict,
                       params: Dict) -> None:
        """
        Save experiment results.
        
        Args:
            experiment_name: Name of the experiment
            model_name: Name of the model
            train_metrics: Training metrics
            test_metrics: Testing metrics
            validation_metrics: Validation metrics
            params: Model parameters
        """
        # Create experiment record
        experiment = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_name': model_name,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'validation_metrics': validation_metrics,
            'parameters': params
        }
        
        # Add to results
        if experiment_name not in self.results:
            self.results[experiment_name] = []
        self.results[experiment_name].append(experiment)
        
        # Save to file
        with open(self.results_file, 'w') as f:
            json.dump(self.results, f, indent=4)
    
    def get_best_experiment(self, 
                          experiment_name: str,
                          metric: str = 'f1',
                          dataset: str = 'validation') -> Dict:
        """Get best experiment based on metric."""
        if experiment_name not in self.results:
            return None
            
        experiments = self.results[experiment_name]
        metric_key = f'{dataset}_metrics'
        
        return max(experiments,
                  key=lambda x: x[metric_key][metric]
                  if metric in x[metric_key] else -float('inf'))
    
    def get_experiment_summary(self, experiment_name: str) -> pd.DataFrame:
        """Get summary of all experiments for plotting."""
        if experiment_name not in self.results:
            return pd.DataFrame()
            
        records = []
        for exp in self.results[experiment_name]:
            record = {
                'timestamp': exp['timestamp'],
                'model': exp['model_name']
            }
            
            # Add metrics
            for dataset in ['train', 'test', 'validation']:
                for metric, value in exp[f'{dataset}_metrics'].items():
                    record[f'{dataset}_{metric}'] = value
            
            records.append(record)
        
        return pd.DataFrame(records)
    
    def plot_experiment_history(self, 
                              experiment_name: str,
                              metric: str = 'f1'):
        """Plot experiment history."""
        import matplotlib.pyplot as plt
        
        df = self.get_experiment_summary(experiment_name)
        if df.empty:
            return
        
        plt.figure(figsize=(12, 6))
        
        for dataset in ['train', 'test', 'validation']:
            metric_col = f'{dataset}_{metric}'
            if metric_col in df.columns:
                plt.plot(df['timestamp'], df[metric_col],
                        marker='o', label=dataset.title())
        
        plt.title(f'{experiment_name} - {metric} History')
        plt.xlabel('Timestamp')
        plt.ylabel(metric.upper())
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()