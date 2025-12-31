"""
Training Logger for Blind Path Detection System
"""

import json
from datetime import datetime
from pathlib import Path


class TrainingLogger:
    """Training log recorder"""

    def __init__(self, log_dir):
        """
        Initialize logger

        Args:
            log_dir: Directory to save logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.log_file = self.log_dir / f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.log_data = {
            "start_time": datetime.now().isoformat(),
            "config": {},
            "experiments": []
        }

    def log_config(self, config_dict):
        """Log configuration"""
        self.log_data["config"] = config_dict

    def log_experiment(self, experiment_name, params, results):
        """Log experiment results"""
        experiment = {
            "name": experiment_name,
            "timestamp": datetime.now().isoformat(),
            "params": params,
            "results": results
        }
        self.log_data["experiments"].append(experiment)
        self.save()

    def save(self):
        """Save logs to file"""
        with open(self.log_file, 'w') as f:
            json.dump(self.log_data, f, indent=2)

    def load(self, log_file=None):
        """Load logs from file"""
        if log_file is None:
            log_file = self.log_file

        with open(log_file, 'r') as f:
            self.log_data = json.load(f)

        return self.log_data