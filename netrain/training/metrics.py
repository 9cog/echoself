"""
Metrics tracking for Deep Tree Echo training
"""

from typing import Dict, Any, List, Optional
import numpy as np
from collections import defaultdict, deque


class MetricsTracker:
    """
    Track and aggregate training metrics.
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics = defaultdict(lambda: deque(maxlen=window_size))
        self.step_metrics = defaultdict(list)
        self.global_step = 0
    
    def update(self, name: str, value: float, step: Optional[int] = None):
        """
        Update a metric value.
        
        Args:
            name: Metric name
            value: Metric value
            step: Optional step number
        """
        if step is None:
            step = self.global_step
        
        self.metrics[name].append(value)
        self.step_metrics[name].append((step, value))
        
        if name == 'global_step':
            self.global_step = int(value)
    
    def get_average(self, name: str) -> float:
        """
        Get the average value of a metric over the window.
        
        Args:
            name: Metric name
            
        Returns:
            Average value
        """
        if name in self.metrics and len(self.metrics[name]) > 0:
            return np.mean(list(self.metrics[name]))
        return 0.0
    
    def get_last(self, name: str) -> float:
        """
        Get the last value of a metric.
        
        Args:
            name: Metric name
            
        Returns:
            Last value
        """
        if name in self.metrics and len(self.metrics[name]) > 0:
            return self.metrics[name][-1]
        return 0.0
    
    def get_all(self) -> Dict[str, Any]:
        """
        Get all metrics.
        
        Returns:
            Dictionary of all metrics
        """
        result = {}
        
        for name in self.metrics:
            result[f"{name}_avg"] = self.get_average(name)
            result[f"{name}_last"] = self.get_last(name)
            result[f"{name}_history"] = list(self.metrics[name])
        
        return result
    
    def get_summary(self) -> str:
        """
        Get a summary string of current metrics.
        
        Returns:
            Summary string
        """
        summary_parts = []
        
        for name in sorted(self.metrics.keys()):
            avg = self.get_average(name)
            last = self.get_last(name)
            summary_parts.append(f"{name}: {last:.4f} (avg: {avg:.4f})")
        
        return " | ".join(summary_parts)
    
    def reset(self):
        """Reset all metrics."""
        self.metrics.clear()
        self.step_metrics.clear()
        self.global_step = 0