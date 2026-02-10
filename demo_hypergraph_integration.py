"""
demo_hypergraph_integration.py

Demonstration of Deep Tree Echo hypergraph encoding system integration.
Shows how the hypergraph system can be used for repository analysis,
cognitive prompting, and adaptive attention allocation.
"""

import os
from pathlib import Path
from typing import List, Dict, Tuple
import json


class CognitiveAnalyzer:
    """
    Cognitive analyzer for repository introspection using hypergraph encoding.
    
    This class demonstrates the practical application of the hypergraph
    system for intelligent repository analysis and cognitive enhancement.
    """
    
    def __init__(self, repo_root: str = None):
        """Initialize the cognitive analyzer."""
        self.repo_root = repo_root or os.getcwd()
        self.cognitive_load = 0.3  # Start with low cognitive load
        self.recent_activity = 0.5  # Medium activity level
    
    def calculate_file_salience(self, filepath: str) -> float:
        """
        Calculate semantic salience score for a file.
        
        This is a Python implementation of the Scheme semantic-salience function,
        demonstrating the cognitive principles without requiring Guile.
        """
        # AtomSpace core files - highest priority
        if "AtomSpace.scm" in filepath or "atomspace" in filepath.lower():
            return 0.95
        
        # Core directories - very high priority
        if "/core/" in filepath:
            return 0.90
        if "/hypergraph/" in filepath:
            return 0.88
        if "/model/" in filepath:
            return 0.85
        
        # Source code - high priority
        if "/src/" in filepath:
            return 0.80
        if filepath.endswith(".scm"):
            return 0.78
        if filepath.endswith(".py"):
            return 0.75
        
        # Documentation - medium-high priority
        if "README" in filepath:
            return 0.70
        if filepath.endswith(".md"):
            return 0.65
        
        # Behavior and cognitive modules
        if "/behavior/" in filepath:
            return 0.75
        if "/eva-model/" in filepath:
            return 0.80
        
        # Configuration files
        if filepath.endswith(".json"):
            return 0.60
        if filepath.endswith((".yml", ".yaml")):
            return 0.58
        
        # Test files
        if filepath.startswith("test_") or "_test." in filepath:
            return 0.55
        
        # Default salience
        return 0.50
    
    def adaptive_attention_threshold(self) -> float:
        """
        Calculate adaptive attention threshold based on current cognitive state.
        
        This implements the adaptive-attention function from the Scheme code.
        """
        base_threshold = 0.50
        load_factor = 0.30
        activity_adjustment = 0.20
        
        return (base_threshold + 
                (self.cognitive_load * load_factor) - 
                (self.recent_activity * activity_adjustment))
    
    def scan_repository(self, threshold: float = None) -> List[Tuple[str, float]]:
        """
        Scan repository and return files meeting attention threshold.
        
        Returns list of (filepath, salience) tuples sorted by salience.
        """
        if threshold is None:
            threshold = self.adaptive_attention_threshold()
        
        files = []
        
        # Walk the repository
        for root, dirs, filenames in os.walk(self.repo_root):
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith('.') 
                      and d not in ['node_modules', '__pycache__', 'dist', 'build']]
            
            for filename in filenames:
                if filename.startswith('.'):
                    continue
                
                filepath = os.path.join(root, filename)
                relative_path = os.path.relpath(filepath, self.repo_root)
                salience = self.calculate_file_salience(relative_path)
                
                if salience >= threshold:
                    files.append((relative_path, salience))
        
        # Sort by salience (highest first)
        return sorted(files, key=lambda x: x[1], reverse=True)
    
    def generate_cognitive_report(self) -> Dict:
        """
        Generate a comprehensive cognitive analysis report.
        
        This demonstrates the hypergraph system's ability to provide
        intelligent insights about repository structure and importance.
        """
        threshold = self.adaptive_attention_threshold()
        files = self.scan_repository(threshold)
        
        # Categorize files by type
        categories = {
            'core': [],
            'models': [],
            'behavior': [],
            'documentation': [],
            'tests': [],
            'configuration': [],
            'other': []
        }
        
        for filepath, salience in files:
            if '/core/' in filepath or '/hypergraph/' in filepath:
                categories['core'].append((filepath, salience))
            elif '/model/' in filepath or '/eva-model/' in filepath:
                categories['models'].append((filepath, salience))
            elif '/behavior/' in filepath:
                categories['behavior'].append((filepath, salience))
            elif filepath.endswith('.md') or 'README' in filepath:
                categories['documentation'].append((filepath, salience))
            elif 'test_' in filepath or '_test.' in filepath:
                categories['tests'].append((filepath, salience))
            elif filepath.endswith(('.json', '.yml', '.yaml')):
                categories['configuration'].append((filepath, salience))
            else:
                categories['other'].append((filepath, salience))
        
        # Calculate statistics
        total_files = len(files)
        avg_salience = sum(s for _, s in files) / total_files if total_files > 0 else 0
        
        report = {
            'cognitive_state': {
                'cognitive_load': self.cognitive_load,
                'recent_activity': self.recent_activity,
                'attention_threshold': threshold
            },
            'statistics': {
                'total_files_analyzed': total_files,
                'average_salience': avg_salience,
                'categories': {
                    cat: len(files) for cat, files in categories.items()
                }
            },
            'categories': categories,
            'top_files': files[:10]  # Top 10 most salient files
        }
        
        return report
    
    def create_cognitive_prompt(self, purpose: str) -> str:
        """
        Create a cognitive prompt for AI processing.
        
        This demonstrates the prompt-template functionality from the
        Scheme implementation.
        """
        threshold = self.adaptive_attention_threshold()
        files = self.scan_repository(threshold)
        
        # Build hypergraph-like representation
        file_context = "\n".join([
            f"(file \"{path}\" salience={salience:.2f})"
            for path, salience in files[:20]  # Limit to top 20 for prompt size
        ])
        
        prompt = f"""DeepTreeEcho Cognitive Process
Purpose: {purpose}
Cognitive Load: {self.cognitive_load}
Recent Activity: {self.recent_activity}
Attention Threshold: {threshold:.2f}

Repository Context (Top Salient Files):
{file_context}

The above files represent the most cognitively salient components
of the repository based on adaptive attention allocation.
"""
        return prompt
    
    def update_cognitive_state(self, load: float, activity: float):
        """Update the cognitive state for adaptive attention."""
        self.cognitive_load = max(0.0, min(1.0, load))
        self.recent_activity = max(0.0, min(1.0, activity))


def demonstrate_hypergraph_system():
    """
    Main demonstration function showing hypergraph system capabilities.
    """
    print("=" * 70)
    print("Deep Tree Echo Hypergraph Encoding System Demonstration")
    print("=" * 70)
    
    # Initialize analyzer
    analyzer = CognitiveAnalyzer()
    
    # 1. Show cognitive state
    print("\n1. Current Cognitive State:")
    print(f"   Cognitive Load: {analyzer.cognitive_load}")
    print(f"   Recent Activity: {analyzer.recent_activity}")
    print(f"   Attention Threshold: {analyzer.adaptive_attention_threshold():.2f}")
    
    # 2. Demonstrate salience scoring
    print("\n2. Semantic Salience Examples:")
    test_paths = [
        "echo/model/AtomSpace.scm",
        "echo/hypergraph/core.scm",
        "src/main.py",
        "README.md",
        "test_basic.py",
        "config.json"
    ]
    
    for path in test_paths:
        salience = analyzer.calculate_file_salience(path)
        print(f"   {path}: {salience:.2f}")
    
    # 3. Scan repository
    print("\n3. Repository Scan (Top 10 Files):")
    files = analyzer.scan_repository()
    for i, (path, salience) in enumerate(files[:10], 1):
        print(f"   {i:2d}. [{salience:.2f}] {path}")
    
    # 4. Generate cognitive report
    print("\n4. Cognitive Analysis Report:")
    report = analyzer.generate_cognitive_report()
    
    print(f"   Total Files: {report['statistics']['total_files_analyzed']}")
    print(f"   Average Salience: {report['statistics']['average_salience']:.2f}")
    print("\n   Category Distribution:")
    for cat, count in report['statistics']['categories'].items():
        if count > 0:
            print(f"     - {cat.capitalize()}: {count} files")
    
    # 5. Demonstrate adaptive attention
    print("\n5. Adaptive Attention Demonstration:")
    print("   Testing different cognitive states:")
    
    test_states = [
        (0.2, 0.8, "Low load, high activity"),
        (0.5, 0.5, "Medium load, medium activity"),
        (0.8, 0.2, "High load, low activity")
    ]
    
    for load, activity, description in test_states:
        analyzer.update_cognitive_state(load, activity)
        threshold = analyzer.adaptive_attention_threshold()
        files_count = len(analyzer.scan_repository())
        print(f"   {description}:")
        print(f"     Threshold: {threshold:.2f}, Files: {files_count}")
    
    # Reset to default state
    analyzer.update_cognitive_state(0.3, 0.5)
    
    # 6. Generate cognitive prompt
    print("\n6. Cognitive Prompt Generation:")
    prompt = analyzer.create_cognitive_prompt(
        "Analyze hypergraph implementation patterns"
    )
    print("   Generated prompt (first 500 chars):")
    print("   " + "-" * 66)
    print("   " + "\n   ".join(prompt[:500].split('\n')))
    print("   " + "-" * 66)
    
    # 7. Save report to file
    print("\n7. Saving Cognitive Analysis Report:")
    report_file = "cognitive_analysis_report.json"
    with open(report_file, 'w') as f:
        # Convert tuples to lists for JSON serialization
        serializable_report = {
            'cognitive_state': report['cognitive_state'],
            'statistics': report['statistics'],
            'top_files': [
                {'path': path, 'salience': salience}
                for path, salience in report['top_files']
            ]
        }
        json.dump(serializable_report, f, indent=2)
    print(f"   Report saved to: {report_file}")
    
    print("\n" + "=" * 70)
    print("Demonstration Complete")
    print("=" * 70)
    print("\nThe hypergraph encoding system successfully demonstrates:")
    print("  ✓ Semantic salience calculation")
    print("  ✓ Adaptive attention allocation")
    print("  ✓ Repository introspection")
    print("  ✓ Cognitive state management")
    print("  ✓ Intelligent prompt generation")
    print("\nThis system enables recursive cognitive evolution through")
    print("dynamic repository analysis and hypergraph-encoded patterns.")


if __name__ == "__main__":
    demonstrate_hypergraph_system()
