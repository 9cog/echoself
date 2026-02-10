"""
hypergraph_bridge.py

Python bridge to the Deep Tree Echo Scheme-based hypergraph encoding system.
Provides a convenient Python API for interacting with the hypergraph modules.

This bridge enables Python-based components to leverage the Scheme-based
cognitive architecture for repository introspection and pattern recognition.
"""

import subprocess
import json
import os
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path


class HypergraphNode:
    """Represents a hypergraph node in Python."""
    
    def __init__(self, node_id: str, node_type: str, content: str, links: List[str]):
        self.id = node_id
        self.type = node_type
        self.content = content
        self.links = links
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary representation."""
        return {
            'id': self.id,
            'type': self.type,
            'content': self.content,
            'links': self.links
        }
    
    def __repr__(self) -> str:
        return f"HypergraphNode(id='{self.id}', type='{self.type}')"


class HypergraphBridge:
    """Bridge between Python and Scheme hypergraph system."""
    
    def __init__(self, repo_root: Optional[str] = None, guile_path: str = "guile"):
        """
        Initialize the hypergraph bridge.
        
        Args:
            repo_root: Root directory of the repository (defaults to current dir)
            guile_path: Path to the Guile Scheme interpreter
        """
        self.repo_root = repo_root or os.getcwd()
        self.guile_path = guile_path
        self.hypergraph_dir = Path(__file__).parent
    
    def _run_scheme(self, scheme_code: str) -> str:
        """
        Execute Scheme code and return the output.
        
        Args:
            scheme_code: Scheme code to execute
            
        Returns:
            Standard output from Scheme execution
        """
        # Prepare the Scheme code with module loading
        full_code = f"""
(add-to-load-path "{self.hypergraph_dir.parent}")
(use-modules (opencog hypergraph))
{scheme_code}
"""
        
        try:
            result = subprocess.run(
                [self.guile_path, "-c", full_code],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"Scheme execution failed: {result.stderr}")
            
            return result.stdout
        except subprocess.TimeoutExpired:
            raise RuntimeError("Scheme execution timed out")
        except FileNotFoundError:
            raise RuntimeError(f"Guile interpreter not found at: {self.guile_path}")
    
    def calculate_salience(self, path: str) -> float:
        """
        Calculate semantic salience for a given file path.
        
        Args:
            path: File path to evaluate
            
        Returns:
            Salience score (0.0 to 1.0)
        """
        scheme_code = f'(display (semantic-salience "{path}"))'
        output = self._run_scheme(scheme_code)
        return float(output.strip())
    
    def adaptive_attention_threshold(
        self, 
        cognitive_load: float, 
        recent_activity: float
    ) -> float:
        """
        Calculate adaptive attention threshold.
        
        Args:
            cognitive_load: Current cognitive load (0.0 to 1.0)
            recent_activity: Recent activity level (0.0 to 1.0)
            
        Returns:
            Attention threshold (0.0 to 1.0)
        """
        scheme_code = f'(display (adaptive-attention {cognitive_load} {recent_activity}))'
        output = self._run_scheme(scheme_code)
        return float(output.strip())
    
    def get_repository_files(
        self, 
        attention_threshold: float = 0.5
    ) -> List[str]:
        """
        Get list of repository files meeting attention threshold.
        
        Args:
            attention_threshold: Minimum salience score (0.0 to 1.0)
            
        Returns:
            List of file paths
        """
        scheme_code = f"""
(define files (repo-file-list "{self.repo_root}" {attention_threshold}))
(for-each
  (lambda (f) (display f) (newline))
  files)
"""
        output = self._run_scheme(scheme_code)
        return [line.strip() for line in output.strip().split('\n') if line.strip()]
    
    def create_cognitive_prompt(
        self,
        purpose: str,
        cognitive_load: float = 0.3,
        recent_activity: float = 0.5,
        root_dir: Optional[str] = None
    ) -> str:
        """
        Create a cognitive prompt with adaptive repository context.
        
        Args:
            purpose: Purpose of the cognitive task
            cognitive_load: Current cognitive load (0.0 to 1.0)
            recent_activity: Recent activity level (0.0 to 1.0)
            root_dir: Root directory to scan (defaults to repo_root)
            
        Returns:
            Complete cognitive prompt string
        """
        root = root_dir or self.repo_root
        scheme_code = f"""
(display (create-cognitive-prompt 
  "{root}"
  {cognitive_load}
  {recent_activity}
  "{purpose}"))
"""
        return self._run_scheme(scheme_code)
    
    def analyze_repository_salience(
        self,
        paths: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Analyze salience scores for multiple paths.
        
        Args:
            paths: List of paths to analyze (optional)
            
        Returns:
            Dictionary mapping paths to salience scores
        """
        if paths is None:
            # Get all files in repository
            paths = []
            for root, dirs, files in os.walk(self.repo_root):
                # Skip hidden directories
                dirs[:] = [d for d in dirs if not d.startswith('.')]
                for file in files:
                    paths.append(os.path.join(root, file))
        
        results = {}
        for path in paths:
            try:
                results[path] = self.calculate_salience(path)
            except Exception as e:
                results[path] = 0.0  # Default for errors
        
        return results
    
    def get_high_salience_files(
        self,
        threshold: float = 0.7
    ) -> List[Tuple[str, float]]:
        """
        Get files with high salience scores.
        
        Args:
            threshold: Minimum salience threshold
            
        Returns:
            List of (path, salience) tuples, sorted by salience
        """
        salience_map = self.analyze_repository_salience()
        high_salience = [
            (path, score) 
            for path, score in salience_map.items() 
            if score >= threshold
        ]
        return sorted(high_salience, key=lambda x: x[1], reverse=True)


# Example usage and testing
if __name__ == "__main__":
    # Initialize bridge
    bridge = HypergraphBridge()
    
    print("=== Deep Tree Echo Hypergraph Bridge ===\n")
    
    # Test 1: Calculate salience for specific paths
    print("Test 1: Semantic Salience")
    test_paths = [
        "./echo/model/AtomSpace.scm",
        "./echo/hypergraph/core.scm",
        "./README.md",
        "./test_basic.py"
    ]
    
    for path in test_paths:
        try:
            salience = bridge.calculate_salience(path)
            print(f"  {path}: {salience:.2f}")
        except Exception as e:
            print(f"  {path}: Error - {e}")
    
    # Test 2: Adaptive attention
    print("\nTest 2: Adaptive Attention Thresholds")
    for load in [0.2, 0.5, 0.8]:
        for activity in [0.2, 0.5, 0.8]:
            threshold = bridge.adaptive_attention_threshold(load, activity)
            print(f"  Load={load}, Activity={activity} => Threshold={threshold:.2f}")
    
    # Test 3: Get high salience files (if repository exists)
    print("\nTest 3: High Salience Files")
    try:
        high_files = bridge.get_high_salience_files(threshold=0.75)
        print(f"  Found {len(high_files)} files with salience >= 0.75")
        for path, salience in high_files[:5]:  # Show top 5
            print(f"    {salience:.2f}: {path}")
    except Exception as e:
        print(f"  Could not scan repository: {e}")
    
    print("\n=== Bridge Test Complete ===")
