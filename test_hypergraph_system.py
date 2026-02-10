"""
test_hypergraph_system.py

Comprehensive test suite for the Deep Tree Echo hypergraph encoding system.
Tests both the Python bridge and validates the Scheme implementation structure.
"""

import unittest
import os
from pathlib import Path
from typing import List, Dict


class TestHypergraphStructure(unittest.TestCase):
    """Test the hypergraph system structure and files."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.repo_root = Path(__file__).parent
        self.hypergraph_dir = self.repo_root / "echo" / "hypergraph"
    
    def test_hypergraph_directory_exists(self):
        """Test that the hypergraph directory was created."""
        self.assertTrue(self.hypergraph_dir.exists())
        self.assertTrue(self.hypergraph_dir.is_dir())
    
    def test_core_module_exists(self):
        """Test that core.scm exists and has content."""
        core_file = self.hypergraph_dir / "core.scm"
        self.assertTrue(core_file.exists())
        
        content = core_file.read_text()
        self.assertIn("make-hypergraph-node", content)
        self.assertIn("hypergraph-node?", content)
        self.assertIn("define-module", content)
    
    def test_attention_module_exists(self):
        """Test that attention.scm exists and has content."""
        attention_file = self.hypergraph_dir / "attention.scm"
        self.assertTrue(attention_file.exists())
        
        content = attention_file.read_text()
        self.assertIn("semantic-salience", content)
        self.assertIn("adaptive-attention", content)
        self.assertIn("filter-by-attention", content)
    
    def test_repo_introspection_module_exists(self):
        """Test that repo-introspection.scm exists and has content."""
        introspection_file = self.hypergraph_dir / "repo-introspection.scm"
        self.assertTrue(introspection_file.exists())
        
        content = introspection_file.read_text()
        self.assertIn("repo-file-list", content)
        self.assertIn("safe-read-file", content)
        self.assertIn("assemble-hypergraph-input", content)
        self.assertIn("MAX-FILE-SIZE", content)
    
    def test_prompt_template_module_exists(self):
        """Test that prompt-template.scm exists and has content."""
        template_file = self.hypergraph_dir / "prompt-template.scm"
        self.assertTrue(template_file.exists())
        
        content = template_file.read_text()
        self.assertIn("prompt-template", content)
        self.assertIn("inject-repo-input-into-prompt", content)
        self.assertIn("create-cognitive-prompt", content)
    
    def test_main_hypergraph_module_exists(self):
        """Test that the main hypergraph.scm module exists."""
        main_file = self.repo_root / "echo" / "hypergraph.scm"
        self.assertTrue(main_file.exists())
        
        content = main_file.read_text()
        self.assertIn("define-module", content)
        self.assertIn("opencog hypergraph", content)
    
    def test_example_usage_exists(self):
        """Test that example-usage.scm exists."""
        example_file = self.hypergraph_dir / "example-usage.scm"
        self.assertTrue(example_file.exists())
        
        content = example_file.read_text()
        self.assertIn("Example 1", content)
        self.assertIn("Example 2", content)
    
    def test_readme_exists(self):
        """Test that README.md exists and has comprehensive content."""
        readme_file = self.hypergraph_dir / "README.md"
        self.assertTrue(readme_file.exists())
        
        content = readme_file.read_text()
        self.assertIn("Deep Tree Echo", content)
        self.assertIn("Hypergraph", content)
        self.assertIn("Architecture", content)
        self.assertIn("Usage Examples", content)
    
    def test_python_bridge_exists(self):
        """Test that the Python bridge exists."""
        bridge_file = self.repo_root / "echo" / "hypergraph_bridge.py"
        self.assertTrue(bridge_file.exists())
        
        content = bridge_file.read_text()
        self.assertIn("HypergraphBridge", content)
        self.assertIn("HypergraphNode", content)


class TestHypergraphNode(unittest.TestCase):
    """Test the HypergraphNode Python class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Import the bridge module
        import sys
        sys.path.insert(0, str(Path(__file__).parent / "echo"))
        from hypergraph_bridge import HypergraphNode
        self.HypergraphNode = HypergraphNode
    
    def test_node_creation(self):
        """Test creating a hypergraph node."""
        node = self.HypergraphNode(
            "test-id",
            "file",
            "test content",
            []
        )
        
        self.assertEqual(node.id, "test-id")
        self.assertEqual(node.type, "file")
        self.assertEqual(node.content, "test content")
        self.assertEqual(node.links, [])
    
    def test_node_to_dict(self):
        """Test converting node to dictionary."""
        node = self.HypergraphNode(
            "test-id",
            "concept",
            "test content",
            ["link1", "link2"]
        )
        
        node_dict = node.to_dict()
        self.assertEqual(node_dict['id'], "test-id")
        self.assertEqual(node_dict['type'], "concept")
        self.assertEqual(node_dict['content'], "test content")
        self.assertEqual(node_dict['links'], ["link1", "link2"])
    
    def test_node_repr(self):
        """Test node string representation."""
        node = self.HypergraphNode("test-id", "file", "content", [])
        repr_str = repr(node)
        
        self.assertIn("HypergraphNode", repr_str)
        self.assertIn("test-id", repr_str)
        self.assertIn("file", repr_str)


class TestHypergraphBridge(unittest.TestCase):
    """Test the HypergraphBridge Python class."""
    
    def setUp(self):
        """Set up test fixtures."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent / "echo"))
        from hypergraph_bridge import HypergraphBridge
        self.HypergraphBridge = HypergraphBridge
    
    def test_bridge_initialization(self):
        """Test initializing the bridge."""
        bridge = self.HypergraphBridge()
        
        self.assertIsNotNone(bridge.repo_root)
        self.assertEqual(bridge.guile_path, "guile")
        self.assertTrue(Path(bridge.repo_root).exists())
    
    def test_bridge_with_custom_root(self):
        """Test initializing bridge with custom root."""
        custom_root = "/tmp/test"
        bridge = self.HypergraphBridge(repo_root=custom_root)
        
        self.assertEqual(bridge.repo_root, custom_root)


class TestSchemeCodeStructure(unittest.TestCase):
    """Test the structure and syntax of Scheme code."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.hypergraph_dir = Path(__file__).parent / "echo" / "hypergraph"
    
    def test_core_scm_syntax(self):
        """Test that core.scm has proper Scheme syntax."""
        core_file = self.hypergraph_dir / "core.scm"
        content = core_file.read_text()
        
        # Check for balanced parentheses
        open_parens = content.count('(')
        close_parens = content.count(')')
        self.assertEqual(open_parens, close_parens, 
                        "Unbalanced parentheses in core.scm")
        
        # Check for module definition
        self.assertIn("(define-module (opencog hypergraph core)", content)
        
        # Check for exported functions
        self.assertIn("#:export", content)
    
    def test_attention_scm_syntax(self):
        """Test that attention.scm has proper Scheme syntax."""
        attention_file = self.hypergraph_dir / "attention.scm"
        content = attention_file.read_text()
        
        # Check for balanced parentheses
        open_parens = content.count('(')
        close_parens = content.count(')')
        self.assertEqual(open_parens, close_parens, 
                        "Unbalanced parentheses in attention.scm")
        
        # Check for key functions
        self.assertIn("(define (semantic-salience", content)
        self.assertIn("(define (adaptive-attention", content)
    
    def test_all_scheme_files_balanced(self):
        """Test that all Scheme files have balanced parentheses."""
        scheme_files = list(self.hypergraph_dir.glob("*.scm"))
        scheme_files.append(Path(__file__).parent / "echo" / "hypergraph.scm")
        
        for scm_file in scheme_files:
            with self.subTest(file=scm_file.name):
                content = scm_file.read_text()
                open_parens = content.count('(')
                close_parens = content.count(')')
                self.assertEqual(open_parens, close_parens,
                               f"Unbalanced parentheses in {scm_file.name}")


class TestDocumentation(unittest.TestCase):
    """Test documentation completeness."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.hypergraph_dir = Path(__file__).parent / "echo" / "hypergraph"
        self.readme = self.hypergraph_dir / "README.md"
    
    def test_readme_sections(self):
        """Test that README has all required sections."""
        content = self.readme.read_text()
        
        required_sections = [
            "Overview",
            "Architecture",
            "Core Components",
            "Usage Examples",
            "Configuration",
            "Testing"
        ]
        
        for section in required_sections:
            with self.subTest(section=section):
                self.assertIn(section, content)
    
    def test_readme_code_examples(self):
        """Test that README includes code examples."""
        content = self.readme.read_text()
        
        # Check for Scheme code blocks
        self.assertIn("```scheme", content)
        
        # Check for example functions
        self.assertIn("make-hypergraph-node", content)
        self.assertIn("semantic-salience", content)
        self.assertIn("adaptive-attention", content)
    
    def test_readme_mermaid_diagram(self):
        """Test that README includes architecture diagram."""
        content = self.readme.read_text()
        self.assertIn("```mermaid", content)
        self.assertIn("graph TD", content)


def run_tests():
    """Run all tests and print results."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestHypergraphStructure))
    suite.addTests(loader.loadTestsFromTestCase(TestHypergraphNode))
    suite.addTests(loader.loadTestsFromTestCase(TestHypergraphBridge))
    suite.addTests(loader.loadTestsFromTestCase(TestSchemeCodeStructure))
    suite.addTests(loader.loadTestsFromTestCase(TestDocumentation))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("Test Summary:")
    print(f"  Tests run: {result.testsRun}")
    print(f"  Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    print("="*70)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
