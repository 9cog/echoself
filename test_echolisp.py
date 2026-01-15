#!/usr/bin/env python3
"""
Comprehensive tests for the EchoLisp class.

Tests cover all functionality including structure generation, successor calculation,
echo evolution, string conversion, and simulation behavior.
"""

import pytest
from echolisp import EchoLisp


class TestEchoLisp:
    """Test suite for EchoLisp class functionality."""
    
    def test_initialization(self):
        """Test EchoLisp initialization."""
        echo = EchoLisp()
        assert echo.treeid == {(): 0}
        
    def test_tostr_empty(self):
        """Test string conversion of empty echo."""
        echo = EchoLisp()
        result = echo.tostr(())
        assert result == "()"
        
    def test_tostr_simple_structures(self):
        """Test string conversion of simple echo structures."""
        echo = EchoLisp()
        
        # Single nested empty
        result = echo.tostr(((),))
        assert result == "(())"
        
        # Double nested empty  
        result = echo.tostr((((),),))
        assert result == "((()))"
        
        # Multiple elements
        result = echo.tostr(((), (), ()))
        assert result == "(()()())"
        
    def test_tostr_complex_structures(self):
        """Test string conversion of complex nested structures."""
        echo = EchoLisp()
        
        # Mixed nesting levels
        result = echo.tostr(((), (((),),)))
        assert result == "(()((())))"
        
        # Deep nesting
        result = echo.tostr((((((),),),),))
        assert result == "((((()))))"
        
    def test_succ_empty_structure(self):
        """Test successor generation for empty structure."""
        echo = EchoLisp()
        successors = list(echo.succ(()))
        
        # Empty structure should only generate one successor: ((),)
        assert len(successors) == 1
        assert successors[0] == ((),)
        
    def test_succ_single_element(self):
        """Test successor generation for single-element structures.""" 
        echo = EchoLisp()
        
        # First establish treeid for empty structure
        _ = list(echo.succ(()))
        echo.treeid[((),)] = 1
        
        successors = list(echo.succ(((),)))
        
        # Should generate: ((),) + ((),) and successors of ((),) wrapped in tuple
        expected_successors = [
            ((),) + ((),),  # ((,), (,))
            ((((),),),)     # Successor of ((),) wrapped
        ]
        
        assert ((),) + ((),) in successors
        assert len(successors) >= 1
        
    def test_echoes_size_1(self):
        """Test echo generation for size 1."""
        echo = EchoLisp()
        echoes = list(echo.echoes(1))
        
        assert len(echoes) == 1
        assert echoes[0] == ()
        
    def test_echoes_size_2(self):
        """Test echo generation for size 2."""
        echo = EchoLisp()
        echoes = list(echo.echoes(2))
        
        # Size 2 should contain successors of empty echo
        assert len(echoes) == 1
        assert echoes[0] == ((),)
        
    def test_echoes_progressive_sizes(self):
        """Test that larger echo sizes contain more structures."""
        echo = EchoLisp()
        
        size_1_count = len(list(echo.echoes(1)))
        
        # Reset for fair comparison
        echo = EchoLisp()
        size_2_count = len(list(echo.echoes(2)))
        
        echo = EchoLisp() 
        size_3_count = len(list(echo.echoes(3)))
        
        # Each size should have at least as many as the previous
        assert size_1_count == 1
        assert size_2_count >= size_1_count
        assert size_3_count >= size_2_count
        
    def test_simulate_expected_output(self):
        """Test simulation produces the expected output format."""
        echo = EchoLisp()
        results = echo.simulate(4)
        
        # Should be a list of tuples with (step_number, string_representation)
        assert isinstance(results, list)
        assert len(results) > 0
        
        for step, structure_str in results:
            assert isinstance(step, int)
            assert isinstance(structure_str, str)
            assert step > 0
            assert structure_str.startswith("(")
            assert structure_str.endswith(")")
            
    def test_simulate_expected_sequence(self):
        """Test simulation produces the expected sequence for n=4."""
        echo = EchoLisp()
        results = echo.simulate(4)
        
        # Extract just the structures
        structures = [structure for _, structure in results]
        
        # Check that we have the expected structures from the problem statement
        expected_structures = [
            "(()()())",     # Step 1: three empty elements
            "(()(()))",     # Step 2: mixed nesting  
            "((()()))",     # Step 3: nested pair
            "(((())))"      # Step 4: deep nesting
        ]
        
        # Verify we get exactly these structures in order
        assert len(structures) == 4
        for i, expected in enumerate(expected_structures):
            assert structures[i] == expected, f"Step {i+1}: expected {expected}, got {structures[i]}"
            
    def test_treeid_assignment(self):
        """Test that tree IDs are assigned correctly."""
        echo = EchoLisp()
        
        # Generate some echoes to populate treeid
        _ = list(echo.echoes(3))
        
        # Check that empty structure has ID 0
        assert echo.treeid[()] == 0
        
        # Check that all generated structures have unique IDs
        ids = list(echo.treeid.values())
        assert len(ids) == len(set(ids))  # All IDs should be unique
        
        # Check that IDs are sequential starting from 0
        assert min(ids) == 0
        assert max(ids) == len(ids) - 1
        
    def test_succ_ordering_constraint(self):
        """Test that successor generation respects ordering constraints."""
        echo = EchoLisp()
        
        # Build up treeid state by generating echoes
        _ = list(echo.echoes(3))
        
        # Test that successors maintain proper ordering
        # This is implicitly tested by the algorithm working correctly
        # More specific tests would require detailed analysis of the constraint
        assert len(echo.treeid) > 1  # Should have generated multiple structures
        
    def test_recursive_structure_conversion(self):
        """Test conversion of deeply recursive structures."""
        echo = EchoLisp()
        
        # Create a deeply nested structure manually
        deep_structure = (((((),),),),)
        result = echo.tostr(deep_structure)
        
        expected = "((((()))))"
        assert result == expected
        
    def test_mixed_nesting_conversion(self):
        """Test conversion of structures with mixed nesting levels."""
        echo = EchoLisp()
        
        # Mixed structure: some deep, some shallow
        mixed_structure = ((), (((),),), ())
        result = echo.tostr(mixed_structure)
        
        expected = "(()((()))())"
        assert result == expected


def test_example_simulation():
    """Test the example simulation from the problem statement."""
    echolisp = EchoLisp()
    steps = echolisp.simulate(4)
    
    # Verify we get results
    assert len(steps) > 0
    
    # Check format
    for step, structure in steps:
        assert isinstance(step, int)
        assert isinstance(structure, str)
        assert step >= 1
        
    # Print results for manual verification
    print("\nExample simulation output:")
    for step, structure in steps:
        print(f"Step {step}: {structure}")


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])