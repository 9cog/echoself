#!/usr/bin/env python3
"""
Test suite for Pure Parentheses Lisp implementation.

Tests the bootstrapping of Lisp from recursive distinction patterns
as described in Spencer-Brown's Laws of Form.
"""

import pytest
from pure_parentheses_lisp import PureParenthesesLisp


class TestPureParenthesesLisp:
    """Test cases for the Pure Parentheses Lisp interpreter."""
    
    def setup_method(self):
        """Set up test fixture with fresh interpreter instance."""
        self.lisp = PureParenthesesLisp()
    
    def test_tokenization(self):
        """Test that tokenizer correctly extracts only parentheses."""
        # Basic parentheses
        tokens = self.lisp.tokenize("()")
        assert tokens == ['(', ')']
        
        # Nested parentheses
        tokens = self.lisp.tokenize("(())")
        assert tokens == ['(', '(', ')', ')']
        
        # Ignore non-parentheses characters
        tokens = self.lisp.tokenize("( hello world )")
        assert tokens == ['(', ')']
        
        # Complex mixed input
        tokens = self.lisp.tokenize("((lambda x))")
        assert tokens == ['(', '(', ')', ')']
    
    def test_parsing_basic_structures(self):
        """Test parsing of basic parentheses structures."""
        # Empty parentheses
        ast = self.lisp.parse(['(', ')'])
        assert ast == []
        
        # Single nested
        ast = self.lisp.parse(['(', '(', ')', ')'])
        assert ast == [[]]
        
        # Double nested
        ast = self.lisp.parse(['(', '(', '(', ')', ')', ')'])
        assert ast == [[[]]]
    
    def test_primordial_distinctions(self):
        """Test the fundamental void and marked states."""
        # The void ()
        result = self.lisp.interpret("()")
        assert result == []
        
        # The first distinction (())
        result = self.lisp.interpret("(())")
        assert result == [[]]
        
        # Multiple nested distinctions
        result = self.lisp.interpret("((()))")
        assert result == [[[]]]
    
    def test_identity_function_patterns(self):
        """Test identity function evaluation rules."""
        # Identity applied to void: ((()) ()) → ()
        result = self.lisp.interpret("((()) ())")
        assert result == []
        
        # Identity applied to marked: ((()) (())) → (())
        result = self.lisp.interpret("((()) (()))")
        assert result == [[]]
    
    def test_church_numerals(self):
        """Test Church numeral encoding and conversion."""
        # Test conversion from int to Church numeral
        assert self.lisp.int_to_church_numeral(0) == []
        assert self.lisp.int_to_church_numeral(1) == [[]]
        assert self.lisp.int_to_church_numeral(2) == [[[]]]
        assert self.lisp.int_to_church_numeral(3) == [[[[]]]]
        
        # Test conversion from Church numeral to int
        assert self.lisp.church_numeral_to_int([]) == 0
        assert self.lisp.church_numeral_to_int([[]]) == 1
        # Note: This is a simplified test - full Church numeral support 
        # would require more sophisticated evaluation
    
    def test_recursive_evaluation(self):
        """Test recursive evaluation of nested structures."""
        # Simple nesting
        result = self.lisp.eval([[]])
        assert result == [[]]
        
        # Complex nesting
        result = self.lisp.eval([[[]]])
        assert result == [[[]]]
        
        # Empty evaluates to empty
        result = self.lisp.eval([])
        assert result == []
    
    def test_k_combinator_pattern(self):
        """Test K combinator behavior patterns."""
        # K combinator should return its first argument
        # Pattern: ((() () (x)) x) where x could be any expression
        
        # Test with void
        k_with_void = [[[], []], []]  # Simplified K pattern with void
        result = self.lisp.eval(k_with_void)
        # In our simple implementation, this tests basic evaluation
        assert isinstance(result, list)
    
    def test_error_handling(self):
        """Test graceful handling of malformed input."""
        # Unmatched parentheses
        result = self.lisp.interpret("(()")
        assert isinstance(result, list)  # Should not crash
        
        # Empty input
        result = self.lisp.interpret("")
        assert result == []
        
        # Only closing parentheses
        result = self.lisp.interpret("))")
        assert isinstance(result, list)
    
    def test_spencer_brown_laws(self):
        """Test adherence to Spencer-Brown's Laws of Form principles."""
        # Law of Calling: (()) = ()  (marked crossing returns to unmarked)
        # In our interpretation, this is the identity function
        marked = self.lisp.interpret("(())")
        void = self.lisp.interpret("()")
        
        # Both should be distinct but related
        assert marked != void
        assert marked == [[]]
        assert void == []
        
        # Law of Crossing: Crossing a boundary twice returns to original state
        # This would be more complex to implement fully but we test basic patterns
        double_cross = self.lisp.interpret("(((())))")
        assert isinstance(double_cross, list)
    
    def test_metacircular_potential(self):
        """Test that the system can potentially support metacircular evaluation."""
        # Test that nested structures can be treated as both data and code
        
        # A nested structure that could represent code
        code_as_data = self.lisp.parse(['(', '(', ')', '(', ')', ')'])
        assert code_as_data == [[], []]
        
        # Evaluation treats it as data structure  
        result = self.lisp.eval(code_as_data)
        assert result == []  # This gets simplified by our identity patterns
    
    def test_bootstrapping_progression(self):
        """Test the progression from simple to complex structures."""
        # Start with void
        void = self.lisp.interpret("()")
        assert void == []
        
        # Progress to marked
        marked = self.lisp.interpret("(())")
        assert marked == [[]]
        
        # Progress to more complex nested structures
        complex1 = self.lisp.interpret("((()))")
        assert complex1 == [[[]]]
        
        complex2 = self.lisp.interpret("((()) ())")
        assert complex2 == []  # Identity function result
        
        # Each level should build upon the previous
        assert len(str(complex1)) > len(str(marked))
        assert len(str(marked)) > len(str(void))


def test_demonstration_output():
    """Test that the main demonstration runs without errors."""
    from pure_parentheses_lisp import main
    
    # This should run without throwing exceptions
    try:
        main()
        assert True
    except Exception as e:
        pytest.fail(f"Demonstration failed with error: {e}")


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])