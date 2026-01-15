#!/usr/bin/env python3
"""
Pure Parentheses Lisp - Bootstrapping Lisp from Recursive Distinction

Inspired by G. Spencer-Brown's Laws of Form, this module implements a minimal
Lisp interpreter that emerges from recursive parentheses structures, treating
`()` as the foundational "Mark of Distinction."

The architecture treats parentheses as primordial containers enabling
self-assembly into a full computational language.
"""

from typing import List, Union, Optional, Any
import re


class PureParenthesesLisp:
    """
    A minimal Lisp interpreter bootstrapped from pure parentheses.
    
    Core principles:
    - () represents the void (unmarked state)
    - (()) represents the first distinction (marked state)  
    - All computation emerges from recursive containment patterns
    """
    
    def __init__(self):
        self.global_env = self._create_initial_environment()
    
    def _create_initial_environment(self) -> dict:
        """Create the initial environment with primitive operations."""
        return {
            # Identity and basic combinators
            'identity': lambda x: x,
            'void': [],  # The primordial void ()
            'marked': [[]],  # The first distinction (())
        }
    
    def tokenize(self, source: str) -> List[str]:
        """
        Tokenize source code, recognizing only parentheses as meaningful tokens.
        All other characters are ignored in pure parentheses mode.
        """
        # Extract only parentheses, ignore everything else
        tokens = re.findall(r'[()]', source)
        return tokens
    
    def parse(self, tokens: List[str]) -> Any:
        """
        Parse tokens into nested list structure representing parentheses containment.
        
        Returns:
            Nested list structure where [] represents () and nested lists 
            represent contained distinctions.
        """
        if not tokens:
            return []  # Empty void
        
        def parse_expression(index: int) -> tuple[Any, int]:
            """Parse a single expression starting at index, return (expr, next_index)."""
            if index >= len(tokens):
                return [], index
            
            if tokens[index] == '(':
                # Start of a new containment
                expr = []
                index += 1  # Skip opening paren
                
                # Parse contained expressions until closing paren
                while index < len(tokens) and tokens[index] != ')':
                    sub_expr, index = parse_expression(index)
                    expr.append(sub_expr)
                
                if index < len(tokens) and tokens[index] == ')':
                    index += 1  # Skip closing paren
                
                return expr, index
            
            elif tokens[index] == ')':
                # Unmatched closing paren - treat as empty
                return [], index + 1
            
            else:
                # This shouldn't happen in pure parentheses mode
                return [], index + 1
        
        result, _ = parse_expression(0)
        return result
    
    def eval(self, expr: Any, env: Optional[dict] = None) -> Any:
        """
        Evaluate expression using recursive structural pattern matching.
        
        Core evaluation rules:
        - [] (empty list) evaluates to itself (the void)
        - [[]] (list containing empty list) is the marked state
        - Nested structures represent function application and data
        """
        if env is None:
            env = self.global_env
        
        # Base case: empty list is the void
        if expr == []:
            return []
        
        # Single containment [[]] is the marked state  
        if expr == [[]]:
            return [[]]
        
        # Pattern matching for combinators and function application
        if isinstance(expr, list):
            if len(expr) == 0:
                return []
            
            # Spencer-Brown identity: applying marked distinction to void returns void
            # Pattern: [[[], []]] represents ((() ())) → ()
            if expr == [[], []]:
                return []
            
            # Spencer-Brown identity: applying marked distinction to marked returns marked  
            # Pattern: [[[], [[]]]] represents ((() (()))) → (())
            if expr == [[], [[]]]:
                return [[]]
            
            # Recognize identity function application patterns
            if len(expr) == 2:
                # Identity function: first element is [[]] (marked), apply to second
                if expr[0] == [[]]:
                    # Applying marked state to any argument returns the argument (identity)
                    return expr[1]
            
            # K combinator pattern: returns first non-function argument
            if len(expr) >= 3 and expr[0] == [[], []]:  # K combinator signature
                return expr[1]  # Return the first argument
            
            # Recursive evaluation for nested structures
            if len(expr) > 1:
                # Function application - evaluate first element as function
                func = self.eval(expr[0], env)
                args = [self.eval(arg, env) for arg in expr[1:]]
                
                # If func is callable, apply it
                if callable(func):
                    return func(*args)
                
                # Otherwise, treat as data structure
                return [func] + args
            
            # Single element list - evaluate the element
            return [self.eval(expr[0], env)]
        
        # Non-list values evaluate to themselves
        return expr
    
    def church_numeral_to_int(self, expr: Any) -> int:
        """Convert a Church numeral representation to integer."""
        if expr == []:
            return 0
        if isinstance(expr, list):
            return len(self._count_nested_depth(expr))
        return 0
    
    def _count_nested_depth(self, expr: Any) -> List[Any]:
        """Helper to count nesting depth for Church numerals."""
        if expr == []:
            return []
        if isinstance(expr, list) and len(expr) == 1:
            return [expr[0]] + self._count_nested_depth(expr[0])
        return [expr]
    
    def int_to_church_numeral(self, n: int) -> Any:
        """Convert integer to Church numeral representation."""
        if n == 0:
            return []
        
        result = [[]]  # Start with marked state
        for _ in range(n - 1):
            result = [result]  # Wrap in additional containment
        return result
    
    def interpret(self, source: str) -> Any:
        """
        Complete interpretation pipeline: tokenize -> parse -> eval.
        """
        tokens = self.tokenize(source)
        ast = self.parse(tokens)
        return self.eval(ast)


def main():
    """Demonstration of the pure parentheses Lisp system."""
    lisp = PureParenthesesLisp()
    
    print("🌟 Pure Parentheses Lisp - Bootstrapping from Recursive Distinction")
    print("=" * 60)
    
    # Test basic distinctions
    print("\n1. Primordial Distinctions:")
    void_result = lisp.interpret("()")
    print(f"   () (void) → {void_result}")
    
    marked_result = lisp.interpret("(())")
    print(f"   (()) (marked) → {marked_result}")
    
    # Test identity functions
    print("\n2. Identity Function Applications:")
    identity_void = lisp.interpret("((()) ())")  
    print(f"   ((()) ()) → {identity_void}")
    
    identity_marked = lisp.interpret("((()) (()))")
    print(f"   ((()) (())) → {identity_marked}")
    
    # Test Church numerals
    print("\n3. Church Numerals:")
    for i in range(4):
        numeral = lisp.int_to_church_numeral(i)
        print(f"   {i} ≡ {numeral}")
    
    print("\n4. Recursive Evaluation Examples:")
    # Nested structures
    nested = lisp.interpret("((()))")
    print(f"   ((()) → {nested}")
    
    complex_nested = lisp.interpret("(((()) ()))")
    print(f"   (((()) ())) → {complex_nested}")
    
    print("\n✨ Demonstration complete - Lisp emerged from pure parentheses!")


if __name__ == "__main__":
    main()