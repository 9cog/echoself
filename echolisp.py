#!/usr/bin/env python3
"""
EchoLisp: A system for generating and simulating echo structures.

This module implements the EchoLisp class which generates hierarchical echo structures
using recursive tree-like patterns. Each echo structure represents a nested hierarchy
that can be converted to Lisp-style string notation.
"""


class EchoLisp:
    """
    EchoLisp generates and manages echo structures - hierarchical tree-like patterns
    that evolve through successive transformations.
    
    The class maintains an internal tracker (treeid) to assign unique identifiers
    to echo structures and provides methods to generate successors, create echoes
    of specific sizes, and simulate echo evolution.
    """
    
    def __init__(self):
        """Initialize EchoLisp with empty echo structure tracker."""
        self.treeid = {(): 0}  # Echo ID tracker - maps echo structures to unique IDs
    
    def succ(self, x):
        """
        Generate successors of an echo structure.
        
        A successor is formed by:
        1. Appending a single-node echo ((),) to any structure
        2. For single-element structures, recursively generating successors of the element
        3. For multi-element structures, generating successors of the head while
           maintaining ordering constraints with respect to the rest
        
        Args:
            x: Tuple representing an echo structure
            
        Yields:
            Tuple: Each successor echo structure
        """
        # Always yield the structure with an appended single-node echo
        yield ((),) + x
        
        # If x is empty, no further successors
        if not x:
            return
        
        # If x has only one element, generate successors of that element
        if len(x) == 1:
            for i in self.succ(x[0]):
                yield (i,)
            return
        
        # For multi-element structures, generate constrained successors
        head, rest = x[0], tuple(x[1:])
        top = self.treeid[rest[0]]
        
        # Generate successors of head that maintain ordering constraint
        for i in [i for i in self.succ(head) if self.treeid[i] <= top]:
            yield (i,) + rest
    
    def echoes(self, n):
        """
        Generate all echoes of size n.
        
        Uses recursive generation starting from base case n=1 (empty echo),
        then builds larger echoes by applying successor operations.
        
        Args:
            n: Integer size of echoes to generate
            
        Yields:
            Tuple: Each echo structure of size n
        """
        # Base case: size 1 is the empty echo
        if n == 1:
            yield ()
            return
        
        # Recursively generate echoes of size n-1, then their successors
        for x in self.echoes(n - 1):
            for a in self.succ(x):
                # Assign unique ID if not seen before
                if a not in self.treeid:
                    self.treeid[a] = len(self.treeid)
                yield a
    
    def tostr(self, x):
        """
        Convert echo structure to a readable Lisp-style string.
        
        Recursively converts nested tuples to parenthesized string notation.
        Empty tuples become "()", nested structures become nested parentheses.
        
        Args:
            x: Tuple representing an echo structure
            
        Returns:
            str: Lisp-style string representation
        """
        return "(" + "".join(map(self.tostr, x)) + ")"
    
    def simulate(self, n):
        """
        Simulate and display echo evolution for echoes up to size n.
        
        Generates all echo structures from size 1 to n, converts them to
        string representation, and returns step-by-step results.
        
        Args:
            n: Maximum size of echoes to simulate
            
        Returns:
            list: List of tuples (step_number, string_representation)
        """
        results = []
        for step, x in enumerate(self.echoes(n)):
            results.append((step + 1, self.tostr(x)))
        return results


# Example usage and demonstration
if __name__ == "__main__":
    # Instantiate and run the simulation
    echolisp = EchoLisp()
    steps = echolisp.simulate(4)
    
    # Print step-by-step evolution
    print("Echo Structure Evolution:")
    print("=" * 40)
    for step, structure in steps:
        print(f"Step {step}: {structure}")
    
    print("\nEcho ID tracker state:")
    print("=" * 40)
    for structure, id_val in echolisp.treeid.items():
        if structure:  # Skip empty structure for cleaner output
            echo_str = echolisp.tostr(structure)
            print(f"ID {id_val}: {structure} -> {echo_str}")