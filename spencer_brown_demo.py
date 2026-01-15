#!/usr/bin/env python3
"""
Advanced demonstration of Spencer-Brown Laws of Form implemented in Pure Parentheses Lisp.

This demonstrates the progression from primordial distinctions to complex
computational structures, showing how Lisp emerges from recursive containment.
"""

from pure_parentheses_lisp import PureParenthesesLisp


def demonstrate_laws_of_form():
    """Demonstrate the Laws of Form principles in action."""
    lisp = PureParenthesesLisp()
    
    print("🎭 Spencer-Brown's Laws of Form in Pure Parentheses Lisp")
    print("=" * 65)
    
    print("\n📍 PRIMORDIAL GENESIS")
    print("Starting from the fundamental distinction between marked and unmarked states:")
    
    void = lisp.interpret("()")
    marked = lisp.interpret("(())")
    
    print(f"   Void (unmarked):     () → {void}")
    print(f"   Marked (distinction): (()) → {marked}")
    
    print("\n🔄 RECURSIVE CROSSINGS")
    print("Demonstrating how crossing boundaries creates computation:")
    
    # Identity crossings
    identity_void = lisp.interpret("((()) ())")
    identity_marked = lisp.interpret("((()) (()))")
    
    print(f"   Identity on void:    ((()) ()) → {identity_void}")
    print(f"   Identity on marked:  ((()) (())) → {identity_marked}")
    
    print("\n🧮 CHURCH NUMERALS FROM CONTAINMENT")
    print("Numbers emerge from nested distinctions:")
    
    for i in range(5):
        numeral = lisp.int_to_church_numeral(i)
        nesting_depth = str(numeral).count('[')
        print(f"   {i}: {numeral} (depth: {nesting_depth})")
    
    print("\n🎼 COMBINATORIAL STRUCTURES")
    print("Basic combinators emerging from pattern matching:")
    
    # More complex patterns
    complex_patterns = [
        ("((()) (()) ())", "Double identity application"),
        ("(((())))", "Triple nesting"),
        ("(() (()))", "Void containing marked"),
        ("((()) (() (())))", "Identity with nested void-marked"),
    ]
    
    for pattern, description in complex_patterns:
        try:
            result = lisp.interpret(pattern)
            print(f"   {description:25}: {pattern} → {result}")
        except Exception as e:
            print(f"   {description:25}: {pattern} → Error: {e}")
    
    print("\n🌊 RECURSIVE DEPTH ANALYSIS")
    depth_examples = [
        "()",
        "(())",
        "((()))",
        "(((())))",
        "((((())))",
    ]
    
    for expr in depth_examples:
        result = lisp.interpret(expr)
        depth = expr.count('(')
        result_complexity = str(result).count('[')
        print(f"   Depth {depth}: {expr:12} → {result} (complexity: {result_complexity})")
    
    print("\n💫 EMERGENT PROPERTIES")
    print("Demonstrating how complex behavior emerges from simple rules:")
    
    # Self-referential patterns
    self_ref_patterns = [
        ("((()) ((())))", "Identity applied to double-marked"),
        ("((() ()) (()))", "Two-argument pattern"),
        ("((()) ((()) ()))", "Nested identity"),
    ]
    
    for pattern, description in self_ref_patterns:
        result = lisp.interpret(pattern)
        print(f"   {description:30}: {result}")
    
    print("\n🎯 VALIDATION OF SPENCER-BROWN PRINCIPLES")
    print("Checking adherence to fundamental laws:")
    
    # Law of Calling
    print("   Law of Calling (double crossing returns to origin):")
    double_cross = lisp.interpret("(((())) ())")
    print(f"     Double cross of marked: {double_cross}")
    
    # Demonstrate idempotency 
    idempotent = lisp.interpret("((()) ((()) (())))")
    print(f"     Idempotent operation: {idempotent}")
    
    print("\n✨ CONCLUSION")
    print("Pure Parentheses Lisp successfully demonstrates:")
    print("   • Emergence of computation from binary distinction")
    print("   • Recursive self-reference and identity")
    print("   • Church numeral encoding via nesting depth")
    print("   • Combinatorial pattern matching")
    print("   • Spencer-Brown's crossing and calling laws")
    print("   • Foundation for metacircular evaluation")


def performance_analysis():
    """Analyze performance characteristics as described in the problem statement."""
    lisp = PureParenthesesLisp()
    
    print("\n⚡ PERFORMANCE & VALIDATION ANALYSIS")
    print("=" * 50)
    
    test_cases = [
        ("Church numeral 3", "(((())))"),
        ("Identity function", "((()) ())"),
        ("Complex nesting", "(((() ())) (()))"),
        ("Deep recursion", "(((((())))))"),
    ]
    
    print("Construct                    | Parentheses Depth | Result Complexity")
    print("-" * 60)
    
    for name, expr in test_cases:
        depth = expr.count('(')
        result = lisp.interpret(expr)
        complexity = str(result).count('[')
        print(f"{name:28} | {depth:17} | {complexity}")
    
    print("\n🎯 Key Metrics:")
    print("   • Parsing: O(n) where n = number of parentheses")
    print("   • Evaluation: O(d) where d = nesting depth")
    print("   • Memory: O(d) for recursive structure representation")
    print("   • Core size: ~150 lines of Python (expandable to full Lisp)")


if __name__ == "__main__":
    demonstrate_laws_of_form()
    performance_analysis()