;
; example-usage.scm
;
; Example usage of the Deep Tree Echo hypergraph encoding system
;
; This script demonstrates how to use the hypergraph system for
; repository introspection and cognitive prompt generation.
;

; Load required modules
(use-modules (opencog)
             (opencog exec)
             (opencog hypergraph))

; ------------------------------------------------------
; Example 1: Basic Hypergraph Node Creation
; ------------------------------------------------------

(display "\n=== Example 1: Basic Hypergraph Node ===\n")

(define example-node 
  (make-hypergraph-node 
    "example-id"           ; ID
    'concept               ; Type
    "Example content"      ; Content
    '()))                  ; Links

(display "Created node: ")
(display example-node)
(newline)

(display "Node ID: ")
(display (hypergraph-node-id example-node))
(newline)

(display "Node type: ")
(display (hypergraph-node-type example-node))
(newline)

; ------------------------------------------------------
; Example 2: Semantic Salience Scoring
; ------------------------------------------------------

(display "\n=== Example 2: Semantic Salience ===\n")

(define test-paths
  '("./echo/model/AtomSpace.scm"
    "./echo/hypergraph/core.scm"
    "./src/main.py"
    "./README.md"
    "./tests/test_basic.py"
    "./config.json"))

(for-each
  (lambda (path)
    (format #t "~a: ~a~%" path (semantic-salience path)))
  test-paths)

; ------------------------------------------------------
; Example 3: Adaptive Attention Allocation
; ------------------------------------------------------

(display "\n=== Example 3: Adaptive Attention ===\n")

(define cognitive-loads '(0.2 0.5 0.8))
(define activities '(0.1 0.5 0.9))

(display "Attention thresholds for different cognitive states:\n")
(for-each
  (lambda (load)
    (for-each
      (lambda (activity)
        (format #t "Load: ~a, Activity: ~a => Threshold: ~a~%"
                load activity (adaptive-attention load activity)))
      activities))
  cognitive-loads)

; ------------------------------------------------------
; Example 4: Repository Introspection
; ------------------------------------------------------

(display "\n=== Example 4: Repository Introspection ===\n")

; Uncomment to run actual repository introspection
; (Note: This will scan the actual repository)
;
; (define repo-root "./echo/hypergraph")
; (define attention-threshold 0.75)
; 
; (display "Files meeting attention threshold:\n")
; (define file-list (repo-file-list repo-root attention-threshold))
; (for-each
;   (lambda (path)
;     (format #t "  - ~a (salience: ~a)~%" 
;             path (semantic-salience path)))
;   file-list)

; ------------------------------------------------------
; Example 5: Hypergraph Input Assembly
; ------------------------------------------------------

(display "\n=== Example 5: Hypergraph Assembly ===\n")

; Uncomment to assemble hypergraph from repository
;
; (define hypergraph-nodes 
;   (assemble-hypergraph-input repo-root attention-threshold))
; 
; (display "Number of nodes: ")
; (display (length hypergraph-nodes))
; (newline)
; 
; (display "\nFirst node:\n")
; (display (car hypergraph-nodes))
; (newline)

; ------------------------------------------------------
; Example 6: Cognitive Prompt Generation
; ------------------------------------------------------

(display "\n=== Example 6: Cognitive Prompt ===\n")

; Uncomment to generate a cognitive prompt
;
; (define cognitive-prompt
;   (create-cognitive-prompt
;     "./echo/hypergraph"    ; Root directory
;     0.3                    ; Current cognitive load
;     0.7                    ; Recent activity
;     "Analyze hypergraph implementation patterns"))
; 
; (display cognitive-prompt)
; (newline)

; ------------------------------------------------------
; Example 7: Simple Prompt Template
; ------------------------------------------------------

(display "\n=== Example 7: Simple Prompt Template ===\n")

(define simple-prompt
  (prompt-template "Analyze the cognitive architecture patterns"))

(display simple-prompt)
(newline)

(display "\n=== Examples Complete ===\n")
