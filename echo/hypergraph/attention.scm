;
; attention.scm
;
; Attention allocation heuristics for Deep Tree Echo hypergraph system
; Implements semantic salience assessment and adaptive attention mechanisms
;
; This module provides the attention allocation mechanism that determines
; which repository files and patterns deserve cognitive focus.
;

(define-module (opencog hypergraph attention)
  #:use-module (opencog)
  #:use-module (opencog exec)
  #:use-module (srfi srfi-1)
  #:use-module (ice-9 regex)
  #:export (
    semantic-salience
    adaptive-attention
    filter-by-attention
  )
)

; ------------------------------------------------------
; Semantic Salience Assessment
; ------------------------------------------------------
; Assigns salience scores to files based on multiple heuristics:
; - Core directories and files
; - Recent changes and activity
; - Configured targets
; - File types and extensions

(define (semantic-salience path)
  "Calculate semantic salience score for a given file path.
   
   The salience score determines how much cognitive attention
   should be allocated to processing this file.
   
   Parameters:
     path - File path to evaluate
   
   Returns:
     A salience score between 0.0 and 1.0"
  (cond
    ; AtomSpace core files - highest priority
    ((string-match "AtomSpace\\.scm$" path) 0.95)
    ((string-match "atomspace" path) 0.92)
    
    ; Core directories - very high priority
    ((string-match "/core/" path) 0.90)
    ((string-match "/hypergraph/" path) 0.88)
    ((string-match "/model/" path) 0.85)
    
    ; Source code - high priority
    ((string-match "/src/" path) 0.80)
    ((string-match "\\.scm$" path) 0.78)
    ((string-match "\\.py$" path) 0.75)
    
    ; Documentation - medium-high priority
    ((string-match "README" path) 0.70)
    ((string-match "\\.md$" path) 0.65)
    
    ; Behavior and cognitive modules
    ((string-match "/behavior/" path) 0.75)
    ((string-match "/eva-model/" path) 0.80)
    
    ; Configuration files
    ((string-match "\\.json$" path) 0.60)
    ((string-match "\\.yml$" path) 0.58)
    ((string-match "\\.yaml$" path) 0.58)
    
    ; Test files
    ((string-match "test_" path) 0.55)
    ((string-match "_test\\." path) 0.55)
    
    ; Default salience for other files
    (else 0.50)))

; ------------------------------------------------------
; Adaptive Attention Allocation
; ------------------------------------------------------
; Dynamically adjusts attention thresholds based on:
; - Current cognitive load
; - Recent repository activity
; - Explicit configuration

(define (adaptive-attention current-load recent-activity)
  "Calculate adaptive attention threshold based on cognitive state.
   
   Higher cognitive load or lower recent activity leads to
   higher thresholds (processing fewer files).
   
   Parameters:
     current-load - Cognitive load (0.0 to 1.0)
     recent-activity - Recent activity level (0.0 to 1.0)
   
   Returns:
     Attention threshold (0.0 to 1.0)"
  (let ((base-threshold 0.50)
        (load-factor 0.30)
        (activity-adjustment 0.20))
    (+ base-threshold
       (* current-load load-factor)
       (- activity-adjustment recent-activity))))

; ------------------------------------------------------
; Attention-Based Filtering
; ------------------------------------------------------

(define (filter-by-attention paths attention-threshold)
  "Filter a list of paths based on attention threshold.
   
   Only paths with salience scores above the threshold are retained.
   
   Parameters:
     paths - List of file paths to filter
     attention-threshold - Minimum salience score (0.0 to 1.0)
   
   Returns:
     Filtered list of paths that meet the attention threshold"
  (filter
    (lambda (path)
      (> (semantic-salience path) attention-threshold))
    paths))
