;
; prompt-template.scm
;
; Prompt template injection and neural-symbolic reasoning integration
; Implements the interface between hypergraph encoding and AI prompts
;
; This module provides the prompt template system for injecting
; repository introspection data into Deep Tree Echo cognitive processes.
;

(define-module (opencog hypergraph prompt-template)
  #:use-module (opencog)
  #:use-module (opencog exec)
  #:use-module (opencog hypergraph core)
  #:use-module (opencog hypergraph attention)
  #:use-module (opencog hypergraph repo-introspection)
  #:use-module (ice-9 format)
  #:export (
    prompt-template
    inject-repo-input-into-prompt
    create-cognitive-prompt
  )
)

; ------------------------------------------------------
; Prompt Template Function
; ------------------------------------------------------

(define (prompt-template input-content)
  "Create a DeepTreeEcho prompt with the given input content.
   
   Parameters:
     input-content - Content to include in the prompt
   
   Returns:
     Formatted prompt string"
  (format #f "DeepTreeEcho Prompt: ~%~a" input-content))

; ------------------------------------------------------
; Repository Input Injection
; ------------------------------------------------------

(define (inject-repo-input-into-prompt root attention-threshold)
  "Inject repository introspection data into prompt template.
   
   This is the main entry point for creating prompts with
   hypergraph-encoded repository context.
   
   Parameters:
     root - Root directory path
     attention-threshold - Minimum salience score (0.0 to 1.0)
   
   Returns:
     Complete prompt with repository context"
  (let ((nodes (assemble-hypergraph-input root attention-threshold)))
    (prompt-template
      (format #f "Inspect these repo files: ~a"
              (hypergraph->string nodes)))))

; ------------------------------------------------------
; Cognitive Prompt Creation
; ------------------------------------------------------

(define (create-cognitive-prompt root current-load recent-activity purpose)
  "Create a cognitive prompt with adaptive attention allocation.
   
   This higher-level function combines attention allocation with
   prompt creation for autonomous cognitive processing.
   
   Parameters:
     root - Root directory path
     current-load - Cognitive load (0.0 to 1.0)
     recent-activity - Recent activity level (0.0 to 1.0)
     purpose - Purpose string describing the cognitive task
   
   Returns:
     Complete prompt with adaptive repository context"
  (let* ((threshold (adaptive-attention current-load recent-activity))
         (nodes (assemble-hypergraph-input root threshold))
         (context (hypergraph->string nodes)))
    (format #f "DeepTreeEcho Cognitive Process~%~
                Purpose: ~a~%~
                Cognitive Load: ~a~%~
                Attention Threshold: ~a~%~
                Repository Context:~%~a"
            purpose current-load threshold context)))
