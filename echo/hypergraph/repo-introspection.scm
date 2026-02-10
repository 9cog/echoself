;
; repo-introspection.scm
;
; Repository introspection and hypergraph encoding for Deep Tree Echo
; Implements recursive repository traversal and adaptive input assembly
;
; This module provides the core repository introspection capabilities
; for building hypergraph representations of the codebase.
;

(define-module (opencog hypergraph repo-introspection)
  #:use-module (opencog)
  #:use-module (opencog exec)
  #:use-module (opencog hypergraph core)
  #:use-module (opencog hypergraph attention)
  #:use-module (srfi srfi-1)
  #:use-module (ice-9 ftw)
  #:use-module (ice-9 rdelim)
  #:export (
    repo-file-list
    safe-read-file
    assemble-hypergraph-input
    hypergraph->string
  )
)

; ------------------------------------------------------
; Configuration Constants
; ------------------------------------------------------

(define MAX-FILE-SIZE 50000) ; 50 KB - maximum file size to read

; ------------------------------------------------------
; Recursive Repository Traversal
; ------------------------------------------------------

(define (repo-file-list root attention-threshold)
  "Recursively traverse repository and filter files by attention.
   
   Parameters:
     root - Root directory path
     attention-threshold - Minimum salience score (0.0 to 1.0)
   
   Returns:
     List of file paths meeting attention criteria"
  (let ((all-files '()))
    ; Use file-system-tree to walk the directory
    (file-system-tree root
      (lambda (path stat result)
        ; Enter predicate - skip .git and other hidden dirs
        (not (or (string-match "/\\." path)
                 (string-match "node_modules" path)
                 (string-match "__pycache__" path))))
      (lambda (path stat result)
        ; Leaf predicate - regular files only
        (eq? (stat:type stat) 'regular))
      (lambda (path stat result)
        ; Down predicate
        result)
      (lambda (path stat result)
        ; Up predicate
        result)
      (lambda (path stat result)
        ; Skip predicate
        result)
      (lambda (path stat errno result)
        ; Error predicate - skip files we can't access
        result)
      '())
    
    ; Get all regular files
    (let* ((file-list (find-files root))
           (filtered (filter-by-attention file-list attention-threshold)))
      filtered)))

(define (find-files dir)
  "Helper function to find all files recursively.
   
   Parameters:
     dir - Directory to search
   
   Returns:
     List of file paths"
  (let ((result '()))
    (ftw dir
      (lambda (path statinfo flag)
        (case flag
          ((regular)
           (set! result (cons path result))
           #t)
          ((directory)
           ; Skip hidden directories and build artifacts
           (not (or (string-match "/\\." path)
                    (string-match "node_modules" path)
                    (string-match "__pycache__" path))))
          (else #t))))
    (reverse result)))

; ------------------------------------------------------
; Adaptive File Reading
; ------------------------------------------------------

(define (safe-read-file path)
  "Read file content with size constraints.
   
   Files exceeding MAX-FILE-SIZE are summarized or omitted.
   
   Parameters:
     path - File path to read
   
   Returns:
     File content string or summary message"
  (catch #t
    (lambda ()
      (let* ((stat (stat path))
             (size (stat:size stat)))
        (if (< size MAX-FILE-SIZE)
            (call-with-input-file path
              (lambda (port)
                (read-string port)))
            "[File too large, summarized or omitted]")))
    (lambda (key . args)
      (format #f "[Error reading file: ~a]" key))))

; ------------------------------------------------------
; Hypergraph Input Assembly
; ------------------------------------------------------

(define (assemble-hypergraph-input root attention-threshold)
  "Assemble hypergraph-encoded representation of repository files.
   
   Parameters:
     root - Root directory path
     attention-threshold - Minimum salience score (0.0 to 1.0)
   
   Returns:
     List of hypergraph nodes representing repository files"
  (let ((files (repo-file-list root attention-threshold)))
    (map
      (lambda (path)
        (make-hypergraph-node
          path              ; id
          'file             ; type
          (safe-read-file path)  ; content
          '()))             ; links (empty for now)
      files)))

; ------------------------------------------------------
; Hypergraph Serialization
; ------------------------------------------------------

(define (hypergraph->string nodes)
  "Serialize hypergraph nodes to string representation.
   
   Parameters:
     nodes - List of hypergraph nodes
   
   Returns:
     String representation of the hypergraph"
  (apply string-append
    (map
      (lambda (node)
        (format #f "~%(file \"~a\" \"~a\")"
                (hypergraph-node-id node)
                (hypergraph-node-content node)))
      nodes)))
