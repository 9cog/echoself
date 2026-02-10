;
; hypergraph.scm
;
; Main hypergraph module for Deep Tree Echo
; Integrates all hypergraph encoding subsystems
;
; This is the top-level module that provides the complete
; hypergraph-encoded cognitive enhancement system inspired by
; the DeepTreeEcho-Eva Self Model Integration.
;

(define-module (opencog hypergraph)
  #:use-module (opencog)
  #:use-module (opencog exec)
  #:use-module (opencog hypergraph core)
  #:use-module (opencog hypergraph attention)
  #:use-module (opencog hypergraph repo-introspection)
  #:use-module (opencog hypergraph prompt-template)
  #:re-export (
    ; Core node operations
    make-hypergraph-node
    hypergraph-node-id
    hypergraph-node-type
    hypergraph-node-content
    hypergraph-node-links
    hypergraph-node?
    
    ; Attention allocation
    semantic-salience
    adaptive-attention
    filter-by-attention
    
    ; Repository introspection
    repo-file-list
    safe-read-file
    assemble-hypergraph-input
    hypergraph->string
    
    ; Prompt templates
    prompt-template
    inject-repo-input-into-prompt
    create-cognitive-prompt
  )
)

; Load all submodules
(load "hypergraph/core.scm")
(load "hypergraph/attention.scm")
(load "hypergraph/repo-introspection.scm")
(load "hypergraph/prompt-template.scm")
