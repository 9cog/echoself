;
; core.scm
;
; Core hypergraph data structures and operations for Deep Tree Echo
; Implements hypergraph node representation and basic operations
;
; Based on the DeepTreeEcho-Eva Self Model Integration
; This module provides the foundational hypergraph encoding for
; neural-symbolic reasoning and cognitive pattern recognition.
;

(define-module (opencog hypergraph core)
  #:use-module (opencog)
  #:use-module (opencog exec)
  #:export (
    make-hypergraph-node
    hypergraph-node-id
    hypergraph-node-type
    hypergraph-node-content
    hypergraph-node-links
    hypergraph-node?
  )
)

; ------------------------------------------------------
; Hypergraph Node Representation
; ------------------------------------------------------
; A hypergraph node is represented as a list with the following structure:
; (node id type content links)
;   - id: unique identifier for the node
;   - type: the type of the node (e.g., 'file, 'concept, 'pattern)
;   - content: the actual content/data of the node
;   - links: list of connections to other nodes

(define (make-hypergraph-node id type content links)
  "Create a hypergraph node with the specified properties.
   
   Parameters:
     id - Unique identifier for the node
     type - Node type (symbol: 'file, 'concept, 'pattern, etc.)
     content - Node content (any data)
     links - List of links to other nodes
   
   Returns:
     A list representing the hypergraph node"
  (list 'node id type content links))

(define (hypergraph-node? obj)
  "Check if an object is a valid hypergraph node.
   
   Parameters:
     obj - Object to check
   
   Returns:
     #t if obj is a hypergraph node, #f otherwise"
  (and (list? obj)
       (>= (length obj) 5)
       (eq? (car obj) 'node)))

(define (hypergraph-node-id node)
  "Extract the ID from a hypergraph node.
   
   Parameters:
     node - A hypergraph node
   
   Returns:
     The node's ID"
  (if (hypergraph-node? node)
      (cadr node)
      (error "Not a valid hypergraph node")))

(define (hypergraph-node-type node)
  "Extract the type from a hypergraph node.
   
   Parameters:
     node - A hypergraph node
   
   Returns:
     The node's type"
  (if (hypergraph-node? node)
      (caddr node)
      (error "Not a valid hypergraph node")))

(define (hypergraph-node-content node)
  "Extract the content from a hypergraph node.
   
   Parameters:
     node - A hypergraph node
   
   Returns:
     The node's content"
  (if (hypergraph-node? node)
      (cadddr node)
      (error "Not a valid hypergraph node")))

(define (hypergraph-node-links node)
  "Extract the links from a hypergraph node.
   
   Parameters:
     node - A hypergraph node
   
   Returns:
     The node's links list"
  (if (hypergraph-node? node)
      (list-ref node 4)
      (error "Not a valid hypergraph node")))
