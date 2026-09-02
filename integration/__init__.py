"""
Integration Module for Echogenesis
===================================

This module provides integration utilities connecting Python
echogenesis components with external systems including:

- Python-TypeScript bridge (REST API)
- AtomSpace adapter (CogPrime compatibility)

Author: Deep Tree Echo
Date: June 2026
"""

from .python_bridge import (
    EchogenesisBridge,
    EchogenesisAPIHandler,
    create_api_server,
    start_api_server,
    NumpyEncoder
)

from .atomspace_adapter import (
    TruthValue,
    IndefiniteTruthValue,
    AttentionValue,
    NodeType,
    LinkType,
    Atom,
    Node,
    Link,
    CognitiveSchematic,
    AtomSpace,
    EchogenesisAtomSpaceAdapter,
    create_atomspace_adapter
)

__all__ = [
    # Python bridge
    'EchogenesisBridge',
    'EchogenesisAPIHandler',
    'create_api_server',
    'start_api_server',
    'NumpyEncoder',
    
    # AtomSpace adapter
    'TruthValue',
    'IndefiniteTruthValue',
    'AttentionValue',
    'NodeType',
    'LinkType',
    'Atom',
    'Node',
    'Link',
    'CognitiveSchematic',
    'AtomSpace',
    'EchogenesisAtomSpaceAdapter',
    'create_atomspace_adapter'
]

# Default ports
DEFAULT_API_PORT = 8765
