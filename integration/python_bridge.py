"""
Python-TypeScript Bridge for Echogenesis
==========================================

REST API wrapper enabling TypeScript services to interact with
Python echogenesis components.

Provides endpoints for:
- Adaptive dimensional embedding
- Optimal cognitive grip
- Perspectival knowing
- Wisdom cultivation
- Full echogenesis cycle

Author: Deep Tree Echo
Date: June 2026
"""

from typing import Dict, Any, Optional, List
import json
import numpy as np
from dataclasses import asdict
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from echogenesis import (
    ECHOGENESIS_CONFIG,
    initialize_echogenesis
)
from echogenesis.adaptive_embedding import AdaptiveDimensionalEmbedding, EmbeddingConfig
from echogenesis.optimal_grip import OptimalGrip, RelevanceContext
from echogenesis.perspectival_knowing import PerspectivalKnowing, FrameType
from echogenesis.wisdom_cultivation import WisdomCultivation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


class EchogenesisBridge:
    """
    Bridge class managing echogenesis components for API access.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or ECHOGENESIS_CONFIG
        
        # Initialize components
        self.embedding = AdaptiveDimensionalEmbedding(EmbeddingConfig(
            **self.config.get('embedding_architecture', {})
        ))
        self.optimal_grip = OptimalGrip(self.config.get('relevance_realization', {}))
        self.perspective = PerspectivalKnowing()
        self.wisdom = WisdomCultivation()
        
        # Full echogenesis core
        self.core = None  # Lazy initialization
        
        logger.info("EchogenesisBridge initialized")
    
    def _ensure_core(self):
        """Lazy initialize echogenesis core."""
        if self.core is None:
            self.core = initialize_echogenesis(self.config)
    
    # Embedding endpoints
    def adaptive_projection(
        self,
        data: np.ndarray,
        cognitive_load: float,
        attention_threshold: float,
        context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Perform adaptive dimensional projection."""
        projected = self.embedding.adaptive_projection(
            data, cognitive_load, attention_threshold, context
        )
        
        return {
            'projected': projected,
            'state': self.embedding.get_state()
        }
    
    def multi_scale_embed(
        self,
        data: np.ndarray,
        scales: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Create multi-scale embeddings."""
        from echogenesis.adaptive_embedding import EmbeddingScale
        
        if scales:
            scale_enums = [EmbeddingScale[s.upper()] for s in scales]
        else:
            scale_enums = None
        
        embeddings = self.embedding.multi_scale_embed(data, scale_enums)
        
        return {
            scale.value: emb for scale, emb in embeddings.items()
        }
    
    def create_embodiment_manifold(
        self,
        sensory: np.ndarray,
        motor: np.ndarray,
        cognitive: np.ndarray
    ) -> Dict[str, Any]:
        """Create unified embodiment manifold."""
        manifold = self.embedding.create_embodiment_manifold(
            sensory, motor, cognitive
        )
        
        return {
            'manifold': manifold,
            'dimension': manifold.shape[-1]
        }
    
    # Optimal grip endpoints
    def realize_relevance(
        self,
        possibilities: List[Dict[str, Any]],
        goals: Optional[List[Dict]] = None,
        constraints: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """Realize relevance across possibilities."""
        if goals:
            self.optimal_grip.set_context(goals=goals)
        if constraints:
            self.optimal_grip.set_context(constraints=constraints)
        
        ranked = self.optimal_grip.realize_relevance(possibilities)
        
        return {
            'ranked': ranked,
            'grip_quality': self.optimal_grip.get_grip_quality(),
            'state': self.optimal_grip.get_full_state()
        }
    
    def get_top_relevant(
        self,
        possibilities: List[Dict[str, Any]],
        k: int = 5
    ) -> Dict[str, Any]:
        """Get top-k most relevant possibilities."""
        top_k = self.optimal_grip.get_top_k(possibilities, k)
        
        return {
            'top_k': top_k,
            'grip_quality': self.optimal_grip.get_grip_quality()
        }
    
    # Perspectival knowing endpoints
    def switch_frame(
        self,
        frame_name: str,
        context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Switch cognitive frame."""
        success = self.perspective.switch_frame(frame_name, context)
        
        return {
            'success': success,
            'current_frame': self.perspective.get_current_frame(),
            'state': self.perspective.get_state()
        }
    
    def perceive(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perceive data through current frame."""
        perceived = self.perspective.perceive(data)
        
        return {
            'perceived': perceived,
            'frame': self.perspective.get_current_frame()
        }
    
    def see_as(
        self,
        data: Dict[str, Any],
        aspect: str,
        pattern_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """See data as particular aspect."""
        perceived = self.perspective.see_as(data, aspect, pattern_type)
        
        return {
            'perceived': perceived,
            'aspect': aspect
        }
    
    def get_available_frames(self) -> List[str]:
        """Get available cognitive frames."""
        return self.perspective.get_frame_types()
    
    # Wisdom cultivation endpoints
    def add_belief(
        self,
        id: str,
        content: str,
        confidence: float = 0.5
    ) -> Dict[str, Any]:
        """Add a belief to track."""
        self.wisdom.add_belief(id, content, confidence)
        
        return {
            'belief_id': id,
            'content': content,
            'confidence': confidence
        }
    
    def examine_self(self) -> Dict[str, Any]:
        """Perform Socratic self-examination."""
        insights = self.wisdom.examine_self()
        
        return {
            'insights': [{'question': i.question, 'discovery': i.discovery} for i in insights],
            'count': len(insights)
        }
    
    def detect_deceptions(self) -> Dict[str, Any]:
        """Scan beliefs for self-deception."""
        deceptions = self.wisdom.detect_deceptions()
        
        return {
            'deceptions': deceptions,
            'count': len(deceptions)
        }
    
    def cultivate_wisdom(self) -> Dict[str, Any]:
        """Run full wisdom cultivation cycle."""
        result = self.wisdom.cultivate()
        
        return {
            'result': result,
            'wisdom_score': self.wisdom.get_wisdom_score()
        }
    
    def get_wisdom_state(self) -> Dict[str, Any]:
        """Get wisdom cultivation state."""
        return self.wisdom.get_full_state()
    
    # Full echogenesis cycle
    def cognitive_cycle(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute complete cognitive cycle."""
        self._ensure_core()
        
        result = self.core.cognitive_cycle(input_data)
        
        return {
            'result': result,
            'state': self.core.get_state()
        }
    
    def get_full_state(self) -> Dict[str, Any]:
        """Get complete system state."""
        return {
            'embedding': self.embedding.get_state(),
            'grip': self.optimal_grip.get_full_state(),
            'perspective': self.perspective.get_state(),
            'wisdom': self.wisdom.get_full_state()
        }


class EchogenesisAPIHandler(BaseHTTPRequestHandler):
    """HTTP request handler for Echogenesis API."""
    
    bridge: EchogenesisBridge = None
    
    def _send_json_response(self, data: Dict, status: int = 200):
        """Send JSON response."""
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        response = json.dumps(data, cls=NumpyEncoder)
        self.wfile.write(response.encode())
    
    def _parse_json_body(self) -> Dict:
        """Parse JSON request body."""
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)
        return json.loads(body) if body else {}
    
    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def do_GET(self):
        """Handle GET requests."""
        if self.path == '/health':
            self._send_json_response({'status': 'healthy', 'service': 'echogenesis'})
        
        elif self.path == '/state':
            state = self.bridge.get_full_state()
            self._send_json_response(state)
        
        elif self.path == '/frames':
            frames = self.bridge.get_available_frames()
            self._send_json_response({'frames': frames})
        
        elif self.path == '/wisdom':
            wisdom = self.bridge.get_wisdom_state()
            self._send_json_response(wisdom)
        
        else:
            self._send_json_response({'error': 'Not found'}, 404)
    
    def do_POST(self):
        """Handle POST requests."""
        try:
            data = self._parse_json_body()
            
            if self.path == '/embedding/project':
                # Convert list to numpy array
                arr = np.array(data.get('data', []))
                result = self.bridge.adaptive_projection(
                    arr,
                    data.get('cognitive_load', 0.5),
                    data.get('attention_threshold', 0.5),
                    data.get('context')
                )
                self._send_json_response(result)
            
            elif self.path == '/embedding/multiscale':
                arr = np.array(data.get('data', []))
                result = self.bridge.multi_scale_embed(arr, data.get('scales'))
                self._send_json_response(result)
            
            elif self.path == '/embedding/manifold':
                result = self.bridge.create_embodiment_manifold(
                    np.array(data.get('sensory', [])),
                    np.array(data.get('motor', [])),
                    np.array(data.get('cognitive', []))
                )
                self._send_json_response(result)
            
            elif self.path == '/grip/realize':
                result = self.bridge.realize_relevance(
                    data.get('possibilities', []),
                    data.get('goals'),
                    data.get('constraints')
                )
                self._send_json_response(result)
            
            elif self.path == '/grip/top':
                result = self.bridge.get_top_relevant(
                    data.get('possibilities', []),
                    data.get('k', 5)
                )
                self._send_json_response(result)
            
            elif self.path == '/perspective/switch':
                result = self.bridge.switch_frame(
                    data.get('frame', 'analytical'),
                    data.get('context')
                )
                self._send_json_response(result)
            
            elif self.path == '/perspective/perceive':
                result = self.bridge.perceive(data.get('data', {}))
                self._send_json_response(result)
            
            elif self.path == '/perspective/see_as':
                result = self.bridge.see_as(
                    data.get('data', {}),
                    data.get('aspect', ''),
                    data.get('pattern_type')
                )
                self._send_json_response(result)
            
            elif self.path == '/wisdom/belief':
                result = self.bridge.add_belief(
                    data.get('id', ''),
                    data.get('content', ''),
                    data.get('confidence', 0.5)
                )
                self._send_json_response(result)
            
            elif self.path == '/wisdom/examine':
                result = self.bridge.examine_self()
                self._send_json_response(result)
            
            elif self.path == '/wisdom/deceptions':
                result = self.bridge.detect_deceptions()
                self._send_json_response(result)
            
            elif self.path == '/wisdom/cultivate':
                result = self.bridge.cultivate_wisdom()
                self._send_json_response(result)
            
            elif self.path == '/cycle':
                result = self.bridge.cognitive_cycle(data.get('input', {}))
                self._send_json_response(result)
            
            else:
                self._send_json_response({'error': 'Not found'}, 404)
        
        except Exception as e:
            logger.error(f"API error: {e}")
            self._send_json_response({'error': str(e)}, 500)


def create_api_server(
    host: str = 'localhost',
    port: int = 8765,
    config: Optional[Dict[str, Any]] = None
) -> HTTPServer:
    """
    Create and return the Echogenesis API server.
    
    Args:
        host: Server host
        port: Server port
        config: Optional echogenesis configuration
        
    Returns:
        Configured HTTP server
    """
    # Initialize bridge
    EchogenesisAPIHandler.bridge = EchogenesisBridge(config)
    
    server = HTTPServer((host, port), EchogenesisAPIHandler)
    logger.info(f"Echogenesis API server created at {host}:{port}")
    
    return server


def start_api_server(
    host: str = 'localhost',
    port: int = 8765,
    config: Optional[Dict[str, Any]] = None,
    threaded: bool = True
) -> Optional[threading.Thread]:
    """
    Start the Echogenesis API server.
    
    Args:
        host: Server host
        port: Server port
        config: Optional echogenesis configuration
        threaded: Run in separate thread
        
    Returns:
        Thread if threaded, None otherwise
    """
    server = create_api_server(host, port, config)
    
    if threaded:
        thread = threading.Thread(target=server.serve_forever)
        thread.daemon = True
        thread.start()
        logger.info(f"Echogenesis API server started in background on {host}:{port}")
        return thread
    else:
        logger.info(f"Echogenesis API server starting on {host}:{port}")
        server.serve_forever()
        return None


if __name__ == '__main__':
    # Run server directly
    import argparse
    
    parser = argparse.ArgumentParser(description='Echogenesis API Server')
    parser.add_argument('--host', default='localhost', help='Server host')
    parser.add_argument('--port', type=int, default=8765, help='Server port')
    
    args = parser.parse_args()
    
    start_api_server(args.host, args.port, threaded=False)
