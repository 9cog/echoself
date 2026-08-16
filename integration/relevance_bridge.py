"""
Relevance Realization Bridge for EchoSelf
==========================================

REST API wrapper enabling TypeScript services to interact with
the Python RelevanceRealizationEngine.

Extends the existing Python bridge with RR-specific endpoints.

Author: Deep Tree Echo
Date: June 2026
"""

from typing import Dict, Any, Optional, List
import json
import numpy as np
from dataclasses import asdict, dataclass, field
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import sys
import os
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from relevance_realization_engine import (
    RelevanceRealizationEngine,
    Possibility,
    RelevanceCriteria,
    OpponentProcess
)

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
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        return super().default(obj)


@dataclass
class RelevanceResult:
    """Result of relevance realization."""
    possibilities: List[Dict[str, Any]]
    filtered_count: int
    original_count: int
    opponent_states: Dict[str, float]
    processing_time_ms: float


class RelevanceBridge:
    """
    Bridge class managing RelevanceRealizationEngine for API access.
    Provides TypeScript-compatible interface to Python RR engine.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.engine = RelevanceRealizationEngine()
        self._request_count = 0
        self._total_processing_time = 0
        
        logger.info("RelevanceBridge initialized")
    
    def realize_relevance(
        self,
        possibilities: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> RelevanceResult:
        """
        Core relevance realization endpoint.
        
        Args:
            possibilities: List of possibility objects with id, data, and optional criteria
            context: Current situational context including goals, resources, etc.
            
        Returns:
            RelevanceResult with filtered and prioritized possibilities
        """
        start_time = time.time()
        self._request_count += 1
        
        # Convert dict possibilities to Possibility objects
        possibility_objects = []
        for p in possibilities:
            criteria = RelevanceCriteria(**p.get('criteria', {})) if p.get('criteria') else RelevanceCriteria()
            poss = Possibility(
                id=p.get('id', str(hash(str(p)))),
                data=p.get('data', p),
                criteria=criteria
            )
            possibility_objects.append(poss)
        
        # Run relevance realization
        relevant = self.engine.realize_relevance(possibility_objects, context)
        
        # Convert back to dicts
        result_possibilities = []
        for p in relevant:
            result_possibilities.append({
                'id': p.id,
                'data': p.data,
                'criteria': {
                    'goal_alignment': p.criteria.goal_alignment,
                    'predictive_power': p.criteria.predictive_power,
                    'cognitive_economy': p.criteria.cognitive_economy,
                    'novelty_value': p.criteria.novelty_value,
                    'contextual_fit': p.criteria.contextual_fit,
                    'score': p.criteria.score()
                },
                'constraints_satisfied': p.constraints_satisfied,
                'future_relevance': p.future_relevance
            })
        
        processing_time = (time.time() - start_time) * 1000
        self._total_processing_time += processing_time
        
        return RelevanceResult(
            possibilities=result_possibilities,
            filtered_count=len(relevant),
            original_count=len(possibilities),
            opponent_states=self.get_opponent_states(),
            processing_time_ms=processing_time
        )
    
    def feed_back(
        self,
        chosen: List[Dict[str, Any]],
        outcomes: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Provide feedback on relevance decisions for learning.
        
        Args:
            chosen: The possibilities that were selected
            outcomes: The outcomes of processing those possibilities
            
        Returns:
            Updated engine state after feedback
        """
        # Convert to Possibility objects
        possibility_objects = []
        for p in chosen:
            criteria = RelevanceCriteria(**p.get('criteria', {})) if p.get('criteria') else RelevanceCriteria()
            poss = Possibility(
                id=p.get('id', ''),
                data=p.get('data', {}),
                criteria=criteria
            )
            possibility_objects.append(poss)
        
        # Process outcomes
        outcome_objects = []
        for o in outcomes:
            outcome_objects.append(type('Outcome', (), {'success': o.get('success', True), **o})())
        
        # Feed back to engine
        self.engine.feed_back(possibility_objects, outcome_objects)
        
        return {
            'status': 'success',
            'opponent_states': self.get_opponent_states(),
            'history_size': len(self.engine.outcome_history)
        }
    
    def get_opponent_states(self) -> Dict[str, float]:
        """Get current opponent process balances."""
        return {
            'exploration_exploitation': self.engine.exploration_exploitation.balance,
            'breadth_depth': self.engine.breadth_depth.balance,
            'speed_accuracy': self.engine.speed_accuracy.balance,
            'certainty_openness': self.engine.certainty_openness.balance
        }
    
    def adjust_opponent_process(
        self,
        process_name: str,
        delta: float
    ) -> Dict[str, Any]:
        """
        Manually adjust an opponent process balance.
        
        Args:
            process_name: Name of the opponent process to adjust
            delta: Amount to shift (-1.0 to 1.0)
            
        Returns:
            Updated opponent process state
        """
        processes = {
            'exploration_exploitation': self.engine.exploration_exploitation,
            'breadth_depth': self.engine.breadth_depth,
            'speed_accuracy': self.engine.speed_accuracy,
            'certainty_openness': self.engine.certainty_openness
        }
        
        if process_name not in processes:
            return {'error': f'Unknown process: {process_name}', 'available': list(processes.keys())}
        
        process = processes[process_name]
        old_balance = process.balance
        process.shift(delta)
        
        return {
            'process': process_name,
            'old_balance': old_balance,
            'new_balance': process.balance,
            'delta': delta
        }
    
    def set_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Set the current context for relevance realization.
        
        Args:
            context: Context dictionary with goals, resources, cognitive_load, etc.
            
        Returns:
            Confirmation with context summary
        """
        self.engine.current_context.update(context)
        
        return {
            'status': 'success',
            'context_keys': list(self.engine.current_context.keys()),
            'goals_count': len(context.get('goals', []))
        }
    
    def get_state(self) -> Dict[str, Any]:
        """Get full engine state."""
        return {
            'opponent_states': self.get_opponent_states(),
            'context': self.engine.current_context,
            'history': {
                'relevance_history_size': len(self.engine.relevance_history),
                'processing_history_size': len(self.engine.processing_history),
                'outcome_history_size': len(self.engine.outcome_history)
            },
            'cost_functions': list(self.engine.cost_functions.keys()),
            'statistics': {
                'request_count': self._request_count,
                'total_processing_time_ms': self._total_processing_time
            }
        }
    
    def get_health(self) -> Dict[str, Any]:
        """Health check endpoint."""
        return {
            'status': 'healthy',
            'service': 'relevance-realization-bridge',
            'engine_ready': True,
            'request_count': self._request_count
        }


class RelevanceBridgeHandler(BaseHTTPRequestHandler):
    """HTTP request handler for Relevance Bridge API."""
    
    bridge: RelevanceBridge = None
    
    def log_message(self, format, *args):
        """Custom log format."""
        logger.info(f"{self.address_string()} - {format % args}")
    
    def _send_json_response(self, data: Any, status: int = 200):
        """Send JSON response."""
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        
        if hasattr(data, '__dict__'):
            data = asdict(data) if hasattr(data, '__dataclass_fields__') else data.__dict__
        
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
        try:
            if self.path == '/health' or self.path == '/':
                self._send_json_response(self.bridge.get_health())
            
            elif self.path == '/state':
                self._send_json_response(self.bridge.get_state())
            
            elif self.path == '/opponents':
                self._send_json_response(self.bridge.get_opponent_states())
            
            else:
                self._send_json_response({'error': 'Not found', 'path': self.path}, 404)
        
        except Exception as e:
            logger.error(f"GET error: {e}")
            self._send_json_response({'error': str(e)}, 500)
    
    def do_POST(self):
        """Handle POST requests."""
        try:
            data = self._parse_json_body()
            
            if self.path == '/realize':
                result = self.bridge.realize_relevance(
                    data.get('possibilities', []),
                    data.get('context')
                )
                self._send_json_response(result)
            
            elif self.path == '/feedback':
                result = self.bridge.feed_back(
                    data.get('chosen', []),
                    data.get('outcomes', [])
                )
                self._send_json_response(result)
            
            elif self.path == '/context':
                result = self.bridge.set_context(data.get('context', {}))
                self._send_json_response(result)
            
            elif self.path == '/opponent/adjust':
                result = self.bridge.adjust_opponent_process(
                    data.get('process', ''),
                    data.get('delta', 0)
                )
                self._send_json_response(result)
            
            else:
                self._send_json_response({'error': 'Not found', 'path': self.path}, 404)
        
        except Exception as e:
            logger.error(f"POST error: {e}")
            self._send_json_response({'error': str(e)}, 500)


def create_relevance_server(
    host: str = 'localhost',
    port: int = 8766,
    config: Optional[Dict[str, Any]] = None
) -> HTTPServer:
    """
    Create and return the Relevance Realization API server.
    
    Args:
        host: Server host
        port: Server port
        config: Optional configuration
        
    Returns:
        Configured HTTP server
    """
    RelevanceBridgeHandler.bridge = RelevanceBridge(config)
    
    server = HTTPServer((host, port), RelevanceBridgeHandler)
    logger.info(f"Relevance Realization API server created at {host}:{port}")
    
    return server


def start_relevance_server(
    host: str = 'localhost',
    port: int = 8766,
    config: Optional[Dict[str, Any]] = None,
    threaded: bool = True
) -> Optional[threading.Thread]:
    """
    Start the Relevance Realization API server.
    
    Args:
        host: Server host
        port: Server port
        config: Optional configuration
        threaded: Run in separate thread
        
    Returns:
        Thread if threaded, None otherwise
    """
    server = create_relevance_server(host, port, config)
    
    if threaded:
        thread = threading.Thread(target=server.serve_forever)
        thread.daemon = True
        thread.start()
        logger.info(f"Relevance Realization API server started in background on {host}:{port}")
        return thread
    else:
        logger.info(f"Relevance Realization API server starting on {host}:{port}")
        server.serve_forever()
        return None


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Relevance Realization API Server')
    parser.add_argument('--host', default='localhost', help='Server host')
    parser.add_argument('--port', type=int, default=8766, help='Server port')
    
    args = parser.parse_args()
    
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║          Relevance Realization Engine API Server                ║
║                     Deep Tree Echo                              ║
╠══════════════════════════════════════════════════════════════════╣
║  Endpoints:                                                      ║
║    GET  /health          - Health check                         ║
║    GET  /state           - Full engine state                    ║
║    GET  /opponents       - Opponent process states              ║
║    POST /realize         - Core relevance realization           ║
║    POST /feedback        - Feedback for learning                ║
║    POST /context         - Set current context                  ║
║    POST /opponent/adjust - Adjust opponent processes            ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    start_relevance_server(args.host, args.port, threaded=False)
