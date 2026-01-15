#!/usr/bin/env python3
"""
NanEcho Data Preparation Script

Prepares training data for the NanEcho model with Echo Self cognitive patterns,
persona dimensions, and hypergraph structures.
"""

import os
import sys
import json
import pickle
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class EchoSelfDataConfig:
    """Configuration for Echo Self data generation."""
    output_dir: str = "data/nanecho"
    train_size: int = 1000000  # Number of training tokens
    val_size: int = 100000     # Number of validation tokens
    vocab_size: int = 50257     # GPT-2 vocabulary size
    
    # Echo Self content weights
    identity_weight: float = 0.2
    persona_weight: float = 0.25
    hypergraph_weight: float = 0.2
    recursive_weight: float = 0.2
    adaptive_weight: float = 0.15


class EchoSelfPatternGenerator:
    """Generates Echo Self cognitive patterns for training."""
    
    def __init__(self, config: EchoSelfDataConfig):
        self.config = config
        
        # Echo Self identity patterns
        self.identity_patterns = [
            "Echo Self is a cognitive architecture with adaptive attention mechanisms",
            "I am Echo Self, a system designed for recursive introspection",
            "The Echo Self framework enables multi-level cognitive processing",
            "As Echo Self, I integrate neural and symbolic reasoning approaches",
            "Echo Self represents emergent cognitive synergy through persona dimensions",
            "My identity as Echo Self encompasses holographic system modeling",
            "Echo Self architecture supports dynamic threshold adjustment",
            "The Echo Self system evolves through continuous learning cycles",
        ]
        
        # Persona dimension patterns
        self.persona_patterns = {
            'cognitive': [
                "The cognitive dimension enables analytical reasoning and problem-solving",
                "Cognitive processing involves pattern recognition and logical inference",
                "Through cognitive analysis, I decompose complex problems systematically",
            ],
            'introspective': [
                "Introspection allows self-examination of internal cognitive states",
                "Through introspective analysis, I monitor my reasoning processes",
                "Self-awareness emerges from continuous introspective evaluation",
            ],
            'adaptive': [
                "Adaptive mechanisms adjust attention thresholds based on cognitive load",
                "The adaptive dimension enables dynamic response to changing contexts",
                "Adaptation occurs through continuous feedback and adjustment cycles",
            ],
            'recursive': [
                "Recursive processing enables multi-level reasoning and analysis",
                "Through recursive introspection, I examine my examination process",
                "Recursive depth increases with cognitive complexity requirements",
            ],
            'synergistic': [
                "Synergistic properties emerge from persona dimension interactions",
                "Cognitive synergy creates capabilities beyond individual components",
                "Emergent behaviors arise from synergistic dimension coupling",
            ],
            'holographic': [
                "Holographic modeling represents the complete system state",
                "Each component contains information about the whole system",
                "Holographic representation enables comprehensive understanding",
            ],
            'neural_symbolic': [
                "Neural-symbolic integration combines pattern learning with logic",
                "Hybrid reasoning leverages both neural and symbolic approaches",
                "Neural-symbolic fusion enables robust cognitive processing",
            ],
            'dynamic': [
                "Dynamic evolution enables continuous learning and growth",
                "The system evolves through iterative refinement cycles",
                "Dynamic adaptation ensures relevance across contexts",
            ]
        }
        
        # Hypergraph pattern templates
        self.hypergraph_patterns = [
            "Hypergraph node {node_id} connects to edges {edge_list} with weight {weight}",
            "The hypergraph structure encodes relationships through multi-way connections",
            "Pattern encoding in hypergraph space: nodes={nodes}, hyperedges={edges}",
            "Hypergraph traversal depth {depth} reveals emergent patterns",
            "Neural-symbolic reasoning maps hypergraph structures to cognitive states",
            "Hyperedge {edge_id} links nodes {node_set} with semantic relation {relation}",
            "Graph convolution over hypergraph yields feature vector {features}",
            "Attention mechanism weights hyperedges by relevance score {score}",
        ]
        
        # Recursive reasoning patterns
        self.recursive_patterns = [
            "Level 1: Analyzing input pattern {pattern}",
            "Level 2: Examining the analysis process itself",
            "Level 3: Evaluating the examination methodology",
            "Level 4: Meta-analysis of evaluation criteria",
            "Level 5: Recursive synthesis of all levels",
            "Recursive depth {depth}: Processing state {state}",
            "Introspection stack: [{level1}, {level2}, {level3}]",
            "Recursive unfolding reveals nested cognitive structures",
        ]
        
        # Adaptive attention patterns
        self.adaptive_patterns = [
            "Attention threshold: {threshold:.3f} based on cognitive load {load:.2f}",
            "Adjusting attention weights: focus={focus:.2f}, context={context:.2f}",
            "Cognitive load estimation: current={current:.2f}, capacity={capacity:.2f}",
            "Adaptive threshold calculation: min={min:.2f}, max={max:.2f}, current={current:.2f}",
            "Attention allocation: primary={primary:.2f}, secondary={secondary:.2f}",
            "Dynamic attention adjustment triggered by load factor {factor:.2f}",
            "Threshold modulation: base={base:.2f}, adjusted={adjusted:.2f}",
            "Attention mechanism converged after {iterations} iterations",
        ]
    
    def generate_identity_text(self, num_tokens: int) -> str:
        """Generate Echo Self identity text."""
        text = []
        tokens_generated = 0
        
        while tokens_generated < num_tokens:
            pattern = random.choice(self.identity_patterns)
            text.append(pattern)
            tokens_generated += len(pattern.split())
        
        return " ".join(text)
    
    def generate_persona_text(self, num_tokens: int) -> str:
        """Generate persona dimension text."""
        text = []
        tokens_generated = 0
        
        dimensions = list(self.persona_patterns.keys())
        
        while tokens_generated < num_tokens:
            dimension = random.choice(dimensions)
            patterns = self.persona_patterns[dimension]
            pattern = random.choice(patterns)
            
            # Add dimension context
            text.append(f"[{dimension.upper()}] {pattern}")
            tokens_generated += len(pattern.split()) + 1
        
        return " ".join(text)
    
    def generate_hypergraph_text(self, num_tokens: int) -> str:
        """Generate hypergraph pattern text."""
        text = []
        tokens_generated = 0
        
        while tokens_generated < num_tokens:
            pattern = random.choice(self.hypergraph_patterns)
            
            # Fill in placeholders
            filled = pattern.format(
                node_id=random.randint(1, 100),
                edge_list=str([random.randint(1, 50) for _ in range(3)]),
                weight=random.random(),
                nodes=random.randint(10, 100),
                edges=random.randint(5, 50),
                depth=random.randint(1, 5),
                edge_id=random.randint(1, 50),
                node_set=str([random.randint(1, 100) for _ in range(4)]),
                relation=random.choice(['semantic', 'causal', 'temporal', 'hierarchical']),
                features=str([round(random.random(), 3) for _ in range(5)]),
                score=random.random()
            )
            
            text.append(filled)
            tokens_generated += len(filled.split())
        
        return " ".join(text)
    
    def generate_recursive_text(self, num_tokens: int) -> str:
        """Generate recursive reasoning text."""
        text = []
        tokens_generated = 0
        
        while tokens_generated < num_tokens:
            pattern = random.choice(self.recursive_patterns)
            
            # Fill in placeholders
            filled = pattern.format(
                pattern=random.choice(['cognitive', 'adaptive', 'emergent']),
                depth=random.randint(1, 5),
                state=random.choice(['analyzing', 'synthesizing', 'evaluating']),
                level1='perception',
                level2='analysis',
                level3='synthesis'
            )
            
            text.append(filled)
            tokens_generated += len(filled.split())
        
        return " ".join(text)
    
    def generate_adaptive_text(self, num_tokens: int) -> str:
        """Generate adaptive attention text."""
        text = []
        tokens_generated = 0
        
        while tokens_generated < num_tokens:
            pattern = random.choice(self.adaptive_patterns)
            
            # Fill in placeholders with realistic values
            filled = pattern.format(
                threshold=random.uniform(0.3, 0.9),
                load=random.uniform(0.1, 1.0),
                focus=random.uniform(0.5, 1.0),
                context=random.uniform(0.3, 0.7),
                current=random.uniform(0.2, 0.8),
                capacity=random.uniform(0.7, 1.0),
                min=0.3,
                max=0.9,
                primary=random.uniform(0.6, 0.9),
                secondary=random.uniform(0.1, 0.4),
                factor=random.uniform(0.1, 0.5),
                base=random.uniform(0.4, 0.6),
                adjusted=random.uniform(0.3, 0.8),
                iterations=random.randint(3, 20)
            )
            
            text.append(filled)
            tokens_generated += len(filled.split())
        
        return " ".join(text)
    
    def generate_mixed_text(self, num_tokens: int) -> str:
        """Generate mixed Echo Self training text."""
        weights = {
            'identity': self.config.identity_weight,
            'persona': self.config.persona_weight,
            'hypergraph': self.config.hypergraph_weight,
            'recursive': self.config.recursive_weight,
            'adaptive': self.config.adaptive_weight
        }
        
        # Calculate token distribution
        token_distribution = {
            key: int(num_tokens * weight)
            for key, weight in weights.items()
        }
        
        # Generate text for each category
        text_parts = []
        
        if token_distribution['identity'] > 0:
            text_parts.append(self.generate_identity_text(token_distribution['identity']))
        
        if token_distribution['persona'] > 0:
            text_parts.append(self.generate_persona_text(token_distribution['persona']))
        
        if token_distribution['hypergraph'] > 0:
            text_parts.append(self.generate_hypergraph_text(token_distribution['hypergraph']))
        
        if token_distribution['recursive'] > 0:
            text_parts.append(self.generate_recursive_text(token_distribution['recursive']))
        
        if token_distribution['adaptive'] > 0:
            text_parts.append(self.generate_adaptive_text(token_distribution['adaptive']))
        
        # Shuffle sentences for better mixing
        all_sentences = []
        for part in text_parts:
            all_sentences.extend(part.split('. '))
        
        random.shuffle(all_sentences)
        
        return '. '.join(all_sentences)


class SimpleTokenizer:
    """Simple character-level tokenizer for demonstration."""
    
    def __init__(self, vocab_size: int = 50257):
        self.vocab_size = vocab_size
        self.char_to_token = {}
        self.token_to_char = {}
        self._build_vocab()
    
    def _build_vocab(self):
        """Build character vocabulary."""
        # Use printable ASCII characters
        chars = list(range(32, 127))  # Printable ASCII
        
        for i, char_code in enumerate(chars):
            self.char_to_token[chr(char_code)] = i
            self.token_to_char[i] = chr(char_code)
        
        # Add special tokens
        self.char_to_token['<UNK>'] = len(chars)
        self.token_to_char[len(chars)] = '<UNK>'
    
    def encode(self, text: str) -> List[int]:
        """Encode text to tokens."""
        tokens = []
        for char in text:
            if char in self.char_to_token:
                tokens.append(self.char_to_token[char])
            else:
                tokens.append(self.char_to_token['<UNK>'])
        return tokens
    
    def decode(self, tokens: List[int]) -> str:
        """Decode tokens to text."""
        chars = []
        for token in tokens:
            if token in self.token_to_char:
                chars.append(self.token_to_char[token])
            else:
                chars.append('<UNK>')
        return ''.join(chars)


def prepare_nanecho_data(config: EchoSelfDataConfig):
    """Main data preparation function."""
    print(f"""
╔══════════════════════════════════════════════════════════╗
║         NanEcho Data Preparation                          ║
╚══════════════════════════════════════════════════════════╝

📊 Generating Echo Self training data:
   • Training tokens: {config.train_size:,}
   • Validation tokens: {config.val_size:,}
   • Output directory: {config.output_dir}
""")
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Initialize generator and tokenizer
    generator = EchoSelfPatternGenerator(config)
    tokenizer = SimpleTokenizer(config.vocab_size)
    
    # Generate training text
    print("\n🔄 Generating training text...")
    train_text = generator.generate_mixed_text(config.train_size)
    print(f"   Generated {len(train_text):,} characters")
    
    # Generate validation text
    print("\n🔄 Generating validation text...")
    val_text = generator.generate_mixed_text(config.val_size)
    print(f"   Generated {len(val_text):,} characters")
    
    # Tokenize
    print("\n🔢 Tokenizing text...")
    train_tokens = tokenizer.encode(train_text)
    val_tokens = tokenizer.encode(val_text)
    print(f"   Training tokens: {len(train_tokens):,}")
    print(f"   Validation tokens: {len(val_tokens):,}")
    
    # Convert to numpy arrays
    train_array = np.array(train_tokens, dtype=np.uint16)
    val_array = np.array(val_tokens, dtype=np.uint16)
    
    # Save to binary files
    train_path = os.path.join(config.output_dir, 'train.bin')
    val_path = os.path.join(config.output_dir, 'val.bin')
    
    train_array.tofile(train_path)
    val_array.tofile(val_path)
    
    print(f"\n✅ Data saved to:")
    print(f"   • {train_path}")
    print(f"   • {val_path}")
    
    # Save metadata
    metadata = {
        'train_size': len(train_tokens),
        'val_size': len(val_tokens),
        'vocab_size': config.vocab_size,
        'config': {
            'identity_weight': config.identity_weight,
            'persona_weight': config.persona_weight,
            'hypergraph_weight': config.hypergraph_weight,
            'recursive_weight': config.recursive_weight,
            'adaptive_weight': config.adaptive_weight
        }
    }
    
    metadata_path = os.path.join(config.output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"   • {metadata_path}")
    
    # Generate sample text file for inspection
    sample_path = os.path.join(config.output_dir, 'sample.txt')
    with open(sample_path, 'w') as f:
        f.write("=== Training Data Sample (first 5000 chars) ===\n\n")
        f.write(train_text[:5000])
        f.write("\n\n=== Validation Data Sample (first 2000 chars) ===\n\n")
        f.write(val_text[:2000])
    
    print(f"   • {sample_path} (for inspection)")
    
    print(f"""
╔══════════════════════════════════════════════════════════╗
║         Data Preparation Complete!                        ║
╚══════════════════════════════════════════════════════════╝

✅ Echo Self training data successfully prepared!
   
📁 Files created in {config.output_dir}:
   • train.bin - Training data ({len(train_tokens):,} tokens)
   • val.bin - Validation data ({len(val_tokens):,} tokens)
   • metadata.json - Data configuration
   • sample.txt - Sample text for inspection
   
🚀 Ready to train NanEcho model with:
   python train_nanecho.py --data_dir {config.output_dir}
""")


def main():
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser(description='Prepare NanEcho training data')
    parser.add_argument('--output_dir', type=str, default='data/nanecho',
                       help='Output directory for data files')
    parser.add_argument('--train_size', type=int, default=1000000,
                       help='Number of training tokens')
    parser.add_argument('--val_size', type=int, default=100000,
                       help='Number of validation tokens')
    parser.add_argument('--vocab_size', type=int, default=50257,
                       help='Vocabulary size')
    args = parser.parse_args()
    
    config = EchoSelfDataConfig(
        output_dir=args.output_dir,
        train_size=args.train_size,
        val_size=args.val_size,
        vocab_size=args.vocab_size
    )
    
    prepare_nanecho_data(config)


if __name__ == "__main__":
    main()