"""
Data preprocessor for Deep Tree Echo training
"""

import torch
import numpy as np
from typing import Dict, Any, List, Optional
import random
import re


class DataPreprocessor:
    """
    Preprocessor for text data, including tokenization and augmentation.
    """
    
    def __init__(
        self,
        preprocessing_config: Dict[str, Any],
        augmentation_config: Dict[str, Any]
    ):
        self.preprocessing_config = preprocessing_config
        self.augmentation_config = augmentation_config
        
        # Initialize tokenizer
        self.tokenizer_name = preprocessing_config.get('tokenizer', 'gpt2')
        self.max_length = preprocessing_config.get('max_length', 2048)
        self.add_special_tokens = preprocessing_config.get('add_special_tokens', True)
        
        # Initialize tokenizer (using tiktoken for GPT-2 compatibility)
        self._init_tokenizer()
        
        # Augmentation settings
        self.enable_augmentation = augmentation_config.get('enable', False)
        self.augmentation_techniques = augmentation_config.get('techniques', [])
    
    def _init_tokenizer(self):
        """Initialize the tokenizer."""
        try:
            import tiktoken
            self.tokenizer = tiktoken.get_encoding("gpt2")
        except ImportError:
            # Fallback to simple character-level tokenization
            print("Warning: tiktoken not available, using simple tokenization")
            self.tokenizer = SimpleTokenizer()
    
    def process_texts(self, texts: List[str]) -> np.ndarray:
        """
        Process a list of texts into tokenized sequences.
        
        Args:
            texts: List of text strings
            
        Returns:
            Numpy array of token IDs
        """
        all_tokens = []
        
        for text in texts:
            # Apply augmentation if enabled
            if self.enable_augmentation and self.training:
                text = self._augment_text(text)
            
            # Tokenize
            tokens = self.tokenize(text)
            all_tokens.extend(tokens)
        
        return np.array(all_tokens, dtype=np.uint16)
    
    def tokenize(self, text: str) -> List[int]:
        """
        Tokenize a single text.
        
        Args:
            text: Input text
            
        Returns:
            List of token IDs
        """
        if hasattr(self.tokenizer, 'encode'):
            tokens = self.tokenizer.encode(text)
        else:
            tokens = self.tokenizer.tokenize(text)
        
        # Truncate if necessary
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        
        return tokens
    
    def _augment_text(self, text: str) -> str:
        """
        Apply data augmentation techniques to text.
        
        Args:
            text: Input text
            
        Returns:
            Augmented text
        """
        augmented = text
        
        for technique in self.augmentation_techniques:
            if isinstance(technique, dict):
                name = list(technique.keys())[0]
                prob = technique[name]
                
                if random.random() < prob:
                    if name == 'token_masking':
                        augmented = self._mask_tokens(augmented, prob)
                    elif name == 'span_masking':
                        augmented = self._mask_spans(augmented, prob)
                    elif name == 'sentence_permutation':
                        augmented = self._permute_sentences(augmented)
        
        return augmented
    
    def _mask_tokens(self, text: str, mask_prob: float) -> str:
        """Randomly mask tokens in the text."""
        words = text.split()
        masked = []
        
        for word in words:
            if random.random() < mask_prob:
                masked.append('[MASK]')
            else:
                masked.append(word)
        
        return ' '.join(masked)
    
    def _mask_spans(self, text: str, span_prob: float) -> str:
        """Mask continuous spans of tokens."""
        words = text.split()
        if len(words) < 3:
            return text
        
        # Randomly select span length and position
        span_length = random.randint(2, min(5, len(words) // 2))
        if len(words) > span_length:
            start = random.randint(0, len(words) - span_length)
            
            masked = words[:start] + ['[MASK]'] + words[start + span_length:]
            return ' '.join(masked)
        
        return text
    
    def _permute_sentences(self, text: str) -> str:
        """Randomly permute sentences in the text."""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) > 1:
            random.shuffle(sentences)
            return '. '.join(sentences) + '.'
        
        return text
    
    @property
    def training(self) -> bool:
        """Check if in training mode (for augmentation)."""
        return True  # Can be made configurable


class SimpleTokenizer:
    """
    Simple fallback tokenizer for when tiktoken is not available.
    """
    
    def __init__(self, vocab_size: int = 50257):
        self.vocab_size = vocab_size
        self.char_to_id = {}
        self.id_to_char = {}
        self._build_vocab()
    
    def _build_vocab(self):
        """Build a simple character-level vocabulary."""
        # Basic ASCII characters
        chars = list(range(32, 127))  # Printable ASCII
        
        for i, char_code in enumerate(chars):
            char = chr(char_code)
            self.char_to_id[char] = i
            self.id_to_char[i] = char
        
        # Add special tokens
        self.char_to_id['<UNK>'] = len(self.char_to_id)
        self.char_to_id['<PAD>'] = len(self.char_to_id)
        self.char_to_id['<EOS>'] = len(self.char_to_id)
        self.char_to_id['<BOS>'] = len(self.char_to_id)
    
    def tokenize(self, text: str) -> List[int]:
        """Convert text to token IDs."""
        tokens = []
        
        for char in text:
            if char in self.char_to_id:
                tokens.append(self.char_to_id[char])
            else:
                tokens.append(self.char_to_id['<UNK>'])
        
        return tokens
    
    def decode(self, tokens: List[int]) -> str:
        """Convert token IDs back to text."""
        chars = []
        
        for token in tokens:
            if token in self.id_to_char:
                chars.append(self.id_to_char[token])
            else:
                chars.append('<UNK>')
        
        return ''.join(chars)