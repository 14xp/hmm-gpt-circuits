"""
This module contains classes for tokenizing text.
"""

from enum import Enum
from typing import Protocol

import tiktoken


class TokenizerType(str, Enum):
    """
    Type of tokenizer.
    """

    TIKTOKEN = "tiktoken"
    ASCII = "ascii"
    INTEGER = "integer"

    def as_tokenizer(self, vocab_size: int = None) -> "Tokenizer":
        """
        Returns the tokenizer corresponding to this type.
        For INTEGER type, vocab_size must be provided.
        """
        if self == TokenizerType.TIKTOKEN:
            return TikTokenTokenizer()
        elif self == TokenizerType.ASCII:
            return ASCIITokenizer()
        elif self == TokenizerType.INTEGER:
            if vocab_size is None:
                raise ValueError("vocab_size must be provided for INTEGER tokenizer")
            return IntegerTokenizer(vocab_size)
        else:
            raise ValueError(f"Invalid tokenizer type: {self}")


class Tokenizer(Protocol):
    """
    Tokenizer interface.
    """

    # The size of the vocabulary.
    vocab_size: int

    def encode(self, text) -> list[int]:
        """
        Tokenizes the input text and returns a list of token IDs.
        """
        ...

    def decode_sequence(self, tokens: list[int]) -> str:
        """
        Converts a list of token IDs to a string.
        """
        ...

    def decode_token(self, token: int) -> str:
        """
        Converts a single token ID to a string.
        """
        ...


class TikTokenTokenizer(Tokenizer):
    """
    Tokenizer for TikToken.
    """

    # 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    vocab_size = 50257

    def __init__(self):
        self.encoding = tiktoken.get_encoding("gpt2")

    def encode(self, text) -> list[int]:
        return self.encoding.encode(text, allowed_special="all")

    def decode_sequence(self, tokens: list[int]) -> str:
        # Note that multiple tokens may have been used to represent a single UTF-8 character.
        return self.encoding.decode(tokens)

    def decode_token(self, token: int) -> str:
        # Decoding falls back to a hex representation if the result isn't printable.
        s = self.encoding.decode_single_token_bytes(token)
        try:
            return s.decode("utf-8")
        except Exception as _:
            return "".join([f"\\x{d:02x}" for d in s])


class ASCIITokenizer(Tokenizer):
    """
    Tokenizer that treats each character in the input as a token.
    """

    vocab_size = 128

    def encode(self, text) -> list[int]:
        # Note that invalid characters are replaced with "?".
        return [ord(c) if ord(c) < 128 else ord("?") for c in text]

    def decode_sequence(self, tokens: list[int]) -> str:
        return "".join([chr(b) for b in tokens])

    def decode_token(self, token: int) -> str:
        return chr(token)


class IntegerTokenizer(Tokenizer):
    """
    Tokenizer for data that is already tokenized as integers.
    This is a pass-through tokenizer that expects input to already be a list of integers.
    """

    def __init__(self, vocab_size: int):
        """
        Initialize the tokenizer with a specific vocabulary size.
        
        Args:
            vocab_size: The number of distinct tokens in the vocabulary
        """
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        self.vocab_size = vocab_size

    def encode(self, tokens) -> list[int]:
        """
        Pass-through encoding for integer sequences.
        
        Args:
            tokens: Either a list of integers or a single integer sequence
            
        Returns:
            List of token IDs, validated to be within vocab_size
        """
        if isinstance(tokens, int):
            tokens = [tokens]
        elif isinstance(tokens, str):
            raise ValueError("IntegerTokenizer expects integer input, not strings. Use encode_sequence for lists.")
        
        # Validate that all tokens are within vocabulary bounds
        for token in tokens:
            if not isinstance(token, int):
                raise ValueError(f"All tokens must be integers, got {type(token)}")
            if token < 0 or token >= self.vocab_size:
                raise ValueError(f"Token {token} is out of vocabulary bounds [0, {self.vocab_size})")
        
        return list(tokens)

    def decode_sequence(self, tokens: list[int]) -> str:
        """
        Convert a list of token IDs to a string representation.
        """
        return " ".join([str(token) for token in tokens])

    def decode_token(self, token: int) -> str:
        """
        Convert a single token ID to a string representation.
        """
        if token < 0 or token >= self.vocab_size:
            return f"<UNK:{token}>"
        return str(token)
