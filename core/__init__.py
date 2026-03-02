"""Core transformer components (config, model, trainer, sampler).

This package provides a namespaced way to import the base TinyLLM
transformer implementation without changing the original flat module
layout. All heavy implementations still live in the top-level modules
(`config.py`, `transformer.py`, etc.) and are re-exported here.
"""

from .config import Config, TransformerTrainingArgs, device
from .transformer import DemoTransformer
from .trainer import TransformerTrainer
from .sampler import TransformerSampler

__all__ = [
    "Config",
    "TransformerTrainingArgs",
    "device",
    "DemoTransformer",
    "TransformerTrainer",
    "TransformerSampler",
]

