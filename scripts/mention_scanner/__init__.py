#!/usr/bin/env python3
"""
Gatto Mention Scanner - Modular Components
Organized scanner components for maintainable code
"""

# Main scanner class
from .scanner import GattoMentionScanner

# Core components  
from .cse_client import CSESearcher
from .content_fetcher import ContentFetcher
from .matching import MentionMatcher, normalize
from .dedup import MentionDeduplicator, dedupe_key

# Utilities
from .scoring import *
from .domains import *
from .config_resolver import load_config, _resolve_thresholds, _rate_limit_delay
from .logging_ext import JSONLWriter, print_stable_summary

__all__ = [
    'GattoMentionScanner',
    'CSESearcher', 
    'ContentFetcher',
    'MentionMatcher',
    'MentionDeduplicator',
    'JSONLWriter',
    'normalize',
    'dedupe_key',
    'load_config',
    'print_stable_summary'
]