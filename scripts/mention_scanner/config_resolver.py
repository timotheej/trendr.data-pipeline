#!/usr/bin/env python3
"""
Configuration resolver for Gatto Mention Scanner
Handles config loading, threshold resolution, and rate limiting configuration
"""
import os
import json
import logging
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

def load_config() -> Optional[Dict[str, Any]]:
    """Load configuration from config.json with fallback handling"""
    config_path = "config.json"
    
    if not os.path.exists(config_path):
        logger.warning(f"Config file {config_path} not found, using defaults")
        return None
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logger.debug(f"Loaded config from {config_path}")
        return config
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        return None

def _resolve_thresholds(args, cfg) -> Tuple[float, float]:
    """Resolve thresholds from CLI args, config, or internal defaults
    
    Priority: CLI override > config["mention_scanner"]["match_score"] > internal constants
    Returns: (high_threshold, mid_threshold)
    """
    # Internal default constants
    DEFAULT_HIGH = 0.82
    DEFAULT_MID = 0.33  # Actually used as geo_score threshold currently
    
    # Try CLI overrides first
    high_threshold = getattr(args, 'threshold_high', None)
    mid_threshold = getattr(args, 'threshold_mid', None)
    
    # Try config if CLI not provided
    if high_threshold is None or mid_threshold is None:
        if cfg and 'mention_scanner' in cfg:
            match_score_cfg = cfg['mention_scanner'].get('match_score', {})
            if high_threshold is None:
                high_threshold = match_score_cfg.get('high')
            if mid_threshold is None:
                mid_threshold = match_score_cfg.get('mid')
    
    # Fallback to internal defaults
    if high_threshold is None:
        high_threshold = DEFAULT_HIGH
    if mid_threshold is None:
        mid_threshold = DEFAULT_MID
    
    # Log resolution source for debugging
    high_source = "CLI" if getattr(args, 'threshold_high', None) else ("config" if cfg else "internal")
    mid_source = "CLI" if getattr(args, 'threshold_mid', None) else ("config" if cfg else "internal")
    
    logger.debug(f"[THRESHOLDS] high={high_threshold} (source: {high_source}), mid={mid_threshold} (source: {mid_source})")
    
    return high_threshold, mid_threshold

def _rate_limit_delay(cfg) -> float:
    """Get rate limiting delay from config or fallback to current values
    
    Reads serp_cost_control (qps/rpm/backoff) from config["mention_scanner"]
    Returns: delay in seconds
    """
    DEFAULT_DELAY = 1.0  # Current hardcoded value
    
    if not cfg:
        logger.debug(f"[RATE_LIMIT] delay={DEFAULT_DELAY:.2f}s (source: internal - no config)")
        return DEFAULT_DELAY
    
    mention_scanner_cfg = cfg.get('mention_scanner', {})
    serp_cost_control = mention_scanner_cfg.get('serp_cost_control', {})
    
    # Try different config formats
    if 'qps' in serp_cost_control:
        qps = serp_cost_control['qps']
        delay = 1.0 / qps if qps > 0 else DEFAULT_DELAY
        logger.debug(f"[RATE_LIMIT] delay={delay:.2f}s (source: config qps={qps})")
        return delay
    elif 'rpm' in serp_cost_control:
        rpm = serp_cost_control['rpm']
        delay = 60.0 / rpm if rpm > 0 else DEFAULT_DELAY
        logger.debug(f"[RATE_LIMIT] delay={delay:.2f}s (source: config rpm={rpm})")
        return delay
    elif 'delay_s' in serp_cost_control:
        delay = float(serp_cost_control['delay_s'])
        logger.debug(f"[RATE_LIMIT] delay={delay:.2f}s (source: config delay_s)")
        return delay
    elif 'backoff_ms' in serp_cost_control:
        delay = serp_cost_control['backoff_ms'] / 1000.0
        logger.debug(f"[RATE_LIMIT] delay={delay:.2f}s (source: config backoff_ms)")
        return delay
    
    # Fallback to DEFAULT_DELAY
    logger.debug(f"[RATE_LIMIT] delay={DEFAULT_DELAY:.2f}s (source: internal - no valid config)")
    return DEFAULT_DELAY