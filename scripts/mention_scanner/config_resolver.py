#!/usr/bin/env python3
"""
Removed unused code: complex threshold systems, legacy resolver chains, experimental config loaders

Unified Config Resolver for GATTO Scanner - Single Source of Truth
Handles CLI > ENV > config > defaults resolution and logs final values
"""
import os
import json
import logging
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

def load_config() -> Optional[Dict[str, Any]]:
    """Load configuration from config.json with fallback handling"""
    # Try multiple paths: current dir, parent dir, project root
    possible_paths = [
        "config.json",
        "../config.json", 
        "../../config.json",
        os.path.join(os.path.dirname(__file__), "../../config.json")
    ]
    
    config_path = None
    for path in possible_paths:
        if os.path.exists(path):
            config_path = path
            break
    
    if not config_path:
        logger.warning(f"Config file not found in any of {possible_paths}, using defaults")
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


def resolve_config(cli_args=None) -> Dict[str, Any]:
    """Single function that returns final config values (CLI > ENV > config > defaults) and logs them"""
    logger = logging.getLogger(__name__)
    
    # Load base config
    base_config = load_config()
    
    # KISS defaults - minimal, let config.json override everything
    final_config = {
        'mention_scanner': {
            'mode': 'balanced',
            'match_score': {
                'high': 0.35,
                'mid': 0.20
            },
            'scoring': {
                'name_weight': 0.60,  # KISS fixed weights
                'geo_weight': 0.25,
                'authority_weight': 0.15
            },
            'limits': {
                'cse_num': 30
            },
            'time_decay': {
                'enabled': False,  # Default disabled
                'tau_days': 90,    # Half-life in days
                'max_age_days': 365  # Max age before 0 score
            },
            'serp_only': {
                'sources': []
            }
        },
        'domain_exclusions': {
            'exclude_from_mentions': False  # Default disabled
        }
    }
    
    # Overlay base config
    if base_config:
        _deep_merge(final_config, base_config)
    
    # Apply CLI overrides first (highest priority)
    if cli_args and hasattr(cli_args, 'cse_num') and cli_args.cse_num:
        cse_num = max(1, min(50, cli_args.cse_num))  # Clamp to 1..50 range
        final_config['mention_scanner']['limits']['cse_num'] = cse_num
    
    if cli_args and hasattr(cli_args, 'time_decay') and cli_args.time_decay is not None:
        final_config['mention_scanner']['time_decay']['enabled'] = cli_args.time_decay
    
    # Apply ENV overrides
    elif os.getenv('CSE_NUM'):
        try:
            cse_num = int(os.getenv('CSE_NUM'))
            # Clamp to 1..50 range
            cse_num = max(1, min(50, cse_num))
            final_config['mention_scanner']['limits']['cse_num'] = cse_num
        except ValueError:
            pass
    
    if os.getenv('SCANNER_MODE'):
        final_config['mention_scanner']['mode'] = os.getenv('SCANNER_MODE')
    
    if os.getenv('THRESHOLD_HIGH'):
        try:
            final_config['mention_scanner']['thresholds']['high'] = float(os.getenv('THRESHOLD_HIGH'))
        except ValueError:
            pass
            
    if os.getenv('THRESHOLD_MID'):
        try:
            final_config['mention_scanner']['thresholds']['mid'] = float(os.getenv('THRESHOLD_MID'))
        except ValueError:
            pass
    
    # Log final resolved config
    scanner_config = final_config['mention_scanner']
    logger.info("📋 RESOLVED CONFIG:")
    logger.info(f"  mode: {scanner_config['mode']}")
    logger.info(f"  limits.cse_num: {scanner_config['limits']['cse_num']}")
    logger.info(f"  time_decay.enabled: {scanner_config['time_decay']['enabled']}")
    
    # Use match_score structure from config.json
    thresholds = scanner_config.get('match_score', {})
    logger.info(f"  thresholds: high={thresholds['high']}, mid={thresholds['mid']}")
    
    # Use scoring structure from config.json  
    scoring = scanner_config.get('scoring', {})
    logger.info(f"  weights: name={scoring['name_weight']}, geo={scoring['geo_weight']}, authority={scoring['authority_weight']}")
    
    # Domain exclusions (nested in mention_scanner in config.json)
    domain_exclusions = scanner_config.get('domain_exclusions', {})
    if domain_exclusions.get('exclude_from_mentions', False):
        excluded_count = len(domain_exclusions.get('social_networks', [])) + len(domain_exclusions.get('review_sites', []))
        logger.info(f"  domain_exclusions: enabled ({excluded_count} domains)")
    else:
        logger.info(f"  domain_exclusions: disabled")
    
    if scanner_config['mode'] == 'serp-only' and scanner_config['serp_only']['sources']:
        logger.info(f"  serp_only.sources: {scanner_config['serp_only']['sources'][:3]}{'...' if len(scanner_config['serp_only']['sources']) > 3 else ''}")
    
    return final_config


def _deep_merge(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    """Deep merge source dict into target dict"""
    for key, value in source.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value