#!/usr/bin/env python3
"""
Domain utilities for Gatto Mention Scanner
Handles domain extraction, apex domain resolution, and subdomain matching
"""
import re
from urllib.parse import urlparse
from typing import Optional

# Pre-compiled regex pattern for domain extraction
RE_DOMAIN_EXTRACT = re.compile(r'^(?:https?://)?([^/]+)')

def extract_apex_domain(domain: str) -> str:
    """Extract apex domain (eTLD+1) from domain"""
    if not domain:
        return ""
    
    # Remove protocol and www prefix
    domain = domain.lower()
    if domain.startswith('http'):
        domain = urlparse(domain).netloc
    
    if domain.startswith('www.'):
        domain = domain[4:]
    
    # Handle common eTLD+1 patterns
    parts = domain.split('.')
    if len(parts) >= 2:
        # Return last two parts as apex domain
        return '.'.join(parts[-2:])
    
    return domain

def domain_of(url: str = None, displayLink: str = None, formattedUrl: str = None) -> str:
    """
    Return a lowercased registrable domain without 'www.'.
    Strategy:
      - Try to parse `url`: if scheme missing, prefix 'http://' then urlparse(url).netloc
      - If still empty, use displayLink (strip port/path).
      - If still empty, regex on formattedUrl/htmlFormattedUrl: r'^(?:https?://)?([^/]+)'
      - Lowercase, strip leading 'www.'
      - Return "" only if all sources missing.
    """
    # Try primary URL first
    if url:
        try:
            if not url.startswith(('http://', 'https://')):
                url = 'http://' + url
            netloc = urlparse(url).netloc
            if netloc:
                domain = netloc.lower()
                if domain.startswith('www.'):
                    domain = domain[4:]
                return domain
        except Exception:
            pass
    
    # Try displayLink
    if displayLink:
        try:
            domain = displayLink.lower()
            # Strip port and path
            domain = domain.split(':')[0].split('/')[0]
            if domain.startswith('www.'):
                domain = domain[4:]
            if domain:
                return domain
        except Exception:
            pass
    
    # Try formattedUrl with regex
    if formattedUrl:
        try:
            match = RE_DOMAIN_EXTRACT.match(formattedUrl)
            if match:
                domain = match.group(1).lower()
                if domain.startswith('www.'):
                    domain = domain[4:]
                return domain
        except Exception:
            pass
    
    return ""

def is_subdomain_match(candidate_domain: str, apex_domain: str) -> bool:
    """Check if candidate domain matches apex domain or is a subdomain"""
    if not candidate_domain or not apex_domain:
        return False
    
    candidate_apex = extract_apex_domain(candidate_domain)
    
    # Direct match
    if candidate_apex == apex_domain:
        return True
    
    # Subdomain match: candidate ends with .apex_domain
    if candidate_domain.endswith('.' + apex_domain):
        return True
    
    return False