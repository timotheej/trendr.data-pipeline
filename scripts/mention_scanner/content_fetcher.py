#!/usr/bin/env python3
"""
Content fetcher for Gatto Mention Scanner
Fetch RSS feeds and HTML lists without external dependencies
"""
import re
import time
import logging
import requests
import xml.etree.ElementTree as ET
from urllib.parse import urljoin
from typing import Dict, Any, List

# Try to import config resolver with fallback
try:
    from .config_resolver import _rate_limit_delay
except ImportError:
    try:
        from config_resolver import _rate_limit_delay
    except ImportError:
        # Fallback
        def _rate_limit_delay(config):
            return 1.0

logger = logging.getLogger(__name__)

# Content fetching constants
FETCH_ENABLED = True
MAX_URLS_PER_SOURCE = 12

# Pre-compiled regex for HTML parsing
RE_HTML_LINKS = re.compile(r'<a[^>]*href=["\']([^"\']+)["\'][^>]*>([^<]+)</a>', re.IGNORECASE)

class ContentFetcher:
    """Fetch RSS feeds and HTML lists without external dependencies"""
    
    def __init__(self, session=None, config=None):
        self.session = session or requests.Session()
        self.session.timeout = 10
        self.config = config
        
    def fetch_rss(self, url: str) -> List[Dict[str, Any]]:
        """Fetch RSS feed using stdlib XML parser"""
        if not FETCH_ENABLED:
            raise RuntimeError("Network disabled")
            
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            # Parse RSS with ElementTree
            root = ET.fromstring(response.content)
            articles = []
            
            # Handle RSS 2.0 and Atom formats
            items = root.findall('.//item') or root.findall('.//{http://www.w3.org/2005/Atom}entry')
            
            for item in items[:MAX_URLS_PER_SOURCE]:
                title_elem = item.find('title') or item.find('{http://www.w3.org/2005/Atom}title')
                link_elem = item.find('link') or item.find('{http://www.w3.org/2005/Atom}link')
                date_elem = item.find('pubDate') or item.find('{http://www.w3.org/2005/Atom}updated')
                
                title = title_elem.text if title_elem is not None else 'Untitled'
                url = link_elem.text if link_elem is not None else (link_elem.get('href') if link_elem is not None else '')
                published_at = date_elem.text if date_elem is not None else None
                
                if title and url:
                    articles.append({
                        'title': title.strip(),
                        'url': url.strip(),
                        'published_at': published_at
                    })
                    
            logger.debug(f"RSS parsed: {len(articles)} articles from {url}")
            return articles
            
        except Exception as e:
            logger.warning(f"RSS fetch failed for {url}: {e}")
            return []
    
    def fetch_html_list(self, url: str, selectors: Dict[str, str] = None) -> List[Dict[str, Any]]:
        """Fetch HTML list using CSS selectors or regex heuristics as fallback"""
        if not FETCH_ENABLED:
            raise RuntimeError("Network disabled")
            
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            html = response.text
            articles = []
            
            if selectors:
                # Use CSS selectors (requires BeautifulSoup or similar)
                # For now, implement basic regex-based parsing since BeautifulSoup not available
                logger.warning(f"CSS selectors not yet implemented, using heuristics for {url}")
                # TODO: Implement proper CSS selector parsing
                
            # Fallback: Heuristic regex parsing
            matches = RE_HTML_LINKS.findall(html)
            
            for href, text in matches[:MAX_URLS_PER_SOURCE]:
                # Filter out navigation/footer links
                if len(text.strip()) > 10 and any(keyword in text.lower() 
                   for keyword in ['restaurant', 'bar', 'café', 'ouvert', 'nouveau', 'adresse']):
                    full_url = urljoin(url, href)
                    articles.append({
                        'title': text.strip(),
                        'url': full_url,
                        'published_at': None  # Not available in HTML list mode
                    })
                    
            mode_desc = "selector-based" if selectors else "heuristic"
            logger.warning(f"HTML parsed ({mode_desc}): {len(articles)} articles from {url}")
            return articles
            
        except Exception as e:
            logger.warning(f"HTML fetch failed for {url}: {e}")
            return []
    
    def rate_limit_sleep(self):
        """Sleep between requests for rate limiting"""
        delay = _rate_limit_delay(self.config)
        time.sleep(delay)