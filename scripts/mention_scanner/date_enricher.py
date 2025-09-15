#!/usr/bin/env python3
"""
KISS Date Enricher for GATTO Scanner
Extracts published_at dates with minimal cost using fallback hierarchy
"""
import re
import logging
import requests
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, Tuple
from urllib.parse import urlparse
import feedparser
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Constants
CONFIDENCE_HIGH = 'high'
CONFIDENCE_MEDIUM = 'medium' 
CONFIDENCE_LOW = 'low'

METHOD_RSS = 'rss'
METHOD_SCHEMA_ORG = 'schema_org'
METHOD_OPENGRAPH = 'opengraph'
METHOD_META_ARTICLE = 'meta_article'
METHOD_URL_INFERENCE = 'url_inference'
METHOD_HTTP_LAST_MODIFIED = 'http_last_modified'
METHOD_SERP = 'serp'
METHOD_UNKNOWN = 'unknown'

class DateEnricher:
    """KISS date enricher with controlled network cost"""
    
    def __init__(self, timeout: float = 5.0, max_content_size: int = 1024 * 1024):
        self.timeout = timeout
        self.max_content_size = max_content_size
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; GATTO-Scanner/1.0; +https://gatto.app)'
        })
    
    def enrich(self, mention: Dict[str, Any], source_catalog_entry: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Enrich mention with published_at using fallback hierarchy
        Returns: mention with added fields: published_at, published_at_confidence, published_at_method
        """
        url = mention.get('url', '')
        if not url:
            return self._set_no_date(mention, 'no_url')
        
        try:
            # Step 1: SERP date (if available) - free
            if self._try_serp_date(mention):
                return mention
            
            # Step 2: RSS feed (if source is confirmed and has RSS) - prioritized GET
            if source_catalog_entry and self._try_rss_date(mention, source_catalog_entry):
                return mention
            
            # Step 3: HTML parsing with single GET (schema.org, opengraph, meta)
            if self._try_html_date(mention, source_catalog_entry):
                return mention
            
            # Step 4: URL inference - free
            if self._try_url_inference(mention):
                return mention
            
            # Step 5: HTTP Last-Modified (HEAD only) - cheap
            if self._try_http_last_modified(mention):
                return mention
            
            # Fallback: no date found
            return self._set_no_date(mention, METHOD_UNKNOWN)
            
        except Exception as e:
            logger.debug(f"Date enrichment failed for {url}: {e}")
            return self._set_no_date(mention, 'error')
    
    def _try_serp_date(self, mention: Dict[str, Any]) -> bool:
        """Try to extract date from SERP snippet/metadata"""
        snippet = mention.get('snippet', '')
        
        # Look for date patterns in snippet
        date_patterns = [
            r'(\d{1,2})\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+(20\d{2})',
            r'(20\d{2})-(\d{2})-(\d{2})',
            r'(\d{1,2})/(\d{1,2})/(20\d{2})'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, snippet, re.IGNORECASE)
            if match:
                try:
                    parsed_date = self._parse_date_from_match(match, pattern)
                    if parsed_date and self._validate_date(parsed_date):
                        self._set_date(mention, parsed_date, CONFIDENCE_LOW, METHOD_SERP)
                        logger.debug(f"SERP date found: {parsed_date} from snippet")
                        return True
                except Exception:
                    continue
        
        return False
    
    def _try_rss_date(self, mention: Dict[str, Any], source_catalog_entry: Dict[str, Any]) -> bool:
        """Try RSS feed for confirmed sources (high priority GET)"""
        rss_url = source_catalog_entry.get('rss_feed_url')
        if not rss_url:
            return False
        
        try:
            logger.debug(f"Fetching RSS feed: {rss_url}")
            response = self.session.get(rss_url, timeout=self.timeout)
            response.raise_for_status()
            
            feed = feedparser.parse(response.content)
            mention_url = mention.get('url', '')
            
            # Find matching RSS entry
            for entry in feed.entries[:50]:  # Limit to recent entries
                if hasattr(entry, 'link') and entry.link == mention_url:
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        published_date = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                        if self._validate_date(published_date):
                            self._set_date(mention, published_date, CONFIDENCE_HIGH, METHOD_RSS)
                            logger.debug(f"RSS date found: {published_date}")
                            return True
            
        except Exception as e:
            logger.debug(f"RSS fetch failed for {rss_url}: {e}")
        
        return False
    
    def _try_html_date(self, mention: Dict[str, Any], source_catalog_entry: Optional[Dict[str, Any]]) -> bool:
        """Try HTML parsing with single GET"""
        url = mention.get('url', '')
        
        try:
            logger.debug(f"Fetching HTML for date extraction: {url}")
            response = self.session.get(
                url, 
                timeout=self.timeout,
                stream=True,
                headers={'Accept': 'text/html,application/xhtml+xml'}
            )
            response.raise_for_status()
            
            # Read limited content
            content = b''
            for chunk in response.iter_content(chunk_size=8192):
                content += chunk
                if len(content) > self.max_content_size:
                    break
            
            soup = BeautifulSoup(content, 'html.parser')
            
            # Try schema.org first (highest confidence)
            if self._try_schema_org(mention, soup):
                return True
            
            # Try custom selectors from source catalog
            if source_catalog_entry and self._try_custom_selectors(mention, soup, source_catalog_entry):
                return True
            
            # Try OpenGraph
            if self._try_opengraph(mention, soup):
                return True
            
            # Try standard meta tags
            if self._try_meta_article(mention, soup):
                return True
            
        except Exception as e:
            logger.debug(f"HTML parsing failed for {url}: {e}")
        
        return False
    
    def _try_schema_org(self, mention: Dict[str, Any], soup: BeautifulSoup) -> bool:
        """Extract from schema.org JSON-LD"""
        scripts = soup.find_all('script', type='application/ld+json')
        
        for script in scripts:
            try:
                import json
                data = json.loads(script.string)
                
                # Handle array or single object
                items = data if isinstance(data, list) else [data]
                
                for item in items:
                    if item.get('@type') in ['Article', 'BlogPosting', 'NewsArticle']:
                        date_published = item.get('datePublished')
                        if date_published:
                            parsed_date = self._parse_iso_date(date_published)
                            if parsed_date and self._validate_date(parsed_date):
                                self._set_date(mention, parsed_date, CONFIDENCE_HIGH, METHOD_SCHEMA_ORG)
                                logger.debug(f"Schema.org date found: {parsed_date}")
                                return True
            except Exception:
                continue
        
        return False
    
    def _try_custom_selectors(self, mention: Dict[str, Any], soup: BeautifulSoup, source_catalog_entry: Dict[str, Any]) -> bool:
        """Try custom date selectors from source catalog"""
        selector = source_catalog_entry.get('html_date_selector')
        if not selector:
            return False
        
        try:
            elements = soup.select(selector)
            for element in elements:
                date_text = element.get_text().strip()
                parsed_date = self._parse_flexible_date(date_text)
                if parsed_date and self._validate_date(parsed_date):
                    self._set_date(mention, parsed_date, CONFIDENCE_HIGH, METHOD_SCHEMA_ORG)
                    logger.debug(f"Custom selector date found: {parsed_date}")
                    return True
        except Exception as e:
            logger.debug(f"Custom selector failed: {e}")
        
        return False
    
    def _try_opengraph(self, mention: Dict[str, Any], soup: BeautifulSoup) -> bool:
        """Extract from OpenGraph meta tags"""
        og_tags = [
            'article:published_time',
            'article:modified_time',
            'og:updated_time'
        ]
        
        for tag in og_tags:
            meta = soup.find('meta', property=tag)
            if meta and meta.get('content'):
                parsed_date = self._parse_iso_date(meta['content'])
                if parsed_date and self._validate_date(parsed_date):
                    self._set_date(mention, parsed_date, CONFIDENCE_MEDIUM, METHOD_OPENGRAPH)
                    logger.debug(f"OpenGraph date found: {parsed_date} from {tag}")
                    return True
        
        return False
    
    def _try_meta_article(self, mention: Dict[str, Any], soup: BeautifulSoup) -> bool:
        """Extract from standard meta tags"""
        meta_names = [
            'date',
            'article:published_time',
            'pubdate',
            'publishdate',
            'publish-date',
            'created',
            'timestamp'
        ]
        
        for name in meta_names:
            meta = soup.find('meta', attrs={'name': name})
            if meta and meta.get('content'):
                parsed_date = self._parse_flexible_date(meta['content'])
                if parsed_date and self._validate_date(parsed_date):
                    self._set_date(mention, parsed_date, CONFIDENCE_MEDIUM, METHOD_META_ARTICLE)
                    logger.debug(f"Meta article date found: {parsed_date} from {name}")
                    return True
        
        return False
    
    def _try_url_inference(self, mention: Dict[str, Any]) -> bool:
        """Extract date from URL patterns"""
        url = mention.get('url', '')
        
        # Common URL date patterns
        patterns = [
            r'/(20\d{2})/(\d{1,2})/(\d{1,2})/',  # /2024/03/15/
            r'/(20\d{2})-(\d{2})-(\d{2})',        # /2024-03-15
            r'/(\d{1,2})-(\d{1,2})-(20\d{2})',   # /15-03-2024
            r'[?&]date=(20\d{2})-(\d{2})-(\d{2})', # ?date=2024-03-15
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                try:
                    groups = match.groups()
                    if len(groups) >= 3:
                        # Handle different group orders
                        if len(groups[0]) == 4:  # Year first
                            year, month, day = int(groups[0]), int(groups[1]), int(groups[2])
                        else:  # Day first
                            day, month, year = int(groups[0]), int(groups[1]), int(groups[2])
                        
                        parsed_date = datetime(year, month, day, tzinfo=timezone.utc)
                        if self._validate_date(parsed_date):
                            self._set_date(mention, parsed_date, CONFIDENCE_LOW, METHOD_URL_INFERENCE)
                            logger.debug(f"URL inference date found: {parsed_date}")
                            return True
                except ValueError:
                    continue
        
        return False
    
    def _try_http_last_modified(self, mention: Dict[str, Any]) -> bool:
        """Extract from HTTP Last-Modified header (HEAD request only)"""
        url = mention.get('url', '')
        
        try:
            logger.debug(f"HEAD request for Last-Modified: {url}")
            response = self.session.head(url, timeout=self.timeout, allow_redirects=True)
            
            last_modified = response.headers.get('Last-Modified')
            if last_modified:
                # Parse HTTP date format
                from email.utils import parsedate_to_datetime
                parsed_date = parsedate_to_datetime(last_modified)
                if parsed_date and self._validate_date(parsed_date):
                    # Convert to UTC if needed
                    if parsed_date.tzinfo is None:
                        parsed_date = parsed_date.replace(tzinfo=timezone.utc)
                    else:
                        parsed_date = parsed_date.astimezone(timezone.utc)
                    
                    self._set_date(mention, parsed_date, CONFIDENCE_LOW, METHOD_HTTP_LAST_MODIFIED)
                    logger.debug(f"HTTP Last-Modified date found: {parsed_date}")
                    return True
        
        except Exception as e:
            logger.debug(f"HTTP HEAD failed for {url}: {e}")
        
        return False
    
    def _parse_iso_date(self, date_str: str) -> Optional[datetime]:
        """Parse ISO format dates"""
        try:
            # Handle various ISO formats
            for fmt in ['%Y-%m-%dT%H:%M:%S%z', '%Y-%m-%dT%H:%M:%SZ', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d']:
                try:
                    if date_str.endswith('Z'):
                        parsed = datetime.strptime(date_str, fmt.replace('%z', 'Z'))
                        return parsed.replace(tzinfo=timezone.utc)
                    else:
                        parsed = datetime.strptime(date_str, fmt)
                        if parsed.tzinfo is None:
                            parsed = parsed.replace(tzinfo=timezone.utc)
                        return parsed
                except ValueError:
                    continue
        except Exception:
            pass
        
        return None
    
    def _parse_flexible_date(self, date_str: str) -> Optional[datetime]:
        """Parse various date formats"""
        if not date_str:
            return None
        
        # Try ISO first
        iso_date = self._parse_iso_date(date_str)
        if iso_date:
            return iso_date
        
        # Try common patterns
        patterns = [
            r'(\d{1,2})\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+(20\d{2})',
            r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+(\d{1,2}),?\s+(20\d{2})',
            r'(\d{1,2})/(\d{1,2})/(20\d{2})',
            r'(20\d{2})-(\d{1,2})-(\d{1,2})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, date_str, re.IGNORECASE)
            if match:
                try:
                    parsed = self._parse_date_from_match(match, pattern)
                    if parsed:
                        return parsed
                except Exception:
                    continue
        
        return None
    
    def _parse_date_from_match(self, match, pattern: str) -> Optional[datetime]:
        """Parse date from regex match based on pattern"""
        try:
            groups = match.groups()
            
            if 'jan|feb|mar' in pattern:  # Month name patterns
                month_names = {
                    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
                }
                
                if len(groups) == 3:
                    if groups[0].isdigit():  # Day Month Year
                        day, month_str, year = groups
                    else:  # Month Day Year
                        month_str, day, year = groups
                    
                    month = month_names.get(month_str.lower())
                    if month:
                        return datetime(int(year), month, int(day), tzinfo=timezone.utc)
            
            elif '/' in pattern or '-' in pattern:  # Numeric patterns
                if len(groups) >= 3:
                    if len(groups[0]) == 4:  # Year first
                        year, month, day = int(groups[0]), int(groups[1]), int(groups[2])
                    elif len(groups[2]) == 4:  # Year last
                        month, day, year = int(groups[0]), int(groups[1]), int(groups[2])
                    else:
                        return None
                    
                    return datetime(year, month, day, tzinfo=timezone.utc)
        
        except (ValueError, IndexError):
            pass
        
        return None
    
    def _validate_date(self, date_obj: datetime) -> bool:
        """Validate date is reasonable"""
        now = datetime.now(timezone.utc)
        
        # Reject future dates (with 24h tolerance for clock skew)
        if date_obj > now + timedelta(hours=24):
            logger.debug(f"Rejecting future date: {date_obj}")
            return False
        
        # Reject very old dates (before 1990)
        if date_obj.year < 1990:
            logger.debug(f"Rejecting too old date: {date_obj}")
            return False
        
        return True
    
    def _set_date(self, mention: Dict[str, Any], date_obj: datetime, confidence: str, method: str):
        """Set date fields in mention"""
        mention['published_at'] = date_obj.isoformat()
        mention['published_at_confidence'] = confidence
        mention['published_at_method'] = method
    
    def _set_no_date(self, mention: Dict[str, Any], reason: str) -> Dict[str, Any]:
        """Set null date fields"""
        mention['published_at'] = None
        mention['published_at_confidence'] = None
        mention['published_at_method'] = METHOD_UNKNOWN
        logger.debug(f"No date found for {mention.get('url', '')}: {reason}")
        return mention