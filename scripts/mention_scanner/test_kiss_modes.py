#!/usr/bin/env python3
"""
KISS Tests for GATTO Scanner - 3 modes coverage
Tests the simplified implementation according to specifications
"""
import json
import sys
import os
from unittest.mock import Mock, patch
from typing import Dict, Any, List

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

# Import the modules
try:
    from collection_router import CollectionRouter
    from scoring import final_score, make_tabular_decision, _calculate_name_score_kiss
    from config_resolver import resolve_config
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

class TestKISSModes:
    """Test the 3 modes: balanced, serp-only, open"""
    
    def setup_method(self):
        """Setup for each test"""
        self.mock_db = Mock()
        self.mock_cse_searcher = Mock()
        self.mock_city_manager = Mock()
        
        # Mock city profile
        mock_city_profile = Mock()
        mock_city_profile.city_names_aliases = ['Paris']
        mock_city_profile.country_code = 'FR'
        mock_city_profile.competing_cities = ['Lyon', 'Marseille']
        self.mock_city_manager.get_profile.return_value = mock_city_profile
        self.mock_city_manager.get_search_locale.return_value = {'gl': 'fr', 'hl': 'fr', 'cr': 'countryFR'}
        
        # Mock config
        self.config = {
            'mention_scanner': {
                'thresholds': {'high': 0.35, 'mid': 0.20},
                'weights': {'name': 0.60, 'geo': 0.25, 'authority': 0.15},
                'limits': {'cse_num': 30}
            }
        }
        
        self.router = CollectionRouter(self.config, self.mock_db, self.mock_cse_searcher, self.mock_city_manager)
        
    def test_balanced_mode_collection(self):
        """Test balanced mode: collect_from_catalog_active_sources() + collect_from_cse()"""
        poi = {'name': 'Le Rigmarole', 'city_slug': 'paris', 'category': 'restaurant'}
        
        # Mock catalog sources
        self.mock_db._load_source_catalog.return_value = [
            {'is_active': True, 'base_url': 'https://lefooding.com'},
            {'is_active': False, 'base_url': 'https://inactive.com'},  # Should be filtered
            {'is_active': True, 'base_url': 'https://timeout.fr'}
        ]
        
        # Mock CSE results for catalog and open queries
        mock_cse_results = [
            {'link': 'https://lefooding.com/article1', 'title': 'Le Rigmarole Paris review', 'snippet': 'Great restaurant'},
            {'link': 'https://timeout.fr/article2', 'title': 'Le Rigmarole guide', 'snippet': 'Must visit'}
        ]
        self.mock_cse_searcher.search.return_value = mock_cse_results
        
        # Test catalog active sources
        catalog_candidates = self.router.collect_from_catalog_active_sources(poi)
        assert len(catalog_candidates) > 0
        
        # Test CSE open queries  
        cse_candidates = self.router.collect_from_cse(poi)
        assert len(cse_candidates) > 0
        
        # Verify queries include poi_name, city_name, category (GOLDEN RULE)
        called_queries = [call[1]['query'] for call in self.mock_cse_searcher.search.call_args_list]
        for query in called_queries:
            assert 'Le Rigmarole' in query
            assert 'Paris' in query  
            assert 'restaurant' in query
    
    def test_serp_only_mode_collection(self):
        """Test serp-only mode: collect_from_catalog_filtered() - no CSE calls"""
        poi = {'name': 'Septime', 'city_slug': 'paris', 'category': 'restaurant'}
        sources = ['lefooding.com', 'timeout.fr']
        
        # Mock catalog for domain resolution
        self.mock_db._load_source_catalog.return_value = [
            {'source_id': 'fooding', 'base_url': 'https://lefooding.com'},
            {'source_id': 'timeout', 'base_url': 'https://timeout.fr'}
        ]
        
        mock_cse_results = [
            {'link': 'https://lefooding.com/septime', 'title': 'Septime Paris restaurant', 'snippet': 'Excellent cuisine'}
        ]
        self.mock_cse_searcher.search.return_value = mock_cse_results
        
        candidates = self.router.collect_from_catalog_filtered(poi, sources)
        
        assert len(candidates) > 0
        # Verify queries include poi_name, city_name, category (GOLDEN RULE)
        called_queries = [call[1]['query'] for call in self.mock_cse_searcher.search.call_args_list]
        for query in called_queries:
            assert 'Septime' in query
            assert 'Paris' in query
            assert 'restaurant' in query
            # Verify site: filtering is used
            assert 'site:' in query
    
    def test_open_mode_collection(self):
        """Test open mode: collect_from_cse() only - no site: filtering"""  
        poi = {'name': 'Le Chateaubriand', 'city_slug': 'paris', 'category': 'restaurant'}
        
        mock_cse_results = [
            {'link': 'https://randomsite.com/chateaubriand', 'title': 'Le Chateaubriand Paris review', 'snippet': 'Amazing food'},
            {'link': 'https://anotherblog.net/paris-resto', 'title': 'Le Chateaubriand experience', 'snippet': 'Unforgettable'}
        ]
        self.mock_cse_searcher.search.return_value = mock_cse_results
        
        candidates = self.router.collect_from_cse(poi)
        
        assert len(candidates) > 0
        # Verify queries include poi_name, city_name, category (GOLDEN RULE) 
        called_queries = [call[1]['query'] for call in self.mock_cse_searcher.search.call_args_list]
        for query in called_queries:
            assert 'Le Chateaubriand' in query
            assert 'Paris' in query
            assert 'restaurant' in query
            # Verify NO site: filtering
            assert 'site:' not in query
    
    def test_kiss_scoring_fixed_weights(self):
        """Test KISS scoring with fixed weights: 0.60*name + 0.25*geo + 0.15*authority"""
        poi_name = "Le Rigmarole"
        title = "Le Rigmarole Paris restaurant review"
        snippet = "Great food in Paris"
        url = "https://lefooding.com/le-rigmarole"
        
        with patch('scoring._calculate_geo_score_kiss', return_value=0.8), \
             patch('scoring.calculate_authority', return_value=0.9):
            
            score, explain = final_score(poi_name, title, snippet, url, 'restaurant', 
                                       config=self.config, debug=True, city_slug='paris')
            
            # Verify fixed weights are applied
            assert explain['weights']['name'] == 0.60
            assert explain['weights']['geo'] == 0.25  
            assert explain['weights']['authority'] == 0.15
            
            # Verify score is calculated correctly
            expected_score = (0.60 * explain['components']['name_match'] + 
                            0.25 * explain['components']['geo_score'] + 
                            0.15 * explain['components']['authority'])
            assert abs(score - expected_score) < 0.01  # Allow small floating point differences
    
    def test_tabular_decision_priority_order(self):
        """Test tabular decision logic with clear priority order"""
        candidate = {'domain': 'lefooding.com', 'url': 'https://lefooding.com/test'}
        
        # Test 1: Auto-accept confirmed domain (no country alert)
        explain = {
            'components': {
                'authority': 1.0,  # Confirmed domain
                'geo_score': 0.1,
                'penalties': {'country_mismatch': 0}
            }
        }
        decision, accepted_by, drop_reasons = make_tabular_decision(0.25, explain, candidate, 0.35, 0.20)
        assert decision == "ACCEPT"
        assert accepted_by == "confirmed_domain"
        
        # Test 2: High score threshold
        explain['components']['authority'] = 0.5
        decision, accepted_by, drop_reasons = make_tabular_decision(0.40, explain, candidate, 0.35, 0.20)
        assert decision == "ACCEPT"
        assert accepted_by == "score_high"
        
        # Test 3: Mid threshold with conditions (geo >= 0.25)
        explain['components']['geo_score'] = 0.30
        decision, accepted_by, drop_reasons = make_tabular_decision(0.25, explain, candidate, 0.35, 0.20)
        assert decision == "REVIEW" 
        assert accepted_by == "mid_conditional"
        
        # Test 4: Country mismatch hard reject
        explain['components']['penalties'] = {'country_mismatch': 1.0}
        decision, accepted_by, drop_reasons = make_tabular_decision(0.50, explain, candidate, 0.35, 0.20)
        assert decision == "REJECT"
        assert "country_mismatch" in drop_reasons
        
    def test_country_mismatch_hard_reject(self):
        """Test country mismatch causes hard reject"""
        poi_name = "Le Bistrot"
        title = "Le Bistrot London UK restaurant"  # Wrong country
        snippet = "Best restaurant in London"
        url = "https://london-dining.co.uk/le-bistrot"
        
        # Mock the penalty calculation to return country mismatch
        def mock_penalty_calc(*args, **kwargs):
            return {'country_mismatch': 1.0, 'city_mismatch': 0.0, 'total': 1.0}
        
        with patch('scoring._calculate_kiss_penalties', side_effect=mock_penalty_calc):
            score, explain = final_score(poi_name, title, snippet, url, 'restaurant',
                                       config=self.config, debug=True, city_slug='paris')  # Expected: FR
            
            # Should have country mismatch penalty
            assert explain['components']['penalties']['country_mismatch'] > 0
            
    def test_city_mismatch_soft_penalty(self):
        """Test city mismatch applies -0.15 penalty"""
        poi_name = "Le Comptoir" 
        title = "Le Comptoir Lyon restaurant"  # Wrong city (competing city)
        snippet = "Great restaurant in Lyon"
        url = "https://lyon-guide.fr/le-comptoir"
        
        # Mock the penalty calculation to return city mismatch
        def mock_penalty_calc(*args, **kwargs):
            return {'country_mismatch': 0.0, 'city_mismatch': 0.15, 'total': 0.15}
        
        with patch('scoring._calculate_kiss_penalties', side_effect=mock_penalty_calc):
            score, explain = final_score(poi_name, title, snippet, url, 'restaurant',
                                       config=self.config, debug=True, city_slug='paris')
            
            # Should have city mismatch penalty
            assert explain['components']['penalties']['city_mismatch'] == 0.15
    
    def test_name_scoring_kiss(self):
        """Test KISS name scoring: 2 signals (fuzzy + trigram) with stopword normalization"""
        # Test exact match
        score1 = _calculate_name_score_kiss("Le Rigmarole", "Le Rigmarole Paris restaurant review", "")
        assert score1 > 0.5  # Should be high for exact match
        
        # Test partial match
        score2 = _calculate_name_score_kiss("Septime", "Septime is a great restaurant in Paris", "")
        assert score2 > 0.3  # Should be decent for good partial match
        
        # Test no match
        score3 = _calculate_name_score_kiss("Le Rigmarole", "Completely different restaurant name", "")
        assert score3 < 0.3  # Should be low for no match
    
    def test_config_resolver_single_source_of_truth(self):
        """Test unified config resolver logs final values"""
        with patch('config_resolver.load_config', return_value=None), \
             patch('os.getenv', return_value=None):
            
            config = resolve_config()
            
            # Verify default KISS values
            scanner_config = config['mention_scanner']
            assert scanner_config['thresholds']['high'] == 0.35
            assert scanner_config['thresholds']['mid'] == 0.20
            assert scanner_config['weights']['name'] == 0.60
            assert scanner_config['weights']['geo'] == 0.25
            assert scanner_config['weights']['authority'] == 0.15
            assert scanner_config['limits']['cse_num'] == 30


def run_all_tests():
    """Run all tests manually without pytest"""
    test_instance = TestKISSModes()
    test_methods = [method for method in dir(test_instance) if method.startswith('test_')]
    
    passed = 0
    failed = 0
    
    for test_method in test_methods:
        try:
            test_instance.setup_method()
            print(f"Running {test_method}...", end=' ')
            getattr(test_instance, test_method)()
            print("✓ PASSED")
            passed += 1
        except Exception as e:
            print(f"✗ FAILED: {e}")
            failed += 1
    
    print(f"\nTest Results: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    # Run tests manually
    success = run_all_tests()
    sys.exit(0 if success else 1)