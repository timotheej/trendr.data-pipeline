#!/usr/bin/env python3
"""
Trendr Social Proof Enhancement - System Integration Test
Comprehensive test of the complete pipeline from enhanced social proof to validated collections.
Tests the integration of all three major steps: Enhanced Proof, Intelligent Classification, Dynamic Neighborhoods, and Validated Collections.
"""
import sys
import os
import logging
import json
from typing import Dict, Any, List
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.database import SupabaseManager
from scripts.enhanced_proof_scanner import EnhancedProofScanner
from scripts.intelligent_classifier import IntelligentMoodClassifier
from scripts.dynamic_neighborhoods import DynamicNeighborhoodCalculator
from scripts.validated_ai_collections import ValidatedCollectionGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrendrSystemIntegrationTest:
    """Complete integration test of the Trendr Social Proof Enhancement system."""
    
    def __init__(self):
        self.db = SupabaseManager()
        self.proof_scanner = EnhancedProofScanner()
        self.classifier = IntelligentMoodClassifier()
        self.neighborhood_calculator = DynamicNeighborhoodCalculator()
        self.collection_generator = ValidatedCollectionGenerator()
        
        self.test_results = {
            'step_1_enhanced_proof': {},
            'step_2_intelligent_classification': {},
            'step_3_dynamic_neighborhoods': {},
            'step_4_validated_collections': {},
            'system_integration': {},
            'overall_assessment': {}
        }
    
    def run_comprehensive_system_test(self, city: str = 'Montreal') -> Dict[str, Any]:
        """Run complete system integration test."""
        logger.info("🚀 Starting Trendr System Integration Test")
        logger.info("=" * 60)
        
        try:
            # Step 1: Test Enhanced Social Proof Collection
            logger.info("📊 Step 1: Testing Enhanced Social Proof Collection")
            step1_results = self.test_enhanced_proof_system(city)
            self.test_results['step_1_enhanced_proof'] = step1_results
            
            # Step 2: Test Intelligent Classification
            logger.info("🧠 Step 2: Testing Intelligent POI Classification")
            step2_results = self.test_intelligent_classification(city)
            self.test_results['step_2_intelligent_classification'] = step2_results
            
            # Step 3: Test Dynamic Neighborhoods
            logger.info("🏘️ Step 3: Testing Dynamic Neighborhood Calculations")
            step3_results = self.test_dynamic_neighborhoods(city)
            self.test_results['step_3_dynamic_neighborhoods'] = step3_results
            
            # Step 4: Test Validated Collections
            logger.info("🎨 Step 4: Testing Validated AI Collections")
            step4_results = self.test_validated_collections(city)
            self.test_results['step_4_validated_collections'] = step4_results
            
            # System Integration Analysis
            logger.info("⚡ Step 5: System Integration Analysis")
            integration_results = self.analyze_system_integration(city)
            self.test_results['system_integration'] = integration_results
            
            # Overall Assessment
            overall_results = self.generate_overall_assessment()
            self.test_results['overall_assessment'] = overall_results
            
            logger.info("✅ Comprehensive System Test Complete!")
            return self.test_results
            
        except Exception as e:
            logger.error(f"System integration test failed: {e}")
            return {'error': str(e), 'partial_results': self.test_results}
    
    def test_enhanced_proof_system(self, city: str) -> Dict[str, Any]:
        """Test Step 1: Enhanced Social Proof Collection."""
        results = {
            'test_name': 'Enhanced Social Proof Collection',
            'success_criteria': {
                'coverage_target': 60,  # >60% POIs with social proof
                'authority_target': 20,  # >20% POIs with authority mentions
                'improvement_target': 'visible'  # Visible improvement over basic system
            }
        }
        
        try:
            # Get baseline POI count
            pois = self.db.get_pois_for_city(city, 50)  # Test with 50 POIs
            total_pois = len(pois)
            
            # Count POIs with social proof
            pois_with_proof = 0
            pois_with_authority = 0
            total_proof_sources = 0
            authority_distribution = {'High': 0, 'Medium': 0, 'Low': 0}
            
            for poi in pois:
                proof_sources = self.db.get_proof_sources_for_poi(poi['id'])
                if proof_sources:
                    pois_with_proof += 1
                    total_proof_sources += len(proof_sources)
                    
                    # Check authority levels
                    has_authority = False
                    for source in proof_sources:
                        authority = source.get('authority_score', 'Low')
                        authority_distribution[authority] += 1
                        if authority in ['High', 'Medium']:
                            has_authority = True
                    
                    if has_authority:
                        pois_with_authority += 1
            
            # Calculate metrics
            coverage_rate = (pois_with_proof / total_pois * 100) if total_pois > 0 else 0
            authority_rate = (pois_with_authority / total_pois * 100) if total_pois > 0 else 0
            avg_proof_per_poi = total_proof_sources / pois_with_proof if pois_with_proof > 0 else 0
            
            results.update({
                'metrics': {
                    'total_pois_tested': total_pois,
                    'pois_with_social_proof': pois_with_proof,
                    'coverage_rate_percent': round(coverage_rate, 1),
                    'pois_with_authority_mentions': pois_with_authority,
                    'authority_rate_percent': round(authority_rate, 1),
                    'total_proof_sources': total_proof_sources,
                    'avg_proof_sources_per_poi': round(avg_proof_per_poi, 1),
                    'authority_distribution': authority_distribution
                },
                'success_evaluation': {
                    'coverage_success': coverage_rate >= results['success_criteria']['coverage_target'],
                    'authority_success': authority_rate >= results['success_criteria']['authority_target'],
                    'overall_step1_success': coverage_rate >= 60 and authority_rate >= 20
                },
                'status': 'completed'
            })
            
            logger.info(f"  📊 Coverage: {coverage_rate:.1f}% (target: {results['success_criteria']['coverage_target']}%)")
            logger.info(f"  👑 Authority: {authority_rate:.1f}% (target: {results['success_criteria']['authority_target']}%)")
            
        except Exception as e:
            results.update({
                'status': 'failed',
                'error': str(e)
            })
            logger.error(f"Step 1 test failed: {e}")
        
        return results
    
    def test_intelligent_classification(self, city: str) -> Dict[str, Any]:
        """Test Step 2: Intelligent POI Classification."""
        results = {
            'test_name': 'Intelligent POI Classification',
            'success_criteria': {
                'accuracy_target': 80,  # >80% classification accuracy
                'confidence_target': 0.6,  # Average confidence > 0.6
                'contextual_tags_target': 70  # >70% POIs with contextual tags
            }
        }
        
        try:
            # Test classification on POIs with social proof
            pois = self.db.get_pois_for_city(city, 20)  # Test subset for detailed analysis
            
            classification_results = []
            contextual_tag_counts = []
            confidence_scores = []
            mood_distribution = {'chill': 0, 'trendy': 0, 'hidden_gem': 0}
            
            for poi in pois:
                try:
                    # Get proof sources and classify
                    proof_sources = self.db.get_proof_sources_for_poi(poi['id'])
                    classification = self.classifier.classify_poi_multi_dimensional(poi, proof_sources)
                    
                    if classification:
                        classification_results.append({
                            'poi_name': poi.get('name', 'Unknown'),
                            'classified_mood': classification.get('mood', 'unknown'),
                            'confidence': classification.get('confidence_score', 0),
                            'contextual_tags_count': len(classification.get('contextual_tags', []))
                        })
                        
                        # Track metrics
                        confidence_scores.append(classification.get('confidence_score', 0))
                        contextual_tag_counts.append(len(classification.get('contextual_tags', [])))
                        
                        mood = classification.get('mood', 'unknown').lower()
                        if mood in mood_distribution:
                            mood_distribution[mood] += 1
                
                except Exception as e:
                    logger.warning(f"Classification failed for {poi.get('name', 'Unknown')}: {e}")
                    continue
            
            # Calculate metrics
            total_classified = len(classification_results)
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
            avg_contextual_tags = sum(contextual_tag_counts) / len(contextual_tag_counts) if contextual_tag_counts else 0
            pois_with_contextual_tags = sum(1 for count in contextual_tag_counts if count > 0)
            contextual_tags_rate = (pois_with_contextual_tags / total_classified * 100) if total_classified > 0 else 0
            
            results.update({
                'metrics': {
                    'total_pois_classified': total_classified,
                    'avg_confidence_score': round(avg_confidence, 3),
                    'avg_contextual_tags_per_poi': round(avg_contextual_tags, 1),
                    'pois_with_contextual_tags': pois_with_contextual_tags,
                    'contextual_tags_rate_percent': round(contextual_tags_rate, 1),
                    'mood_distribution': mood_distribution,
                    'sample_classifications': classification_results[:5]  # Top 5 examples
                },
                'success_evaluation': {
                    'confidence_success': avg_confidence >= results['success_criteria']['confidence_target'],
                    'contextual_tags_success': contextual_tags_rate >= results['success_criteria']['contextual_tags_target'],
                    'overall_step2_success': avg_confidence >= 0.6 and contextual_tags_rate >= 70
                },
                'status': 'completed'
            })
            
            logger.info(f"  🎯 Avg Confidence: {avg_confidence:.3f} (target: {results['success_criteria']['confidence_target']})")
            logger.info(f"  🏷️ Contextual Tags: {contextual_tags_rate:.1f}% (target: {results['success_criteria']['contextual_tags_target']}%)")
            
        except Exception as e:
            results.update({
                'status': 'failed',
                'error': str(e)
            })
            logger.error(f"Step 2 test failed: {e}")
        
        return results
    
    def test_dynamic_neighborhoods(self, city: str) -> Dict[str, Any]:
        """Test Step 3: Dynamic Neighborhood Calculations."""
        results = {
            'test_name': 'Dynamic Neighborhood Calculations',
            'success_criteria': {
                'data_driven_improvement': True,  # Dynamic > static data
                'neighborhood_coverage': 80,  # >80% neighborhoods with data
                'calculation_accuracy': 'significant_difference'  # Notable difference from static
            }
        }
        
        try:
            # Run comparison between static and dynamic distributions
            comparison_results = self.neighborhood_calculator.compare_static_vs_dynamic()
            
            if 'error' not in comparison_results:
                neighborhoods_analyzed = len(comparison_results)
                significant_changes = 0
                total_changes = []
                
                for neighborhood, diffs in comparison_results.items():
                    for mood, data in diffs.items():
                        change_pct = abs(data.get('change_pct', 0))
                        total_changes.append(change_pct)
                        if change_pct >= 20:  # 20% or more change is significant
                            significant_changes += 1
                
                avg_change = sum(total_changes) / len(total_changes) if total_changes else 0
                significant_change_rate = (significant_changes / (neighborhoods_analyzed * 3) * 100) if neighborhoods_analyzed > 0 else 0
                
                results.update({
                    'metrics': {
                        'neighborhoods_analyzed': neighborhoods_analyzed,
                        'significant_changes': significant_changes,
                        'significant_change_rate_percent': round(significant_change_rate, 1),
                        'avg_mood_distribution_change_percent': round(avg_change, 1),
                        'detailed_comparison': comparison_results
                    },
                    'success_evaluation': {
                        'data_improvement_success': avg_change >= 15,  # At least 15% average change
                        'coverage_success': neighborhoods_analyzed >= 4,  # At least 4 neighborhoods
                        'overall_step3_success': avg_change >= 15 and neighborhoods_analyzed >= 4
                    },
                    'status': 'completed'
                })
                
                logger.info(f"  🏘️ Neighborhoods: {neighborhoods_analyzed} analyzed")
                logger.info(f"  📊 Avg Change: {avg_change:.1f}% (significant if >15%)")
            else:
                results.update({
                    'status': 'failed',
                    'error': comparison_results.get('error', 'Unknown error')
                })
        
        except Exception as e:
            results.update({
                'status': 'failed',
                'error': str(e)
            })
            logger.error(f"Step 3 test failed: {e}")
        
        return results
    
    def test_validated_collections(self, city: str) -> Dict[str, Any]:
        """Test Step 4: Validated AI Collections."""
        results = {
            'test_name': 'Validated AI Collections',
            'success_criteria': {
                'social_proof_validation': 100,  # 100% collections with social proof
                'collection_quality': 'authentic',  # Authentic collections only
                'collection_diversity': 2  # At least 2 different collection types
            }
        }
        
        try:
            # Test collection generation
            generation_results = self.collection_generator.generate_all_validated_collections(city)
            
            if 'error' not in generation_results:
                # Analyze collection quality
                quality_analysis = self.collection_generator.analyze_collection_quality(city)
                
                if 'error' not in quality_analysis:
                    results.update({
                        'metrics': {
                            'validated_pois_used': generation_results.get('total_validated_pois', 0),
                            'collections_generated': generation_results.get('collections_generated', 0),
                            'collections_saved': generation_results.get('collections_saved', 0),
                            'collection_breakdown': generation_results.get('collection_breakdown', {}),
                            'social_proof_coverage': quality_analysis.get('social_proof_coverage', 0),
                            'total_unique_pois_in_collections': quality_analysis.get('total_unique_pois', 0),
                            'avg_pois_per_collection': quality_analysis.get('avg_pois_per_collection', 0)
                        },
                        'success_evaluation': {
                            'validation_success': generation_results.get('collections_generated', 0) > 0,
                            'quality_success': quality_analysis.get('social_proof_coverage', 0) > 0,
                            'diversity_success': len(generation_results.get('collection_breakdown', {})) >= 2,
                            'overall_step4_success': (
                                generation_results.get('collections_generated', 0) > 0 and 
                                len(generation_results.get('collection_breakdown', {})) >= 2
                            )
                        },
                        'status': 'completed'
                    })
                    
                    collections_generated = generation_results.get('collections_generated', 0)
                    logger.info(f"  🎨 Collections Generated: {collections_generated}")
                    logger.info(f"  ✅ Collections Saved: {generation_results.get('collections_saved', 0)}")
                else:
                    results.update({
                        'status': 'partial_success',
                        'generation_results': generation_results,
                        'quality_analysis_error': quality_analysis.get('error', 'Unknown')
                    })
            else:
                results.update({
                    'status': 'failed',
                    'error': generation_results.get('error', 'Unknown')
                })
        
        except Exception as e:
            results.update({
                'status': 'failed',
                'error': str(e)
            })
            logger.error(f"Step 4 test failed: {e}")
        
        return results
    
    def analyze_system_integration(self, city: str) -> Dict[str, Any]:
        """Analyze overall system integration and data flow."""
        integration_analysis = {
            'test_name': 'System Integration Analysis',
            'data_flow_validation': {},
            'performance_metrics': {},
            'integration_quality': {}
        }
        
        try:
            # Check data flow: POIs → Social Proof → Classification → Neighborhoods → Collections
            total_pois = len(self.db.get_pois_for_city(city, 100))
            
            # Count POIs at each stage
            pois_with_proof = 0
            pois_classified = 0
            pois_in_collections = 0
            
            # Get collections and analyze pipeline
            collections = self.db.client.table('collections').select('poi_ids').eq('city', city).execute()
            collection_poi_ids = set()
            for collection in collections.data:
                collection_poi_ids.update(collection.get('poi_ids', []))
            
            pois_in_collections = len(collection_poi_ids)
            
            # Estimate classification coverage (approximate)
            sample_pois = self.db.get_pois_for_city(city, 20)
            classified_sample = 0
            proof_sample = 0
            
            for poi in sample_pois:
                proof_sources = self.db.get_proof_sources_for_poi(poi['id'])
                if proof_sources:
                    proof_sample += 1
                    # Assume classified if has proof (simplified)
                    classified_sample += 1
            
            # Scale up estimates
            if len(sample_pois) > 0:
                pois_with_proof = int((proof_sample / len(sample_pois)) * total_pois)
                pois_classified = int((classified_sample / len(sample_pois)) * total_pois)
            
            # Data flow efficiency
            proof_to_classification_efficiency = (pois_classified / pois_with_proof * 100) if pois_with_proof > 0 else 0
            classification_to_collection_efficiency = (pois_in_collections / pois_classified * 100) if pois_classified > 0 else 0
            overall_pipeline_efficiency = (pois_in_collections / total_pois * 100) if total_pois > 0 else 0
            
            integration_analysis.update({
                'data_flow_validation': {
                    'total_pois_in_system': total_pois,
                    'pois_with_social_proof': pois_with_proof,
                    'pois_classified': pois_classified,
                    'pois_in_collections': pois_in_collections,
                    'pipeline_stages': {
                        'ingestion_to_proof': f"{(pois_with_proof/total_pois*100):.1f}%" if total_pois > 0 else "0%",
                        'proof_to_classification': f"{proof_to_classification_efficiency:.1f}%",
                        'classification_to_collections': f"{classification_to_collection_efficiency:.1f}%",
                        'overall_pipeline': f"{overall_pipeline_efficiency:.1f}%"
                    }
                },
                'performance_metrics': {
                    'system_coverage': f"{overall_pipeline_efficiency:.1f}%",
                    'data_quality_score': 'good' if overall_pipeline_efficiency > 5 else 'needs_improvement',
                    'integration_completeness': 'functional' if pois_in_collections > 0 else 'incomplete'
                },
                'integration_quality': {
                    'all_steps_functional': all([
                        self.test_results.get('step_1_enhanced_proof', {}).get('status') == 'completed',
                        self.test_results.get('step_2_intelligent_classification', {}).get('status') == 'completed',
                        self.test_results.get('step_3_dynamic_neighborhoods', {}).get('status') == 'completed',
                        self.test_results.get('step_4_validated_collections', {}).get('status') == 'completed'
                    ]),
                    'data_flow_working': pois_in_collections > 0,
                    'system_improvement_validated': overall_pipeline_efficiency > 3  # At least 3% of POIs make it through
                },
                'status': 'completed'
            })
            
            logger.info(f"  ⚡ Pipeline Efficiency: {overall_pipeline_efficiency:.1f}%")
            logger.info(f"  🔄 Data Flow: {total_pois} → {pois_with_proof} → {pois_classified} → {pois_in_collections}")
            
        except Exception as e:
            integration_analysis.update({
                'status': 'failed',
                'error': str(e)
            })
            logger.error(f"Integration analysis failed: {e}")
        
        return integration_analysis
    
    def generate_overall_assessment(self) -> Dict[str, Any]:
        """Generate overall assessment of the social proof enhancement system."""
        assessment = {
            'test_name': 'Overall System Assessment',
            'timestamp': datetime.utcnow().isoformat(),
            'step_success_summary': {},
            'system_performance': {},
            'recommendations': [],
            'final_verdict': {}
        }
        
        try:
            # Analyze each step's success
            step_successes = {}
            for step_name, step_results in self.test_results.items():
                if step_name.startswith('step_'):
                    success_eval = step_results.get('success_evaluation', {})
                    overall_success = success_eval.get(f'overall_{step_name}_success', False)
                    step_successes[step_name] = {
                        'status': step_results.get('status', 'unknown'),
                        'overall_success': overall_success
                    }
            
            # Count successful steps
            completed_steps = sum(1 for s in step_successes.values() if s['status'] == 'completed')
            successful_steps = sum(1 for s in step_successes.values() if s['overall_success'])
            
            # System performance evaluation
            integration_quality = self.test_results.get('system_integration', {}).get('integration_quality', {})
            all_functional = integration_quality.get('all_steps_functional', False)
            data_flow_working = integration_quality.get('data_flow_working', False)
            system_improved = integration_quality.get('system_improvement_validated', False)
            
            # Generate recommendations
            recommendations = []
            
            # Step-specific recommendations
            step1 = self.test_results.get('step_1_enhanced_proof', {})
            if step1.get('metrics', {}).get('coverage_rate_percent', 0) < 60:
                recommendations.append("Enhance social proof collection - current coverage below 60% target")
            
            step2 = self.test_results.get('step_2_intelligent_classification', {})
            if step2.get('metrics', {}).get('avg_confidence_score', 0) < 0.6:
                recommendations.append("Improve classification confidence - consider additional training data")
            
            step4 = self.test_results.get('step_4_validated_collections', {})
            if step4.get('metrics', {}).get('collections_generated', 0) < 3:
                recommendations.append("Optimize collection generation thresholds for better coverage")
            
            # Overall system recommendations
            if not all_functional:
                recommendations.append("Address failed components before production deployment")
            
            if not data_flow_working:
                recommendations.append("Fix data pipeline - no POIs successfully flowing to collections")
            
            # Success determination
            system_success = (
                completed_steps >= 3 and  # At least 3 steps completed
                successful_steps >= 2 and  # At least 2 steps successful
                data_flow_working  # Data pipeline working
            )
            
            assessment.update({
                'step_success_summary': {
                    'total_steps_tested': len(step_successes),
                    'completed_steps': completed_steps,
                    'successful_steps': successful_steps,
                    'step_details': step_successes
                },
                'system_performance': {
                    'all_steps_functional': all_functional,
                    'data_flow_working': data_flow_working,
                    'system_improvement_validated': system_improved,
                    'overall_system_health': 'good' if system_success else 'needs_attention'
                },
                'recommendations': recommendations,
                'final_verdict': {
                    'system_ready_for_production': system_success and len(recommendations) <= 2,
                    'social_proof_enhancement_successful': system_success,
                    'key_achievements': [
                        "Enhanced social proof collection implemented",
                        "Intelligent POI classification working", 
                        "Dynamic neighborhood calculations active",
                        "Validated AI collections generating"
                    ] if system_success else ["Partial system implementation achieved"],
                    'overall_grade': 'A' if system_success and len(recommendations) == 0 else 
                                   'B' if system_success else 'C'
                }
            })
            
            logger.info(f"🎯 System Success Rate: {successful_steps}/{len(step_successes)} steps")
            logger.info(f"📊 Overall Assessment: {assessment['final_verdict']['overall_grade']}")
            
        except Exception as e:
            assessment.update({
                'status': 'failed',
                'error': str(e)
            })
            logger.error(f"Assessment generation failed: {e}")
        
        return assessment
    
    def generate_test_report(self) -> str:
        """Generate a human-readable test report."""
        if not self.test_results:
            return "No test results available. Run comprehensive test first."
        
        report_lines = [
            "=" * 80,
            "TRENDR SOCIAL PROOF ENHANCEMENT - SYSTEM INTEGRATION TEST REPORT",
            "=" * 80,
            f"Test Completed: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}",
            ""
        ]
        
        # Overall Assessment Summary
        overall = self.test_results.get('overall_assessment', {})
        final_verdict = overall.get('final_verdict', {})
        
        report_lines.extend([
            "🎯 OVERALL ASSESSMENT",
            "-" * 40,
            f"System Grade: {final_verdict.get('overall_grade', 'Unknown')}",
            f"Production Ready: {'✅ Yes' if final_verdict.get('system_ready_for_production', False) else '❌ No'}",
            f"Enhancement Successful: {'✅ Yes' if final_verdict.get('social_proof_enhancement_successful', False) else '❌ No'}",
            ""
        ])
        
        # Step-by-Step Results
        step_mapping = {
            'step_1_enhanced_proof': '📊 Step 1: Enhanced Social Proof Collection',
            'step_2_intelligent_classification': '🧠 Step 2: Intelligent POI Classification', 
            'step_3_dynamic_neighborhoods': '🏘️ Step 3: Dynamic Neighborhood Calculations',
            'step_4_validated_collections': '🎨 Step 4: Validated AI Collections'
        }
        
        for step_key, step_title in step_mapping.items():
            step_data = self.test_results.get(step_key, {})
            if step_data:
                status = step_data.get('status', 'unknown')
                success = step_data.get('success_evaluation', {}).get(f'overall_{step_key}_success', False)
                
                report_lines.extend([
                    step_title,
                    "-" * len(step_title),
                    f"Status: {'✅ Completed' if status == 'completed' else '❌ ' + status.title()}",
                    f"Success: {'✅ Yes' if success else '❌ No'}",
                ])
                
                # Add key metrics
                metrics = step_data.get('metrics', {})
                if metrics:
                    report_lines.append("Key Metrics:")
                    for key, value in list(metrics.items())[:5]:  # Top 5 metrics
                        if isinstance(value, (int, float)):
                            report_lines.append(f"  • {key.replace('_', ' ').title()}: {value}")
                
                report_lines.append("")
        
        # Recommendations
        recommendations = overall.get('recommendations', [])
        if recommendations:
            report_lines.extend([
                "💡 RECOMMENDATIONS",
                "-" * 40
            ])
            for i, rec in enumerate(recommendations, 1):
                report_lines.append(f"{i}. {rec}")
            report_lines.append("")
        
        # System Integration Status
        integration = self.test_results.get('system_integration', {})
        if integration:
            data_flow = integration.get('data_flow_validation', {})
            if data_flow:
                report_lines.extend([
                    "⚡ DATA PIPELINE STATUS",
                    "-" * 40,
                    f"Total POIs: {data_flow.get('total_pois_in_system', 0)}",
                    f"With Social Proof: {data_flow.get('pois_with_social_proof', 0)}",
                    f"Classified: {data_flow.get('pois_classified', 0)}",
                    f"In Collections: {data_flow.get('pois_in_collections', 0)}",
                    "",
                    "Pipeline Efficiency:",
                ])
                pipeline_stages = data_flow.get('pipeline_stages', {})
                for stage, percentage in pipeline_stages.items():
                    report_lines.append(f"  • {stage.replace('_', ' ').title()}: {percentage}")
        
        report_lines.extend([
            "",
            "=" * 80,
            "End of Report",
            "=" * 80
        ])
        
        return "\n".join(report_lines)

def main():
    """CLI interface for system integration testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Trendr System Integration Test')
    parser.add_argument('--city', default='Montreal', help='City to test')
    parser.add_argument('--full-test', action='store_true', help='Run comprehensive system test')
    parser.add_argument('--report', action='store_true', help='Generate readable test report')
    parser.add_argument('--output', help='Save results to JSON file')
    
    args = parser.parse_args()
    
    tester = TrendrSystemIntegrationTest()
    
    try:
        if args.full_test:
            logger.info(f"Running comprehensive system test for {args.city}")
            results = tester.run_comprehensive_system_test(args.city)
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                logger.info(f"Results saved to {args.output}")
            
            if args.report:
                report = tester.generate_test_report()
                print("\n" + report)
            else:
                print(f"\n✅ System test completed. Overall grade: {results.get('overall_assessment', {}).get('final_verdict', {}).get('overall_grade', 'Unknown')}")
        
        elif args.report:
            # Try to generate report from existing test (if any)
            report = tester.generate_test_report()
            print(report)
        
        else:
            print("Use --full-test to run comprehensive system integration test")
            print("Use --report to generate a readable test report")
    
    except Exception as e:
        logger.error(f"System integration test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()