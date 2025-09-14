#!/usr/bin/env python
import sys
import os
import logging
import json
import argparse
import subprocess
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TrendrPipelineOrchestrator:
    """Main orchestrator for Trendr data pipeline - Gatto version"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.stats = {
            'ingested': 0,
            'mentions_processed': 0, 
            'classified': 0,
            'photos_processed': 0,
            'errors': []
        }
        self.config = {}
        self.merged_params = {}
        
    def load_config(self, config_path: str = "config.json") -> Dict[str, Any]:
        """Load pipeline configuration from JSON file"""
        if not os.path.exists(config_path):
            logger.error(f"❌ Configuration file not found: {config_path}")
            sys.exit(1)
            
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"✅ Configuration loaded from {config_path}")
            return config
        except json.JSONDecodeError as e:
            logger.error(f"❌ Invalid JSON in configuration file: {e}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"❌ Error loading configuration: {e}")
            sys.exit(1)
    
    def merge_params(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Merge CLI arguments with config.json parameters"""
        merged = {}
        
        # Extract relevant config sections
        pipeline_config = self.config.get('pipeline_config', {})
        api_config = self.config.get('api_cost_optimization', {})
        social_config = self.config.get('social_proof_config', {})
        
        # Cities: CLI takes precedence, then config
        if args.city:
            merged['cities'] = [args.city]
        else:
            merged['cities'] = pipeline_config.get('cities', ['Paris'])
        
        # Other parameters
        merged['mode'] = 'full' if args.mode == 'collections' else args.mode  # collections -> full mapping
        merged['batch_size'] = args.limit if args.limit else pipeline_config.get('batch_size', 25)
        merged['dry_run'] = args.dry_run
        merged['fields'] = args.fields
        merged['serp_only'] = args.serp_only
        merged['cse_num'] = args.cse_num  # No cap limit
        merged['explain'] = args.explain
        merged['debug'] = args.debug
        
        # sources optionnelles depuis config (neutre si absent)
        merged['sources'] = (self.config.get('social_proof_config', {}) or {}).get('sources')
        
        # Seed pipeline parameters
        merged['seed_poi_name'] = args.seed_poi_name
        merged['seed_place_id'] = args.seed_place_id
        merged['seed_city'] = args.seed_city
        merged['seed_commit'] = not args.no_commit
        merged['seed_stdout_json'] = args.seed_stdout_json
        
        # Config values
        merged['poi_categories'] = pipeline_config.get('poi_categories', [])
        merged['photos_enabled'] = pipeline_config.get('photos_enabled', True)
        merged['incremental_mode'] = pipeline_config.get('incremental_mode', True)
        merged['daily_api_limit'] = api_config.get('daily_api_limit', 150)
        merged['daily_api_budget'] = api_config.get('daily_api_budget', 1.5)
        merged['max_requests_per_day'] = api_config.get('max_requests_per_day', 950)
        merged['max_queries_per_poi'] = social_config.get('max_queries_per_poi', 3)
        merged['cache_enabled'] = social_config.get('cache_enabled', True)
        merged['cache_ttl_hours'] = social_config.get('cache_ttl_hours', 48)
        
        return merged
    
    def log_effective_params(self):
        """Log the effective parameters after CLI + config merge"""
        logger.info("🔧 EFFECTIVE PARAMETERS:")
        logger.info(f"   Cities: {self.merged_params['cities']}")
        logger.info(f"   Mode: {self.merged_params['mode']}")
        logger.info(f"   Batch size: {self.merged_params['batch_size']}")
        logger.info(f"   Fields: {self.merged_params['fields']}")
        logger.info(f"   Dry run: {self.merged_params['dry_run']}")
        logger.info(f"   Photos enabled: {self.merged_params['photos_enabled']}")
        logger.info(f"   Daily API budget: ${self.merged_params['daily_api_budget']}")
        logger.info(f"   Max requests/day: {self.merged_params['max_requests_per_day']}")
        if self.merged_params['dry_run']:
            logger.info("🔥 DRY-RUN: aucune requête externe ni écriture DB")
    
    def run_subprocess(self, cmd: List[str], step_name: str) -> bool:
        """Run subprocess and handle errors gracefully"""
        step_start = time.time()
        logger.info(f"[{step_name}] 🚀 Starting: {' '.join(cmd)}")
        
        try:
            env = os.environ.copy()
            root = os.getcwd()
            env['PYTHONPATH'] = f"{root}:{root}/scripts" + ((":" + env['PYTHONPATH']) if 'PYTHONPATH' in env else "")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800,  # 30 minutes timeout
                env=env
            )
            
            duration = time.time() - step_start
            
            if result.returncode == 0:
                logger.info(f"[{step_name}] ✅ Completed in {duration:.1f}s")
                
                # Parse output for statistics if available
                if result.stdout:
                    self._parse_step_stats(result.stdout, step_name)
                    # Log last few lines of output for visibility
                    lines = result.stdout.strip().split('\n')
                    for line in lines[-3:]:
                        if line.strip():
                            logger.info(f"[{step_name}] {line.strip()}")
                
                return True
            else:
                logger.error(f"[{step_name}] ❌ Failed with exit code {result.returncode}")
                logger.error(f"[{step_name}] Error output: {result.stderr}")
                
                # Check for unsupported flag error and retry (only once)
                if ("unrecognized arguments" in result.stderr or "invalid choice" in result.stderr) and "RETRY" not in step_name:
                    logger.warning(f"[{step_name}] Flag non supporté par le script, relance sans flags optionnels")
                    return self._retry_without_optional_flags(cmd, step_name)
                
                self.stats['errors'].append(f"{step_name}: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"[{step_name}] ❌ Timeout after 30 minutes")
            self.stats['errors'].append(f"{step_name}: Timeout")
            return False
        except Exception as e:
            logger.error(f"[{step_name}] ❌ Subprocess error: {e}")
            self.stats['errors'].append(f"{step_name}: {str(e)}")
            return False
    
    def _retry_without_optional_flags(self, original_cmd: List[str], step_name: str) -> bool:
        """Retry command with only essential flags for compatibility"""
        # Handle new module invocation: python3 -m scripts.mention_scanner
        if len(original_cmd) >= 3 and original_cmd[1] == '-m' and original_cmd[2] == 'scripts.mention_scanner':
            essential_cmd = ['python3', '-m', 'scripts.mention_scanner']
            
            # Keep only essential mention scanner flags
            skip_next = False
            for i, arg in enumerate(original_cmd[3:], 3):
                if skip_next:
                    skip_next = False
                    continue
                    
                # Essential flags that mention scanner needs
                if arg in ['--mode', '--city-slug', '--poi-name', '--cse-num', '--limit-per-poi']:
                    if i + 1 < len(original_cmd):
                        essential_cmd.extend([arg, original_cmd[i + 1]])
                        skip_next = True
        else:
            # Fallback for old script invocation: python3 script.py
            essential_cmd = [original_cmd[0], original_cmd[1]]  # python3 script.py
            
            skip_next = False
            for i, arg in enumerate(original_cmd[2:], 2):
                if skip_next:
                    skip_next = False
                    continue
                    
                # Only keep the most basic flags that are likely to work
                if arg in ['--city-slug', '--scan', '--score-city', '--backfill']:
                    if arg in ['--scan', '--backfill']:
                        essential_cmd.append(arg)
                    elif i + 1 < len(original_cmd):
                        essential_cmd.extend([arg, original_cmd[i + 1]])
                        skip_next = True
        
        logger.info(f"[{step_name}] 🔄 Retry with: {' '.join(essential_cmd)}")
        return self.run_subprocess(essential_cmd, f"{step_name}-RETRY")
    
    def run_subprocess_with_output(self, cmd: List[str], step_name: str) -> tuple[bool, str]:
        """Run subprocess and return both success status and stdout"""
        step_start = time.time()
        logger.info(f"[{step_name}] 🚀 Starting: {' '.join(cmd)}")
        
        try:
            env = os.environ.copy()
            root = os.getcwd()
            env['PYTHONPATH'] = f"{root}:{root}/scripts" + ((":" + env['PYTHONPATH']) if 'PYTHONPATH' in env else "")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800,  # 30 minutes timeout
                env=env
            )
            
            duration = time.time() - step_start
            
            if result.returncode == 0:
                logger.info(f"[{step_name}] ✅ Completed in {duration:.1f}s")
                return True, result.stdout
            else:
                logger.error(f"[{step_name}] ❌ Failed with exit code {result.returncode}")
                logger.error(f"[{step_name}] Error output: {result.stderr}")
                self.stats['errors'].append(f"{step_name}: {result.stderr}")
                return False, result.stdout
                
        except subprocess.TimeoutExpired:
            logger.error(f"[{step_name}] ❌ Timeout after 30 minutes")
            self.stats['errors'].append(f"{step_name}: Timeout")
            return False, ""
        except Exception as e:
            logger.error(f"[{step_name}] ❌ Subprocess error: {e}")
            self.stats['errors'].append(f"{step_name}: {str(e)}")
            return False, ""
    
    def _parse_poi_id_from_output(self, output: str) -> Optional[str]:
        """Extract POI ID from ingester output"""
        import re
        # Look for pattern "UPSERT ok: <uuid> | <city> | <name>"
        pattern = r"UPSERT ok: ([a-f0-9-]{36})"
        match = re.search(pattern, output)
        if match:
            return match.group(1)
        return None
    
    def _parse_json_output(self, output: str) -> tuple[Optional[str], Optional[str]]:
        import json
        # Parse ligne par ligne pour trouver le JSON valide
        for line in output.strip().split('\n'):
            line = line.strip()
            if line.startswith('{') and line.endswith('}'):
                try:
                    data = json.loads(line)
                    status = data.get('status')
                    if status == 'upserted':
                        poi_info = data.get('poi', {}) or data.get('data', {}).get('poi', {})
                        poi_id = poi_info.get('id')
                        poi_name = poi_info.get('name')
                        logger.info(f"✅ JSON: POI upserted with ID: {poi_id}")
                        return poi_id, poi_name
                    if status == 'dry_run':
                        poi_preview = data.get('poi_preview', {}) or data.get('data', {}).get('poi_preview', {})
                        poi_name = poi_preview.get('name')
                        logger.info(f"📝 JSON: POI preview name: {poi_name}")
                        return None, poi_name
                    if status == 'error':
                        logger.error(f"❌ JSON: Seed ingestion error: {data.get('message','Unknown')} (code: {data.get('code','unknown')})")
                        return None, None
                except json.JSONDecodeError:
                    continue
        logger.warning("No valid JSON found in ingester output")
        return None, None
    
    def run_seed_ingestion(self) -> tuple[bool, Optional[str], Optional[str]]:
        """Run seed POI ingestion step and return success, poi_id, poi_name"""
        if self.merged_params['seed_place_id']:
            # Seed by Place ID
            cmd = [
                'python3', 'scripts/google_places_ingester.py',
                '--place-id', self.merged_params['seed_place_id']
            ]
            if self.merged_params['seed_city']:
                cmd.extend(['--city', self.merged_params['seed_city']])
        elif self.merged_params['seed_poi_name']:
            # Seed by POI name
            cmd = [
                'python3', 'scripts/google_places_ingester.py',
                '--poi-name', self.merged_params['seed_poi_name'],
                '--city', self.merged_params['seed_city']
            ]
        else:
            return False, None, None
            
        # Add commit flag if enabled
        if self.merged_params['seed_commit']:
            cmd.append('--commit')
        
        # Add stdout-json flag if enabled
        if self.merged_params['seed_stdout_json']:
            cmd.append('--stdout-json')
            
        success, output = self.run_subprocess_with_output(cmd, 'SEED-INGEST')
        
        if success:
            # Handle JSON output parsing if enabled
            if self.merged_params['seed_stdout_json']:
                poi_id, poi_name = self._parse_json_output(output)
                return True, poi_id, poi_name
            else:
                # Legacy heuristic parsing
                poi_id = None
                if self.merged_params['seed_commit']:
                    poi_id = self._parse_poi_id_from_output(output)
                
                # Return POI name for mention scanner
                poi_name = self.merged_params['seed_poi_name'] if self.merged_params['seed_poi_name'] else None
                
                return True, poi_id, poi_name
        else:
            return False, None, None
    
    def run_seed_mention_scan(self, poi_id: Optional[str], poi_name: Optional[str]) -> bool:
        """Run mention scanner for the seeded POI"""
        city = (self.merged_params['seed_city'] or 'paris').lower()
        
        # Use balanced mode from config, fallback to balanced if not specified
        mention_config = self.config.get('mention_scanner', {})
        scan_mode = mention_config.get('mode', 'balanced')
        if self.merged_params.get('serp_only'):
            scan_mode = 'serp-only'
        
        cmd = ['python3','-m','scripts.mention_scanner',
                '--mode', scan_mode,
                '--city-slug', city,
                '--cse-num', str(self.merged_params.get('cse_num',50))]
        
        # Add sources ONLY for serp-only mode
        # balanced mode uses source_catalog automatically, don't override with manual sources
        if scan_mode == 'serp-only' and (self.config.get('social_proof_config') or {}).get('sources'):
            cmd += ['--sources', ','.join(self.config['social_proof_config']['sources'])]
        
        # Debug extras - always enable for seed pipeline to get visibility
        debug_mode = (logger.level == logging.DEBUG) or os.getenv("SCAN_DEBUG") == "1" or self.merged_params.get('debug', False)
        if debug_mode:
            cmd += [
                '--debug',
                '--log-drop-reasons',
                '--jsonl-out', 'out/seed_scan.jsonl',
                '--dump-serp', 'out/serp_dumps',
                '--no-cache'
            ]
            os.makedirs('out/serp_dumps', exist_ok=True)
        
        if poi_name:
            if poi_id:
                logger.info(f"SEED-MENTIONS: using poi-name {poi_name} (poi-id {poi_id})")
            else:
                logger.info(f"SEED-MENTIONS: using poi-name {poi_name}")
            cmd += ['--poi-name', poi_name]
        elif poi_id:
            # Lookup POI name from database using poi_id
            try:
                from utils.database import SupabaseManager
                db = SupabaseManager()
                result = db.client.table('poi').select('name').eq('id', poi_id).limit(1).execute()
                if result.data and len(result.data) > 0:
                    poi_name = result.data[0]['name']
                    logger.info(f"SEED-MENTIONS: resolved poi-name {poi_name} from poi-id {poi_id}")
                    cmd += ['--poi-name', poi_name]
                else:
                    logger.error(f"POI ID {poi_id} not found in database")
                    return False
            except Exception as e:
                logger.error(f"Failed to lookup POI name for ID {poi_id}: {e}")
                return False
        else:
            logger.error("No POI ID or name available for mention scan")
            return False
            
        return self.run_subprocess(cmd, 'SEED-MENTIONS')
    
    def run_seed_pipeline(self) -> bool:
        """Execute SEED → SCAN pipeline"""
        logger.info("🌱 Starting SEED → SCAN pipeline")
        
        # Step 1: Seed POI ingestion
        logger.info("📍 STEP 1: POI Ingestion")
        success, poi_id, poi_name = self.run_seed_ingestion()
        
        if not poi_id and poi_name:
            try:
                from utils.database import SupabaseManager
                db = SupabaseManager()
                # Use raw SQL query via Supabase client
                result = db.client.table('poi').select('id').ilike('name', poi_name).eq('city_slug', (self.merged_params['seed_city'] or 'paris').lower()).order('updated_at', desc=True).limit(1).execute()
                if result.data and len(result.data) > 0:
                    poi_id = result.data[0]['id']
                    logger.info(f"🔍 Fallback DB: resolved poi_id={poi_id} for '{poi_name}'")
            except Exception as e:
                logger.warning(f"Fallback DB failed: {e}")
        
        if not success:
            logger.error("❌ Seed ingestion failed, stopping pipeline")
            return False
        
        # Handle JSON parsing errors (both poi_id and poi_name are None)
        if self.merged_params['seed_stdout_json'] and poi_id is None and poi_name is None:
            logger.error("❌ JSON parsing failed or error status received, stopping pipeline")
            return False
            
        # Log what was extracted
        if poi_id:
            logger.info(f"✅ POI ingested with ID: {poi_id}")
        if poi_name:
            logger.info(f"📝 POI name for mentions: {poi_name}")
            
        # Step 2: Mention scanning
        logger.info("🔍 STEP 2: Mention Scanning")
        success = self.run_seed_mention_scan(poi_id, poi_name)
        
        if not success:
            logger.error("❌ Mention scanning failed")
            return False
            
        logger.info("✅ SEED → SCAN pipeline completed successfully")
        return True
    
    def _parse_step_stats(self, output: str, step_name: str):
        """Parse statistics from subprocess output"""
        try:
            lines = output.split('\n')
            for line in lines:
                # Look for common patterns
                if 'ingested' in line.lower() and any(c.isdigit() for c in line):
                    numbers = [int(s) for s in line.split() if s.isdigit()]
                    if numbers and step_name == 'INGEST':
                        self.stats['ingested'] += numbers[0]
                elif 'classified' in line.lower() and any(c.isdigit() for c in line):
                    numbers = [int(s) for s in line.split() if s.isdigit()]
                    if numbers and step_name == 'CLASSIFY':
                        self.stats['classified'] += numbers[0]
                elif 'processed' in line.lower() and any(c.isdigit() for c in line):
                    numbers = [int(s) for s in line.split() if s.isdigit()]
                    if numbers:
                        if step_name == 'MENTIONS':
                            self.stats['mentions_processed'] += numbers[0]
                        elif step_name == 'PHOTOS':
                            self.stats['photos_processed'] += numbers[0]
        except Exception:
            pass  # Ignore parsing errors
    
    def run_ingestion_step(self, city: str) -> bool:
        """Run ingestion step"""
        cmd = [
            'python3', 'scripts/google_places_ingester.py',
            '--seed',
            '--city-slug', city.lower(),
            '--limit', str(self.merged_params['batch_size'])
        ]
        
        if logger.level == logging.DEBUG:
            cmd.append('--debug')
        
        if self.merged_params['dry_run']:
            cmd.append('--dry-run')
            
        return self.run_subprocess(cmd, 'INGEST')
    
    def run_mentions_step(self, city: str) -> bool:
        """Run mentions scanning step"""
        # Use balanced mode from config, fallback to balanced if not specified
        mention_config = self.config.get('mention_scanner', {})
        scan_mode = mention_config.get('mode', 'balanced')
        if self.merged_params.get('serp_only'):
            scan_mode = 'serp-only'
            
        cmd = ['python3','-m','scripts.mention_scanner',
                '--mode', scan_mode,
                '--city-slug', city.lower(),
                '--cse-num', str(self.merged_params.get('cse_num',50))]
        
        # Add sources ONLY for serp-only mode
        # balanced mode uses source_catalog automatically, don't override with manual sources
        if scan_mode == 'serp-only' and (self.config.get('social_proof_config') or {}).get('sources'):
            cmd += ['--sources', ','.join(self.config['social_proof_config']['sources'])]
            
        # Debug options
        debug_mode = (logger.level == logging.DEBUG) or self.merged_params.get('debug', False)
        if debug_mode:
            cmd += ['--debug', '--log-drop-reasons']
            
        return self.run_subprocess(cmd, 'MENTIONS')
    
    def run_classification_step(self, city: str) -> bool:
        """Run classification step"""
        # Note: intelligent_classifier doesn't support --limit, it processes all POIs for a city
        cmd = [
            'python3', 'scripts/intelligent_classifier.py',
            '--score-city', city.lower()
        ]
        
        if logger.level == logging.DEBUG or self.merged_params['explain']:
            cmd.append('--debug')
        
        # Add force flag for testing to override recent scoring
        if self.merged_params['dry_run'] or self.merged_params['batch_size'] <= 10:
            cmd.append('--force')
            
        return self.run_subprocess(cmd, 'CLASSIFY')
    
    def run_photos_step(self, city: str) -> bool:
        """Run photo processing step"""
        if not self.merged_params['photos_enabled']:
            logger.info("[PHOTOS] ⏭️  Photos disabled in config")
            return True
            
        cmd = [
            'python3', 'scripts/photo_processor.py',
            '--backfill',
            '--city', city,
            '--limit', str(self.merged_params['batch_size']),
            '--rate-limit', '1.0'
        ]
            
        return self.run_subprocess(cmd, 'PHOTOS')
    
    def run_pipeline_mode(self) -> bool:
        """Execute pipeline according to selected mode"""
        success = True
        
        # Check if this is a seed pipeline execution
        if self.merged_params['seed_poi_name'] or self.merged_params['seed_place_id']:
            logger.info("🌱 SEED → SCAN Pipeline Mode")
            return self.run_seed_pipeline()
        
        for city in self.merged_params['cities']:
            logger.info(f"🏙️  Processing city: {city}")
            
            if self.merged_params['mode'] in ['full', 'collections']:
                # Full pipeline: all steps
                if not self.run_ingestion_step(city):
                    success = False
                    break
                if not self.run_mentions_step(city):
                    success = False
                    break  
                if not self.run_classification_step(city):
                    success = False
                    break
                if not self.run_photos_step(city):
                    success = False
                    break
                    
            elif self.merged_params['mode'] == 'ingest':
                if not self.run_ingestion_step(city):
                    success = False
                    break
                    
            elif self.merged_params['mode'] == 'mentions':
                if not self.run_mentions_step(city):
                    success = False
                    break
                    
            elif self.merged_params['mode'] == 'classify':
                if not self.run_classification_step(city):
                    success = False
                    break
                    
            elif self.merged_params['mode'] == 'photos':
                if not self.run_photos_step(city):
                    success = False
                    break
        
        return success
    
    def print_final_summary(self):
        """Print final execution summary"""
        duration = datetime.now() - self.start_time
        
        print("\n" + "="*60)
        print("📊 TRENDR PIPELINE EXECUTION SUMMARY")
        print("="*60)
        print(f"⏱️  Duration: {duration}")
        print(f"🏙️  Cities processed: {len(self.merged_params['cities'])}")
        print(f"📍 POIs ingested: {self.stats['ingested']}")
        print(f"🔍 Mentions processed: {self.stats['mentions_processed']}")
        print(f"🤖 POIs classified: {self.stats['classified']}")
        print(f"📸 Photos processed: {self.stats['photos_processed']}")
        print(f"❌ Errors: {len(self.stats['errors'])}")
        
        if self.stats['errors']:
            print("\n🚨 ERRORS ENCOUNTERED:")
            for error in self.stats['errors'][:5]:
                print(f"  • {error}")
            if len(self.stats['errors']) > 5:
                print(f"  ... and {len(self.stats['errors']) - 5} more errors")
        
        print("="*60)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Trendr Data Pipeline - Gatto Orchestrator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 run_pipeline.py --city Paris --mode full --limit 5 --dry-run
  python3 run_pipeline.py --mode mentions --city Paris --serp-only --cse-num 10  
  python3 run_pipeline.py --mode classify --city Paris --limit 50 --explain
  
Seed Pipeline (SEED → SCAN):
  python3 run_pipeline.py --mode mentions --seed-poi-name "Septime" --seed-city Paris
  python3 run_pipeline.py --mode mentions --seed-place-id "ChIJ..." 
  python3 run_pipeline.py --mode mentions --seed-poi-name "Frenchie" --seed-city Paris --seed-stdout-json
        """
    )
    
    # Main arguments
    parser.add_argument('--city', help='City to process (if not provided, uses config.json cities)')
    parser.add_argument('--mode', choices=['full', 'ingest', 'mentions', 'classify', 'photos', 'collections'],
                       default='full', help='Pipeline execution mode (collections=alias for full)')
    parser.add_argument('--limit', type=int, help='Batch size for each step (overrides config.json batch_size)')
    parser.add_argument('--dry-run', action='store_true', help='Log actions without network/DB calls')
    
    # Field and optimization arguments
    parser.add_argument('--fields', choices=['basic', 'contact', 'all'], default='basic',
                       help='Field mask for ingestion API calls')
    
    # Mentions-specific arguments  
    parser.add_argument('--serp-only', action='store_true', help='Use only SERP API for mentions')
    parser.add_argument('--cse-num', type=int, default=50, help='Number of CSE results (default 50, no max limit)')
    
    # Classification-specific arguments
    parser.add_argument('--explain', action='store_true', help='Show detailed score components in classification')
    
    # Debug arguments
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with detailed logging and output files')
    
    # Seed POI arguments for chained SEED → SCAN pipeline
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument('--seed-poi-name', help='Seed POI by name for targeted ingestion then mention scan')
    seed_group.add_argument('--seed-place-id', help='Seed POI by Google Place ID for targeted ingestion then mention scan')
    parser.add_argument('--seed-city', help='City for seed POI (required with --seed-poi-name)')
    parser.add_argument('--no-commit', action='store_true', help='Run seed ingestion in dry-run mode (default: commit)')
    parser.add_argument('--seed-stdout-json', action='store_true', help='Use JSON output parsing for SEED→SCAN pipeline')
    
    args = parser.parse_args()
    
    # Validate seed arguments
    if args.seed_poi_name and not args.seed_city:
        parser.error("--seed-city is required when using --seed-poi-name")
    
    # Initialize orchestrator
    orchestrator = TrendrPipelineOrchestrator()
    
    try:
        # Load configuration
        orchestrator.config = orchestrator.load_config()
        
        # Merge CLI + config parameters
        orchestrator.merged_params = orchestrator.merge_params(args)
        
        # Log effective parameters
        orchestrator.log_effective_params()
        
        # Execute pipeline
        logger.info("🚀 STARTING TRENDR PIPELINE EXECUTION")
        success = orchestrator.run_pipeline_mode()
        
        # Print summary
        orchestrator.print_final_summary()
        
        if success:
            logger.info("✅ PIPELINE COMPLETED SUCCESSFULLY")
            return 0
        else:
            logger.error("❌ PIPELINE FAILED - CHECK ERRORS ABOVE")
            return 1
            
    except KeyboardInterrupt:
        logger.info("\n⚠️  Pipeline interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"❌ CRITICAL ERROR: {e}")
        if logger.level == logging.DEBUG:
            import traceback
            logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())