#!/bin/bash

# Trendr Data Pipeline - Automatic Deployment Script
# Configures and prepares the pipeline for execution on VPS

set -e  # Stop on error

echo "🚀 TRENDR DATA PIPELINE DEPLOYMENT"
echo "===================================="

# Configuration variables
PROJECT_DIR="/var/trendr/data-pipeline"
PYTHON_VERSION="3.9"
VENV_NAME="trendr_env"
LOG_DIR="/var/log/trendr"
DATA_DIR="/var/data/trendr"

# Colors for display
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 not found. Installation required."
        exit 1
    fi
    
    # Check pip
    if ! command -v pip3 &> /dev/null; then
        print_error "pip3 not found. Installation required."
        exit 1
    fi
    
    # Check git (optional)
    if ! command -v git &> /dev/null; then
        print_warning "Git not found. Manual repo clone required."
    fi
    
    print_status "✅ Prerequisites checked"
}

# Create necessary directories
create_directories() {
    print_status "Creating directories..."
    
    sudo mkdir -p "$PROJECT_DIR"
    sudo mkdir -p "$LOG_DIR"
    sudo mkdir -p "$DATA_DIR"
    sudo mkdir -p "$DATA_DIR/cache"
    sudo mkdir -p "$DATA_DIR/backups"
    
    # Permissions
    sudo chown -R $USER:$USER "$PROJECT_DIR" "$LOG_DIR" "$DATA_DIR"
    
    print_status "✅ Directories created"
}

# Python environment setup
setup_python_env() {
    print_status "Setting up Python environment..."
    
    cd "$PROJECT_DIR"
    
    # Create virtual environment
    if [ ! -d "$VENV_NAME" ]; then
        python3 -m venv "$VENV_NAME"
        print_status "Virtual environment created"
    else
        print_status "Virtual environment already exists"
    fi
    
    # Activate environment
    source "$VENV_NAME/bin/activate"
    
    # Update pip
    pip install --upgrade pip
    
    # Install dependencies
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        print_status "✅ Dependencies installed from requirements.txt"
    else
        print_warning "requirements.txt not found, installing dependencies manually..."
        
        # Main dependencies
        pip install supabase python-dotenv requests tenacity dateutil openai anthropic
        
        print_status "✅ Base dependencies installed"
    fi
}

# Environment variables setup
setup_environment() {
    print_status "Setting up environment variables..."
    
    ENV_FILE="$PROJECT_DIR/.env"
    ENV_EXAMPLE="$PROJECT_DIR/.env.example"
    
    # Create .env.example if not existing
    if [ ! -f "$ENV_EXAMPLE" ]; then
        cat > "$ENV_EXAMPLE" << EOF
# Supabase Configuration
SUPABASE_URL=your_supabase_url_here
SUPABASE_KEY=your_supabase_key_here

# Google APIs
GOOGLE_PLACES_API_KEY=your_google_places_api_key_here
GOOGLE_CUSTOM_SEARCH_API_KEY=your_google_custom_search_api_key_here
GOOGLE_CUSTOM_SEARCH_ENGINE_ID=your_search_engine_id_here

# AI APIs (Optional)
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Pipeline Configuration
PIPELINE_LOG_LEVEL=INFO
PIPELINE_DATA_DIR=$DATA_DIR
PIPELINE_CACHE_DIR=$DATA_DIR/cache
EOF
        print_status ".env.example created"
    fi
    
    # Check if .env exists
    if [ ! -f "$ENV_FILE" ]; then
        print_warning ".env not found. Copying from .env.example..."
        cp "$ENV_EXAMPLE" "$ENV_FILE"
        print_warning "⚠️  IMPORTANT: Configure the variables in $ENV_FILE"
    else
        print_status "✅ .env found"
    fi
}

# Cron setup for automatic execution
setup_cron() {
    print_status "Setting up cron for automatic execution..."
    
    CRON_SCRIPT="/usr/local/bin/trendr_pipeline"
    
    # Create execution script
    sudo tee "$CRON_SCRIPT" > /dev/null << EOF
#!/bin/bash

# Trendr pipeline automatic execution script
export PATH="/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="$PROJECT_DIR"

cd "$PROJECT_DIR"
source "$VENV_NAME/bin/activate"

# Log with timestamp
echo "\$(date): Starting Trendr pipeline" >> "$LOG_DIR/cron.log"

# Execute pipeline
python run_pipeline.py --config config.json >> "$LOG_DIR/pipeline.log" 2>&1

# Exit status
if [ \$? -eq 0 ]; then
    echo "\$(date): Pipeline completed successfully" >> "$LOG_DIR/cron.log"
else
    echo "\$(date): Pipeline completed with error" >> "$LOG_DIR/cron.log"
fi
EOF

    # Execution permissions
    sudo chmod +x "$CRON_SCRIPT"
    
    # Add to crontab (daily execution at 6 AM)
    CRON_JOB="0 6 * * * $CRON_SCRIPT"
    
    # Check if job already exists
    if ! crontab -l 2>/dev/null | grep -q "$CRON_SCRIPT"; then
        (crontab -l 2>/dev/null; echo "$CRON_JOB") | crontab -
        print_status "✅ Cron job added (daily execution at 6 AM)"
    else
        print_status "✅ Cron job already exists"
    fi
}

# Log rotation setup
setup_log_rotation() {
    print_status "Setting up log rotation..."
    
    LOGROTATE_CONFIG="/etc/logrotate.d/trendr"
    
    sudo tee "$LOGROTATE_CONFIG" > /dev/null << EOF
$LOG_DIR/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 644 $USER $USER
    postrotate
        # Signal to restart if necessary
        /bin/true
    endscript
}
EOF

    print_status "✅ Log rotation configured"
}

# Installation validation
validate_installation() {
    print_status "Validating installation..."
    
    cd "$PROJECT_DIR"
    source "$VENV_NAME/bin/activate"
    
    # Basic import test
    python -c "
import sys
sys.path.append('.')
try:
    from utils.database import SupabaseManager
    from scripts.google_places_ingester import GooglePlacesIngester
    from scripts.enhanced_proof_scanner import EnhancedProofScanner
    print('✅ Main imports OK')
except Exception as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
"
    
    if [ $? -eq 0 ]; then
        print_status "✅ Installation validated"
        
        # Database schema check
        echo ""
        print_status "🔍 Checking database schema..."
        python scripts/database_schema_updater.py --check
        if [ $? -ne 0 ]; then
            print_warning "⚠️ Database schema needs updating"
            echo ""
            echo "📋 Next steps after deployment:"
            echo "1. Run: python scripts/database_schema_updater.py --update-guide"
            echo "2. Apply the SQL schema update in Supabase"
            echo "3. Test pipeline: python run_pipeline.py --dry-run"
        else
            print_status "✅ Database schema is up to date"
        fi
    else
        print_error "❌ Validation problem"
        exit 1
    fi
}

# Main menu
show_menu() {
    echo
    echo "Deployment options:"
    echo "1. Complete installation (recommended)"
    echo "2. Environment setup only"
    echo "3. Cron setup only"
    echo "4. Installation validation"
    echo "5. Test pipeline"
    echo "q. Quit"
    echo
}

# Pipeline test
test_pipeline() {
    print_status "Testing pipeline..."
    
    cd "$PROJECT_DIR"
    source "$VENV_NAME/bin/activate"
    
    # Test dry-run
    python run_pipeline.py --dry-run
    
    if [ $? -eq 0 ]; then
        print_status "✅ Pipeline test successful"
    else
        print_error "❌ Pipeline test failed"
    fi
}

# Complete installation
full_install() {
    print_status "🚀 Complete Trendr pipeline installation..."
    
    check_prerequisites
    create_directories
    setup_python_env
    setup_environment
    setup_cron
    setup_log_rotation
    validate_installation
    
    print_status "🎉 Complete installation finished!"
    
    echo
    echo "📋 NEXT STEPS:"
    echo "1. Configure variables in $PROJECT_DIR/.env"
    echo "2. Test pipeline: ./deploy.sh -> option 5"
    echo "3. Pipeline will run automatically every day at 6 AM"
    echo
    echo "📁 Important files:"
    echo "- Configuration: $PROJECT_DIR/.env"
    echo "- Logs: $LOG_DIR/pipeline.log"
    echo "- Cache: $DATA_DIR/cache"
    echo
}

# Interactive menu
if [ "$1" = "--auto" ]; then
    # Automatic mode
    full_install
else
    # Interactive mode
    while true; do
        show_menu
        read -p "Choose an option: " choice
        case $choice in
            1) full_install; break ;;
            2) setup_python_env; setup_environment ;;
            3) setup_cron ;;
            4) validate_installation ;;
            5) test_pipeline ;;
            q|Q) print_status "Goodbye!"; exit 0 ;;
            *) print_warning "Invalid option" ;;
        esac
    done
fi

exit 0