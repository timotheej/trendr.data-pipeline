#!/bin/bash

# 🐳 Entrypoint for Trendr Docker containers
# This script handles different startup modes for the containers

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Starting Trendr container...${NC}"

# Ensure proper permissions for volumes
if [ -d "/app/logs" ]; then
    chown -R trendr:trendr /app/logs
fi

if [ -d "/app/data" ]; then
    chown -R trendr:trendr /app/data
fi

# Set timezone
export TZ=${TZ:-Europe/Paris}

echo -e "${GREEN}✅ Container initialized${NC}"
echo -e "${BLUE}📋 Running command: $@${NC}"

# Switch to trendr user and execute the command
exec gosu trendr "$@"