#!/bin/bash

# ==========================================
# LangExtract Benchmark Automation Script
# ==========================================
# Usage: ./run.sh config1.yaml config2.yaml ...
# This script will:
# 1. Set up Python 3.12 environment
# 2. Install dependencies
# 3. Install Ollama
# 4. Pull required models from config files
# 5. Run extractions for each config
# 6. Zip the results folder
# ==========================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if config files were provided
if [ $# -eq 0 ]; then
    echo -e "${RED}Error: No config files provided${NC}"
    echo "Usage: $0 config1.yaml config2.yaml ..."
    echo "Example: $0 configs/stixnet_gemma3-12b_config.yaml configs/stixnet_deepseek-r1-14b_config.yaml"
    exit 1
fi

# Verify all config files exist
echo -e "${BLUE}[0/7] Verifying config files...${NC}"
for config in "$@"; do
    if [ ! -f "$config" ]; then
        echo -e "${RED}Error: Config file not found: $config${NC}"
        exit 1
    fi
    echo -e "${GREEN}  ✓ Found: $config${NC}"
done

# ==========================================
# 1. Python 3.12 Setup
# ==========================================
echo ""
echo -e "${BLUE}[1/7] Setting up Python 3.12 environment...${NC}"

# Check if running in Colab
if [ -d "/content" ]; then
    echo "Detected Google Colab environment"

    # Install Python 3.12 using deadsnakes PPA
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update
    sudo apt-get install -y python3.12 python3.12-venv python3.12-dev

    # Create virtual environment with Python 3.12
    python3.12 -m venv venv312
    source venv312/bin/activate
else
    echo "Local environment detected"
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv312" ]; then
        echo "Creating virtual environment..."
        python3.12 -m venv venv312
    else
        echo "Virtual environment already exists"
    fi
    source venv312/bin/activate
fi

echo -e "${GREEN}Python version: $(python --version)${NC}"

# ==========================================
# 2. Install Dependencies
# ==========================================
echo ""
echo -e "${BLUE}[2/7] Installing Python dependencies from requirements.txt...${NC}"

if [ -f "requirements.txt" ]; then
    pip install --upgrade pip
    pip install -r requirements.txt
    echo -e "${GREEN}Dependencies installed successfully${NC}"
else
    echo -e "${YELLOW}Warning: requirements.txt not found, skipping...${NC}"
fi

# ==========================================
# 3. Install Ollama
# ==========================================
echo ""
echo -e "${BLUE}[3/7] Installing Ollama...${NC}"

if command -v ollama &> /dev/null; then
    echo -e "${GREEN}Ollama is already installed${NC}"
    ollama --version
else
    echo "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    echo -e "${GREEN}Ollama installed successfully${NC}"
fi

# Start Ollama service if not running
if ! pgrep -x "ollama" > /dev/null; then
    echo "Starting Ollama service..."
    ollama serve &
    OLLAMA_PID=$!
    sleep 5  # Give it time to start
    echo -e "${GREEN}Ollama service started (PID: $OLLAMA_PID)${NC}"
else
    echo -e "${GREEN}Ollama service is already running${NC}"
fi

# ==========================================
# 4. Extract and Pull Required Models
# ==========================================
echo ""
echo -e "${BLUE}[4/7] Extracting model names from config files...${NC}"

# Create array to store unique model names
declare -A models

# Extract model_id from each YAML config file
for config in "$@"; do
    echo "Processing: $config"

    # Try multiple methods to extract model_id
    # Method 1: Using grep and awk (most portable)
    model_id=$(grep -E "^\s*model_id:" "$config" | awk -F': ' '{print $2}' | tr -d '"' | tr -d "'" | xargs)

    if [ -n "$model_id" ]; then
        models["$model_id"]=1
        echo -e "  ${GREEN}✓ Found model: $model_id${NC}"
    else
        echo -e "  ${YELLOW}⚠ Could not extract model_id from $config${NC}"
    fi
done

# Pull each unique model
echo ""
echo -e "${BLUE}Pulling required models...${NC}"
for model in "${!models[@]}"; do
    echo ""
    echo -e "${BLUE}Pulling model: $model${NC}"

    # Check if model already exists
    if ollama list | grep -q "$model"; then
        echo -e "${GREEN}  ✓ Model already exists: $model${NC}"
    else
        echo "  Downloading $model (this may take a while)..."
        ollama pull "$model"
        echo -e "${GREEN}  ✓ Successfully pulled: $model${NC}"
    fi
done

# ==========================================
# 5. Run Extractions for Each Config
# ==========================================
echo ""
echo -e "${BLUE}[5/7] Running extractions...${NC}"

total_configs=$#
current=0

for config in "$@"; do
    current=$((current + 1))
    echo ""
    echo -e "${BLUE}======================================${NC}"
    echo -e "${BLUE}Processing config $current/$total_configs: $config${NC}"
    echo -e "${BLUE}======================================${NC}"

    # Run the extraction
    python run_extraction.py --config "$config"

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Successfully completed: $config${NC}"
    else
        echo -e "${RED}✗ Failed: $config${NC}"
        echo -e "${YELLOW}Continuing with next config...${NC}"
    fi
done

# ==========================================
# 6. Create Results Archive
# ==========================================
echo ""
echo -e "${BLUE}[6/7] Creating results archive...${NC}"

timestamp=$(date +"%Y%m%d_%H%M%S")
archive_name="results_${timestamp}.zip"

if [ -d "results" ]; then
    echo "Zipping results folder..."
    zip -r "$archive_name" results/
    echo -e "${GREEN}✓ Results archived to: $archive_name${NC}"
    echo -e "${GREEN}  Archive size: $(du -h "$archive_name" | cut -f1)${NC}"
else
    echo -e "${YELLOW}Warning: results/ folder not found, skipping archive creation${NC}"
fi

# ==========================================
# 7. Summary
# ==========================================
echo ""
echo -e "${BLUE}[7/7] Execution Summary${NC}"
echo -e "${BLUE}======================================${NC}"
echo -e "${GREEN}✓ Python environment: $(python --version)${NC}"
echo -e "${GREEN}✓ Ollama version: $(ollama --version 2>&1 | head -n1)${NC}"
echo -e "${GREEN}✓ Configs processed: $total_configs${NC}"
echo -e "${GREEN}✓ Models pulled: ${#models[@]}${NC}"
if [ -f "$archive_name" ]; then
    echo -e "${GREEN}✓ Results archive: $archive_name${NC}"
fi
echo -e "${BLUE}======================================${NC}"
echo -e "${GREEN}All tasks completed successfully!${NC}"
echo ""

# Deactivate virtual environment
deactivate 2>/dev/null || true