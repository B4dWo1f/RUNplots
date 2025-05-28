#!/bin/bash
dirscript=`dirname $0`

set -e  # Exit on error

echo "🔧 Installing system dependencies..."
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update
pyVERSION="3.10"
PKG_NAME="python$pyVERSION"
sudo apt-get install -y \
    "$PKG_NAME" \
    "$PKG_NAME"-dev \
    "$PKG_NAME"-venv \
    "$PKG_NAME"-distutils \
    "$PKG_NAME"-lib2to3 \
    "$PKG_NAME"-tk \
    build-essential \
    libffi-dev \
    libssl-dev \
    curl

echo "🐍 Setting up virtual environment..."
"$PKG_NAME" -m venv .venv3.10 --prompt py3.10
source ".venv3.10/bin/activate"

echo "⬆️ Installing pip-tools..."
pip install --upgrade pip
pip install pip-tools

echo "📦 Compiling and installing Python requirements..."
pip-compile requirements.in
pip install -r requirements.txt

echo "🎭 Installing Playwright drivers..."
playwright install

GREEN='\033[0;32m'
NC='\033[0m' # No Color
echo "${GREEN}✅ All done!${NC}"
