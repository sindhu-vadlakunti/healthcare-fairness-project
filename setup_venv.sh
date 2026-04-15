#!/usr/bin/env bash
# ===================================================================
# Healthcare Fairness ML Project Setup (Linux/Mac)
# ===================================================================
# Exit immediately if a command exits with a non-zero status.
set -e

echo "🚀 Starting project setup for Linux/Mac..."

# 1. Create a virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment 'venv'..."
    python3 -m venv venv
else
    echo "✅ Virtual environment already exists."
fi

# 2. Activate the environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# 3. Install dependencies
echo "⬇️ Installing required packages..."
pip install --upgrade pip
pip install -r requirements.txt

# 4. Success message & Optional execution
echo "==================================================================="
echo "🎉 Setup complete!"
echo "==================================================================="
echo "To run the research pipeline immediately, press 'y'. To exit, press any other key."
read -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "▶️ Running pipeline..."
    python run.py
else
    echo "You can run the project later by executing:"
    echo "  1. source venv/bin/activate"
    echo "  2. python run.py"
fi
