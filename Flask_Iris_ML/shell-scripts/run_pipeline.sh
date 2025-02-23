#!/bin/bash

# Navigate to project directory
cd "$(dirname "$0")/.." || exit

# Virtual Environment Setup
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3.8 -m venv venv
fi

echo "Activating virtual environment..."
source venv/Scripts/activate

# install dependencies
pip install -r requirements.txt

# Train
python train-scripts/train.py

# Run app
python src/app.py

# give this file excecutable permissions
# =================================================================
# chmod +x shell-scripts/run_pipeline.sh
#
# Now you can run the pipeline using the command:
#
# shell-scripts//run_pipeline.sh
#
# This will create a virtual environment, install the required dependencies,
# train the model, and start the Flask application.