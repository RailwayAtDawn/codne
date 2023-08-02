#!/bin/bash
echo "Checking environment..."
if ! command -v pip &> /dev/null
then
    echo "pip could not be found"
    exit
fi
if ! command -v python &> /dev/null
then
    echo "python could not be found"
    exit
fi


echo "Checking for libraries..."
if ! python -c "import transformers" &> /dev/null; then
    echo "Transformers library not installed. Installing..."
    pip install transformers
fi
if ! python -c "import gradio" &> /dev/null; then
    echo "Gradio library not installed. Installing..."
    pip install gradio
fi
if ! python -c "import torch" &> /dev/null; then
    echo "Torch library not installed. Installing..."
    pip install torch
fi
echo "Checking for libraries..."

echo "Start running..."
python ./app.py