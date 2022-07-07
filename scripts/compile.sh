#!/bin/bash

# Build the training and batch prediction pipeline specs
python -m src.pipelines.$1.$2.pipeline compile --template_path "$1_$2.json"
