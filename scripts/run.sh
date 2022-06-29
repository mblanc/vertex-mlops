#!/bin/bash

# Build the training and batch prediction pipeline specs
python -m src.pipelines.$1.$2.pipeline compile --template_path "$1_$2.json"

# Build the training and batch prediction pipeline specs
python -m src.pipelines.$1.$2.pipeline run  --project "svc-demo-vertex" --region "us-central1" --template_path "$1_$2.json" --pipeline_root "gs://svc-demo-vertex/vertex-mlops/pipeline_root"