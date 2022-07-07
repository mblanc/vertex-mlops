# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from kfp.v2 import compiler, dsl
from kfp.v2.dsl import Artifact, component, Output

from src.components.metrics.automl import interpret_automl_classification_metrics


@component(
    base_image="python:3.9",
    packages_to_install=["google-cloud-aiplatform"],
)
def return_unmanaged_model(
    artifact_uri: str, resource_name: str, model: Output[Artifact]
):
    model.metadata["resourceName"] = resource_name
    model.uri = artifact_uri


@dsl.pipeline(name="tabular-classification-automl--evaluation-pipeline")
def pipeline(
    project: str,
    region: str,
):
    import_model_op = return_unmanaged_model(
        artifact_uri="https://us-central1-aiplatform.googleapis.com/v1/projects/125188993477/locations/us-central1/models/2223665510153715712",
        resource_name="projects/125188993477/locations/us-central1/models/2223665510153715712",
    )

    _ = interpret_automl_classification_metrics(
        project,
        region,
        import_model_op.outputs["model"],
    )


def compile(package_path: str):
    """Compile the pipeline"""
    compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path=package_path,
    )
