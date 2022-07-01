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

import argparse
import json
from os import path
from typing import Dict, Any

from google.cloud import aiplatform
from kfp.v2 import compiler, dsl
from kfp.v2.dsl import Output, Artifact, component

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

    model_eval_task = interpret_automl_classification_metrics(
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


def run_job(
    template_path: str,
    pipeline_root: str,
    project: str,
    region: str,
    pipeline_params: Dict[str, Any] = {},
):
    """Run the pipeline"""
    job = aiplatform.PipelineJob(
        display_name="tabular_classification_automl_evaluation_pipeline",
        template_path=template_path,
        pipeline_root=pipeline_root,
        parameter_values=pipeline_params,
        project=project,
        location=region,
        enable_caching=False,
    )
    job.run()


def parse_args() -> argparse.Namespace:
    """Parse arguments"""
    parser = argparse.ArgumentParser(
        description=f"tabular classification evalauation pipeline operations."
    )

    commands = parser.add_subparsers(help="commands", dest="command", required=True)

    # compile command arguments
    cmd_compile = commands.add_parser(
        "compile", help="compile pipeline function to a json package file."
    )
    cmd_compile.add_argument(
        "--template_path", required=True, help="path to compiled pipeline package file."
    )

    # run command arguments
    cmd_run_job = commands.add_parser("run", help="run pipeline job on AI platform.")
    cmd_run_job.add_argument("--project", required=True, help="project ID.")
    cmd_run_job.add_argument("--region", required=True, help="region.")
    cmd_run_job.add_argument(
        "--template_path", required=True, help="path to compiled pipeline package file."
    )
    cmd_run_job.add_argument(
        "--pipeline_root",
        required=True,
        help="GCS root directory for files generated by pipeline job.",
    )

    return parser.parse_args()


def main(args):
    if args.command == "compile":
        compile(args.template_path)
    elif args.command == "run":
        aiplatform.init()

        basepath = path.dirname(__file__)
        filepath = path.abspath(path.join(basepath, "params.json"))
        print(filepath)
        with open(filepath) as json_file:
            pipeline_params = json.load(json_file)

        print(pipeline_params)

        run_job(
            template_path=args.template_path,
            pipeline_root=args.pipeline_root,
            project=args.project,
            region=args.region,
            pipeline_params=pipeline_params,
        )
    else:
        print(f"Command not implemented: {args.command}")


if __name__ == "__main__":
    main(parse_args())