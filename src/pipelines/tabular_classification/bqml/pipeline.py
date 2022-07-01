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
from google_cloud_pipeline_components.types import artifact_types
from google_cloud_pipeline_components.v1.bigquery import (
    BigqueryCreateModelJobOp,
    BigqueryEvaluateModelJobOp,
    BigqueryExportModelJobOp,
    BigqueryPredictModelJobOp,
)
from google_cloud_pipeline_components.v1.endpoint import EndpointCreateOp, ModelDeployOp
from google_cloud_pipeline_components.v1.model import ModelUploadOp
from kfp.v2 import compiler, dsl
from kfp.v2.components import importer_node

from src.components.metrics.bqml import interpret_bqml_evaluation_metrics


@dsl.pipeline(name="tabular-classification-bqml-pipeline")
def pipeline(
    project: str,
    bq_location: str,
    region: str,
    bq_table: str,
    label: str,
    model: str,
    artifact_uri: str,
    display_name: str,
):
    bq_model = BigqueryCreateModelJobOp(
        project=project,
        location=bq_location,
        query=f"CREATE OR REPLACE MODEL {model} OPTIONS (model_type='dnn_classifier', labels=['{label}']) AS SELECT * FROM `{bq_table}`",
    )

    bq_eval_model_op = BigqueryEvaluateModelJobOp(
        project=project, location=bq_location, model=bq_model.outputs["model"]
    ).after(bq_model)

    _ = interpret_bqml_evaluation_metrics(
        bq_eval_model_op.outputs["evaluation_metrics"]
    )

    _ = BigqueryPredictModelJobOp(
        project=project,
        location=bq_location,
        model=bq_model.outputs["model"],
        table_name=f"`{bq_table}`",
        # query_statement=f"SELECT * EXCEPT ({label}) FROM {bq_table} WHERE body_mass_g IS NOT NULL AND sex IS NOT NULL"
        job_configuration_query={
            "destinationTable": {
                "projectId": "svc-demo-vertex",
                "datasetId": "pipeline_us",
                "tableId": "results_1",
            },
            "createDisposition": "CREATE_IF_NEEDED",
            "writeDisposition": "WRITE_TRUNCATE",
        },
    ).after(bq_model)

    bq_export = BigqueryExportModelJobOp(
        project=project,
        location=bq_location,
        model=bq_model.outputs["model"],
        model_destination_path=artifact_uri,
    ).after(bq_model)

    import_unmanaged_model_task = importer_node.importer(
        artifact_uri=artifact_uri,
        artifact_class=artifact_types.UnmanagedContainerModel,
        metadata={
            "containerSpec": {
                "imageUri": "us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-6:latest",
            },
        },
    ).after(bq_export)

    model_upload = ModelUploadOp(
        project=project,
        display_name=display_name,
        unmanaged_container_model=import_unmanaged_model_task.outputs["artifact"],
    ).after(import_unmanaged_model_task)

    endpoint = EndpointCreateOp(
        project=project,
        location=region,
        display_name=display_name,
    ).after(model_upload)

    _ = ModelDeployOp(
        model=model_upload.outputs["model"],
        endpoint=endpoint.outputs["endpoint"],
        dedicated_resources_min_replica_count=1,
        dedicated_resources_max_replica_count=1,
        dedicated_resources_machine_type="n1-standard-2",
        traffic_split={"0": 100},
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
        display_name="tabular_classification_bqml_pipeline",
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
        description=f"tabular classification pipeline operations."
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
