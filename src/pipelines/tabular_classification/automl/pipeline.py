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

from google_cloud_pipeline_components import aiplatform as gcc_aip
from google_cloud_pipeline_components.v1.endpoint import EndpointCreateOp, ModelDeployOp
from kfp.v2 import dsl

from src.components.metrics.automl import interpret_automl_classification_metrics
from src.pipelines.trigger.pipeline import VertexPipeline


class TabularClassificationAutoMLPipeline(VertexPipeline):

    display_name = "tabular_classification_automl_pipeline"

    @dsl.pipeline(name="tabular-classification-automl-pipeline")
    def pipeline(
        self,
        project: str,
        region: str,
        bq_table: str,
        label: str,
        display_name: str,
    ):
        dataset_create_op = gcc_aip.TabularDatasetCreateOp(
            project=project,
            location=region,
            display_name=display_name,
            bq_source=bq_table,
        )

        training_op = gcc_aip.AutoMLTabularTrainingJobRunOp(
            project=project,
            location=region,
            display_name=display_name,
            optimization_prediction_type="classification",
            dataset=dataset_create_op.outputs["dataset"],
            target_column=label,
        )

        _ = interpret_automl_classification_metrics(
            project,
            region,
            training_op.outputs["model"],
        )

        endpoint_op = EndpointCreateOp(
            project=project,
            location=region,
            display_name=display_name,
        )

        ModelDeployOp(
            model=training_op.outputs["model"],
            endpoint=endpoint_op.outputs["endpoint"],
            dedicated_resources_machine_type="n1-standard-2",
            dedicated_resources_min_replica_count=1,
            dedicated_resources_max_replica_count=1,
        )


if __name__ == "__main__":
    pipeline = TabularClassificationAutoMLPipeline()
    pipeline.main(pipeline.parse_args())
