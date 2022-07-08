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

from google_cloud_pipeline_components.v1.bigquery import (
    BigqueryCreateModelJobOp,
    BigqueryEvaluateModelJobOp,
    BigqueryPredictModelJobOp,
)
from kfp.v2 import dsl

from src.components.metrics.bqml import interpret_bqml_evaluation_metrics
from src.pipelines.trigger.pipeline import VertexPipeline


class TabularRegressionBQMLPipeline(VertexPipeline):

    display_name = "tabular_regression_bqml_pipeline"

    @dsl.pipeline(name="tabular-regression-bqml-pipeline")
    def pipeline(
        self,
        project: str,
        bq_location: str,
        region: str,
        bq_table: str,
        label: str,
        model: str,
    ):
        bq_model = BigqueryCreateModelJobOp(
            project=project,
            location=bq_location,
            query=f"CREATE OR REPLACE MODEL {model} OPTIONS (model_type='BOOSTED_TREE_REGRESSOR', labels=['{label}']) AS SELECT * FROM `{bq_table}`",
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
                    "tableId": "results_abalone",
                },
                "createDisposition": "CREATE_IF_NEEDED",
                "writeDisposition": "WRITE_TRUNCATE",
            },
        ).after(bq_model)


if __name__ == "__main__":
    pipeline = TabularRegressionBQMLPipeline()
    pipeline.main(pipeline.parse_args())
