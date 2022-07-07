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

from google_cloud_pipeline_components.experimental.bigquery import (
    BigqueryCreateModelJobOp,
    BigqueryExplainForecastModelJobOp,
    BigqueryMLArimaEvaluateJobOp,
)
from kfp.v2 import compiler, dsl


@dsl.pipeline(name="forecasting-bqml-pipeline")
def pipeline(
    project: str,
    region: str,
    bq_location: str,
    bq_table: str,
    model: str,
    label: str,
    time_column: str,
    id_column: str,
    data_frequency: str,
    forecast_horizon: int,
):
    bq_model = BigqueryCreateModelJobOp(
        project=project,
        location=bq_location,
        query=f"""
        CREATE OR REPLACE MODEL {model}
        OPTIONS(
          MODEL_TYPE='ARIMA_PLUS',
          TIME_SERIES_TIMESTAMP_COL='{time_column}',
          TIME_SERIES_DATA_COL='{label}',
          TIME_SERIES_ID_COL='{id_column}',
          DATA_FREQUENCY='{data_frequency}') AS
        SELECT {id_column}, {label}, {time_column}  FROM `{bq_table}`
        """,
    )

    _ = BigqueryMLArimaEvaluateJobOp(
        project=project,
        location=bq_location,
        model=bq_model.outputs["model"],
        job_configuration_query={
            "destinationTable": {
                "projectId": "svc-demo-vertex",
                "datasetId": "pipeline_us",
                "tableId": "forecast_eval",
            },
            "createDisposition": "CREATE_IF_NEEDED",
            "writeDisposition": "WRITE_TRUNCATE",
        },
    ).after(bq_model)

    _ = BigqueryExplainForecastModelJobOp(
        project=project,
        location=bq_location,
        model=bq_model.outputs["model"],
        horizon=forecast_horizon,
        job_configuration_query={
            "destinationTable": {
                "projectId": "svc-demo-vertex",
                "datasetId": "pipeline_us",
                "tableId": "forecast_results",
            },
            "createDisposition": "CREATE_IF_NEEDED",
            "writeDisposition": "WRITE_TRUNCATE",
        },
    ).after(bq_model)


def compile(package_path: str):
    """Compile the pipeline"""
    compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path=package_path,
    )
