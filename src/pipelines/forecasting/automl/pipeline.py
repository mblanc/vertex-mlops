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
from kfp.v2 import dsl

from src.pipelines.trigger.pipeline import VertexPipeline


class TabularForecastingAutoMLPipeline(VertexPipeline):

    display_name = "forecasting_automl_pipeline"

    @dsl.pipeline(name="forecasting-automl-pipeline")
    def pipeline(
        self,
        project: str,
        region: str,
        display_name: str,
        bq_table: str,
        label: str,
        time_column: str,
        id_column: str,
        available_at_forecast_columns: list,
        unavailable_at_forecast_columns: list,
        time_series_attribute_columns: list,
        forecast_horizon: int,
        context_window: int,
        data_granularity_unit: str,
        data_granularity_count: int,
        column_specs: dict,
        bigquery_source_input_uri: str,
        bigquery_destination_output_uri: str,
    ):
        dataset_create_op = gcc_aip.TimeSeriesDatasetCreateOp(
            project=project,
            location=region,
            display_name=display_name,
            bq_source=bq_table,
        )

        training_op = gcc_aip.AutoMLForecastingTrainingJobRunOp(
            project=project,
            location=region,
            display_name=display_name,
            dataset=dataset_create_op.outputs["dataset"],
            target_column=label,
            time_column=time_column,
            time_series_identifier_column=id_column,
            available_at_forecast_columns=available_at_forecast_columns,
            unavailable_at_forecast_columns=unavailable_at_forecast_columns,
            time_series_attribute_columns=time_series_attribute_columns,
            forecast_horizon=forecast_horizon,
            context_window=context_window,
            data_granularity_unit=data_granularity_unit,
            data_granularity_count=data_granularity_count,
            column_specs=column_specs,
            export_evaluated_data_items=True,
        )

        _ = gcc_aip.ModelBatchPredictOp(
            project=project,
            location=region,
            job_display_name=display_name,
            model=training_op.outputs["model"],
            instances_format="bigquery",
            bigquery_source_input_uri=bigquery_source_input_uri,
            predictions_format="bigquery",
            bigquery_destination_output_uri=bigquery_destination_output_uri,
            generate_explanation=True,
        )


if __name__ == "__main__":
    pipeline = TabularForecastingAutoMLPipeline()
    pipeline.main(pipeline.parse_args())
