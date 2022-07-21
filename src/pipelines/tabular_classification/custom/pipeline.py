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

from kfp.v2 import dsl

from src.components.bigquery.bq_import import import_csv_to_bigquery
from src.pipelines.trigger.pipeline import VertexPipeline


class TabularClassificationCustomPipeline(VertexPipeline):

    display_name = "tabular_classification_custom_pipeline"

    @dsl.pipeline(name="tabular-classification-custom-pipeline")
    def pipeline(
        self,
        project: str,
        gcs_input_file_uri: str,
        region: str,
        bq_dataset: str,
        bq_location: str,
    ) -> None:
        # Imports data to BigQuery using a custom component.
        _ = import_csv_to_bigquery(project, bq_location, bq_dataset, gcs_input_file_uri)


if __name__ == "__main__":
    pipeline = TabularClassificationCustomPipeline()
    pipeline.main(pipeline.parse_args())
