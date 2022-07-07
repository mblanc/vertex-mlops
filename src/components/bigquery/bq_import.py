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

from kfp.v2.dsl import Artifact, component, Output


@component(base_image="python:3.9", packages_to_install=["google-cloud-bigquery"])
def import_csv_to_bigquery(
    project: str,
    bq_location: str,
    bq_dataset: str,
    gcs_csv_uri: str,
    raw_dataset: Output[Artifact],
    table_name_prefix: str = "abalone",
):
    from google.cloud import bigquery

    # Construct a BigQuery client object.
    client = bigquery.Client(project=project, location=bq_location)

    def load_dataset(gcs_uri, table_id):
        job_config = bigquery.LoadJobConfig(
            autodetect=True,
            # schema=[
            #     bigquery.SchemaField("Sex", "STRING"),
            #     bigquery.SchemaField("Length", "NUMERIC"),
            #     bigquery.SchemaField("Diameter", "NUMERIC"),
            #     bigquery.SchemaField("Height", "NUMERIC"),
            #     bigquery.SchemaField("Whole_weight", "NUMERIC"),
            #     bigquery.SchemaField("Shucked_weight", "NUMERIC"),
            #     bigquery.SchemaField("Viscera_weight", "NUMERIC"),
            #     bigquery.SchemaField("Shell_weight", "NUMERIC"),
            #     bigquery.SchemaField("Rings", "NUMERIC"),
            # ],
            # skip_leading_rows=1,
            # The source format defaults to CSV, so the line below is optional.
            source_format=bigquery.SourceFormat.CSV,
        )
        print(f"Loading {gcs_uri} into {table_id}")
        load_job = client.load_table_from_uri(
            gcs_uri, table_id, job_config=job_config
        )  # Make an API request.

        load_job.result()  # Waits for the job to complete.
        destination_table = client.get_table(table_id)  # Make an API request.
        print("Loaded {} rows.".format(destination_table.num_rows))

    def create_dataset_if_not_exist(bq_dataset_id, bq_location):
        print(
            "Checking for existence of bq dataset. If it does not exist, it creates one"
        )
        dataset = bigquery.Dataset(bq_dataset_id)
        dataset.location = bq_location
        dataset = client.create_dataset(dataset, exists_ok=True, timeout=300)
        print(f"Created dataset {dataset.full_dataset_id} @ {dataset.location}")

    bq_dataset_id = f"{project}.{bq_dataset}"
    create_dataset_if_not_exist(bq_dataset_id, bq_location)

    raw_table_name = f"{table_name_prefix}_raw"
    table_id = f"{project}.{bq_dataset}.{raw_table_name}"
    print("Deleting any tables that might have the same name on the dataset")
    client.delete_table(table_id, not_found_ok=True)
    print("will load data to table")
    load_dataset(gcs_csv_uri, table_id)

    raw_dataset_uri = f"bq://{table_id}"
    raw_dataset.uri = raw_dataset_uri
