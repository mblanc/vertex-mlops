from kfp.v2.dsl import (
    Artifact,
    ClassificationMetrics,
    component,
    Input,
    Metrics,
    Output,
)


@component(
    base_image="python:3.9",
    packages_to_install=[
        "google-cloud-aiplatform",
    ],
)
def interpret_automl_classification_metrics(
    project: str,
    region: str,
    model: Input[Artifact],
    metrics: Output[Metrics],
    classificationMetrics: Output[ClassificationMetrics],
):
    import json
    import logging

    from google.cloud import aiplatform as aip

    def get_eval_info(client, model_name):
        from google.protobuf.json_format import MessageToDict

        response = client.list_model_evaluations(parent=model_name)
        metrics_list = []
        for evaluation in response:
            print("model_evaluation")
            print(" name:", evaluation.name)
            print(" metrics_schema_uri:", evaluation.metrics_schema_uri)
            metrics = MessageToDict(evaluation._pb.metrics)
            for metric in metrics.keys():
                logging.info("metric: %s, value: %s", metric, metrics[metric])
            metrics_list.append(metrics)

        return metrics_list

    def log_metrics(metrics_list, classificationMetrics):
        test_confusion_matrix = metrics_list[0]["confusionMatrix"]
        logging.info("rows: %s", test_confusion_matrix["rows"])

        # log the ROC curve
        fpr = []
        tpr = []
        thresholds = []
        for item in metrics_list[0]["confidenceMetrics"]:
            fpr.append(item.get("falsePositiveRate", 0.0))
            tpr.append(item.get("recall", 0.0))
            thresholds.append(item.get("confidenceThreshold", 0.0))
        print(f"fpr: {fpr}")
        print(f"tpr: {tpr}")
        print(f"thresholds: {thresholds}")
        classificationMetrics.log_roc_curve(fpr, tpr, thresholds)

        # log the confusion matrix
        annotations = []
        for item in test_confusion_matrix["annotationSpecs"]:
            annotations.append(item["displayName"])
        logging.info("confusion matrix annotations: %s", annotations)
        classificationMetrics.log_confusion_matrix(
            annotations,
            test_confusion_matrix["rows"],
        )

        # log textual metrics info as well
        for metric in metrics_list[0].keys():
            if metric != "confidenceMetrics":
                val_string = json.dumps(metrics_list[0][metric])
                metrics.log_metric(metric, val_string)
        # metrics.metadata["model_type"] = "AutoML Tabular classification"

    logging.getLogger().setLevel(logging.INFO)
    aip.init(project=project)
    # extract the model resource name from the input Model Artifact
    model_resource_path = model.metadata["resourceName"]
    logging.info("model path: %s", model_resource_path)

    client_options = {"api_endpoint": f"{region}-aiplatform.googleapis.com"}
    # Initialize client that will be used to create and send requests.
    client = aip.gapic.ModelServiceClient(client_options=client_options)
    metrics_list = get_eval_info(client, model_resource_path)
    log_metrics(metrics_list, classificationMetrics)


@component(
    base_image="python:3.9",
    packages_to_install=[
        "google-cloud-aiplatform",
    ],
)
def interpret_automl_regression_metrics(
    project: str, region: str, model: Input[Artifact], metrics: Output[Metrics]
):
    import google.cloud.aiplatform as aip

    model = aip.Model(
        model_name=model.metadata["resourceName"], project=project, location=region
    )

    model_evaluations = model.list_model_evaluations()
    model_evaluation = list(model_evaluations)[0]

    available_metrics = [
        "meanAbsoluteError",
        "meanAbsolutePercentageError",
        "rSquared",
        "rootMeanSquaredError",
        "rootMeanSquaredLogError",
    ]
    output = dict()
    for x in available_metrics:
        val = model_evaluation.metrics.get(x)
        output[x] = val
        metrics.log_metric(str(x), float(val))
    print(output)
