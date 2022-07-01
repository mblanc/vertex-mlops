from kfp.v2.dsl import Artifact, Input, Metrics, Output, component


@component(base_image="python:3.9")
def interpret_bqml_evaluation_metrics(
    bqml_evaluation_metrics: Input[Artifact], metrics: Output[Metrics]
):
    import math

    metadata = bqml_evaluation_metrics.metadata
    for r in metadata["rows"]:
        rows = r["f"]
        schema = metadata["schema"]["fields"]
        output = {}
        for metric, value in zip(schema, rows):
            metric_name = metric["name"]
            val = float(value["v"])
            output[metric_name] = val
            metrics.log_metric(metric_name, val)
            if metric_name == "mean_squared_error":
                rmse = math.sqrt(val)
                metrics.log_metric("root_mean_squared_error", rmse)

    metrics.log_metric("framework", "BQML")

    print(output)
