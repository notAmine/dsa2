# pylint: disable=C0321,C0103,E1221,C0301,E1305,E1121,C0302,C0330
# -*- coding: utf-8 -*-

import mlflow


def register(run_name, params, metrics):
    """
    :run_name: Name of model
    :log_params: dict with model params
    :metrics: dict with model evaluation metrics
    :return:
    """
    mlflow.set_tracking_uri("sqlite:///local.db")

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_uuid
        experiment_id = run.info.experiment_id

        mlflow.log_params(params)
        metrics.apply(lambda x: mlflow.log_metric(x.metric_name, x.metric_val), axis=1)
        print(run_id, experiment_id)
    mlflow.end_run()
