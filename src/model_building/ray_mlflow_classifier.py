import mlflow
import mlflow.pytorch
import pandas as pd
import ray
import torch


INPUT_COLUMNS = [
    "vibration_x",
    "vibration_y",
    "vibration_z",
    "speed",
    "score_mild",
    "score_standard",
    "score_strict",
]

TARGET_COLUMN = "label"

RAY_ADDRESS = "ray://cluster-kuberay-head-svc.ray.svc.cluster.local:10001"
MLFLOW_TRACKING_URI = "http://mlflow-svc.mlflow.svc.cluster.local:5000"
MLFLOW_EXPERIMENT_NAME = "road_quality_poc"


@ray.remote
def train_model(dataset: pd.DataFrame):

    # Daten in PyTorch-Tensoren umwandeln
    x = torch.tensor(
        dataset[INPUT_COLUMNS].values,
        dtype=torch.float32,
    )
    y = torch.tensor(
        dataset[TARGET_COLUMN].values,
        dtype=torch.long,
    )

    # Sehr einfaches neuronales Netzwerk
    model = torch.nn.Sequential(
        torch.nn.Linear(len(INPUT_COLUMNS), 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 3),
    )

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training
    epochs = 10

    for _ in range(epochs):
        optimizer.zero_grad()

        predictions = model(x)
        loss = loss_function(predictions, y)

        loss.backward()
        optimizer.step()

    # This code runs on a Ray worker. Therefore MLflow must also be configured
    # inside the remote task rather than only in the local driver process.
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("learning_rate", 0.001)
        mlflow.set_tag("execution_backend", "ray")

        mlflow.log_metric("train_loss", loss.item())

        mlflow.pytorch.log_model(
            model,
            name="model",
        )

        return {
            "train_loss": loss.item(),
            "run_id": run.info.run_id,
            "experiment_name": MLFLOW_EXPERIMENT_NAME,
        }


def train(dataset: pd.DataFrame):
    started_ray = False

    if not ray.is_initialized():
        ray.init(
            address=RAY_ADDRESS,
            runtime_env={
                "pip": [
                    "numpy==2.2.6",
                    "torch==2.11.0",
                    "pandas==2.3.3",
                    "mlflow==3.11.1",
                ]
            }
    )
        started_ray = True

    try:
        return ray.get(train_model.remote(dataset))
    finally:
        if started_ray:
            ray.shutdown()