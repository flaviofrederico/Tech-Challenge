"""Promote the latest registered model version to the 'champion' alias.

This implements a simple automated governance gate, following the practices
described in MLOps model governance literature: a new model version is only
promoted to 'champion' if it outperforms the current champion on the primary
metric (ROC AUC), or if no champion exists yet. Approval metadata (who/when)
is recorded as a model version tag and description for auditability.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from purchase_intention.config import PipelineConfig

MODEL_NAME = "purchase_intention_classifier"
CHAMPION_ALIAS = "champion"
PRIMARY_METRIC = "roc_auc"


def get_latest_version(client: MlflowClient, model_name: str) -> str:
    """Return the version number of the most recently registered model version.

    Args:
        client: An initialized MLflow tracking client.
        model_name: Name of the registered model.

    Returns:
        The version number (as a string) of the latest registered version.
    """
    versions = client.search_model_versions(f"name='{model_name}'")
    latest = max(versions, key=lambda version: int(version.version))
    return latest.version


def get_metric(client: MlflowClient, run_id: str, metric_name: str) -> float:
    """Fetch a single metric value logged for a given MLflow run."""
    run = client.get_run(run_id)
    return run.data.metrics[metric_name]


def get_champion_metric(
    client: MlflowClient, model_name: str, metric_name: str
) -> float | None:
    """Return the primary metric of the current champion version, if any exists.

    Returns:
        The metric value, or None if no version currently holds the champion alias.
    """
    try:
        champion = client.get_model_version_by_alias(model_name, CHAMPION_ALIAS)
    except MlflowException:
        return None
    return get_metric(client, champion.run_id, metric_name)


def promote_latest_version(config_path: str) -> None:
    """Promote the latest registered model version to champion if it qualifies.

    Args:
        config_path: Path to the pipeline YAML configuration file.
    """
    config = PipelineConfig.from_yaml(config_path)
    client = MlflowClient(tracking_uri=config.mlflow.tracking_uri)

    latest_version = get_latest_version(client, MODEL_NAME)
    latest_metadata = client.get_model_version(MODEL_NAME, latest_version)
    candidate_metric = get_metric(client, latest_metadata.run_id, PRIMARY_METRIC)
    champion_metric = get_champion_metric(client, MODEL_NAME, PRIMARY_METRIC)

    should_promote = champion_metric is None or candidate_metric >= champion_metric
    if not should_promote:
        print(
            f"Versao {latest_version} ({PRIMARY_METRIC}={candidate_metric:.4f}) "
            f"nao superou o champion atual ({PRIMARY_METRIC}={champion_metric:.4f}). "
            "Promocao ignorada."
        )
        return

    _apply_promotion(client, latest_version, candidate_metric, champion_metric)
    print(
        f"Versao {latest_version} promovida a '{CHAMPION_ALIAS}' "
        f"({PRIMARY_METRIC}={candidate_metric:.4f})."
    )


def _apply_promotion(
    client: MlflowClient,
    version: str,
    candidate_metric: float,
    previous_champion_metric: float | None,
) -> None:
    """Set the champion alias and record approval metadata on the model version."""
    client.set_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS, version)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    comparison = (
        f"(anterior: {previous_champion_metric:.4f})"
        if previous_champion_metric is not None
        else "(primeiro champion)"
    )
    client.update_model_version(
        name=MODEL_NAME,
        version=version,
        description=(
            f"Promovido a champion em {timestamp}. "
            f"{PRIMARY_METRIC}={candidate_metric:.4f} {comparison}"
        ),
    )
    client.set_model_version_tag(MODEL_NAME, version, "approved_by", "Flavio Frederico")
    client.set_model_version_tag(MODEL_NAME, version, "approval_date", timestamp)


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for this script."""
    parser = argparse.ArgumentParser(
        description="Promote the latest model version to champion."
    )
    parser.add_argument("--config", default="configs/config.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    promote_latest_version(args.config)