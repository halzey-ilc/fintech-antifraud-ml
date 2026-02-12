from __future__ import annotations

from pathlib import Path

import typer

from app.core.config import AppConfig, resolve_paths
from app.core.logging import configure_logging, get_logger
from app.pipeline.eval_pipeline import run_eval
from app.pipeline.train_pipeline import run_train


log = get_logger("cli")

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Antifraud ML CLI: train/eval pipelines and artifacts.",
)


def _config_option() -> typer.models.OptionInfo:
    return typer.Option(
        "configs/dev.yaml",
        "--config",
        "-c",
        help="Path to YAML config (relative to project root or absolute).",
        show_default=True,
    )


def _resolve_config_path(config: str) -> Path:
    """
    Accepts relative path (from project root) or absolute path.
    """
    p = Path(config)
    if p.is_absolute():
        return p
    project_root = Path(__file__).resolve().parents[3]
    return (project_root / p).resolve()


@app.command("train")
def train_cmd(config: str = _config_option()) -> None:
    """
    Train antifraud model and persist artifacts.
    """
    configure_logging()

    config_path = _resolve_config_path(config)
    cfg = AppConfig.from_yaml(config_path)
    project_root = config_path.parent.parent if config_path.parts[-2] == "configs" else config_path.parent
    paths = resolve_paths(cfg, project_root)

    log.info("train.start config=%s", str(config_path))
    res = run_train(cfg, paths, project_root)
    log.info("train.done model=%s metrics=%s", res.model_path, res.metrics_path)


@app.command("eval")
def eval_cmd(config: str = _config_option()) -> None:
    """
    Evaluate model on eval dataset and persist reports.
    """
    configure_logging()

    config_path = _resolve_config_path(config)
    cfg = AppConfig.from_yaml(config_path)
    project_root = config_path.parent.parent if config_path.parts[-2] == "configs" else config_path.parent
    paths = resolve_paths(cfg, project_root)

    log.info("eval.start config=%s", str(config_path))
    out = run_eval(cfg, paths, project_root)
    log.info("eval.done outputs=%s", out)


def app_entry() -> None:
    """
    Poetry entry point: `poetry run antifraud ...`
    """
    app()


if __name__ == "__main__":
    app_entry()
