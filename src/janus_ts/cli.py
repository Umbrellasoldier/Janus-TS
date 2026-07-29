"""Command-line entry points for Janus-TS."""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer

from .config import load_config
from .preprocessing import (
    audit_processed_dataset,
    expected_processed_path,
    inventory_sources,
    load_pinned_tokenizer,
    preprocess_transition1x,
)

app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)
data_app = typer.Typer(no_args_is_help=True)
train_app = typer.Typer(no_args_is_help=True)
evaluate_app = typer.Typer(no_args_is_help=True)
infer_app = typer.Typer(no_args_is_help=True)
app.add_typer(data_app, name="data")
app.add_typer(train_app, name="train")
app.add_typer(evaluate_app, name="evaluate")
app.add_typer(infer_app, name="infer")


ConfigOption = Annotated[
    Path,
    typer.Option("--config", exists=True, dir_okay=False, readable=True),
]


class ExplorationSplit(str, Enum):
    """Splits available to the explicitly non-formal inference command."""

    val = "val"
    test = "test"


@data_app.command("preprocess")
def data_preprocess(
    config_path: ConfigOption = Path("configs/transition1x.yaml"),
    check_token_lengths: Annotated[
        bool,
        typer.Option("--check-token-lengths/--skip-token-lengths"),
    ] = True,
    local_files_only: Annotated[bool, typer.Option("--local-files-only")] = False,
) -> None:
    """Build the pinned Transition1x Arrow dataset and run its hard audit."""

    config = load_config(config_path)
    tokenizer = (
        load_pinned_tokenizer(config, local_files_only=local_files_only)
        if check_token_lengths
        else None
    )
    result = preprocess_transition1x(config, tokenizer=tokenizer)
    typer.echo(
        json.dumps(
            {
                "output_path": str(result.output_path),
                "fingerprint": result.fingerprint,
                "audit": result.audit,
            },
            indent=2,
            sort_keys=True,
        )
    )


@data_app.command("audit")
def data_audit(
    config_path: ConfigOption = Path("configs/transition1x.yaml"),
    check_token_lengths: Annotated[
        bool,
        typer.Option("--check-token-lengths/--skip-token-lengths"),
    ] = True,
    local_files_only: Annotated[bool, typer.Option("--local-files-only")] = False,
) -> None:
    """Reopen the content-addressed dataset and repeat every data hard gate."""

    config = load_config(config_path)
    sources = inventory_sources(config)
    path = expected_processed_path(config, sources)
    tokenizer = (
        load_pinned_tokenizer(config, local_files_only=local_files_only)
        if check_token_lengths
        else None
    )
    audit = audit_processed_dataset(path, config, tokenizer=tokenizer, write=True)
    typer.echo(json.dumps({"path": str(path), "audit": audit}, indent=2, sort_keys=True))


def _transient_exit(error: BaseException) -> None:
    from .workflow import TRANSIENT_EXIT_CODE, is_proven_transient_exception

    if not is_proven_transient_exception(error):
        raise error
    typer.echo(f"TRANSIENT: {error}", err=True)
    raise typer.Exit(TRANSIENT_EXIT_CODE) from error


def _prepared_summary(prepared: object) -> dict[str, object]:
    return {
        "status": "ready",
        "run_fingerprint": prepared.run_identity.fingerprint,  # type: ignore[attr-defined]
        "run_path": str(prepared.paths.project),  # type: ignore[attr-defined]
        "processed_path": str(prepared.processed_path),  # type: ignore[attr-defined]
        "pissa_bundle": str(prepared.bundle_dir),  # type: ignore[attr-defined]
        "micro_batch_size_per_gpu": prepared.micro_batch_size_per_gpu,  # type: ignore[attr-defined]
        "gradient_accumulation_steps": prepared.gradient_accumulation_steps,  # type: ignore[attr-defined]
    }


@train_app.command("smoke")
def train_smoke(
    config_path: ConfigOption = Path("configs/transition1x.yaml"),
) -> None:
    """Run every frozen CPU/GPU launch gate and choose the one batch geometry."""

    from .workflow import prepare_smoke_workflow

    try:
        prepared = prepare_smoke_workflow(config_path)
    except BaseException as exc:
        _transient_exit(exc)
        raise AssertionError("unreachable") from exc
    typer.echo(json.dumps(_prepared_summary(prepared), indent=2, sort_keys=True))


@train_app.command("run")
def train_run(
    config_path: ConfigOption = Path("configs/transition1x.yaml"),
) -> None:
    """Resume five epochs, select on validation, then run four test inference modes."""

    from .workflow import run_full_workflow

    try:
        prepared = run_full_workflow(config_path)
    except BaseException as exc:
        _transient_exit(exc)
        raise AssertionError("unreachable") from exc
    summary = _prepared_summary(prepared)
    summary["status"] = "complete"
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


@train_app.command("worker", hidden=True)
def train_worker(
    config_path: ConfigOption,
    processed_path: Annotated[
        Path, typer.Option("--processed-path", exists=True, file_okay=False, readable=True)
    ],
    bundle_dir: Annotated[
        Path, typer.Option("--bundle-dir", exists=True, file_okay=False, readable=True)
    ],
    identity_path: Annotated[
        Path, typer.Option("--identity-path", exists=True, dir_okay=False, readable=True)
    ],
    output_dir: Annotated[Path, typer.Option("--output-dir")],
    micro_batch_size: Annotated[int, typer.Option("--micro-batch-size", min=1, max=2)],
    gradient_accumulation_steps: Annotated[
        int, typer.Option("--gradient-accumulation-steps", min=1)
    ],
    resume_from_checkpoint: Annotated[
        Path | None,
        typer.Option("--resume-from-checkpoint", exists=True, file_okay=False, readable=True),
    ] = None,
) -> None:
    """Internal torchrun worker; never invoke outside the high-level launcher."""

    from .workflow import run_distributed_training_worker

    run_distributed_training_worker(
        config_path=config_path,
        processed_path=processed_path,
        bundle_dir=bundle_dir,
        identity_path=identity_path,
        output_dir=output_dir,
        micro_batch_size_per_gpu=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        resume_from_checkpoint=resume_from_checkpoint,
    )


@evaluate_app.command("run")
def evaluate_run(
    config_path: ConfigOption = Path("configs/transition1x.yaml"),
) -> None:
    """Resume validation, selection, and the four ordered test inference modes."""

    from .workflow import prepare_smoke_workflow, run_evaluation_phase

    try:
        prepared = prepare_smoke_workflow(config_path)
        selected = run_evaluation_phase(prepared)
    except BaseException as exc:
        _transient_exit(exc)
        raise AssertionError("unreachable") from exc
    typer.echo(
        json.dumps(
            {
                **_prepared_summary(prepared),
                "status": "complete",
                "selected_epoch": selected.epoch,
                "selected_global_step": selected.global_step,
                "selected_checkpoint_role": selected.kind,
                "selected_checkpoint_train_loss": selected.train_loss,
                "selected_checkpoint_fingerprint": selected.checkpoint_fingerprint,
            },
            indent=2,
            sort_keys=True,
        )
    )


@infer_app.command("thinking")
def infer_thinking(
    config_path: ConfigOption = Path("configs/transition1x.yaml"),
    processed_path: Annotated[
        Path,
        typer.Option(
            "--processed-path",
            exists=True,
            file_okay=False,
            readable=True,
            help="Content-verified processed DatasetDict used by the checkpoint.",
        ),
    ] = ...,
    checkpoint_dir: Annotated[
        Path,
        typer.Option(
            "--checkpoint-dir",
            exists=True,
            file_okay=False,
            readable=True,
            help="Selected durable epoch checkpoint containing the portable adapter.",
        ),
    ] = ...,
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            file_okay=False,
            help="Parent for the isolated thinking-exploratory artifact tree.",
        ),
    ] = ...,
    split: Annotated[
        ExplorationSplit,
        typer.Option("--split", help="Dataset split to inspect without formal scoring."),
    ] = ExplorationSplit.val,
    reaction_ids: Annotated[
        list[str] | None,
        typer.Option(
            "--reaction-id",
            help="Exact reaction ID; repeat this option to select multiple records.",
        ),
    ] = None,
    limit: Annotated[
        int | None,
        typer.Option(
            "--limit",
            min=1,
            help="Limit selected records; defaults to two when no IDs are supplied.",
        ),
    ] = None,
) -> None:
    """Run optional thinking-mode sampling without touching formal experiment state."""

    from .workflow import (
        DelegatedProcessError,
        run_locked_gpu_command,
        torchrun_command,
    )

    config = load_config(config_path)
    arguments = [
        "--config",
        str(config_path.resolve()),
        "--processed-path",
        str(processed_path.resolve()),
        "--checkpoint-dir",
        str(checkpoint_dir.resolve()),
        "--output-dir",
        str(output_dir.resolve()),
        "--split",
        split.value,
    ]
    for reaction_id in reaction_ids or ():
        arguments.extend(("--reaction-id", reaction_id))
    if limit is not None:
        arguments.extend(("--limit", str(limit)))
    command = torchrun_command("janus_ts.thinking_inference", *arguments)
    try:
        outcome = run_locked_gpu_command(
            "exploratory-thinking-inference",
            command,
            lock_path=config.runtime.gpu_lock_path,
            required_gpu_count=config.runtime.required_gpu_count,
            timeout_seconds=None,
            require_clean_after=True,
        )
        if outcome.returncode != 0:
            raise DelegatedProcessError(outcome.phase, outcome.command, outcome.returncode)
    except BaseException as exc:
        _transient_exit(exc)
        raise AssertionError("unreachable") from exc


if __name__ == "__main__":
    app()
