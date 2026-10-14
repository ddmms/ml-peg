"""Commandline interface for the optimised inference speed benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from typer import Option, Typer

inference_speed_app = Typer(
    name="inference-speed",
    no_args_is_help=True,
    epilog="Try 'ml_peg inference-speed COMMAND --help' for subcommand options",
)


@inference_speed_app.command(name="stage", help="Stage a campaign input directory")
def stage(
    base_dir: Annotated[
        Path, Option(help="Directory containing <material>.extxyz base cells.")
    ],
    out: Annotated[Path, Option(help="Campaign directory to create.")],
    template: Annotated[
        Path | None, Option(help="Spec template. Default is the benchmark's.")
    ] = None,
    campaign_id: Annotated[
        str | None, Option(help="Campaign ID. Default is derived from the inputs.")
    ] = None,
    max_atoms: Annotated[
        int | None, Option(help="Skip sizes larger than this many atoms.")
    ] = None,
) -> None:
    """
    Stage a campaign input directory.

    Parameters
    ----------
    base_dir
        Directory containing base cells.
    out
        Campaign directory to create.
    template
        Spec template.
    campaign_id
        Campaign ID.
    max_atoms
        Skip sizes larger than this many atoms.
    """
    from ml_peg.calcs.molecular_dynamics.optimized_inference_speed.stage import (
        SPEC_TEMPLATE,
        stage_campaign,
    )

    spec = stage_campaign(
        base_dir,
        out,
        template=template or SPEC_TEMPLATE,
        campaign_id=campaign_id,
        max_atoms=max_atoms,
    )
    for system in spec["systems"]:
        sizes = ", ".join(str(size["n_atoms"]) for size in system["sizes"])
        print(f"{system['material']}: {sizes}")
    print(f"Staged campaign {spec['campaign']['id']} in {out}")


@inference_speed_app.command(name="build", help="Build a container image")
def build(
    model: Annotated[str, Option(help="Container name in contrib.")],
    cache_dir: Annotated[
        Path | None, Option(help="Directory for images. Default is $MLPEG_SIF_CACHE.")
    ] = None,
) -> None:
    """
    Build a container image.

    Parameters
    ----------
    model
        Container name.
    cache_dir
        Directory for images.
    """
    from ml_peg.calcs.molecular_dynamics.optimized_inference_speed.build import (
        DEFAULT_CACHE,
        build_container,
    )

    record = build_container(model, cache_dir or DEFAULT_CACHE)
    print(f"Built {record['sif']} (sha256 {record['sha256']})")


@inference_speed_app.command(name="run", help="Run containers on a campaign")
def run(
    campaign: Annotated[Path, Option(help="Campaign directory.")],
    models: Annotated[
        str | None, Option(help="Comma-separated container names. Default is all.")
    ] = None,
    runtime: Annotated[
        Literal["apptainer", "local"],
        Option(help="Run with Apptainer, or run the local entry without it."),
    ] = "apptainer",
    cache_dir: Annotated[
        Path | None, Option(help="Directory for images. Default is $MLPEG_SIF_CACHE.")
    ] = None,
    precisions: Annotated[
        str | None, Option(help="Comma-separated precision modes. Default is all.")
    ] = None,
    repeats: Annotated[
        int | None, Option(help="Number of repeats. Default is from the spec.")
    ] = None,
    timeout: Annotated[
        float | None, Option(help="Timeout per run in seconds. Default from spec.")
    ] = None,
    container_env: Annotated[
        list[str] | None,
        Option(help="Extra KEY=VALUE environment variable for containers."),
    ] = None,
    seed: Annotated[int, Option(help="Seed for shuffling the run order.")] = 0,
    gpu_index: Annotated[int, Option(help="Index of the GPU to sample.")] = 0,
) -> None:
    """
    Run containers on a campaign.

    Parameters
    ----------
    campaign
        Campaign directory.
    models
        Comma-separated container names.
    runtime
        ``apptainer`` or ``local``.
    cache_dir
        Directory for images.
    precisions
        Comma-separated precision modes.
    repeats
        Number of repeats.
    timeout
        Timeout per run in seconds.
    container_env
        Extra environment variables for containers.
    seed
        Seed for shuffling the run order.
    gpu_index
        Index of the GPU to sample.
    """
    from ml_peg.calcs.molecular_dynamics.optimized_inference_speed.build import (
        DEFAULT_CACHE,
    )
    from ml_peg.calcs.molecular_dynamics.optimized_inference_speed.run import (
        run_campaign,
    )

    records = run_campaign(
        campaign,
        models=models.split(",") if models else None,
        runtime=runtime,
        cache_dir=cache_dir or DEFAULT_CACHE,
        precisions=precisions.split(",") if precisions else None,
        repeats=repeats,
        timeout=timeout,
        container_env=container_env or (),
        seed=seed,
        gpu_index=gpu_index,
    )
    statuses: dict[str, int] = {}
    for record in records:
        statuses[record["status"]] = statuses.get(record["status"], 0) + 1
    print(f"Finished {len(records)} runs: {statuses}")


@inference_speed_app.command(name="analyse", help="Validate, aggregate and plot")
def analyse(
    campaign: Annotated[Path, Option(help="Campaign directory.")],
) -> None:
    """
    Validate, aggregate and plot a campaign.

    Parameters
    ----------
    campaign
        Campaign directory.
    """
    from ml_peg.analysis.molecular_dynamics.optimized_inference_speed.aggregate import (  # noqa: E501
        analyse_campaign,
    )

    results = analyse_campaign(campaign)
    runs = results["runs"]
    print(runs[["model", "precision", "run_id", "status", "valid", "reasons"]])
    print(f"Wrote results to {campaign / 'analysis'}")
