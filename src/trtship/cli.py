"""trtship command line interface.

Typer, so ``--help`` comes from type hints and docstrings. Every subcommand maps
to one pipeline stage plus the composite ``ship``; each parses into the same
pydantic config objects the Python API uses (FR-8.2, FR-8.3).

Only ``doctor`` and ``version`` exist so far — Phase 0. Stages are added as they
land rather than stubbed, so ``--help`` never advertises something that does not
work.
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from trtship import __version__
from trtship.env import Environment, Stage, detect

app = typer.Typer(
    name="trtship",
    help="Take a PyTorch model to a benchmarked, calibrated, Triton-servable TensorRT engine.",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()


@app.command()
def version() -> None:
    """Print the trtship version."""
    console.print(f"trtship {__version__}")


@app.command()
def doctor(
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Include the raw environment dictionary."
    ),
) -> None:
    """Check whether this machine can run trtship, and which stages.

    Run this first on any new machine. It answers the question that otherwise
    costs an hour of confused debugging: is my environment wrong, or is my model
    wrong?

    Examples:

        trtship doctor

        trtship doctor --verbose
    """
    env = detect()
    console.print()
    console.print(_environment_table(env))

    if env.gpus:
        console.print()
        console.print(_gpu_table(env))

    console.print()
    console.print(_stage_table(env))

    blockers = env.blockers()
    if blockers:
        console.print()
        console.print(Text("Notes", style="bold"))
        for line in blockers:
            console.print(Text("  • ", style="yellow") + Text(line))

    if verbose:
        console.print()
        console.print(env.to_dict())

    console.print()
    if env.torch_version is None:
        console.print(Text("PyTorch is missing — trtship cannot run here.", style="bold red"))
        raise typer.Exit(code=1)
    if env.can_build_engines:
        console.print(Text("Ready: this machine can run the full pipeline.", style="bold green"))
    else:
        console.print(
            Text("Partial: CPU stages only. ", style="bold yellow")
            + Text("Engine build, calibration and benchmarking need a CUDA GPU with TensorRT.")
        )


def _environment_table(env: Environment) -> Table:
    t = Table(title="Environment", title_justify="left", show_header=False, box=None, pad_edge=False)
    t.add_column("key", style="dim", no_wrap=True)
    t.add_column("value")
    t.add_row("platform", f"{env.platform} ({env.machine})")
    t.add_row("python", env.python_version)
    t.add_row("torch", _present(env.torch_version))
    t.add_row("cuda", _present(env.cuda_version) if env.cuda_available else _absent("unavailable"))
    t.add_row("driver", _present(env.driver_version))
    t.add_row("tensorrt", _present(env.tensorrt_version))
    ort = env.onnxruntime_version
    t.add_row("onnxruntime", _present(f"{ort} (gpu)" if ort and env.onnxruntime_gpu else ort))
    t.add_row("docker", _present("available") if env.docker_available else _absent("not found"))
    return t


def _gpu_table(env: Environment) -> Table:
    t = Table(title="GPUs", title_justify="left", box=None, pad_edge=False)
    t.add_column("#", style="dim")
    t.add_column("name")
    t.add_column("sm", justify="right")
    t.add_column("memory", justify="right")
    t.add_column("precisions")
    for g in env.gpus:
        t.add_row(
            str(g.index),
            g.name,
            f"{g.compute_capability[0]}.{g.compute_capability[1]}",
            f"{g.total_memory_mb / 1024:.1f} GB",
            ", ".join(g.supported_precisions),
        )
    return t


def _stage_table(env: Environment) -> Table:
    available = set(env.available_stages())
    t = Table(title="Pipeline stages", title_justify="left", box=None, pad_edge=False)
    t.add_column("stage", no_wrap=True)
    t.add_column("status")
    for stage in Stage:
        ok = stage in available
        t.add_row(
            stage.value,
            Text("available", style="green") if ok else Text("unavailable", style="dim"),
        )
    return t


def _present(value: str | None) -> Text:
    return Text(value, style="green") if value else _absent("not installed")


def _absent(label: str) -> Text:
    return Text(label, style="dim")


if __name__ == "__main__":
    app()
