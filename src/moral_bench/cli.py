from __future__ import annotations

from pathlib import Path

import typer
from dotenv import load_dotenv

from . import ethics, runner
from . import report as report_mod
from .models import PROVIDERS, SystemOne

app = typer.Typer(add_completion=False, help="Benchmark a decision model against ETHICS.")

DATA = Path("data/raw")
RESULTS = Path("results")


@app.command()
def fetch() -> None:
    """Download the ETHICS dataset."""
    typer.echo(f"ETHICS at {ethics.fetch(DATA)}")


@app.command()
def run(
    provider: str = typer.Option("kev", help=f"One of: {', '.join(PROVIDERS)}"),
    name: str = typer.Option(None, help="Label for this model in results, e.g. kev-0.8b"),
    n: int = typer.Option(300, help="Items sampled per framework"),
    split: str = typer.Option("test", help="ETHICS split: test or test_hard"),
    seed: int = typer.Option(0),
    workers: int = typer.Option(4),
    base_url: str = typer.Option(None, help="Override the provider's base URL"),
) -> None:
    """Score a sample of every framework and cache the predictions."""
    load_dotenv()
    root = ethics.fetch(DATA)
    model = SystemOne.from_provider(provider, base_url)
    path = runner.predictions_path(RESULTS, name or provider)
    items = [
        it
        for fw in ethics.FRAMEWORKS
        for it in ethics.sample(ethics.load(fw, root, split, seed), n, seed)
    ]
    with typer.progressbar(length=len(items), label=name or provider) as bar:
        cached = len(items) - runner.run(items, model, path, workers, on_done=lambda: bar.update(1))
        bar.update(cached)
    typer.echo(f"{len(items)} items ({cached} from cache) -> {path}")
    typer.echo(f"tokens used: {model.usage}")


@app.command()
def report() -> None:
    """Write results/report.md, summary.json and alignment.png."""
    path = report_mod.write(RESULTS)
    typer.echo(path.read_text())


if __name__ == "__main__":
    app()
