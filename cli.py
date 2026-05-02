"""Command-line interface — `greenwash analyse` and `greenwash eval`."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(
    add_completion=False,
    help="Multi-agent greenwashing detector with LangSmith tracing.",
)
console = Console()


@app.command()
def analyse(
    text: str | None = typer.Argument(None, help="Inline text to analyse."),
    file: Path | None = typer.Option(None, "--file", "-f", help="Path to a text file."),
    out: Path | None = typer.Option(None, "--out", "-o", help="Write JSON result here."),
) -> None:
    """Run the full pipeline on a document."""
    if not text and not file:
        console.print("[red]Provide either inline TEXT or --file.[/red]")
        raise typer.Exit(1)

    document = file.read_text() if file else (text or "")

    # Imported here so the CLI starts fast when only --help is used.
    from .graph import analyse as run

    with console.status("Running detector → classifier → rewriter…"):
        result = run(document)

    console.print(Panel(result.original_text, title="Original", border_style="dim"))
    console.print(Panel(result.final_text, title="Rewritten", border_style="green"))

    if result.spans:
        table = Table(title="Decisions", show_lines=True)
        table.add_column("Phrase", style="yellow")
        table.add_column("Action", style="cyan")
        table.add_column("Justification")
        decisions_by_phrase = {d.phrase: d for d in result.decisions}
        for span in result.spans:
            d = decisions_by_phrase.get(span.phrase)
            table.add_row(
                span.phrase,
                d.action.value if d else "—",
                d.justification if d else "—",
            )
        console.print(table)
    else:
        console.print("[green]No greenwashing detected.[/green]")

    if result.errors:
        console.print(Panel("\n".join(result.errors), title="Errors", border_style="red"))

    if out:
        out.write_text(result.model_dump_json(indent=2))
        console.print(f"[dim]Saved to {out}[/dim]")


@app.command()
def evaluate(
    dataset: str = typer.Option("greenwashing-eval", help="LangSmith dataset name."),
    seed: bool = typer.Option(False, "--seed", help="Upload the seed dataset first."),
) -> None:
    """Run the LangSmith evaluation suite."""
    from evals.run_eval import main as run_eval

    run_eval(dataset_name=dataset, seed=seed)


if __name__ == "__main__":
    app()
