"""CLI commands for GVCA SAC Survey Analysis."""

import asyncio
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

app = typer.Typer(
    name="gvca",
    help="GVCA SAC Survey Analysis CLI",
    add_completion=False,
)
console = Console()


def run_async(coro):
    """Run an async function in the event loop."""
    return asyncio.get_event_loop().run_until_complete(coro)


@app.command()
def import_data(
    year: str = typer.Argument(..., help="Survey year (e.g., 2025)"),
    file: Path = typer.Option(..., "--file", "-f", help="Path to CSV file"),
    replace: bool = typer.Option(False, "--replace", "-r", help="Replace existing data"),
):
    """Import survey data from a CSV file."""
    from app.core.transform import transform_and_persist

    if not file.exists():
        console.print(f"[red]Error:[/red] File not found: {file}")
        raise typer.Exit(1)

    with open(file, "rb") as f:
        file_content = f.read()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Importing data for year {year}...", total=None)

        try:
            result = run_async(
                transform_and_persist(file_content, year, replace_existing=replace)
            )
            progress.update(task, completed=True)

            console.print(f"\n[green]Import successful![/green]")
            console.print(f"  Total responses: {result['total_responses']}")
            console.print(f"  Free responses extracted: {result['free_responses_extracted']}")
            console.print(f"  Replaced existing: {result['replaced_existing']}")

        except Exception as e:
            progress.update(task, completed=True)
            console.print(f"\n[red]Error:[/red] {e}")
            raise typer.Exit(1)


@app.command()
def export_data(
    year: str = typer.Argument(..., help="Survey year (e.g., 2025)"),
    output: Path = typer.Option(
        Path("export.csv"),
        "--output",
        "-o",
        help="Output CSV file path",
    ),
):
    """Export survey data to a CSV file."""
    from app.services.pocketbase_client import pb_client
    import polars as pl

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Exporting data for year {year}...", total=None)

        try:
            # Fetch data from PocketBase
            responses = run_async(
                pb_client.get_full_list(
                    "survey_responses",
                    filter_str=f'year = "{year}"',
                )
            )

            if not responses:
                progress.update(task, completed=True)
                console.print(f"[yellow]No data found for year {year}[/yellow]")
                raise typer.Exit(1)

            # Convert to DataFrame and export
            df = pl.DataFrame(responses)
            df.write_csv(output)

            progress.update(task, completed=True)
            console.print(f"\n[green]Export successful![/green]")
            console.print(f"  Exported {len(responses)} responses to {output}")

        except Exception as e:
            progress.update(task, completed=True)
            console.print(f"\n[red]Error:[/red] {e}")
            raise typer.Exit(1)


@app.command()
def tag_batch(
    year: str = typer.Argument(..., help="Survey year (e.g., 2025)"),
    retag: bool = typer.Option(False, "--retag", help="Retag already tagged responses"),
    wait: bool = typer.Option(False, "--wait", "-w", help="Wait for batch to complete"),
):
    """Start batch AI tagging for survey responses."""
    from app.services.openai_batch import openai_batch_client
    from app.services.pocketbase_client import pb_client
    import time

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Preparing batch tagging...", total=None)

        try:
            # Get responses to tag
            if retag:
                free_responses = run_async(
                    pb_client.get_full_list(
                        "free_responses",
                        filter_str=f'year = "{year}"',
                    )
                )
            else:
                free_responses = run_async(
                    pb_client.get_untagged_free_responses(year)
                )

            if not free_responses:
                progress.update(task, completed=True)
                console.print(f"[yellow]No responses to tag for year {year}[/yellow]")
                raise typer.Exit(0)

            progress.update(task, description=f"Creating batch input ({len(free_responses)} responses)...")

            # Create batch input file
            input_file_id = run_async(
                openai_batch_client.create_batch_input_file(free_responses)
            )

            progress.update(task, description="Submitting batch to OpenAI...")

            # Submit batch
            batch_result = run_async(
                openai_batch_client.submit_batch(input_file_id)
            )

            progress.update(task, completed=True)

            console.print(f"\n[green]Batch submitted![/green]")
            console.print(f"  Batch ID: {batch_result['batch_id']}")
            console.print(f"  Status: {batch_result['status']}")
            console.print(f"  Total items: {len(free_responses)}")

            if wait:
                console.print("\n[yellow]Waiting for batch completion...[/yellow]")
                console.print("(This may take up to 24 hours for large batches)")

                while True:
                    status = run_async(
                        openai_batch_client.check_batch_status(batch_result['batch_id'])
                    )

                    if status['status'] == 'completed':
                        console.print(f"\n[green]Batch completed![/green]")
                        console.print(f"  Completed: {status['request_counts']['completed']}")
                        console.print(f"  Failed: {status['request_counts']['failed']}")
                        break
                    elif status['status'] in ('failed', 'expired', 'cancelled'):
                        console.print(f"\n[red]Batch {status['status']}[/red]")
                        raise typer.Exit(1)
                    else:
                        console.print(
                            f"  Status: {status['status']} - "
                            f"{status['request_counts']['completed']}/{status['request_counts']['total']} completed"
                        )
                        time.sleep(60)  # Poll every minute

        except typer.Exit:
            raise
        except Exception as e:
            progress.update(task, completed=True)
            console.print(f"\n[red]Error:[/red] {e}")
            raise typer.Exit(1)


@app.command()
def generate_charts(
    year: str = typer.Argument(..., help="Survey year (e.g., 2025)"),
    chart_type: str = typer.Option(
        "all",
        "--type",
        "-t",
        help="Chart type: all, demographic, trend, sentiment",
    ),
    output_dir: Path = typer.Option(
        Path("artifacts"),
        "--output-dir",
        "-o",
        help="Output directory for charts",
    ),
):
    """Generate visualization charts for survey data."""
    from app.core.charts import (
        generate_demographic_comparison_chart,
        generate_trend_comparison_chart,
        generate_sentiment_chart,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        charts_generated = []

        if chart_type in ("all", "demographic"):
            task = progress.add_task("Generating demographic comparison charts...", total=None)
            try:
                # Generate demographic comparison charts
                segments = [
                    ("year1_family", "not_year1_family"),
                    ("minority", "not_minority"),
                    ("support", "not_support"),
                ]
                for seg_a, seg_b in segments:
                    result = run_async(
                        generate_demographic_comparison_chart(year, seg_a, seg_b, export_png=True)
                    )
                    if result.get("file_path"):
                        charts_generated.append(result["file_path"])
                progress.update(task, completed=True)
            except Exception as e:
                progress.update(task, completed=True)
                console.print(f"[yellow]Warning: Could not generate demographic charts: {e}[/yellow]")

        if chart_type in ("all", "sentiment"):
            task = progress.add_task("Generating sentiment chart...", total=None)
            try:
                result = run_async(
                    generate_sentiment_chart(year, export_png=True)
                )
                if result.get("file_path"):
                    charts_generated.append(result["file_path"])
                progress.update(task, completed=True)
            except Exception as e:
                progress.update(task, completed=True)
                console.print(f"[yellow]Warning: Could not generate sentiment chart: {e}[/yellow]")

        if chart_type in ("all", "trend"):
            task = progress.add_task("Generating trend comparison chart...", total=None)
            try:
                # Get all available years
                from app.services.pocketbase_client import pb_client
                years = run_async(pb_client.get_distinct_years("survey_responses"))
                if len(years) >= 2:
                    result = run_async(
                        generate_trend_comparison_chart(years, export_png=True)
                    )
                    if result.get("file_path"):
                        charts_generated.append(result["file_path"])
                progress.update(task, completed=True)
            except Exception as e:
                progress.update(task, completed=True)
                console.print(f"[yellow]Warning: Could not generate trend chart: {e}[/yellow]")

    console.print(f"\n[green]Charts generated: {len(charts_generated)}[/green]")
    for path in charts_generated:
        console.print(f"  - {path}")


@app.command()
def status(
    year: Optional[str] = typer.Argument(None, help="Survey year (optional)"),
):
    """Show status of survey data and processing."""
    from app.services.pocketbase_client import pb_client

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Fetching status...", total=None)

        try:
            # Get years with data
            years = run_async(pb_client.get_distinct_years("survey_responses"))

            if year and year not in years:
                progress.update(task, completed=True)
                console.print(f"[yellow]No data found for year {year}[/yellow]")
                console.print(f"Available years: {', '.join(years) if years else 'None'}")
                raise typer.Exit(0)

            years_to_show = [year] if year else years

            progress.update(task, completed=True)

            # Create status table
            table = Table(title="Survey Data Status")
            table.add_column("Year", style="cyan")
            table.add_column("Survey Responses", justify="right")
            table.add_column("Free Responses", justify="right")
            table.add_column("Tagged", justify="right")
            table.add_column("Clustered", justify="right")

            for y in years_to_show:
                # Count records
                survey_count = run_async(
                    pb_client.count("survey_responses", f'year = "{y}"')
                )
                free_count = run_async(
                    pb_client.count("free_responses", f'year = "{y}"')
                )
                tagged_count = run_async(
                    pb_client.count("tagging_results", f'year = "{y}"')
                )
                clustered_count = run_async(
                    pb_client.count("clustering_results", f'year = "{y}"')
                )

                table.add_row(
                    y,
                    str(survey_count),
                    str(free_count),
                    str(tagged_count),
                    str(clustered_count),
                )

            console.print()
            console.print(table)

        except typer.Exit:
            raise
        except Exception as e:
            progress.update(task, completed=True)
            console.print(f"\n[red]Error:[/red] {e}")
            raise typer.Exit(1)


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
