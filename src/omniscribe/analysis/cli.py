"""CLI entry point for transcript quality analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.text import Text

console = Console()


def _suspicion_color(score: float) -> str:
    if score >= 0.6:
        return "red"
    if score >= 0.3:
        return "yellow"
    return "green"


def _format_ts(secs: int) -> str:
    hh, rem = divmod(secs, 3600)
    mm, ss = divmod(rem, 60)
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


def print_report(headers: dict, analyses: list, threshold_db: float) -> None:
    from .analyzer import SegmentAnalysis

    total = len(analyses)
    suspicious = [a for a in analyses if a.suspicion_score >= 0.3]
    likely = [a for a in analyses if a.suspicion_score >= 0.6]

    console.print()
    console.print("[bold]TRANSCRIPT QUALITY ANALYSIS[/bold]", style="cyan")
    console.print("─" * 60)
    if headers:
        console.print(f"  Model:     {headers.get('Model', '?')}")
        console.print(f"  Device:    {headers.get('Device', '?')}")
        console.print(f"  Threshold: {headers.get('Silence Threshold', '?')}")
        console.print(f"  Logprob:   {headers.get('Min Logprob', '?')}")
    console.print()

    # Summary table
    console.print("[bold]SUMMARY[/bold]")
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_row("Total segments", str(total))
    table.add_row("Suspicious (score ≥ 0.3)", f"[yellow]{len(suspicious)}[/yellow]")
    table.add_row("Likely hallucinations (score ≥ 0.6)", f"[red]{len(likely)}[/red]")
    if total:
        table.add_row("Clean segments", f"[green]{total - len(suspicious)}[/green]")
    console.print(table)
    console.print()

    if not suspicious:
        console.print("[green]No suspicious segments found.[/green]")
        return

    # Threshold recommendation
    silent_dbs = [a.rms_db for a in analyses if a.suspicion_score >= 0.6]
    if silent_dbs:
        # Recommend a threshold between the hallucinated segments' energy and the median
        all_dbs = sorted(a.rms_db for a in analyses)
        p25 = all_dbs[len(all_dbs) // 4]
        recommended = max(max(silent_dbs) + 2, threshold_db)
        console.print(
            f"[bold]Threshold recommendation:[/bold] "
            f"current [yellow]{threshold_db:.1f} dB[/yellow] → "
            f"try [cyan]{recommended:.1f} dB[/cyan]  "
            f"(25th-pct of all segments: {p25:.1f} dB)"
        )
        console.print()

    def _print_segment(a: "SegmentAnalysis") -> None:
        color = _suspicion_color(a.suspicion_score)
        seg = a.segment
        score_pct = int(a.suspicion_score * 100)
        label_ts = f"[{_format_ts(seg.timestamp_secs)}] {seg.label}"
        console.print(
            f"[{color}]●[/{color}] [bold]{label_ts}[/bold]  "
            f"[dim]score {score_pct}%[/dim]"
        )
        text_display = seg.text if len(seg.text) <= 80 else seg.text[:77] + "..."
        console.print(f"  [italic]{text_display}[/italic]")
        console.print(
            f"  [dim]RMS {a.rms_db:+.1f} dB | "
            f"peak {a.peak_db:+.1f} dB | "
            f"energy std {a.energy_std_db:.1f} dB | "
            f"{a.silence_ratio*100:.0f}% silence | "
            f"{a.words_per_sec:.1f} w/s | "
            f"window {a.audio_duration_secs:.1f}s[/dim]"
        )
        for flag in a.flags:
            console.print(f"  [dim]→ {flag}[/dim]")
        console.print()

    by_score = sorted(suspicious, key=lambda x: -x.suspicion_score)
    likely = [a for a in by_score if a.suspicion_score >= 0.6]
    borderline = [a for a in by_score if a.suspicion_score < 0.6]

    if likely:
        console.print(f"[bold red]LIKELY HALLUCINATIONS[/bold red]  ({len(likely)} of {total})")
        console.print("─" * 60)
        for a in likely:
            _print_segment(a)

    if borderline:
        console.print(f"[bold yellow]SUSPICIOUS[/bold yellow]  ({len(borderline)} of {total})")
        console.print("─" * 60)
        for a in borderline:
            _print_segment(a)


def plot_analysis(
    analyses: list,
    channels: dict,
    sr: int,
    threshold_db: float,
    output_path: Path | None,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        console.print("[red]matplotlib required for plots: pip install matplotlib[/red]")
        return

    from .analyzer import energy_curve

    fig, axes = plt.subplots(
        len(channels), 1,
        figsize=(18, 4 * len(channels)),
        sharex=True,
    )
    if len(channels) == 1:
        axes = [axes]

    label_colors = {"red": "#e74c3c", "yellow": "#f39c12", "green": "#27ae60"}

    # Per-channel subplot
    for ax, (ch_label, audio) in zip(axes, channels.items()):
        times, db_curve = energy_curve(audio, sr)
        ax.plot(times, db_curve, color="#3498db", linewidth=0.6, alpha=0.8, label="RMS energy")
        ax.axhline(threshold_db, color="orange", linewidth=1.0, linestyle="--",
                   label=f"silence threshold ({threshold_db:.0f} dB)")
        ax.set_ylabel("dB")
        ax.set_title(ch_label)
        ax.set_ylim(-70, 0)
        ax.grid(True, alpha=0.2)

        # Overlay transcript segments for this channel
        ch_analyses = [a for a in analyses if a.segment.label == ch_label]
        for a in ch_analyses:
            color = label_colors[_suspicion_color(a.suspicion_score)]
            x = a.segment.timestamp_secs
            ax.axvline(x, color=color, linewidth=0.8, alpha=0.6)
            if a.suspicion_score >= 0.3:
                ax.annotate(
                    f"[{_format_ts(a.segment.timestamp_secs)}]",
                    xy=(x, threshold_db - 3),
                    fontsize=5,
                    color=color,
                    rotation=90,
                    va="top",
                )

        legend_patches = [
            mpatches.Patch(color=label_colors["red"], label="likely hallucination (≥0.6)"),
            mpatches.Patch(color=label_colors["yellow"], label="suspicious (≥0.3)"),
            mpatches.Patch(color=label_colors["green"], label="clean"),
        ]
        ax.legend(handles=[ax.lines[0], ax.lines[1]] + legend_patches,
                  loc="upper right", fontsize=7)

    axes[-1].set_xlabel("Time (seconds)")
    fig.suptitle("Audio Energy vs Transcript Segments", fontsize=12, fontweight="bold")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        console.print(f"[green]Plot saved: {output_path}[/green]")
    else:
        plt.show()


def export_csv(analyses: list, output_path: Path) -> None:
    import csv
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "timestamp_secs", "label", "word_count",
            "rms_db", "peak_db", "energy_std_db", "silence_ratio",
            "audio_duration_secs", "words_per_sec",
            "expected_speech_secs", "suspicion_score", "flags", "text",
        ])
        for a in analyses:
            s = a.segment
            w.writerow([
                s.timestamp_secs, s.label, s.word_count,
                f"{a.rms_db:.2f}", f"{a.peak_db:.2f}", f"{a.energy_std_db:.2f}",
                f"{a.silence_ratio:.3f}",
                f"{a.audio_duration_secs:.2f}", f"{a.words_per_sec:.2f}",
                f"{s.expected_speech_secs:.2f}", f"{a.suspicion_score:.2f}",
                "; ".join(a.flags), s.text,
            ])
    console.print(f"[green]CSV saved: {output_path}[/green]")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze transcript quality by correlating audio energy with transcribed segments."
    )
    parser.add_argument("wav", type=Path, help="WAV recording file")
    parser.add_argument(
        "transcript", type=Path, nargs="?",
        help="Transcript .txt file (defaults to wav path with .txt extension)"
    )
    parser.add_argument(
        "--silence-threshold-db", type=float, default=-50.0,
        help="RMS dB below which audio is considered silent (default: -50.0)",
    )
    parser.add_argument(
        "--mic-label", default="You",
        help="Label used for microphone in transcript (default: You)",
    )
    parser.add_argument(
        "--system-label", default="Them",
        help="Label used for system audio in transcript (default: Them)",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Show interactive energy plot with transcript overlays",
    )
    parser.add_argument(
        "--plot-output", type=Path, default=None,
        help="Save plot to file instead of displaying (e.g. analysis.png)",
    )
    parser.add_argument(
        "--csv", type=Path, default=None,
        help="Export per-segment data to CSV",
    )
    parser.add_argument(
        "--min-suspicion", type=float, default=0.3,
        help="Minimum suspicion score to include in report (default: 0.3)",
    )
    args = parser.parse_args()

    wav_path = args.wav
    transcript_path = args.transcript or wav_path.with_suffix(".txt")

    if not wav_path.exists():
        console.print(f"[red]WAV file not found: {wav_path}[/red]")
        return 1
    if not transcript_path.exists():
        console.print(f"[red]Transcript not found: {transcript_path}[/red]")
        return 1

    console.print(f"[dim]Loading {wav_path.name} …[/dim]")

    from .analyzer import analyze
    headers, analyses, channels, sr, skipped, drift_scale = analyze(
        wav_path,
        transcript_path,
        silence_threshold_db=args.silence_threshold_db,
        mic_label=args.mic_label,
        system_label=args.system_label,
    )

    # Override threshold with value from transcript header if present
    if "Silence Threshold" in headers:
        try:
            threshold_db = float(headers["Silence Threshold"].split()[0])
        except ValueError:
            threshold_db = args.silence_threshold_db
    else:
        threshold_db = args.silence_threshold_db

    if drift_scale is not None:
        console.print(
            f"[yellow]Warning: transcript timestamps were wall-clock inflated "
            f"(old recording, fixed in current version). "
            f"Applied {drift_scale:.3f}× correction to align with audio.[/yellow]"
        )
    if skipped:
        console.print(
            f"[dim]Note: {skipped} segment(s) skipped — timestamps beyond audio length.[/dim]"
        )
    print_report(headers, analyses, threshold_db)

    if args.csv:
        export_csv(analyses, args.csv)

    if args.plot or args.plot_output:
        plot_analysis(analyses, channels, sr, threshold_db, args.plot_output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
