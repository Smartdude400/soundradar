"""``python -m soundradar`` -- launch the window, or scan a file from a shell."""
from __future__ import annotations

import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] not in {"-h", "--help", "gui"}:
        from .offline import analyse_file

        sensitivity = 0.35
        if len(argv) > 1 and argv[1].startswith("--sensitivity="):
            sensitivity = float(argv.pop(1).split("=", 1)[1])
        report = analyse_file(argv[0], sensitivity=sensitivity)
        print(f"{report.path}")
        print(f"{report.duration:.2f} s · {report.rate} Hz · {report.channels} ch · "
              f"background {report.background_db:.1f} dBFS")
        print(f"{len(report.events)} events")
        print(f"{'time':>9}  {'type':<10} {'freq':>9}  {'level':>7}  {'above':>6}  {'dur':>7}")
        for row in report.to_rows():
            print(f"{row['time_s']:9.3f}  {row['type']:<10} {row['freq_hz']:8.0f}Hz  "
                  f"{row['level_dbfs']:6.1f}  {row['snr_db']:6.1f}  "
                  f"{row['duration_ms']:6.0f}ms")
        if len(argv) > 1:
            report.write_csv(Path(argv[1]))
            print(f"wrote {argv[1]}")
        return 0
    from .ui.app import main as run_app

    run_app()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
