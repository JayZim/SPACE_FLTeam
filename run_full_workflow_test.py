import sys
import os
import argparse
import subprocess
from pathlib import Path
from datetime import datetime, timedelta

try:
    from skyfield.api import load, utc
except Exception:
    load = None
    utc = None


def find_project_root(start: Path) -> Path:
    cur = start.resolve()
    for parent in [cur] + list(cur.parents):
        if (parent / "main.py").exists() and (parent / "requirements.txt").exists():
            return parent
    return cur


def find_latest(path: Path, pattern: str) -> Path | None:
    matches = sorted(path.glob(pattern), key=lambda p: p.stat().st_ctime, reverse=True)
    return matches[0] if matches else None


def _parse_tle_file(tle_path: Path) -> dict:
    lines = [ln.strip() for ln in tle_path.read_text().splitlines() if ln.strip()]
    tle_dict = {}
    i = 0
    while i < len(lines):
        name = lines[i]
        if i + 2 < len(lines):
            l1 = lines[i + 1]
            l2 = lines[i + 2]
            tle_dict[name] = [l1, l2]
            i += 3
        else:
            break
    return tle_dict


def _generate_sat_sim_from_tles(project_root: Path, tle_path: Path) -> Path:
    try:
        from sat_sim.sat_sim import SatSim
    except Exception as e:
        print(f"Failed to import SatSim: {e}")
        sys.exit(1)

    if load is None:
        print("skyfield not available. Please install requirements first.")
        sys.exit(1)

    tle_dict = _parse_tle_file(tle_path)
    if not tle_dict:
        print(f"No valid TLE records parsed from {tle_path}")
        sys.exit(1)

    # Timing options (interactive)
    print("\nConfigure SatSim time options (real-time now; 1 minute per step):")
    steps_str = input("Timesteps (integer, default 10): ").strip() or "10"
    out_type = "txt"

    try:
        steps = int(steps_str)
        minutes_per_step = 1
    except ValueError:
        steps = 10
        minutes_per_step = 1

    # Build times
    ts = load.timescale()
    # Use current real time
    dt_start = datetime.now(tz=utc)
    start_time = ts.from_datetime(dt_start)
    end_time = ts.from_datetime((dt_start + timedelta(minutes=steps * minutes_per_step)))

    # Run SatSim
    sim = SatSim(start_time=start_time, end_time=end_time, timestep=minutes_per_step, output_file_type=out_type, gui_enabled=False, tle_data=tle_dict, output_to_file=True)
    # Ensure the internal output module uses the desired file type (txt)
    try:
        sim.set_output_file_type(out_type)
    except Exception:
        pass
    sim.run_with_adj_matrix()

    # Find latest generated file under sat_sim/output
    out_dir = project_root / "sat_sim" / "output"
    # Try both txt and csv to avoid mismatch when the inner output defaults to csv
    candidates = []
    for pat in ["sat_sim_*.txt", "sat_sim_*.csv"]:
        p = find_latest(out_dir, pat)
        if p is not None:
            candidates.append(p)
    latest = max(candidates, key=lambda p: p.stat().st_ctime) if candidates else None
    if latest is None:
        print("Failed to locate generated sat_sim output file.")
        sys.exit(1)
    print(f"Generated SatSim file: {latest}")
    return latest


def run_workflow(project_root: Path, sat_sim_file: Path, timesteps: int | None, custom_duration: str | None, quiet: bool = False, interactive_fl_output: bool = False) -> int:
    cmd = [
        sys.executable,
        str(project_root / "main.py"),
        "flomps",
        str(sat_sim_file),
    ]

    if timesteps is not None:
        cmd += ["--timesteps", str(timesteps)]
    if custom_duration is not None:
        cmd += ["--custom-duration", custom_duration]
    # Allow interactive FLAM selection to ensure correct FL input during FL stage
    cmd += ["--select-flam"]
    if interactive_fl_output:
        cmd += ["--interactive-fl-output"]

    print("Running:", " ".join(cmd))
    if quiet:
        proc = subprocess.run(cmd, cwd=str(project_root), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # Keep a short summary to avoid flooding the terminal
        if proc.stdout:
            lines = proc.stdout.strip().splitlines()
            tail = "\n".join(lines[-50:])  # last 50 lines as summary
            print(tail)
        if proc.stderr:
            print(proc.stderr)
    else:
        proc = subprocess.run(cmd, cwd=str(project_root))
    return proc.returncode


def _interactive_choose_sat_sim(project_root: Path) -> Path:
    candidates_dir = project_root / "sat_sim" / "output"
    files = sorted([p for p in candidates_dir.glob("*.txt")], key=lambda p: p.name)
    if not files:
        print(f"No SatSim .txt files found under {candidates_dir}")
        sys.exit(1)

    print("\nSelect a SatSim output file:")
    for i, p in enumerate(files, 1):
        print(f"  {i}) {p.name}")

    while True:
        choice = input(f"Enter number (1-{len(files)}): ").strip()
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(files):
                return files[idx - 1]
        print("Invalid selection. Try again.")


def _interactive_options() -> tuple[int | None, str | None]:
    print("\nOptional parameters (press Enter to accept defaults):")
    ts = input("Timesteps (integer, each = 1 minute) [default empty]: ").strip()
    cd = input("Custom duration (HH:MM:SS) [default 00:10:00]: ").strip()

    timesteps = int(ts) if ts.isdigit() else None
    custom_duration = cd if cd else "00:10:00"
    return timesteps, custom_duration


def main():
    parser = argparse.ArgumentParser(description="Run full FLOMPS workflow and print output locations")
    parser.add_argument("sat_sim_file", nargs="?", type=str, help="Path to SatSim output .txt file")
    parser.add_argument("--timesteps", type=int, default=None, help="Custom timesteps (each = 1 minute)")
    parser.add_argument("--custom-duration", type=str, default=None, help="Custom duration in HH:MM(:SS)")
    parser.add_argument("--auto", action="store_true", help="Run with sensible defaults without prompts (use latest SatSim or generate from 8-sat TLE)")
    parser.add_argument("--quiet", action="store_true", help="Reduce console output; show only a short summary")
    parser.add_argument("--interactive-fl-output", action="store_true", help="Enable interactive prompts for GIF generation and dashboard creation")
    args = parser.parse_args()

    start_path = Path(__file__).parent
    project_root = find_project_root(start_path)

    generated_via_tle = False
    using_existing_adjacency_txt = False
    # Determine sat_sim_file (arg or interactive/auto)
    if args.sat_sim_file:
        sat_sim_file = Path(args.sat_sim_file)
        if not sat_sim_file.exists():
            print(f"Error: SatSim file not found: {sat_sim_file}")
            sys.exit(1)
        # Detect adjacency txt (starts with 'Number of satellites:')
        try:
            if sat_sim_file.suffix == ".txt":
                with open(sat_sim_file, 'r') as f:
                    first_line = f.readline().strip()
                if first_line.startswith('Number of satellites:'):
                    using_existing_adjacency_txt = True
        except Exception:
            pass
    else:
        if args.auto:
            # Prefer latest existing sat_sim, else generate from 8-sat TLE with 10 steps
            out_dir = project_root / "sat_sim" / "output"
            latest_existing = find_latest(out_dir, "sat_sim_*.txt")
            if latest_existing is not None:
                sat_sim_file = latest_existing
            else:
                tle_dir = project_root / "TLEs"
                tle_file = tle_dir / "SatCount8.tle"
                sat_sim_file = _generate_sat_sim_from_tles(project_root, tle_file)
                generated_via_tle = True
        else:
            # Offer to generate from TLEs or pick existing
            print("\nChoose input source:")
            print("  1) Use existing sat_sim/output/*.txt")
            print("  2) Generate from TLEs (interactive)")
            choice = input("Enter 1 or 2: ").strip()
            if choice == "2":
                # Generate from TLEs
                tle_dir = project_root / "TLEs"
                candidates = [
                    ("1", "SatCount1.tle"),
                    ("3", "SatCount3.tle"),
                    ("4", "SatCount4.tle"),
                    ("8", "SatCount8.tle"),
                    ("40", "SatCount40.tle"),
                    ("NovaSar", "NovaSar.tle"),
                    ("Walker", "Walker.tle"),
                ]
                available = [(label, fname) for (label, fname) in candidates if (tle_dir / fname).exists()]
                if not available:
                    print(f"No TLE files found in {tle_dir}; falling back to existing sat_sim outputs.")
                    sat_sim_file = _interactive_choose_sat_sim(project_root)
                else:
                    print("\nSelect satellite set:")
                    for i, (label, fname) in enumerate(available, 1):
                        print(f"  {i}) {label} -> {fname}")
                    idx = None
                    while idx is None:
                        s = input(f"Enter number (1-{len(available)}): ").strip()
                        if s.isdigit() and 1 <= int(s) <= len(available):
                            idx = int(s) - 1
                    _, tle_file = available[idx]
                    sat_sim_file = _generate_sat_sim_from_tles(project_root, tle_dir / tle_file)
                    generated_via_tle = True
            else:
                sat_sim_file = _interactive_choose_sat_sim(project_root)
                # Detect adjacency txt
                try:
                    with open(sat_sim_file, 'r') as f:
                        first_line = f.readline().strip()
                    if first_line.startswith('Number of satellites:'):
                        using_existing_adjacency_txt = True
                except Exception:
                    pass

    # Determine options (use provided flags or interactive prompts if both missing)
    timesteps = args.timesteps
    custom_duration = args.custom_duration
    # If just generated SatSim from TLEs, or using existing adjacency txt, don't override or prompt
    if not generated_via_tle and not using_existing_adjacency_txt:
        if timesteps is None and custom_duration is None:
            timesteps, custom_duration = _interactive_options()
    else:
        timesteps = None
        custom_duration = None

    rc = run_workflow(project_root, sat_sim_file, timesteps, custom_duration, quiet=args.quiet, interactive_fl_output=args.interactive_fl_output)
    if rc != 0:
        print(f"Workflow exited with code {rc}")
        sys.exit(rc)

    synth_dir = project_root / "synth_FLAMs"
    latest_flam = find_latest(synth_dir, "flam_*.csv")

    results_root = project_root / "federated_learning" / "results_from_output"
    latest_run = None
    if results_root.exists():
        runs = [p for p in results_root.iterdir() if p.is_dir()]
        latest_run = sorted(runs, key=lambda p: p.stat().st_ctime, reverse=True)[0] if runs else None

    print("\n=== Outputs ===")
    if latest_flam is not None:
        print(f"Latest FLAM: {latest_flam}")
    else:
        print("Latest FLAM: <not found>")

    if latest_run is not None:
        print(f"Results dir: {latest_run}")
        dash = latest_run / "dashboard.html"
        metrics = find_latest(latest_run, "metrics_*.json")
        model = find_latest(latest_run, "model_*.pt")
        log = find_latest(latest_run, "results_*.log")
        if dash.exists():
            print(f"Dashboard: {dash}")
        if metrics is not None:
            print(f"Metrics JSON: {metrics}")
        if model is not None:
            print(f"Model Weights: {model}")
        if log is not None:
            print(f"Log: {log}")
    else:
        print("Results dir: <not found>")


if __name__ == "__main__":
    main()


