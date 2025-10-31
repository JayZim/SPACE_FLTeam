import os
import sys
import webbrowser
from pathlib import Path

# ensure project root is on sys.path so package imports resolve when running file directly
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from federated_learning.fl_output import FLOutput

def _ensure_animation(metrics_path, kind, forced_name=None):
    """
    Ensure the needed animation GIF exists for a metrics file.
    kind: "accuracy" or "participation"
    returns absolute path to the gif
    """
    metrics_path = os.path.abspath(metrics_path)
    if not os.path.isfile(metrics_path):
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    out_dir = os.path.dirname(metrics_path)
    if forced_name:
        gif_path = os.path.join(out_dir, forced_name)
    else:
        gif_path = os.path.join(out_dir, "accuracy_progress.gif" if kind == "accuracy" else "client_participation.gif")

    # If gif already exists, keep it. Otherwise try to (re)generate
    if not os.path.isfile(gif_path):
        try:
            if kind == "accuracy":
                FLOutput.animate_accuracy_progress(metrics_path, save_path=gif_path)
            else:
                FLOutput.animate_client_participation(metrics_path, save_path=gif_path)
        except Exception as e:
            raise RuntimeError(f"Failed to generate {kind} animation for {metrics_path}: {e}")
    return os.path.abspath(gif_path)

def _file_uri(path):
    p = Path(path).absolute()
    return p.as_uri()

def run_dashboard_creator():
    """
    Prompt user for two metrics JSON files (FedAvg, FLOMPS), ensure animations exist,
    then generate and open a simple HTML dashboard with side-by-side animations.
    """
    print("\n=== FL Comparison Dashboard Creator ===")
    fedavg_metrics = input("Enter path to FedAvg metrics JSON file:\n").strip()
    if not os.path.isfile(fedavg_metrics):
        print(f"Error: File not found: {fedavg_metrics}")
        return

    flomps_metrics = input("Enter path to FLOMPS metrics JSON file (or press Enter to use last run):\n").strip()
    if flomps_metrics == "":
        print("No FLOMPS path provided; aborting dashboard creation.")
        return
    if not os.path.isfile(flomps_metrics):
        print(f"Error: File not found: {flomps_metrics}")
        return

    try:
        print("Ensuring animations for FedAvg...")
        fed_acc = _ensure_animation(fedavg_metrics, "accuracy", forced_name="accuracy_fedavg.gif")
        fed_part = _ensure_animation(fedavg_metrics, "participation", forced_name="participation_fedavg.gif")

        print("Ensuring animations for FLOMPS...")
        flomps_acc = _ensure_animation(flomps_metrics, "accuracy", forced_name="accuracy_flomps.gif")
        flomps_part = _ensure_animation(flomps_metrics, "participation", forced_name="participation_flomps.gif")
    except Exception as e:
        print(f"Error generating/locating animations: {e}")
        return

    # Build simple HTML side-by-side dashboard
    out_html_dir = os.path.dirname(os.path.abspath(flomps_metrics))
    dashboard_path = os.path.join(out_html_dir, "fl_comparison_dashboard.html")

    html = f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8"/>
      <title>Federated Learning Comparison Dashboard</title>
      <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .container {{ display: flex; gap: 24px; align-items: flex-start; }}
        .col {{ flex: 1; min-width: 300px; }}
        .card {{ border: 1px solid #ccc; padding: 12px; border-radius: 6px; }}
        .title {{ font-weight: bold; margin-bottom: 8px; }}
        img {{ max-width: 100%; height: auto; display: block; margin-bottom: 10px; }}
        .label {{ color: #555; font-size: 0.9em; margin-bottom: 6px; }}
      </style>
    </head>
    <body>
      <h1>Federated Learning Comparison Dashboard</h1>
      <p>Left column: FedAvg &nbsp;&nbsp; | &nbsp;&nbsp; Right column: FLOMPS</p>
      <div class="container">
        <div class="col card">
          <div class="title">FedAvg — Accuracy Progress</div>
          <img src="{_file_uri(fed_acc)}" alt="FedAvg Accuracy">
          <div class="title">FedAvg — Client Participation</div>
          <img src="{_file_uri(fed_part)}" alt="FedAvg Participation">
        </div>
        <div class="col card">
          <div class="title">FLOMPS — Accuracy Progress</div>
          <img src="{_file_uri(flomps_acc)}" alt="FLOMPS Accuracy">
          <div class="title">FLOMPS — Client Participation</div>
          <img src="{_file_uri(flomps_part)}" alt="FLOMPS Participation">
        </div>
      </div>
      <p>Dashboard generated at: {dashboard_path}</p>
    </body>
    </html>
    """

    try:
        with open(dashboard_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"Dashboard written to: {dashboard_path}")
        webbrowser.open(Path(dashboard_path).absolute().as_uri())
    except Exception as e:
        print(f"Failed to write/open dashboard: {e}")

if __name__ == "__main__":
    run_dashboard_creator()