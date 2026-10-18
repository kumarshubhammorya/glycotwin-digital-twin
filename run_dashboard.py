"""
Launch the GlycoTwin Streamlit doctor dashboard.

    python run_dashboard.py            # http://localhost:8501
    python run_dashboard.py 8601       # another port

Works from any working directory (paths resolve from this file). Equivalent to
`streamlit run dashboard/app.py` from the repository root, plus the teal theme.
Synthetic data only. Research prototype, not a medical device.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
APP = ROOT / "dashboard" / "app.py"

REQUIRED = {
    "data/synthetic_ehr.csv": "python data/generate_datasets.py",
    "data/synthetic_wearables.csv": "python data/generate_datasets.py",
    "models/glycotwin_model.joblib": "python models/train_fusion_model.py",
    "models/metrics.json": "python models/evaluate.py",
}


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else "8501"
    missing = [(p, cmd) for p, cmd in REQUIRED.items() if not (ROOT / p).exists()]
    for p, cmd in missing:
        print(f"[warning] {p} is missing. Create it with: {cmd}")

    from streamlit.web import cli as stcli

    sys.argv = [
        "streamlit", "run", str(APP),
        "--server.port", str(port),
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
        "--theme.primaryColor", "#0D9488",
    ]
    print(f"Launching GlycoTwin dashboard on http://localhost:{port} ...")
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
