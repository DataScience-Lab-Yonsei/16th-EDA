from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
commands = [
    [sys.executable, str(ROOT / "src" / "run_pipeline.py"), "--config", str(ROOT / "config.yaml")],
    [sys.executable, str(ROOT / "src" / "compare_directional_no_availability.py")],
    [sys.executable, str(ROOT / "src" / "compare_absolute_no_availability.py")],
    [sys.executable, str(ROOT / "src" / "compare_three_sensitivity_models.py")],
    [sys.executable, str(ROOT / "src" / "generate_submission_comparison.py")],
]

for cmd in commands:
    result = subprocess.call(cmd, cwd=ROOT)
    if result:
        raise SystemExit(result)

raise SystemExit(0)
