"""config.yaml의 색상·폰트를 바꾼 뒤 전체 결과를 안전하게 다시 만드는 진입점."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
raise SystemExit(subprocess.call([sys.executable, str(ROOT / "run_all.py")], cwd=ROOT))
