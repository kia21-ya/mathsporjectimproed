import os
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENGINE_DIR = ROOT / "python-engine"

os.chdir(ENGINE_DIR)

port = os.environ.get("PORT", "8000")

subprocess.run(
    [
        sys.executable,
        "-m",
        "uvicorn",
        "main:app",
        "--host",
        "0.0.0.0",
        "--port",
        port,
    ],
    check=True,
)