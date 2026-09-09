import os
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENGINE = ROOT / "python-engine"
REQ = ENGINE / "requirements.txt"

def main():
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(REQ)],
        check=True,
    )
    os.chdir(ENGINE)
    subprocess.run(
        [sys.executable, "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"],
        check=True,
    )

if __name__ == "__main__":
    main()