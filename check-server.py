#!/usr/bin/env python3
"""Diagnose why the web server might not start."""
import sys
from pathlib import Path

# Add project root
root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "src"))

print("Python:", sys.executable)
print("Version:", sys.version)
print("Project root:", root)
print()

# Check .env
env_file = root / ".env"
print(".env exists:", env_file.exists())
print()

# Try importing
print("Importing web app...")
try:
    from web.app import app
    print("  OK - app imported")
except Exception as e:
    print("  FAILED:", e)
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Try uvicorn
print()
print("Starting uvicorn (will run until you press Ctrl+C)...")
import uvicorn
uvicorn.run(app, host="0.0.0.0", port=8000)
