import os
import subprocess
import sys

if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    local_venv_python = os.path.join(base_dir, "venv", "Scripts", "python.exe")
    ui_script = os.path.join(os.path.dirname(__file__), "ui", "app.py")

    # If the project-local virtual environment exists and we're not already
    # using it, relaunch with that interpreter to avoid dependency/version drift.
    if os.path.exists(local_venv_python) and os.path.abspath(sys.executable) != os.path.abspath(local_venv_python):
        print("Switching to project virtual environment...")
        subprocess.run([local_venv_python, __file__], cwd=base_dir)
        sys.exit(0)

    print("Starting Streamlit Application...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", ui_script], cwd=base_dir)
