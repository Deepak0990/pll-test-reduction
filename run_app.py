import subprocess
import sys
import time

def main():
    print("Starting FastAPI Backend...")
    # Start uvicorn in the background
    backend_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.main:app", "--reload"]
    )

    # Give the backend a couple of seconds to boot up
    time.sleep(3)

    print("Starting Streamlit Frontend...")
    # Start streamlit
    frontend_process = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py"]
    )

    try:
        # Keep the script running while both services are active
        backend_process.wait()
        frontend_process.wait()
    except KeyboardInterrupt:
        print("\nStopping all services...")
        backend_process.terminate()
        frontend_process.terminate()

if __name__ == "__main__":
    main()