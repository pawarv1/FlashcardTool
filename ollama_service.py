import subprocess
import time
import atexit
import urllib.request
import os

_ollama_process = None

def is_ollama_running(host="http://localhost:11434") -> bool:
    """Checks if the Ollama server is already running and responding."""
    try:
        req = urllib.request.urlopen(f"{host}/api/tags", timeout=2)
        return req.getcode() == 200
    except Exception:
        return False

def start_ollama():
    """Starts 'ollama serve' if it isn't already running."""
    global _ollama_process

    if is_ollama_running():
        print("🟢 Ollama service is already active.")
        return

    print("🚀 Starting background Ollama server...")
    try:
        _ollama_process = subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid if hasattr(os, "setsid") else None
        )
        
        # Poll up to 10 seconds
        for _ in range(10):
            if is_ollama_running():
                print("✅ Ollama server started successfully.")
                break
            time.sleep(1)
        else:
            print("⚠️ Ollama started but isn't responding yet on port 11434.")
            
    except Exception as e:
        print(f"❌ Failed to start Ollama server: {e}")

def stop_ollama():
    """Stops the Ollama server process started by this script."""
    global _ollama_process
    if _ollama_process is not None:
        print("\n🛑 Shutting down Ollama server process...")
        try:
            _ollama_process.terminate()
            _ollama_process.wait(timeout=3)
        except Exception:
            _ollama_process.kill()
        _ollama_process = None
        print("✅ Ollama server stopped.")

# Automatically register stop_ollama to run on Python exit
atexit.register(stop_ollama)