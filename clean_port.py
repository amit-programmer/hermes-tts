import sys
import subprocess

def free_port(port: int = 8006):
    """Cleanly terminates any lingering Windows process holding the specified port."""
    try:
        output = subprocess.check_output(f'netstat -ano | findstr ":{port}"', shell=True).decode()
        pids = set()
        for line in output.strip().splitlines():
            parts = line.split()
            if len(parts) >= 5:
                # Check for LISTENING state
                state = parts[3].upper() if len(parts) > 3 else ""
                pid = parts[-1]
                if pid.isdigit() and int(pid) > 0 and (state == "LISTENING" or ":8006" in parts[1]):
                    pids.add(int(pid))
        
        if not pids:
            print(f"[PORT-CLEANUP] Port {port} is clean and free.")
            return

        for pid in pids:
            print(f"[PORT-CLEANUP] Terminating process {pid} on port {port}...")
            subprocess.run(f"taskkill /F /PID {pid}", shell=True, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"[PORT-CLEANUP] Process {pid} killed successfully.")
            
    except subprocess.CalledProcessError:
        print(f"[PORT-CLEANUP] Port {port} is completely free.")

if __name__ == "__main__":
    target_port = int(sys.argv[1]) if len(sys.argv) > 1 else 8006
    free_port(target_port)
