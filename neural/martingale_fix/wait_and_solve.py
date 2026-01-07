
import time
import os
import subprocess
import sys

def main():
    target_file = sys.argv[1]
    cmd = sys.argv[2:]
    
    print(f"Waiting for {target_file}...")
    while not os.path.exists(target_file):
        time.sleep(60)
        print(f"Waiting for {target_file}...")
        
    print(f"File found! Waiting 60s for write completion stability...")
    time.sleep(60) 
    
    print(f"Launching: {' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
