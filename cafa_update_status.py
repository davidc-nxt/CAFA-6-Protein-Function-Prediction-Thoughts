#!/usr/bin/env python3
"""
Project Status Auto-Updater.
Scans logs, submissions, and processes to report current state.
"""

import os
import subprocess
import glob

def get_kaggle_scores():
    try:
        cmd = "venv/bin/kaggle competitions submissions -c cafa-6-protein-function-prediction -v | head -10"
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return res.stdout
    except:
        return "Kaggle API unreachable"

def get_running_processes():
    cmd = "ps aux | grep cafa | grep -v grep | grep -v update_status"
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return res.stdout.strip()

def get_latest_submission_files():
    files = glob.glob("submission_v*.tsv")
    files.sort(key=os.path.getmtime, reverse=True)
    return files[:5]

def main():
    print("# CAFA-6 Project Status Update")
    print("\n## 🏃 Active Processes")
    procs = get_running_processes()
    if procs:
        print("```bash")
        print(procs)
        print("```")
    else:
        print("No background jobs running.")
        
    print("\n## 📊 Recent Submissions")
    print(get_kaggle_scores())
    
    print("\n## 📁 Latest Files")
    for f in get_latest_submission_files():
        print(f"- {f}")

if __name__ == "__main__":
    main()
