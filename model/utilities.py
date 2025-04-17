import numpy as np
import pandas as pd
import os
import glob

folder_path = {"./data/20380357","./data/221609641"}
all_files = []

for path in folder_paths:
    try:
        files = os.listdir(path)
        all_files.extend([os.path.join(path, file) for file in files if file.endswith('.csv')])
    except FileNotFoundError:
        print(f"Folder {path} not found. Skipping.")

print(all_files)

