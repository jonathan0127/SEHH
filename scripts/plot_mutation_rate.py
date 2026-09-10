#!/usr/bin/env python3

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if os.path.exists(os.path.join(PROJECT_ROOT, "Makefile")):
    os.chdir(PROJECT_ROOT)

sys.path.insert(0, SCRIPT_DIR)
from plot import plot_mutation_rate

if __name__ == "__main__":
    # 設定路徑
    mutation_rate_folder = "result/output"
    image_folder = "result/images/mutation_rate"
    
    print(f"Looking for mutation rate files in: {mutation_rate_folder}")
    print(f"Output images will be saved to: {image_folder}")
    
    # 繪製 mutation rate 圖
    plot_mutation_rate(mutation_rate_folder, image_folder)