import sys
from sys import argv
import os
import re
from typing import Optional
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if os.path.exists(os.path.join(PROJECT_ROOT, "Makefile")):
    os.chdir(PROJECT_ROOT)

sys.path.insert(0, SCRIPT_DIR)
from plot import plot_convergence, plot_ranking, plot_csv, plot_md, plot_wrst, plot_mutation_rate

RECORDS_DIR = "result/records"
IMAGES_DIR = "result/images"
CSV_DIR = "result/csv"
MD_DIR = "result/md"
ERROR_DIR = "result/error"

PARA_SETTING_PATH = os.path.join("src", "config", "para_setting.h")

def clean_workspace():
    # 移除主要生成檔案
    for fname in ["in.txt", "bin/main", "bin/train_SEHH", "bin/export_config", "main", "train", "train_SEHH", "hyper_result", "forpy", "export_config"]:
        if os.path.exists(fname):
            os.remove(fname)

    # 清理 result/records 下所有 .txt 檔
    records_dir = "result/records"
    if os.path.exists(records_dir):
        for file in os.listdir(records_dir):
            if file.endswith(".txt"):
                os.remove(os.path.join(records_dir, file))
    
    official_dir = "result/2022official"
    if os.path.exists(official_dir):
        for file in os.listdir(official_dir):
            if file.endswith(".txt"):
                os.remove(os.path.join(official_dir, file))
    
    official_dir = "result/2024official"
    if os.path.exists(official_dir):
        for file in os.listdir(official_dir):
            if file.endswith(".txt"):
                os.remove(os.path.join(official_dir, file))

    # 清理 result/images/convergence 下所有 .png 檔
    images_con_dir = "result/images/convergence"
    if os.path.exists(images_con_dir):
        for file in os.listdir(images_con_dir):
            if file.endswith(".png"):
                os.remove(os.path.join(images_con_dir, file))

    error_dir = "result/error"
    if os.path.exists(error_dir):
        for file in os.listdir(error_dir):
            if file.endswith(".csv"):
                os.remove(os.path.join(error_dir, file))

    images_con_dir = "result/images/mutation_rate"
    if os.path.exists(images_con_dir):
        for file in os.listdir(images_con_dir):
            if file.endswith(".png"):
                os.remove(os.path.join(images_con_dir, file))

    print("Workspace cleaned.")

def setup_directories():
    directories = ["records", "images", "output", "2022official", "2024official", "error"]
    for directory in directories:
        dir_path = "result/" + directory
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"Directory '{dir_path}' created.")
    
    image_dirs = ["convergence", "ranking", "mutation_rate"]
    for img_dir in image_dirs:
        dir_path = os.path.join(IMAGES_DIR, img_dir)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"Directory '{dir_path}' created.")


def read_algorithms(file_path) -> tuple[list, int, dict, str, int, Optional[str]]:
    algorithms = []
    func_num = -1
    functions = {}
    base_algorithm_marker: Optional[str] = None

    read_instance = False

    with open(file_path, "r") as file:
        
        for line in file.readlines():
            line = line.strip()

            if read_instance:
                index, name, dim = line.split(',', 2)
                functions[int(index)] = (name, int(dim))

            elif line[0].isdigit():
                read_instance = True
                func_num, benchmark_name, dim = line.split(',', 2)

            else:
                if line.startswith("#BASE_ALGORITHM"):
                    parts = line.split(maxsplit=1)
                    if len(parts) == 2:
                        base_algorithm_marker = parts[1]
                    continue
                algorithms.append(line)
    
    base_algorithm = base_algorithm_marker

    return algorithms, int(func_num), functions, benchmark_name, int(dim), base_algorithm
    

if __name__ == "__main__":

    run_optimization = True
    setup_directories()

    if len(argv) > 1:
        for arg in argv[1:]:
            if arg in ("-c", "clean", 'clear'):
                clean_workspace()
                exit()
            if arg in ('-p'):
                run_optimization = False
                print("Skipping optimization run, only plotting results...")
                break
        else:
            raise ValueError("Invalid argument.\nUse '-c' to clean the workspace.\nUse '-p' to do plotting only.")

    if os.name == 'nt':
        subprocess.run(["mingw32-make", 'main_cpp'], check=True)
    else:
        subprocess.run(["make", 'main_cpp'], check=True)
    
    temp_file = "in.txt"
    export_bin = "./bin/export_config" if os.path.exists("./bin/export_config") else "./export_config"
    os.system(f"{export_bin} > {temp_file}")

    algorithms, func_num, functions, benchmark_name, dim, base_algorithm = read_algorithms(temp_file)
    print("Base Algorithm:", base_algorithm)
    algo_num = len(algorithms)
    
    # 生成標準算法的輸入
    if run_optimization:
        with open(temp_file, "w") as file:
            total_tests = func_num * algo_num
            file.write(f"{total_tests}\n")
            
            # 標準算法
            for i in range(1, func_num + 1):
                for j in range(1, algo_num + 1):
                    file.write(f"{i} {j}\n")

        print("Running standard optimization algorithms...")
        main_bin = "./bin/main" if os.path.exists("./bin/main") else "./main"
        os.system(f"{main_bin} < {temp_file}")
        
    # 繪製包括所有算法的結果
    print("Plotting all results...")
    plot_convergence(RECORDS_DIR, IMAGES_DIR + "/convergence", algorithms, functions)
    plot_ranking(RECORDS_DIR, IMAGES_DIR + "/ranking", algorithms, functions, benchmark_name)
    plot_csv(RECORDS_DIR, CSV_DIR, algorithms, functions, benchmark_name)
    plot_md(CSV_DIR, MD_DIR)
    plot_wrst(ERROR_DIR, IMAGES_DIR + "/wrst", base_algorithm, algorithms, functions, benchmark_name)
    
    # 繪製 mutation rate 收斂圖
    mutation_rate_folder = "result/output"
    mutation_rate_image_folder = IMAGES_DIR + "/mutation_rate"
    plot_mutation_rate(mutation_rate_folder, mutation_rate_image_folder)