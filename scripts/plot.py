import csv
import math
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from typing import Optional
from tqdm import tqdm
from scipy.stats import wilcoxon

def plot_convergence(result_folder, image_folder, algorithms: list[str], functions: dict):

    missing_files = []
    
    print(f"Ploting Functions...")
    for func_num, function in tqdm(functions.items()):
        
        func_name, dim = function
        avg_fig, (avg_solution_ax, avg_hyper_ax) = plt.subplots(2, 1, figsize=(10, 8), sharex=True , height_ratios=[3, 1])
        best_fig, (best_solution_ax, best_hyper_ax) = plt.subplots(2, 1, figsize=(10, 8), sharex=True , height_ratios=[3, 1])

        # get func minima within all algorithms
        avg_global_min = None
        best_global_min = None
        for algo in algorithms:
            file_name = f"{func_num}_{dim}_{algo}.txt"
            file_path = os.path.join(result_folder, file_name)
            if not os.path.exists(file_path):
                continue
            with open(file_path, 'r') as f:
                lines = f.readlines()[1:]  # 跳過第一行
                for line in lines:
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            avg_solution, best_solution = map(float, parts[1:])
                            if (avg_global_min is None) or (avg_solution < avg_global_min):
                                avg_global_min = avg_solution
                            if (best_global_min is None) or (best_solution < best_global_min):
                                best_global_min = best_solution
                        except ValueError:
                            continue
        
        iterations = []
        for algo in algorithms:
            file_name = f"{func_num}_{dim}_{algo}.txt"
            
            # print("reading file:", file_name)
            file_path = os.path.join(result_folder, file_name)
            
            # 檢查文件是否存在
            if not os.path.exists(file_path):
                missing_files.append(file_name)
                continue
                
            iterations = []
            avg_solutions = []
            best_solutions = []
            
            # 繪製收斂圖
            with open(file_path, 'r') as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            iteration, avg_solution, best_solution = map(float, parts[:3])
                            iterations.append(iteration)
                            avg_solutions.append(avg_solution)
                            best_solutions.append(best_solution)
                        except ValueError:
                            print(f"無法解析檔案 {file_name} 中的行: {line}")

                # 跳過第一行
                iterations = iterations[1:]
                avg_solutions = avg_solutions[1:]
                best_solutions = best_solutions[1:]

                if avg_solutions:  # 確保 avg_solutions 不為空
                    avg_solutions = [np.log10(s - avg_global_min + 1e-6) for s in avg_solutions]
                if best_solutions:  # 確保 best_solutions 不為空
                    best_solutions = [np.log10(s - best_global_min + 1e-6) for s in best_solutions]

                # 為超啟發式算法使用特殊標記
                if "SEHH" in algo or "Hybrid-SE" in algo:
                    marker = 'D'
                    markersize = 5
                    linestyle = '--'
                elif algo == "LSHADE":
                    linestyle = '--'

                avg_solution_ax.plot(iterations, avg_solutions, label=algo, marker=marker, markersize=markersize, linestyle=linestyle)
                best_solution_ax.plot(iterations, best_solutions, label=algo, marker=marker, markersize=markersize, linestyle=linestyle)

        # 繪製超啟發式算法的序列
        if iterations:
            for algo in algorithms[::-1]:

                if not (algo.startswith('Hybrid') or algo.startswith('SEHH')):
                    continue
                    
                sequence_num = algo.split('-')[-1]
                seq_file = f'result/hyper_sequences/SEHH_sequences{sequence_num}.txt'
                if not os.path.exists(seq_file):
                    hyper_method = algo.split('-')[-2] if len(algo.split('-')) >= 2 else "SE"
                    seq_file = f'result/hyper_sequences/hyper{hyper_method}_sequences{sequence_num}.txt'
                if not os.path.exists(seq_file):
                    continue

                with open(seq_file, 'r') as f_hyper:
                    max_iter = max(iterations)
                    methods = []
                    time_weights = []

                    # 讀取每行，尋找支援的元啟發式方法
                    lines = f_hyper.readlines()
                    i = 0
                    while i < len(lines):
                        if lines[i].strip() in ['GA', 'DE', 'PSO', 'LSHADE', 'LSRTDE', 'RDE']:
                            method = lines[i].strip()
                            # 跳過參數名稱行
                            i += 2
                            if i < len(lines):
                                params = lines[i].strip().split()
                                time_weight = float(params[-1])  # timeWeight 總是最後一個參數
                                
                                # 如果是 DE，根據 strategy 參數確定具體策略
                                if method == 'DE':

                                    de_strategies = {
                                        1: "DE_r1",
                                        2: "DE_r2", 
                                        3: "DE_b1",
                                        4: "DE_b2"
                                    }

                                    strategy = int(float(params[2]))  # strategy 是第三個參數
                                    method = de_strategies[strategy]
                                
                                methods.append(method)
                                time_weights.append(time_weight)
                        i += 1

                # 設定 tab10 色彩
                tab10 = plt.cm.tab10.colors
                algo_enum = {algo : i for i, algo in enumerate(["PSO", "GA", "DE_r1", "DE_b1", "DE_r2", "DE_b2", "LSHADE", "LSRTDE", "RDE"])}

                # 計算時間權重
                if hyper_method == 'SE':
                    time_weights = np.ones(len(methods))  # 每個方法的時間權重相等
                time_weights = np.array(time_weights) / sum(time_weights) * max_iter
                cumulative_weights = np.cumsum(np.concatenate((np.array([0]), time_weights)))

                # 繪製超啟發式序列
                for i, (method, weight) in enumerate(zip(methods, time_weights)):
                    start = cumulative_weights[i]

                    def plot_bar(ax):
                        ax.barh(f'Hybrid({hyper_method}) {sequence_num}', weight, color = tab10[algo_enum[method]], left=start, height=0.5)
                        ax.text(
                            start + weight / 2,                                 # x 位置：長條圖中央
                            f'Hybrid({hyper_method}) {sequence_num}',           # y 位置
                            method,                                             # 顯示文字
                            va='center', ha='center', fontsize=8, color='black', fontweight='bold'
                        )

                    plot_bar(avg_hyper_ax)
                    plot_bar(best_hyper_ax)
        
        if not iterations:
            plt.close(avg_fig)
            plt.close(best_fig)
            continue

        # avg_ax 設定
        avg_solution_ax.set_title(f"{func_name} (Dim: {dim}) Avg")
        avg_hyper_ax.set_xlabel("Evaluations")
        avg_solution_ax.set_ylabel("Solution (log10)")
        avg_hyper_ax.set_ylabel("Hyper Heuristic Sequence")
        avg_hyper_ax.grid(True)
        avg_solution_ax.grid(True)
        avg_solution_ax.legend()

        avg_fig.tight_layout()
        avg_fig.savefig(os.path.join(image_folder, f"{func_num}_{dim}_avg.png"), dpi=300, bbox_inches='tight')

        # best_ax 設定
        best_solution_ax.set_title(f"{func_name} (Dim: {dim}) Best")
        best_hyper_ax.set_xlabel("Evaluations")
        best_solution_ax.set_ylabel("Solution (log10)")
        best_hyper_ax.set_ylabel("Hyper Heuristic Sequence")
        best_hyper_ax.grid(True)
        best_solution_ax.grid(True)
        best_solution_ax.legend()
        
        best_fig.tight_layout()
        best_fig.savefig(os.path.join(image_folder, f"{func_num}_{dim}_best.png"), dpi=300, bbox_inches='tight')

        plt.close()

    if missing_files:
        for file in set(missing_files):
            print(f"不存在的文件: {file}")

    print(f"Ploting Convergence Complete!")

def plot_ranking(result_folder, image_folder, algorithms: list[str], functions: dict, benchmark_name: str):
    print("Calculating Rankings...")

    avg_rankings = {algo: [] for algo in algorithms}
    best_rankings = {algo: [] for algo in algorithms}
    
    for func_num, function in tqdm(functions.items()):

        func_name, dim = function
        avg_all_evals = []
        best_all_evals = []

        for algo in algorithms:
            file_name = f"{func_num}_{dim}_{algo}.txt"
            file_path = os.path.join(result_folder, file_name)
            if not os.path.exists(file_path):
                continue

            avg_evalutations = []
            best_evalutations = []
            
            with open(file_path, 'r') as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            iteration, avg_fitness, best_finess = map(float, parts[:3])
                            avg_evalutations.append(avg_fitness)
                            best_evalutations.append(best_finess)
                        except ValueError:
                            print(f"無法解析檔案 {file_name} 中的行: {line}")

            avg_all_evals.append((algo, avg_evalutations[::-1]))
            best_all_evals.append((algo, best_evalutations[::-1]))

        avg_sorted_evals = sorted(avg_all_evals, key=lambda x: x[1])
        best_sorted_evals = sorted(best_all_evals, key=lambda x: x[1])

        for rank in range(len(avg_sorted_evals)):
            algo, _ = avg_sorted_evals[rank]
            avg_rankings[algo].append(rank + 1)

        for rank in range(len(best_sorted_evals)):
            algo, _ = best_sorted_evals[rank]
            best_rankings[algo].append(rank + 1)

    # 設定 tab20 色彩
    tab20 = plt.cm.tab20.colors
    algo_enum = {algo : i for i, algo in enumerate(algorithms)}

    # Avg 的平均排名
    avg_average_rankings = {algo: np.mean(ranks) for algo, ranks in avg_rankings.items()}
    plt.figure(figsize=(12, 6))
    sorted_rankings = sorted(avg_average_rankings.items(), key=lambda x: x[1])
    for algo, avg_rank in sorted_rankings:
        bar = plt.bar(algo, avg_rank, color=tab20[algo_enum[algo]], label=algo)
        plt.bar_label(bar, fmt='%.2f')
    plt.title(f"Average Rankings on {benchmark_name} Avg")
    plt.ylabel("Average Rank")
    plt.ylim(0, len(algorithms))
    plt.tight_layout()
    plt.legend(loc = 'upper left', ncol=2)
    plt.savefig(os.path.join(image_folder, f"{benchmark_name}_avg.png"), dpi=300)
    
    # Avg 的平均排名
    best_average_rankings = {algo: np.mean(ranks) for algo, ranks in best_rankings.items()}
    plt.figure(figsize=(12, 6))
    sorted_rankings = sorted(best_average_rankings.items(), key=lambda x: x[1])
    for algo, avg_rank in sorted_rankings:
        bar = plt.bar(algo, avg_rank, color=tab20[algo_enum[algo]], label=algo)
        plt.bar_label(bar, fmt='%.2f')
    plt.title(f"Average Rankings on {benchmark_name} Best")
    plt.ylabel("Average Rank")
    plt.ylim(0, len(algorithms))
    plt.tight_layout()
    plt.legend(loc = 'upper left', ncol=2)
    plt.savefig(os.path.join(image_folder, f"{benchmark_name}_best.png"), dpi=300)

    print("Ploting Rankings Complete!")
    
def plot_csv(result_folder, csv_folder, algorithms: list[str], functions: dict, benchmark_name: str):
    os.makedirs(csv_folder, exist_ok=True)

    function_ids = sorted(functions.keys())
    headers = ["Algorithm"] + [f"F{fid}" for fid in function_ids]

    avg_data = {algo: {fid: None for fid in function_ids} for algo in algorithms}
    best_data = {algo: {fid: None for fid in function_ids} for algo in algorithms}

    missing_files = []

    for fid in function_ids:
        _, dim = functions[fid]
        for algo in algorithms:
            file_name = f"{fid}_{dim}_{algo}.txt"
            file_path = os.path.join(result_folder, file_name)

            if not os.path.exists(file_path):
                missing_files.append(file_name)
                continue

            final_iteration = None
            final_avg = None
            final_best = None

            with open(file_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 3:
                        continue

                    try:
                        iteration, avg_val, best_val = map(float, parts[:3])
                    except ValueError:
                        continue

                    final_iteration = iteration
                    final_avg = avg_val
                    final_best = best_val

            if final_iteration is None:
                continue

            avg_data[algo][fid] = final_avg
            best_data[algo][fid] = final_best

    if missing_files:
        print("Missing record files for CSV:")
        for file in sorted(set(missing_files)):
            print(f"  {file}")

    def write_csv(file_path: str, data: dict):
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            for algo in algorithms:
                row = [algo]
                for fid in function_ids:
                    value = data[algo][fid]
                    if value is None:
                        row.append('')
                    else:
                        row.append(f"{value:.10g}")
                writer.writerow(row)

    avg_csv_path = os.path.join(csv_folder, f"{benchmark_name}_avg.csv")
    best_csv_path = os.path.join(csv_folder, f"{benchmark_name}_best.csv")

    write_csv(avg_csv_path, avg_data)
    write_csv(best_csv_path, best_data)

    print(f"CSV files generated at {csv_folder}")

def plot_md(csv_folder, md_folder):
    os.makedirs(md_folder, exist_ok=True)

    if not os.path.exists(csv_folder):
        print(f"CSV folder '{csv_folder}' does not exist, skip markdown export.")
        return

    csv_files = sorted(f for f in os.listdir(csv_folder) if f.endswith('.csv'))

    if not csv_files:
        print(f"No CSV files found in '{csv_folder}', skip markdown export.")
        return

    for csv_file in csv_files:
        csv_path = os.path.join(csv_folder, csv_file)

        with open(csv_path, 'r', newline='') as f:
            reader = list(csv.reader(f))

        if len(reader) < 2:
            continue

        headers = reader[0]
        data_rows = reader[1:]

        if len(headers) < 2 or not data_rows:
            continue

        col_best_value: dict[int, float] = {}
        col_best_rows: dict[int, list[int]] = {}

        for row_idx, row in enumerate(data_rows):
            for col_idx in range(1, len(headers)):
                if col_idx >= len(row):
                    continue

                cell = row[col_idx].strip()
                if cell == '':
                    continue

                try:
                    value = float(cell)
                except ValueError:
                    continue

                best_val = col_best_value.get(col_idx)
                if best_val is None or value < best_val - 1e-12:
                    col_best_value[col_idx] = value
                    col_best_rows[col_idx] = [row_idx]
                elif math.isclose(value, best_val, rel_tol=1e-9, abs_tol=1e-12):
                    col_best_rows.setdefault(col_idx, []).append(row_idx)

        md_lines: list[str] = []
        title = os.path.splitext(csv_file)[0]
        md_lines.append(f"# {title}")
        md_lines.append('')

        md_lines.append('| ' + ' | '.join(headers) + ' |')
        md_lines.append('| ' + ' | '.join(['---'] * len(headers)) + ' |')

        for row_idx, row in enumerate(data_rows):
            cells = []
            for col_idx in range(len(headers)):
                cell = row[col_idx].strip() if col_idx < len(row) else ''
                if col_idx >= 1 and cell:
                    best_rows = col_best_rows.get(col_idx, [])
                    if row_idx in best_rows:
                        cell = f"**{cell}**"
                cells.append(cell)
            md_lines.append('| ' + ' | '.join(cells) + ' |')

        md_path = os.path.join(md_folder, f"{title}.md")
        with open(md_path, 'w', encoding='utf-8') as md_file:
            md_file.write('\n'.join(md_lines) + '\n')

        print(f"Markdown file generated: {md_path}")

def plot_wrst(error_folder, image_folder, base_algorithm: Optional[str], algorithms: list[str], functions: dict, benchmark_name: str, alpha: float = 0.05):
    if not base_algorithm:
        print("No base algorithm specified; skipping Wilcoxon analysis.")
        return

    if not os.path.exists(error_folder):
        print(f"Error folder '{error_folder}' does not exist; skipping Wilcoxon analysis.")
        return

    comparison_algorithms = [algo for algo in algorithms if algo != base_algorithm]
    if not comparison_algorithms:
        print("No algorithms available for comparison with base algorithm; skipping Wilcoxon analysis.")
        return

    os.makedirs(image_folder, exist_ok=True)

    function_ids = sorted(functions.keys())

    def load_errors(file_path: str) -> list[float]:
        if not os.path.exists(file_path):
            return []
        values: list[float] = []
        with open(file_path, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader, None)  # skip header
            for row in reader:
                if len(row) < 2:
                    continue
                cell = row[1].strip()
                if not cell:
                    continue
                try:
                    values.append(float(cell))
                except ValueError:
                    continue
        return values

    summary: dict[str, dict[str, int]] = {algo: {"Better": 0, "No Difference": 0, "Worse": 0} for algo in comparison_algorithms}

    for fid in function_ids:
        _, dim = functions[fid]
        base_file = os.path.join(error_folder, f"{fid}_{dim}_{base_algorithm}_error.csv")
        base_errors = load_errors(base_file)
        if not base_errors:
            print(f"Base error file missing or empty: {base_file}")
            continue

        for algo in comparison_algorithms:
            other_file = os.path.join(error_folder, f"{fid}_{dim}_{algo}_error.csv")
            other_errors = load_errors(other_file)
            if not other_errors:
                print(f"Comparison error file missing or empty: {other_file}")
                continue

            paired_length = min(len(base_errors), len(other_errors))
            if paired_length < 1:
                summary[algo]["No Difference"] += 1
                continue

            base_arr = np.array(base_errors[:paired_length])
            other_arr = np.array(other_errors[:paired_length])

            diff = other_arr - base_arr
            if np.allclose(diff, 0):
                summary[algo]["No Difference"] += 1
                continue

            try:
                _, pvalue = wilcoxon(base_arr, other_arr, zero_method='wilcox', alternative='two-sided')
            except ValueError:
                summary[algo]["No Difference"] += 1
                continue

            if pvalue < alpha:
                if np.mean(base_arr) < np.mean(other_arr):
                    summary[algo]["Better"] += 1
                elif np.mean(base_arr) > np.mean(other_arr):
                    summary[algo]["Worse"] += 1
                else:
                    summary[algo]["No Difference"] += 1
            else:
                summary[algo]["No Difference"] += 1

    better_counts = [summary[algo]["Better"] for algo in comparison_algorithms]
    nodiff_counts = [summary[algo]["No Difference"] for algo in comparison_algorithms]
    worse_counts = [summary[algo]["Worse"] for algo in comparison_algorithms]

    indices = np.arange(len(comparison_algorithms))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    bars_better = ax.bar(indices - width, better_counts, width, label="Better", color="#2ca02c")
    bars_nodiff = ax.bar(indices, nodiff_counts, width, label="No Difference", color="#7f7f7f")
    bars_worse = ax.bar(indices + width, worse_counts, width, label="Worse", color="#d62728")

    for bars in (bars_better, bars_nodiff, bars_worse):
        ax.bar_label(bars, fmt='%d', padding=3)

    ax.set_xticks(indices)
    ax.set_xticklabels(comparison_algorithms, rotation=45, ha='right')
    ax.set_ylabel("Number of Test Functions")
    ax.set_title(f"Wilcoxon Signed-Rank Test (vs {base_algorithm})")
    ax.legend()
    fig.tight_layout()

    output_path = os.path.join(image_folder, f"{benchmark_name}_wilcoxon.png")
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Wilcoxon grouped bar chart saved to {output_path}")

def plot_mutation_rate(mutation_rate_folder, image_folder):
    """
    繪製 mutation rate 的收斂圖

    Args:
        mutation_rate_folder: mutation rate 資料的資料夾路徑
        image_folder: 輸出圖片的資料夾路徑
    """
    if not os.path.exists(mutation_rate_folder):
        print(f"Mutation rate folder '{mutation_rate_folder}' does not exist.")
        return

    os.makedirs(image_folder, exist_ok=True)

    mutation_rate_file = os.path.join(mutation_rate_folder, "mutation_rate.txt")

    evaluations = []
    mutation_rates = []

    if not os.path.exists(mutation_rate_file):
        print(f"File not found: {mutation_rate_file}")
        return

    with open(mutation_rate_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            # 支援 tab 或空白分隔
            parts = re.split(r'[\t\s]+', line)
            if len(parts) >= 2:
                try:
                    eval_count = int(parts[0])
                    mut_rate = float(parts[1])
                    evaluations.append(eval_count)
                    mutation_rates.append(mut_rate)
                except ValueError:
                    continue

    if not evaluations or not mutation_rates:
        print("No valid mutation rate data found.")
        return

    plt.figure(figsize=(12, 8))
    plt.plot(evaluations, mutation_rates, marker='o', markersize=3, linewidth=2, label='Mutation Rate')

    plt.xlabel('Evaluations')
    plt.ylabel('Mutation Rate')
    plt.title('Mutation Rate Convergence')
    plt.grid(True, alpha=0.3)
    plt.legend()

    y_min = min(mutation_rates) * 0.95
    y_max = max(mutation_rates) * 1.05
    plt.ylim(y_min, y_max)

    plt.tight_layout()

    output_path = os.path.join(image_folder, "mutation_rate_convergence.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Mutation rate convergence plot saved to {output_path}")

if __name__ == "__main__":
    
    print("This module is not intended to be run directly.")
    print("Please use 'run.py -p' to skip the full optimization process and only generate plots.")