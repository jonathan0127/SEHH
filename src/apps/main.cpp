#include <iostream>
#include <vector>
#include <functional>
#include <iomanip>
#include <cmath>
#include <limits>
#include <string>
#include <fstream>
#include <cstring>
#include <cstdint>
#include <stdexcept>
#include <chrono>
#include "metaheuristic/PSO.h"
#include "metaheuristic/GA.h"
#include "metaheuristic/DE.h"
#include "metaheuristic/LSHADE.h"
#include "hyper/hyperSequence.h"
#include "utils/instance.h"
#include "CEC2020.h"
#include "CEC2022.h"
#include "CEC2024.h"
#include <filesystem>

using namespace std;

static vector<uint32_t> loadSeeds(const string& path) {
    ifstream in(path);
    vector<uint32_t> seeds;
    seeds.reserve(1000);

    double value = 0.0;
    while(in >> value) {
        seeds.push_back(static_cast<uint32_t>(value));
    }

    return seeds;
}

static uint32_t computeRunSeed(int problemSize, int funcNo, int runId, const vector<uint32_t>& seeds) {
    const int runs = RUNS;
    long long int seedIndex = (problemSize / 10 * funcNo * runs + runId) - runs;
    seedIndex = ((seedIndex % 1000) + 1000) % 1000; // normalize to [0, 999]

    return seeds[static_cast<size_t>(seedIndex)];
}

// Function information structure
struct FunctionInfo {
    string name;
    function<double(vector<double>&)> func;
    int dimension;
    vector<double> lowerBounds;
    vector<double> upperBounds;
};

// Display vector values for output
void printVector(const vector<double>& vec) {
    cout << "[ ";
    for (size_t i = 0; i < vec.size(); ++i) {
        cout << setprecision(6) << vec[i];
        if (i < vec.size() - 1) {
            cout << ", ";
        }
    }
    cout << " ]" << endl;
}

// Lambda function to save midBest to file
void saveMidBestToFile(const vector<pair<int, double>>& avg_midBest, const vector<pair<int, double>>& best_midBest, const string& filename) {

    const int length = min(avg_midBest.size(), best_midBest.size());
    bool flag = false;

    ofstream midOutFile(filename);
    for(int i = 0; i < length; ++i){
        const int iter = avg_midBest[i].first;
        const double avg = avg_midBest[i].second;
        const double best = best_midBest[i].second;

        if(i == length-1 && avg < best) flag = true;

        midOutFile << iter << " " << avg << " " << best << endl;
    }

    if(flag){
        cerr << "[Warning] avg_midBest has better values than best_midBest in " << filename << endl;
    }

    midOutFile.close();
};

void saveAllMidBestToFile2022(const double all_midbest_2022[17][RUNS + 1], const string& filename) {
    ofstream outFile(filename);
    if (!outFile.is_open()) {
        cerr << "Error opening file for writing: " << filename << endl;
        return;
    }
    // set precision 8
    outFile << fixed << setprecision(8);
    for (int i = 0; i < 17; ++i) {
        for (int run = 0; run < RUNS; ++run) {
            outFile << all_midbest_2022[i][run];
            if (run < RUNS - 1) {
                outFile << " ";
            }
        }
        outFile << endl;
    }

    outFile.close();
}

void saveAllMidBestToFile2024(const double all_midbest_2024[1005][RUNS + 1], const string& filename) {
    ofstream outFile(filename);
    if (!outFile.is_open()) {
        cerr << "Error opening file for writing: " << filename << endl;
        return;
    }
    // set precision 8
    outFile << fixed << setprecision(8);
    for (int i = 0; i < 1000; ++i) {
        for (int run = 0; run < RUNS; ++run) {
            outFile << all_midbest_2024[i][run];
            if (run < RUNS - 1) {
                outFile << " ";
            }
        }
        outFile << endl;
    }

    outFile.close();
}

void runMetaAlgorithm(const Instance& instance,
                      string filename,
                      pair<vector<double>, double> &result,
                      vector<pair<int, double>>& midBest,
                      const string& algorithm,
                      uint32_t runSeed){

    // Generate appropriate dimension bounds
    vector<double> lowerBounds = instance.getLowerBounds();
    vector<double> upperBounds = instance.getUpperBounds();

    // Create algorithm instance, execute, save results
    if(algorithm == "PSO"){
        PSO pso(instance.getDimension(), instance.getObjFunction(), lowerBounds, upperBounds,
                META_POPULATION_SIZE, META_MAX_EVALUATIONS, 0.7, 1.5, 1.5, runSeed);
        result = pso.run(false);
        midBest = pso.getMidBest();
    }else if(algorithm == "GA"){
        GA ga(instance.getDimension(), instance.getObjFunction(), lowerBounds, upperBounds,
              META_POPULATION_SIZE, 0.9, 0.01, 0.04, 1, 1, META_MAX_EVALUATIONS, runSeed);
        result = ga.run(false);
        midBest = ga.getMidBest();
    }else if(algorithm == "DE_r1"){
        DE de(instance.getDimension(), instance.getObjFunction(), lowerBounds, upperBounds, 1,
              META_POPULATION_SIZE, META_MAX_EVALUATIONS, 0.5, 0.9, runSeed);
        result = de.run(false);
        midBest = de.getMidBest();
    }else if(algorithm == "DE_b1"){
        DE de(instance.getDimension(), instance.getObjFunction(), lowerBounds, upperBounds, 2,
              META_POPULATION_SIZE, META_MAX_EVALUATIONS, 0.5, 0.9, runSeed);
        result = de.run(false);
        midBest = de.getMidBest();
    }else if(algorithm == "DE_r2"){
        DE de(instance.getDimension(), instance.getObjFunction(), lowerBounds, upperBounds, 3,
              META_POPULATION_SIZE, META_MAX_EVALUATIONS, 0.5, 0.9, runSeed);
        result = de.run(false);
        midBest = de.getMidBest();
    }else if(algorithm == "DE_b2"){
        DE de(instance.getDimension(), instance.getObjFunction(), lowerBounds, upperBounds, 4,
              META_POPULATION_SIZE, META_MAX_EVALUATIONS, 0.5, 0.9, runSeed);
        result = de.run(false);
        midBest = de.getMidBest();
    }else if(algorithm == "LSHADE"){
        LSHADE lshade(instance.getDimension(), instance.getObjFunction(), lowerBounds, upperBounds,
                         META_POPULATION_SIZE, META_POPULATION_SIZE / 4, META_MAX_EVALUATIONS, 6, 0.5, 0.5, runSeed);
        result = lshade.run(false);
        midBest = lshade.getMidBest();
    }else if(algorithm.rfind("SEHH", 0) == 0 || algorithm.rfind("Hybrid-SE", 0) == 0){
        string seq_num = algorithm.substr(algorithm.rfind('-') + 1);
        string hyper_seq = "./result/hyper_sequences/SEHH_sequences" + seq_num + ".txt";
        HyperSequence hyperSequence(hyper_seq);
        double bestFitness = hyperSequence.calaulateDummyFitness(instance, runSeed);
        result = {hyperSequence.getBestSolution(), bestFitness};
        midBest = hyperSequence.getMidBest();
    }else{
        cerr << "Unknown algorithm: " << algorithm << endl;
        return;
    }
}

/*
要輸入 ./main -d 才會顯示選擇的函數和算法
*/
int main(int argc, char* argv[]) {
    bool display = false;

    int opt;
    while((opt = getopt(argc, argv, "d")) != -1){
        switch(opt){
            case 'd':
                display = true;
            default:
                break;
        }
    }

    int n;
    cerr << "Enter the number try: ";
    cin >> n;

    const vector<Instance>& instances = BenchmarkSetting::benchmark.getInstances();

    vector<uint32_t> precomputedSeeds;
    uint64_t baseTimeSeed = 0;
    if (BENCH_YEAR == 2024) {
        // For CEC2024 we derive seeds from current time, so no precomputed table is needed.
        baseTimeSeed = static_cast<uint64_t>(chrono::high_resolution_clock::now().time_since_epoch().count());
    } else {
        try {
            precomputedSeeds = loadSeeds("benchmarks/data/data_2022/Rand_Seeds.txt");
        } catch(const exception& e) {
            cerr << "[Seed Error] " << e.what() << endl;
            return 1;
        }
    }

    while(n--) {
        // Display available functions
        if(display){
            cerr << "Available optimization functions:" << endl;
            for (int i = 0; i < (int)instances.size(); ++i) {
                cerr << (i+1) << ". " << instances[i].getName() << endl;
            }
        }
        
        // Select function
        int funcChoice;
        if(display) cerr << "Select function (1-" << (int)instances.size() << "): ";
        cin >> funcChoice;

        if (funcChoice <= 0 || funcChoice > static_cast<int>((int)instances.size())) {
            cerr << "Invalid choice, using default: 1" << endl;
            funcChoice = 1;
        }

        Instance selected_instance = instances[funcChoice - 1];

        // Display available optimization algorithms
        if(display){
            for(size_t i = 0; i < algorithmName.size(); ++i) {
                cerr << (i+1) << ". " << algorithmName[i] << endl;
            }
        } 
        
        // Select optimization algorithm
        int algoChoice;
        if(display) cerr << "Select optimization algorithm (1-" << algorithmName.size() << "): ";
        cin >> algoChoice;

        cerr << "\nOptimizing function: " << selected_instance.getName() << endl;

        // Execute selected optimization algorithm
        string func_path = "result/records/" + to_string(funcChoice) + "_" + to_string(selected_instance.getDimension()) + "_";
        pair<vector<double>, double> result;
        vector<pair<int, double>> current_midBest;
        vector<pair<int, double>> avg_midBest(META_MAX_EVALUATIONS / 100 + 1, {0, 0});
        vector<pair<int, double>> best_midBest; // 儲存最佳那次的 midBest
        // Track best-of-runs
        double bestOfRunsFitness = numeric_limits<double>::infinity();
        vector<double> bestOfRunsSolution;

        filesystem::create_directories("result/error");
        vector<double> runErrors;
        runErrors.reserve(RUNS);

        double optimumFitness = 0.0;
        if(funcChoice > 0 && funcChoice < static_cast<int>(sizeof(BENCH::detail::biases) / sizeof(double))){
            optimumFitness = BENCH::detail::biases[funcChoice];
        }

        if (algoChoice <= 0 || algoChoice > static_cast<int>(algorithmName.size())) {
            cerr << "Invalid choice, using default: 1 (PSO)" << endl;
            algoChoice = 1;
        }

        for(int i=0;i<=META_MAX_EVALUATIONS / 100;++i) {
            avg_midBest[i].first = i * 100;
        }

        string algorithm = algorithmName[algoChoice - 1];
        cerr << "\nRunning " << algorithm << " algorithm..." << endl;

        const int problemSize = selected_instance.getDimension();
        const int funcNo = funcChoice;
        cerr << "funNO: " << funcNo << ", dim: " << problemSize << endl;

        tqdm runBar;
        double sumRunFitness = 0.0;  // 累計每次 run 的最終 fitness（用於平均）
        double all_midbest_2022[17][RUNS + 1];
        double all_midbest_2024[1005][RUNS + 1];
        for(int run = 0; run < RUNS; ++run) {
            runBar.progress(run+1, RUNS);
            uint32_t runSeed = 0;
            if (BENCH_YEAR == 2024) {
                const uint64_t jitter = static_cast<uint64_t>(run + 1) * 0x9e3779b97f4a7c15ULL;
                runSeed = static_cast<uint32_t>((baseTimeSeed ^ jitter) & 0xffffffffULL);
            } else {
                runSeed = computeRunSeed(problemSize, funcNo, run + 1, precomputedSeeds);
            }
            runMetaAlgorithm(selected_instance, func_path + algorithm + ".txt", result, current_midBest, algorithm, runSeed);
            if (current_midBest.size() < avg_midBest.size()) {
                double last_val = current_midBest.empty() ? 0.0 : current_midBest.back().second;
                while (current_midBest.size() < avg_midBest.size()) {
                    current_midBest.push_back({(int)current_midBest.size() * 100, last_val});
                }
            }
            double runFitness = result.second;
            runErrors.push_back(runFitness - optimumFitness);
            int length = min((int)current_midBest.size(), (int)avg_midBest.size());
            // cerr << "current midBest size: " << current_midBest.size() << ", avg_midBest size: " << avg_midBest.size() << endl;
            int idx_for_check = 0;
            double exponent = (static_cast<double>(idx_for_check) / 5.0) - 3.0;
            int check_point = static_cast<int>(floor(pow(10.0, exponent) * static_cast<double>(META_MAX_EVALUATIONS)));
            // cerr << "check_point: " << check_point << endl;
            bool save_last = false;
            for(int i = 0; i < length; i++) {
                avg_midBest[i].second += current_midBest[i].second;
                if(i % 3 == 0){
                    all_midbest_2024[i / 3][run] = current_midBest[i].second - optimumFitness;
                }
                if(algorithm.rfind("SEHH", 0) != 0 && algorithm.rfind("Hybrid-SE", 0) != 0) continue;
                if(current_midBest[i].second - optimumFitness <= EQUAL_ZERO_THRESHOLD && !save_last) {
                    all_midbest_2022[16][run] = i * 100;
                    save_last = true;
                }
                if(check_point <= i * 100) {
                    if(current_midBest[i].second - optimumFitness <= EQUAL_ZERO_THRESHOLD && !save_last) {
                        all_midbest_2022[16][run] = i * 100;
                        all_midbest_2022[idx_for_check][run] = 0.0;
                        save_last = true;
                    }
                    else if(current_midBest[i].second - optimumFitness <= EQUAL_ZERO_THRESHOLD && save_last) {
                        all_midbest_2022[idx_for_check][run] = 0.0;
                    }
                    else {
                        all_midbest_2022[idx_for_check][run] = current_midBest[i].second - optimumFitness;
                    }
                    
                    idx_for_check++;
                    exponent = (static_cast<double>(idx_for_check) / 5.0) - 3.0;
                    check_point = static_cast<int>(floor(pow(10.0, exponent) * static_cast<double>(META_MAX_EVALUATIONS)));
                }
                
            }
            if(!save_last) {
                all_midbest_2022[16][run] = (length - 1) * 100;
            }
            sumRunFitness += runFitness; // 累加本次 run 最終 fitness

            // Update best-of-runs
            for(int i = length-1; i >= 0; i--){
                
                if(best_midBest.empty() || current_midBest[i].second < best_midBest[i].second){

                    bestOfRunsFitness = runFitness;
                    bestOfRunsSolution = result.first;
                    best_midBest = current_midBest; // 記下最佳 run 的曲線

                }
                if(best_midBest[i].second < current_midBest[i].second){
                    break;
                }
            }
        }
        runBar.finish();

        // Average midBest over runs
        for(auto& [iter, best] : avg_midBest) {
            best /= RUNS;
        }

        // 計算並輸出平均 fitness 到 summary 檔
        double avgRunFitness = sumRunFitness / RUNS;
        filesystem::create_directories("result/records");
        ofstream summary("result/records/average_fitness.txt", ios::app);
        if(summary.tellp() == 0) {
            summary << "Algorithm\tAverageFitness\n";
        }
        summary << algorithm << "\t" << setprecision(10) << avgRunFitness << "\n";
        

        // After RUNS, set result to best-of-runs
        if(!bestOfRunsSolution.empty() || bestOfRunsFitness < numeric_limits<double>::infinity()){
            result = {bestOfRunsSolution, bestOfRunsFitness};
        }
        // Save error log
        string error_file_path = "result/error/" + to_string(funcChoice) + "_" + to_string(selected_instance.getDimension()) + "_" + algorithm + "_error.csv";
        ofstream errorFile(error_file_path);
        if(errorFile.is_open()){
            errorFile << "Run,Error" << '\n';
            for(size_t i = 0; i < runErrors.size(); ++i){
                errorFile << (i + 1) << "," << setprecision(10) << runErrors[i] << '\n';
            }
            errorFile.close();
        }else{
            cerr << "Failed to write error log: " << error_file_path << endl;
        }

        // Save midBest：改為輸出平均&最佳曲線
        saveMidBestToFile(avg_midBest, best_midBest, func_path + algorithmName[algoChoice - 1] + ".txt");
        if(algorithm.rfind("SEHH", 0) == 0 || algorithm.rfind("Hybrid-SE", 0) == 0){
            if(BENCH_YEAR == 2024){
                string midbest_saveAll = "result/2024official/"+ algorithm + "_" + to_string(funcChoice) + "_" + to_string(selected_instance.getDimension())  + ".txt";
                saveAllMidBestToFile2024(all_midbest_2024, midbest_saveAll);
            }
            else{
                string midbest_saveAll = "result/2022official/"+ algorithm + "_" + to_string(funcChoice) + "_" + to_string(selected_instance.getDimension())  + ".txt";
                saveAllMidBestToFile2022(all_midbest_2022, midbest_saveAll);
            }

            
        }
        
        // Display results
        if(display){
            auto [bestSolution, bestFitness] = result;
            cout << "\nOptimization Results:" << endl;
            cout << "Best Solution: ";
            printVector(bestSolution);
            cout << "Best Fitness: " << setprecision(10) << bestFitness << endl;
        }

        cerr << "Optimization completed!" << endl;
    }
    return 0;
}