#ifndef SEHH_H
#define SEHH_H

#include "utils/sample.h"
#include "config/para_setting.h"
#include "utils/tqdm.h"

#include "hyperSequence.h"
#include "segment.h"

#include <queue>
#include <random>
#include <set>
#include <algorithm>
#include <string>
#include <fstream>
#include <iostream>
#include <chrono>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <functional>
#include <atomic>
#include <mutex>
#include <cstdint>

typedef vector<HyperSequence> s1d;
typedef vector<vector<HyperSequence>> s2d;
typedef vector<vector<vector<HyperSequence>>> s3d;

using namespace std;


class SEHH {
public:

    SEHH(
        const vector<Instance>& instances,
        const int maxEvaluations = HYPER_MAX_EVALUATIONS,
        const double mutationRate = 0.2,
        const double crossoverRate = 0.9,
        const int seqence_length = 5,
        const int num_searchers = 4,
        const int init_num_samples = 4,
        const int min_sample_per_region = 2,
        const int max_sample_per_region = 8,
        const int num_players = 1
    );

    unordered_map<int, string> metaMap = {
        {0, "PSO"},
        {1, "GA"},
        {2, "DE"},
        {3, "LSHADE"},
        {4, "LSRTDE"},
        {5, "RDE"}
    };

    unordered_map<string, int> metaEnum = {
        {"PSO", 0},
        {"GA", 1},
        {"DE", 2},
        {"LSHADE", 3},
        {"LSRTDE", 4},
        {"RDE", 5}
    };

    void run(const string &outFile);

private:

    // Size Parameters
    atomic<int> evaluations{0}; // 記錄 evaluate_fitness 執行次數 (thread-safe)
    const int maxEvaluations;
    double mutationRate; // 會動態調整
    double mutationLowerBound;
    double mutationUpperBound;
    double mutationStep;
    int stepToChange; // 幾個 iteration 沒動調整一次 mutation rate
    const double crossoverRate;
    const int seqence_length;
    const int num_searchers;
    const int num_regions;
    const int init_num_samples;
    const int min_sample_per_region;
    const int max_sample_per_region;
    const int num_players;
    int howLongNoImprovement; // 記錄多少個 generation 沒有改善
    
    // Instance & SEHH sequences
    vector<Instance> instances;
    s1d searcher_sol;                     // [searcher]
    s2d sample_sol;               // [region][sample]
    s1d sample_sol_best;                  // [region]
    s3d sampleV_sol;      // [searcher][region][sample]

    vector<int> searcher_region_id;     // [searcher] 指派的 region
    vector<int> region_meta;            // [region] 主 metaheuristic

    vector<pair<double, HyperSequence>> all_time_best_sequences; // Three best sequences of all time

    // SE bookkeeping
    vector<double> taj; // times region j has been invested (searched)
    vector<double> tbj; // times region j has not been invested (reset to 1 if invested)

    // Exepected Value
    vector<vector<double>> expectedValues; // [searcher][region]
    vector<double> Tj;
    vector<double> Mj;

    tqdm bar;
    vector<pair<int, double>> midBest;
    mt19937 rng;
    vector<uint32_t> precomputedSeeds;
    // training logs
    string iter_ratio_file_path; // log path for iter_ratio
    mutex iter_ratio_log_mu;     // guard writes

    // Fitness cache: 避免重複評估相同的 sequence
    unordered_map<size_t, double> fitnessCache;
    mutex cacheMutex;
    atomic<size_t> cacheHits{0};
    atomic<size_t> cacheMisses{0};
    static constexpr size_t MAX_CACHE_SIZE = 10000;

    // 計算 sequence 的 hash
    size_t hashSequence(const HyperSequence& hs) const {
        size_t h = 0;
        
        // 量化函數：四捨五入到 2 位小數
        auto quantize = [](double val) -> int64_t {
            return static_cast<int64_t>(round(val * 100.0));  // 0.01 精度
        };
        
        for(const auto& seg : hs.getSequence()){
            // 組合 metaheuristic 類型
            h ^= hash<string>{}(seg.getTemplateName()) + 0x9e3779b9 + (h << 6) + (h >> 2);
            
            // 組合量化後的參數
            const auto& params = seg.getSegmentParams();
            if(holds_alternative<SEHH_PSOParams>(params)){
                const auto& p = get<SEHH_PSOParams>(params);
                h ^= hash<int64_t>{}(quantize(p.w)) + 0x9e3779b9 + (h << 6) + (h >> 2);
                h ^= hash<int64_t>{}(quantize(p.c1)) + 0x9e3779b9 + (h << 6) + (h >> 2);
                h ^= hash<int64_t>{}(quantize(p.c2)) + 0x9e3779b9 + (h << 6) + (h >> 2);
            } else if(holds_alternative<SEHH_GAParams>(params)){
                const auto& p = get<SEHH_GAParams>(params);
                h ^= hash<int64_t>{}(quantize(p.mutationRate)) + 0x9e3779b9 + (h << 6) + (h >> 2);
                h ^= hash<int64_t>{}(quantize(p.crossoverRate)) + 0x9e3779b9 + (h << 6) + (h >> 2);
                h ^= hash<int>{}(p.selectionMethod) + 0x9e3779b9 + (h << 6) + (h >> 2);
            } else if(holds_alternative<SEHH_DEParams>(params)){
                const auto& p = get<SEHH_DEParams>(params);
                h ^= hash<int64_t>{}(quantize(p.F)) + 0x9e3779b9 + (h << 6) + (h >> 2);
                h ^= hash<int64_t>{}(quantize(p.CR)) + 0x9e3779b9 + (h << 6) + (h >> 2);
                h ^= hash<int>{}(p.strategy) + 0x9e3779b9 + (h << 6) + (h >> 2);
            } else if(holds_alternative<SEHH_LSHADEParams>(params)){
                const auto& p = get<SEHH_LSHADEParams>(params);
                h ^= hash<int>{}(p.memorySize) + 0x9e3779b9 + (h << 6) + (h >> 2);
                h ^= hash<int64_t>{}(quantize(p.CR)) + 0x9e3779b9 + (h << 6) + (h >> 2);
                h ^= hash<int64_t>{}(quantize(p.F)) + 0x9e3779b9 + (h << 6) + (h >> 2);
            } else if(holds_alternative<SEHH_LSRTDEParams>(params)){
                const auto& p = get<SEHH_LSRTDEParams>(params);
                h ^= hash<int>{}(p.memorySize) + 0x9e3779b9 + (h << 6) + (h >> 2);
                h ^= hash<int64_t>{}(quantize(p.sigmaCR)) + 0x9e3779b9 + (h << 6) + (h >> 2);
                h ^= hash<int64_t>{}(quantize(p.sigmaF)) + 0x9e3779b9 + (h << 6) + (h >> 2);
            } else if(holds_alternative<SEHH_RDEParams>(params)){
                const auto& p = get<SEHH_RDEParams>(params);
                h ^= hash<int64_t>{}(quantize(p.psizeParam)) + 0x9e3779b9 + (h << 6) + (h >> 2);
                h ^= hash<int64_t>{}(quantize(p.G_flag)) + 0x9e3779b9 + (h << 6) + (h >> 2);
                h ^= hash<int64_t>{}(quantize(p.EB_flag)) + 0x9e3779b9 + (h << 6) + (h >> 2);
                h ^= hash<int>{}(p.selectionMode) + 0x9e3779b9 + (h << 6) + (h >> 2);
            }
        }
        return h;
    }

    // 查詢 cache
    bool tryGetCachedFitness(const HyperSequence& hs, double& outFitness) {
        size_t h = hashSequence(hs);
        lock_guard<mutex> lk(cacheMutex);
        auto it = fitnessCache.find(h);
        if(it != fitnessCache.end()){
            outFitness = it->second;
            cacheHits.fetch_add(1, memory_order_relaxed);
            return true;
        }
        return false;
    }

    // 存入 cache
    void storeFitnessInCache(const HyperSequence& hs, double fitness) {
        size_t h = hashSequence(hs);
        lock_guard<mutex> lk(cacheMutex);
        if(fitnessCache.size() >= MAX_CACHE_SIZE){
            // 清除一半的 cache（簡單策略）
            auto it = fitnessCache.begin();
            for(size_t i = 0; i < MAX_CACHE_SIZE / 2 && it != fitnessCache.end(); ++i){
                it = fitnessCache.erase(it);
            }
        }
        fitnessCache[h] = fitness;
        cacheMisses.fetch_add(1, memory_order_relaxed);
    }

    // Initialization
    void initialize();

    // Resource Arrangement
    void resource_arrangement();

    // Vision Search
    void vision_search();
    void transit();
    HyperSequence crossoverSequences(const HyperSequence& a, const HyperSequence& b);
    HyperSequence mutateSequence(const HyperSequence& seq);
    HyperSequence argumentMutation(const HyperSequence& seq);
    int calculateRegion(const HyperSequence& seq);
    virtual void compute_expected_value();
    void parallelEvaluate(vector<HyperSequence*>& seqs, const string& label); 
    double evaluate_fitness(HyperSequence& hs);
    double evaluate_fitness_with_iteration(HyperSequence& hs);
    void determination();

    // Marketing Survey
    void marketing_survey();
    
    // helpers
    HyperSequence generate_random_sequence(int nSegments);
    void ensureSizes();
    void updateFile(const string &outFile);
    static vector<uint32_t> loadPrecomputedSeeds(const string& path);
    uint32_t computeRunSeed(int problemSize, int funcNo, int runId) const;
    void logIterRatio(double iter_ratio);
};

/* -------------------- Implementation -------------------- */

SEHH::SEHH(
    const vector<Instance>& instances,
    const int maxEvaluations,
    const double mutationRate,
    const double crossoverRate,
    const int seqence_length,
    const int num_searchers,
    const int init_num_samples,
    const int min_sample_per_region,
    const int max_sample_per_region,
    const int num_players
) : 
    instances(instances),
    maxEvaluations(maxEvaluations),
    mutationRate(mutationRate),
    crossoverRate(crossoverRate),
    seqence_length(seqence_length),
    num_searchers(num_searchers),
    num_regions(metaMap.size()),
    init_num_samples(init_num_samples),
    min_sample_per_region(min_sample_per_region),
    max_sample_per_region(max_sample_per_region),
    num_players(num_players),
    rng(random_device{}()) {}

void SEHH::ensureSizes(){

    searcher_sol.resize(num_searchers);
    sample_sol.resize(num_regions);
    for(int j=0; j<num_regions; ++j)
        sample_sol[j].resize(init_num_samples);
    sample_sol_best.resize(num_regions);
    sampleV_sol.resize(num_searchers);
    for(int i=0; i<num_searchers; ++i){
        sampleV_sol[i].resize(num_regions);
        for(int j=0; j<num_regions; ++j)
            sampleV_sol[i][j].resize(init_num_samples);
    }

    taj.assign(num_regions, 0.0);
    tbj.assign(num_regions, 1.0);

    expectedValues.assign(num_searchers, vector<double>(num_regions, 0.0));
    Tj.assign(num_regions, 0.0);
    Mj.assign(num_regions, 0.0);

}

void SEHH::run(const string &outFile){

    if(instances.empty()){
        cerr << "[SEHH] No instances provided." << endl;
        return;
    }

    initialize();
    resource_arrangement();

    // for(int i=0;i<num_searchers;++i){
    //     cout << "Searcher " << i << " fitness=" << searcher_sol[i].getFitness() << " region=" << metaMap[region_meta[searcher_region_id[i]]] << "\n";
    //     cout << searcher_sol[i] << "\n";
    // }
    
    auto start_total = chrono::steady_clock::now();

    string mutation_rate_file = "./result/output/mutation_rate.txt";
    string midBest_file = "./result/output/midBest.txt";
    iter_ratio_file_path = "./result/output/iter_ratio.txt";

    ofstream mutation_rate_ofs(mutation_rate_file);
    ofstream iter_ratio_ofs(iter_ratio_file_path, ios::trunc);

    // bar.progress(evaluations, maxEvaluations);
    while(evaluations.load() < maxEvaluations){

        auto t0 = chrono::steady_clock::now();

        if(all_time_best_sequences.size() == 0){
            cerr << "\n[SEHH] evaluations=" << evaluations.load() << "/" << maxEvaluations << " best_fitness=inf" << endl;
        }else{
            cerr << "\n[SEHH] evaluations=" << evaluations.load() << "/" << maxEvaluations << " best_fitness=" << all_time_best_sequences[0].first << endl;
        }
        cerr << "mutation rate=" << mutationRate << endl;
        
        // 輸出 evaluation 次數和 mutation rate 到檔案
        mutation_rate_ofs << evaluations.load() << " " << mutationRate << endl;
        
        cerr << "In vision search..." << endl;
        vision_search();
        cerr << "In marketing survey..." << endl;
        marketing_survey();

        auto t1 = chrono::steady_clock::now();
        double elapsed_ms = chrono::duration_cast<chrono::milliseconds>(t1 - t0).count();


        // 每次完成迴圈後輸出目前結果
        updateFile(outFile);
        howLongNoImprovement++;
        if(howLongNoImprovement >= stepToChange){
            // 超過指定次數沒改善，調整 mutation rate
            mutationRate += mutationStep;
            if(mutationRate > mutationUpperBound)
                mutationRate = mutationUpperBound;
            howLongNoImprovement = 0; // 重置計數器
        }
    }
    // bar.finish();

    // 輸出每個 searcher 的結果
    updateFile(outFile);

    // 輸出 midBest 到檔案
    ofstream midBest_ofs(midBest_file);
    if(midBest_ofs.is_open()){
        for(const auto& [iter, best] : midBest){
            midBest_ofs << iter << " " << best << endl;
        }
        midBest_ofs.close();
    }

    // close mutation rate 檔案
    if(mutation_rate_ofs.is_open()){
        mutation_rate_ofs.close();
    }

    auto end_total = chrono::steady_clock::now();
    double total_ms = chrono::duration_cast<chrono::milliseconds>(end_total - start_total).count();
    cerr << "[SEHH] finished. evaluations=" << evaluations.load() << " time_ms=" << total_ms << " output=" << outFile << endl;
}

// init
void SEHH::initialize(){

    // 處理 mutation rate
    mutationLowerBound = MutationRateParameter::mutationLowerBound;
    mutationRate = mutationLowerBound; 
    mutationUpperBound = MutationRateParameter::mutationUpperBound;
    mutationStep = MutationRateParameter::mutationStep;
    stepToChange = MutationRateParameter::stepToChange; // generation 幾個 iteration 沒動調整一次 mutation rate
    howLongNoImprovement = 0;
    precomputedSeeds = loadPrecomputedSeeds("benchmarks/data/data_2022/Rand_Seeds.txt");
    ensureSizes();
    evaluations.store(0);
    all_time_best_sequences.clear();

    for(int i=0; i<num_searchers; ++i){
        searcher_sol[i] = generate_random_sequence(seqence_length);
    }
}

// resource_arrangement
void SEHH::resource_arrangement(){
    // assign main metaheuristic to each region
    region_meta.resize(num_regions);
    for(int j=0;j<num_regions;++j){
        region_meta[j] = j; // assign each region a primary metaheuristic
    }

    // initialize sample_sol of each region
    for(int j=0;j<num_regions;++j){
        int main_meta = region_meta[j];
        for(int k=0;k<init_num_samples;++k){
            vector<Segment> segs;
            int main_count = seqence_length / 2 + 1;
            HyperSequence candidate_seq;
            do{
                segs.clear();
                for(int s=0;s<seqence_length;++s){
                    if(s < main_count)
                        segs.emplace_back(Segment(main_meta)); // main metaheuristic
                    else
                        segs.emplace_back(Segment(rng() % num_regions)); // other random
                }
                shuffle(segs.begin(), segs.end(), rng);
                candidate_seq = HyperSequence(segs);
            }
            while(calculateRegion(candidate_seq) != j); // ensure candidate belongs to region j
            
            sample_sol[j][k] = candidate_seq;
            sample_sol[j][k].setRegion(j);
        }
    }

    // assign searcher to region and initialize sequence
    searcher_region_id.resize(num_searchers);
    for(int i=0;i<num_searchers;++i){
        int region_id = i % num_regions;
        int main_meta = region_meta[region_id];
        vector<Segment> segs;
        int main_count = seqence_length / 2 + 1;
        HyperSequence candidate_seq;
        do{
            segs.clear();
            for(int s=0;s<seqence_length;++s){
                if(s < main_count)
                    segs.emplace_back(Segment(main_meta));
                else
                    segs.emplace_back(Segment(rng()%num_regions));
            }
            shuffle(segs.begin(), segs.end(), rng);
            candidate_seq = HyperSequence(segs);
        }
        while(calculateRegion(candidate_seq) != region_id); // 確保分配的 sequence 屬於該 region
        
        searcher_sol[i] = candidate_seq;
        searcher_region_id[i] = region_id;
        searcher_sol[i].setRegion(region_id);
    }

    // 4. 初始化 investment
    for (int i = 0; i < num_searchers; i++) {
        int r = searcher_region_id[i];
        taj[r]++;
        tbj[r] = 1.0;
    }
    
}

// vision_search wrapper
void SEHH::vision_search(){

    cerr << "In transition..." << endl;
    transit();

    cerr << "In calculation of expected value..." << endl;
    compute_expected_value();

    cerr << "In vision selection..." << endl;
    determination();
}

// transit -> crossover
void SEHH::transit(){
    for(int i=0;i<num_searchers;++i){
        s2d temp_sampleV_sol = sampleV_sol[i]; // [region][sample]
        for (int o = 0; o < num_regions; o++) {
            temp_sampleV_sol[o].clear();
        }

        for(int j=0;j<num_regions;++j){
            int num_samples = (int)sample_sol[j].size();
            for(int k=0;k<num_samples;++k){
                // 在 crossover 和 mutation 回傳的 seq 不一定位於 region j
                HyperSequence new_seq;
                new_seq = crossoverSequences(searcher_sol[i], sample_sol[j][k]);
                new_seq = mutateSequence(new_seq); // 前 75% 迭代
                new_seq = argumentMutation(new_seq); // 後 25% 迭代
                int region_id = calculateRegion(new_seq);
                temp_sampleV_sol[region_id].push_back(new_seq);
                temp_sampleV_sol[region_id].back().setRegion(region_id);
            }
        }

        // 處理 sampleV_sol 過多過少的問題
        for(int j=0;j<num_regions;++j){
            int current_size = (int)temp_sampleV_sol[j].size();
            if(current_size < min_sample_per_region){
                // 不足，補齊
                for (int m = current_size; m < min_sample_per_region; ++m) {
                    vector<Segment> new_segs;
                    int main_meta = j;
                    int main_count = seqence_length / 2 + 1;
                    do{
                        new_segs.clear();
                        for(int s=0; s<seqence_length; ++s){
                            if(s < main_count)
                                new_segs.emplace_back(Segment(main_meta));
                            else
                                new_segs.emplace_back(Segment(rng() % num_regions));
                        }
                        shuffle(new_segs.begin(), new_segs.end(), rng);
                    }
                    while(calculateRegion(HyperSequence(new_segs)) != j); // 確保分配的 sequence 屬於該 region

                   
                    temp_sampleV_sol[j].push_back(HyperSequence(new_segs));
                    temp_sampleV_sol[j].back().setRegion(j);
                }
            }else if(current_size > max_sample_per_region){
                // 過多，隨機刪減
                for (int m = current_size; m > max_sample_per_region; --m) {
                    uniform_int_distribution<int> dis(0, m - 1);
                    int idx = dis(rng);
                    temp_sampleV_sol[j].erase(temp_sampleV_sol[j].begin() + idx);
                }
            }
        }
        
        // 存入 sampleV_sol
        swap(sampleV_sol[i], temp_sampleV_sol);
    }
}

// crossoverSequences: segment-level 50/50
HyperSequence SEHH::crossoverSequences(const HyperSequence& a, const HyperSequence& b){
    const auto& sa = a.getSequence();
    const auto& sb = b.getSequence();
    vector<Segment> out;

    // one point crossover
    bernoulli_distribution coin(crossoverRate);
    if (coin(rng)) {
        uniform_int_distribution<int> dist(1, seqence_length - 1);
        int crossover_point = dist(rng);
        for(int t = 0; t < seqence_length; ++t) {
            if(t < crossover_point)
                out.push_back(sa[t]);
            else
                out.push_back(sb[t]);
        }
    }else{
        out = sb;
    }
    return HyperSequence(out);
}

// mutateSequences: segment-level mutation
HyperSequence SEHH::mutateSequence(const HyperSequence& seq){
    const vector<Segment>& s = seq.getSequence();
    vector<Segment> out = s;
    double localMutationRate;
    // 設定 mutation rate
    if(evaluations.load() < maxEvaluations * 0.75){
        localMutationRate = mutationRate; // 前 75% 迭代
    }else{
        localMutationRate = mutationLowerBound / 2.0; // 後 25% 迭代
    }
    // 改變 sequence 中的 segment
    bernoulli_distribution coin(localMutationRate);
    if (coin(rng)) {
        uniform_int_distribution<int> dist(0, seqence_length - 1);
        out[dist(rng)] = Segment(rng() % num_regions); // choose any registered low-level metaheuristic
    }

    // 交換兩個 segment 的位置
    bernoulli_distribution swapCoin(localMutationRate);
    if(swapCoin(rng) && seqence_length >= 2) {
        // 隨機挑選兩個不同的位置交換
        uniform_int_distribution<int> dist(0, seqence_length - 1);
        int idx1 = dist(rng);
        int idx2 = dist(rng);
        
        // 確保兩個 index 不同
        while(idx1 == idx2){
            idx2 = dist(rng);
        }

        // 執行交換
        swap(out[idx1], out[idx2]);
    }

    bernoulli_distribution twoOptCoin(localMutationRate);

    if(twoOptCoin(rng) && seqence_length >= 3) { // 至少長度 3 才看得出反轉效果
        // 隨機選擇兩個切點 i 和 j
        uniform_int_distribution<int> dist(0, seqence_length - 1);
        int idx1 = dist(rng);
        int idx2 = dist(rng);

        // 確保 idx1 < idx2
        if(idx1 > idx2) swap(idx1, idx2);
        
        if(idx2 > idx1){
            // 使用 reverse 反轉區間 [idx1, idx2]
            reverse(out.begin() + idx1, out.begin() + idx2 + 1);
        }
    }
    return HyperSequence(out);
}

// argumentMutation: parameter-level mutation
HyperSequence SEHH::argumentMutation(const HyperSequence& seq){

    double localMutationRate;
    if(evaluations.load() < maxEvaluations * 0.75){
        localMutationRate = mutationLowerBound / 2.0; // 前 75% 迭代
    }else{
        localMutationRate = mutationRate; // 後 25% 迭代
    }

    bernoulli_distribution coin(localMutationRate);
    auto s = seq.getSequence();
    double progress = (double)evaluations.load() / maxEvaluations;
    // resetProb 從 1.0 降至 0.2
    double resetProb = clamp(1.0 - 0.8 * progress, 0.2, 1.0);
    bernoulli_distribution doReset(resetProb);

    for(auto& segment : s){
        if(coin(rng)){
            if(doReset(rng)){
                segment.RandominitP();
            } else {
                // 整個 range 範圍為 6 個標準差
                segment.perturbParams(rng, 6);
            }
        }
    }
    return HyperSequence(s);
}

int SEHH::calculateRegion(const HyperSequence& seq) {
    vector<int> time_weight_sum(num_regions, 0); // 計算每個 metaheuristic 的 time_weight 總和
    for (const auto& segment : seq.getSequence()) {
        time_weight_sum[metaEnum[segment.getTemplateName()]] += segment.getTimeWeight();
    }

    // 找出最大 time_weight 總和
    int max_weight = 0;
    for (int j = 0; j < num_regions; j++) {
        if (time_weight_sum[j] > max_weight) {
            max_weight = time_weight_sum[j];
        }
    }

    // 收集所有達到最大 time_weight 的 regions（處理平手情況）
    vector<int> candidates;
    for (int j = 0; j < num_regions; j++) {
        if (time_weight_sum[j] == max_weight) {
            candidates.push_back(j);
        }
    }

    // 如果有多個平手，隨機選擇一個
    if (candidates.size() == 1) {
        return candidates[0];
    } else {
        uniform_int_distribution<int> dist(0, candidates.size() - 1);
        return candidates[dist(rng)];
    }
}

// compute_expected_value: evaluate all v and compute e
// 每個 Thread 完成一個 Task 後再去 Queue 領下一個
void SEHH::parallelEvaluate(vector<HyperSequence*>& seqs, const string& label){
    if(seqs.empty()) return;
    
    // 先檢查 cache，分離需要評估的
    vector<HyperSequence*> needEval;
    needEval.reserve(seqs.size());
    size_t cachedCount = 0;
    
    for(auto* seq : seqs){
        double cachedFit;
        if(tryGetCachedFitness(*seq, cachedFit)){
            seq->setFitness(cachedFit);
            cachedCount++;
        } else {
            needEval.push_back(seq);
        }
    }
    
    if(cachedCount > 0){
        cerr << "[" << label << "] " << cachedCount << "/" << seqs.size() << " from cache" << endl;
    }
    
    if(needEval.empty()){
        return;
    }
    
    unsigned int num_threads = thread::hardware_concurrency();
    if(num_threads == 0) num_threads = 4;
    size_t N = needEval.size();
    
    cerr << "[" << label << "] evaluating " << N << " sequences with dynamic scheduling..." << endl;
    
    atomic<size_t> taskIndex{0};  // 下一個待處理
    atomic<size_t> done{0};       // 已完成的任務數
    
    auto worker = [&](){
        while(true){
            // 領取下一個任務
            size_t i = taskIndex.fetch_add(1, memory_order_relaxed);
            
            // 沒有更多任務了，退出
            if(i >= N) break;
            
            double f;
            if(FITNESS_METHOD_CHOICE == 0){
                f = evaluate_fitness(*needEval[i]); 
            }
            else if(FITNESS_METHOD_CHOICE == 1){
                f = evaluate_fitness_with_iteration(*needEval[i]); 
            }
            needEval[i]->setFitness(f);
            
            // 存入 cache
            storeFitnessInCache(*needEval[i], f);
            
            size_t completed = done.fetch_add(1, memory_order_relaxed) + 1;
            
            // 每 10 個輸出進度
            if(completed % 10 == 0){
                cerr << "\r[" << label << "] " << completed << "/" << N << flush;
            }
        }
    };
    
    vector<thread> threads; 
    threads.reserve(num_threads);
    for(unsigned int t = 0; t < num_threads && t < N; ++t){ 
        threads.emplace_back(worker);
    } 
    
    for(auto &th : threads) th.join();
    
    cerr << "\r[" << label << "] " << N << "/" << N << " done. (cache hits: " << cacheHits.load() << ", misses: " << cacheMisses.load() << ")" << endl;
}

void SEHH::compute_expected_value(){

    // 1. evaluate all sampleV_sol

    int total_progress = 0;
    int progress = 0;

    if (evaluations.load() == 0) {
        vector<HyperSequence*> batch;
        batch.reserve(num_searchers + num_regions * init_num_samples);
        for(int i=0;i<num_searchers;++i) batch.push_back(&searcher_sol[i]);
        for(int j=0;j<num_regions;++j){
            for(int k=0;k<init_num_samples;++k) batch.push_back(&sample_sol[j][k]);
        }
        parallelEvaluate(batch, "First evaluation (parallel)");
        for(int i=0;i<num_searchers;++i) searcher_sol[i].setRegion(searcher_region_id[i]);
        for(int j=0;j<num_regions;++j){
            for(int k=0;k<init_num_samples;++k) sample_sol[j][k].setRegion(j);
        }
    }
    else {
        struct Task {
            int i;
            int j;
            int k;
            HyperSequence* seq;
            double fitness;
        };
        vector<Task> tasks;
        tasks.reserve(num_searchers * init_num_samples);

        for(int i=0;i<num_searchers;++i){
            int j = searcher_region_id[i];

            for(int k=0;k<init_num_samples;++k){
                if(sampleV_sol[i][j].empty()) continue;
                uniform_int_distribution<int> dis(0, (int)sampleV_sol[i][j].size() - 1);
                int n = dis(rng);
                tasks.push_back({i,j,k,&sampleV_sol[i][j][n],numeric_limits<double>::infinity()});
            }
        }
        vector<HyperSequence*> evalPtrs;
        evalPtrs.reserve(tasks.size());

        for(auto& t : tasks)
            evalPtrs.push_back(t.seq);

        parallelEvaluate(evalPtrs, "Update evaluation (parallel)");

        for(auto& t : tasks){
            t.fitness = t.seq->getFitness();
        }

        for(auto& t : tasks){
            double f = t.fitness;
            int i=t.i, j=t.j, k=t.k;
            if(f < searcher_sol[i].getFitness()){
                searcher_sol[i] = *t.seq;
                searcher_sol[i].setFitness(f);
                searcher_sol[i].setRegion(j);
            }
            if(f < sample_sol[j][k].getFitness()){
                sample_sol[j][k] = *t.seq;
                sample_sol[j][k].setFitness(f);
                sample_sol[j][k].setRegion(j);
            }
        }
    }
    if(total_progress > 0 && progress > 0){
        bar.finish();
    } else {
        bar.reset();
    }

    bar.set_label("Summing up region fitness");
    total_progress = num_regions;
    progress = 0;
    vector<double> regionBestFitness(num_regions, 0.0);
    double sumRegionFitness = 0.0;
    for (int j = 0; j < num_regions; j++) {
        double rbj = numeric_limits<double>::infinity();
        int b = -1;

        for (int k = 0; k < init_num_samples; k++) {
            sumRegionFitness += sample_sol[j][k].getFitness();
            // update fbj
            if (sample_sol[j][k].getFitness() < rbj) {
                b = k;
                rbj = sample_sol[j][k].getFitness();
            }
        }
        if (b >= 0) {
            regionBestFitness[j] = rbj;
            sample_sol_best[j] = sample_sol[j][b];
        }

        progress++;
        bar.progress(progress, total_progress);
    }
    if(sumRegionFitness <= 0.0) sumRegionFitness = 1.0;
    bar.finish();
    bar.reset() ;

    bar.set_label("Calculating expected values");
    total_progress = num_searchers * num_regions;
    progress = 0;

    for(int i=0;i<num_searchers;++i){
        for(int j=0;j<num_regions;++j){
            double f1 = double(taj[j]) / double(tbj[j]); // Tj
            double sumv = 0.0;
            // for(int k=0;k<num_samples;++k)
            //     sumv += evaluate_fitness(sampleV_sol[i][j][k]);
            // double f2 = double(num_samples) / sumv;
            double f3 = sumRegionFitness / regionBestFitness[j]; // M_j
            expectedValues[i][j] = f1 * f3;

            progress++;
            bar.progress(progress, total_progress);
        }
    }
    bar.finish();
    bar.reset();

    // debug: print expected values
    // for(int i=0;i<num_searchers;++i){
    //     for(int j=0;j<num_regions;++j){
    //         cerr << expectedValues[i][j] << "\t";
    //     }
    //     cerr << endl;
    // }
}

// evaluate_fitness (median) with early stopping
double SEHH::evaluate_fitness(HyperSequence& hs){ 
    
    vector<double> vals;
    vals.reserve(ITERATIONS_FOR_EVALUATION);
    int evalCount = 0;
    
    // 早停機制：如果有歷史最佳解，用它的倍數
    double earlyStopThreshold = numeric_limits<double>::infinity();
    if(!all_time_best_sequences.empty()){
        earlyStopThreshold = all_time_best_sequences[0].first * 3.0;  // 3 倍
    }
    const int earlyStopCheckPoint = max(1, ITERATIONS_FOR_EVALUATION / 3);  // 前 1/3 檢查
    double runningSum = 0.0;
    
    for(int iter = 0; iter < ITERATIONS_FOR_EVALUATION; ++iter){
        double total_fit = 0.0;
        try {
            for(int idx = 0; idx < (int)instances.size(); ++idx){
                const Instance& inst = instances[idx];
                int dim = inst.getDimension();
                int funcNo = inst.getFuncNo();
                if(dim <= 0 || funcNo <= 0){
                    throw runtime_error("Invalid instance parameters.");
                }
                uint32_t runSeed = computeRunSeed(dim, funcNo, iter + 1);
                pair<int, double> result = hs.calaulateDummyFitnessForTraining(inst, runSeed);
                total_fit += result.second;
            }
            total_fit /= instances.size();
        } catch(const exception& e){
            cerr << "[SEHH] evaluate exception: " << e.what() << endl;
            total_fit = numeric_limits<double>::infinity();
        }
        
        vals.push_back(total_fit);
        runningSum += total_fit;
        ++evalCount;
        
        if(iter >= earlyStopCheckPoint - 1 && earlyStopThreshold < numeric_limits<double>::infinity()){
            double avgSoFar = runningSum / (iter + 1);
            if(avgSoFar > earlyStopThreshold){
                break;
            }
        }
    }
    
    // 計算中位數
    double med = numeric_limits<double>::infinity();
    if(!vals.empty()){
        size_t m = vals.size()/2;
        nth_element(vals.begin(), vals.begin()+m, vals.end());
        if(vals.size() % 2 == 1){
            med = vals[m];
        }else{
            double a = *max_element(vals.begin(), vals.begin()+m);
            double b = vals[m];
            med = 0.5 * (a + b);
        }
    }

    evaluations.fetch_add(evalCount, memory_order_relaxed);
    return med;
}

double SEHH::evaluate_fitness_with_iteration(HyperSequence& hs){ 
    // 以 pair<avg_fitness, avg_best_iter> 收集每次重複的結果
    vector<pair<double,int>> vals;
    vals.reserve(ITERATIONS_FOR_EVALUATION);
    int evalCount = 0;

    for(int iter = 0; iter < ITERATIONS_FOR_EVALUATION; ++iter){
        double total_fit = 0.0;
        long long total_iter = 0; // 可能跨多 instance，先用較大型別

        try {
            for(int idx = 0; idx < (int)instances.size(); ++idx){
                const Instance& inst = instances[idx];
                int dim = inst.getDimension();
                int funcNo = inst.getFuncNo();
                if(dim <= 0 || funcNo <= 0){
                    throw runtime_error("Invalid instance parameters.");
                }
                uint32_t runSeed = computeRunSeed(dim, funcNo, iter + 1);
                auto result = hs.calaulateDummyFitnessForTraining(inst, runSeed);
                // result.first  = bestIter（越小越好）
                // result.second = dummyFitness（越小越好）
                total_fit  += result.second;
                total_iter += result.first;
            }

            // 對多個 instance 取平均
            double avg_fit  = total_fit  / max<size_t>(1, instances.size());
            int avg_iter = (int)(total_iter / max<size_t>(1, instances.size()));
            vals.emplace_back(avg_fit, avg_iter);

        } catch(const exception& e){
            cerr << "[SEHH] evaluate exception: " << e.what() << endl;
            // 失敗時給極差值，iteration 也置最大
            vals.emplace_back(numeric_limits<double>::infinity(), META_MAX_EVALUATIONS);
        }

        ++evalCount;
    }

    // 以先比 fitness，再比 iteration取中位數
    auto lexcmp = [](const pair<double,int>& a, const pair<double,int>& b){
        if (a.first < b.first) return true;
        if (a.first > b.first) return false;
        return a.second < b.second;
    };

    double med_combined = numeric_limits<double>::infinity();
    double med_iter_ratio = 0.0;
    if(!vals.empty()){
        size_t m = vals.size()/2;
        nth_element(vals.begin(), vals.begin()+m, vals.end(), lexcmp);

        auto to_scalar = [](const pair<double,int>& x){
            double r = clamp((double)x.second / (double)META_MAX_EVALUATIONS, 0.0, 1.0);
            return x.first + kEps * r;
        };
        auto iter_ratio_of = [](const pair<double,int>& x){
            return clamp((double)x.second / (double)META_MAX_EVALUATIONS, 0.0, 1.0);
        };

        if(vals.size() % 2 == 1){
            med_combined = to_scalar(vals[m]);
            med_iter_ratio = iter_ratio_of(vals[m]);
        }else{
            auto a_it = max_element(vals.begin(), vals.begin()+m, lexcmp);
            auto b     = vals[m];
            med_combined = 0.5 * (to_scalar(*a_it) + to_scalar(b));
            med_iter_ratio = 0.5 * (iter_ratio_of(*a_it) + iter_ratio_of(b));
        }
    }

    evaluations.fetch_add(evalCount, memory_order_relaxed);
    logIterRatio(med_iter_ratio);
    return med_combined;
}

// vision_selection
void SEHH::determination(){

    for (int j = 0; j < num_regions; j++)
        tbj[j]++;

    uniform_int_distribution<int> dist(0, num_regions - 1);

    // find index of the best vij
    for (int i = 0; i < num_searchers; i++) {
        int j = dist(rng);
        double ev = expectedValues[i][j];
        for (int p = 0; p < num_players-1; p++) {
            int c = dist(rng);
            if (expectedValues[i][c] > ev) {
                j = c;
                ev = expectedValues[i][j];
            }
        }

        // assign searcher i to region j
        searcher_region_id[i] = j;

        // update ta[j] and tb[j];
        taj[j]++;
        tbj[j] = 1;
    }
}

// marketing_survey
void SEHH::marketing_survey(){

    // 1. tbj > 1 的 region，ta[j] 歸 1
    for(int j=0;j<num_regions;++j){
        if(tbj[j] > 1)
            taj[j] = 1.0;
    }

    // 2. 更新最佳解
    const auto pre_all_time_best = all_time_best_sequences;
    for(int i=0;i<num_searchers;++i){
        double fit = searcher_sol[i].getFitness();
        if(all_time_best_sequences.size() < 3 || fit < all_time_best_sequences[2].first){
            all_time_best_sequences.push_back({fit, searcher_sol[i]});
            all_time_best_sequences.back().second.setRegion(calculateRegion(searcher_sol[i]));
            all_time_best_sequences.back().second.setFitness(fit);

            sort(all_time_best_sequences.begin(), all_time_best_sequences.end(), [](const auto& a, const auto& b){
                return a.first < b.first;
            });

            all_time_best_sequences.erase(unique(all_time_best_sequences.begin(), all_time_best_sequences.end(), [](const auto& a, const auto& b){
                return a.second == b.second;
            }), all_time_best_sequences.end());

            if(all_time_best_sequences.size() > 3)
                all_time_best_sequences.pop_back();
        }
    }
    for(int j=0;j<num_regions;++j){
        for(int k=0;k<init_num_samples;++k){
            double fit = sample_sol[j][k].getFitness();
            if(all_time_best_sequences.size() < 3 || fit < all_time_best_sequences[2].first){
                all_time_best_sequences.push_back({fit, sample_sol[j][k]});
                all_time_best_sequences.back().second.setRegion(j);
                all_time_best_sequences.back().second.setFitness(fit);

                sort(all_time_best_sequences.begin(), all_time_best_sequences.end(), [](const auto& a, const auto& b){
                    return a.first < b.first;
                });

                all_time_best_sequences.erase(unique(all_time_best_sequences.begin(), all_time_best_sequences.end(), [](const auto& a, const auto& b){
                    return a.second == b.second;
                }), all_time_best_sequences.end());

                if(all_time_best_sequences.size() > 3)
                    all_time_best_sequences.pop_back();
            }
        }
    }

    // 如果有改善，重置 howLongNoImprovement
    if(pre_all_time_best != all_time_best_sequences){
        howLongNoImprovement = -1; // 因為下面會 +1
        mutationRate = mutationLowerBound; // reset mutation rate
    }

}

inline vector<uint32_t> SEHH::loadPrecomputedSeeds(const string& path){
    if(path.empty()){
        return {};
    }

    ifstream in(path);
    if(!in.is_open()){
        cerr << "[SEHH] Failed to open seed file: " << path << endl;
        return {};
    }

    vector<uint32_t> seeds;
    seeds.reserve(1000);
    double value = 0.0;
    while(in >> value){
        seeds.push_back(static_cast<uint32_t>(value));
    }
    return seeds;
}

inline uint32_t SEHH::computeRunSeed(int problemSize, int funcNo, int runId) const {
    if(precomputedSeeds.empty()){
        throw runtime_error("[SEHH] No precomputed seeds available.");
    }

    const int runs = RUNS;
    long long seedIndex = static_cast<long long>(problemSize / 10) * funcNo * runs + runId - runs;
    seedIndex = ((seedIndex % 1000) + 1000) % 1000;
    return precomputedSeeds[seedIndex];
}

// helper: generate random HyperSequence with n segments
HyperSequence SEHH::generate_random_sequence(int nSegments){

    vector<Segment> segs;
    segs.reserve(nSegments);

    for(int i=0;i<nSegments;++i)
        segs.emplace_back(Segment());

    return HyperSequence(segs);
}

void SEHH::updateFile(const string &outFile){
    ofstream ofs(outFile, ios::trunc);
    if(!ofs){
        cerr << "[SEHH] cannot open output file: " << outFile << endl;
        return;
    } else {

        ofs << "================= Top Three Best Solutions ================\n";

        // Output best solution
        for(size_t i=0;i<all_time_best_sequences.size() && i<3;++i){
            ofs << "Rank " << (i+1) << " fitness=" << all_time_best_sequences[i].first << " region=" << metaMap[region_meta[all_time_best_sequences[i].second.getRegion()]] << "\n";
            ofs << all_time_best_sequences[i].second << "\n";
        }

        ofs << "================= All Searchers =================\n";

        // Output Every searcher
        for(int i=0;i<num_searchers;++i){
            ofs << "Searcher " << i << " fitness=" << searcher_sol[i].getFitness() << " region=" << metaMap[region_meta[searcher_sol[i].getRegion()]] << "\n";
            ofs << "Seaching region: " << metaMap[region_meta[searcher_region_id[i]]] << "\n";
            ofs << searcher_sol[i] << "\n";
        }
        ofs.close();
    }
}

inline void SEHH::logIterRatio(double iter_ratio){
    lock_guard<mutex> lk(iter_ratio_log_mu);
    ofstream ofs(iter_ratio_file_path, ios::app);
    if(!ofs.is_open()) return;
    ofs << evaluations.load() << " " << iter_ratio << "\n";
}

#endif /* SEHH_H */
