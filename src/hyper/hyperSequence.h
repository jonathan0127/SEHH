#ifndef HYPERSEQUENCE_H
#define HYPERSEQUENCE_H

#include "segment.h"
#include "metaheuristic/PSO.h"
#include "metaheuristic/GA.h"
#include "metaheuristic/DE.h"
#include "metaheuristic/LSHADE.h"
#include "metaheuristic/LSRTDE.h"
#include "metaheuristic/RDE.h"
#include "utils/instance.h"
#include "config/para_setting.h"
#include "utils/search_state.h"
#include <vector>
#include <variant>
#include <iostream>
#include <sstream>
#include <fstream>
#include <functional>
#include <cstdint>
#include <random>
#include <algorithm>
#include <numeric>
#include <cmath>

using namespace std;

// time_weight helpers
inline int clampTimeWeight(int w){
    if(w < TIME_WEIGHT_RANGE.first) return TIME_WEIGHT_RANGE.first;
    if(w > TIME_WEIGHT_RANGE.second) return TIME_WEIGHT_RANGE.second;
    return w;
}

inline vector<int> allocateEvaluationsByWeights(const vector<int>& rawWeights,
                                                     int totalEvaluations,
                                                     bool ensureMinOne = true){
    const int n = (int)rawWeights.size();
    vector<int> weights(n);
    long long sumW = 0;
    for(int i=0;i<n;++i){
        weights[i] = clampTimeWeight(rawWeights[i]);
        sumW += weights[i];
    }
    vector<int> result(n, 0);
    if(n==0 || totalEvaluations<=0){
        return result;
    }
    if(sumW<=0){
        // fallback: equal split
        int base = totalEvaluations / n;
        int rem = totalEvaluations - base*n;
        for(int i=0;i<n;++i){ result[i] = base + (i<rem); }
        return result;
    }
    if(ensureMinOne && totalEvaluations < n){
        // give one to top weights
        vector<int> idx(n); iota(idx.begin(), idx.end(), 0);
        stable_sort(idx.begin(), idx.end(), [&](int a,int b){
            if(weights[a] != weights[b]) return weights[a] > weights[b];
            return a < b;
        });
        for(int k=0;k<totalEvaluations;++k) result[idx[k]] = 1;
        return result;
    }
    const int reserve = ensureMinOne ? n : 0;
    const int alloc = totalEvaluations - reserve;
    result.assign(n, ensureMinOne ? 1 : 0);
    vector<double> residual(n,0.0);
    long long sumFloor = 0;
    for(int i=0;i<n;++i){
        const double quota = (double)alloc * (double)weights[i] / (double)sumW;
        const int f = (int)floor(quota);
        result[i] += f;
        residual[i] = quota - f;
        sumFloor += f;
    }
    int R = alloc - (int)sumFloor;
    if(R>0){
        vector<int> idx(n); iota(idx.begin(), idx.end(), 0);
        stable_sort(idx.begin(), idx.end(), [&](int a,int b){
            if(abs(residual[a]-residual[b]) > 1e-12) return residual[a] > residual[b];
            if(weights[a] != weights[b]) return weights[a] > weights[b];
            return a < b;
        });
        for(int k=0;k<R && k<n;++k) result[idx[k]] += 1;
    }
    int sumNow = accumulate(result.begin(), result.end(), 0);
    if(sumNow != totalEvaluations){
        int diff = totalEvaluations - sumNow;
        for(int i=0; diff>0 && i<n; ++i, --diff) result[i]++;
        for(int i=0; diff<0 && i<n; ++i){ if(result[i] > (ensureMinOne?1:0)){ result[i]--; ++diff; } }
    }
    return result;
}

class HyperSequence{
public:
    HyperSequence(){
        fitness = numeric_limits<double>::infinity(); 
        region = -1;
        length = 0;
        sequence.clear();
        midBest.clear();
        globalBestSolution.clear();
    }

    HyperSequence(const vector<Segment>& seq)
        : sequence(seq),
          region(-1),
          length((int)seq.size()),
          fitness(numeric_limits<double>::infinity()){
        midBest.clear();
        globalBestSolution.clear();
    }

    HyperSequence(string file){
        ifstream inFile(file);
        if (!inFile) {
            cerr << "無法開啟序列檔案: " << file << endl;
            return;
        };
        
        vector<vector<Segment>> sequences;
        string line;
        int numSequences = 0;
        int currentSequenceIndex = -1; // 從-1開始，因為讀到第一個序列標記時會+1變為0
        bool readingSequence = false;
        vector<Segment> currentSequence;

        // 讀取序列數量
        // getline(inFile, line);
        // numSequences = stoi(line);
        
        while (getline(inFile, line)) {
            // 跳過空行
            if (line.empty()) continue;
            
            // 讀取基因類型
            if ((line == "PSO" || line == "GA" || line == "DE" || line == "LSHADE" || line == "LSRTDE" || line == "RDE")) {
                string geneType = line;
                string paramsLine;
                string valuesLine;
                
                // 讀取參數名稱和值
                getline(inFile, paramsLine);
                getline(inFile, valuesLine);
                
                stringstream paramsSS(paramsLine);
                stringstream valuesSS(valuesLine);
                
                if (geneType == "PSO") {
                    string w_name, c1_name, c2_name, tw_name;
                    double w, c1, c2; int timeWeight;
                    paramsSS >> w_name >> c1_name >> c2_name >> tw_name;
                    valuesSS >> w >> c1 >> c2;
                    if(!(valuesSS >> timeWeight)){
                        timeWeight = (TIME_WEIGHT_RANGE.first + TIME_WEIGHT_RANGE.second)/2;
                    }
                    SEHH_PSOParams params = {w, c1, c2, clampTimeWeight(timeWeight)};
                    currentSequence.push_back(Segment(geneType, params));
                } 
                else if (geneType == "GA") {
                    string mr_name, cr_name, er_name, sm_name, cm_name, tw_name;
                    double mutationRate, crossoverRate, elitismRate;
                    int selectionMethod, crossoverMethod;
                    int timeWeight;
                    
                    paramsSS >> mr_name >> cr_name >> er_name >> sm_name >> cm_name >> tw_name;
                    valuesSS >> mutationRate >> crossoverRate >> elitismRate >> selectionMethod >> crossoverMethod;
                    if(!(valuesSS >> timeWeight)){
                        timeWeight = (TIME_WEIGHT_RANGE.first + TIME_WEIGHT_RANGE.second)/2;
                    }
                    
                    SEHH_GAParams params = {mutationRate, crossoverRate, elitismRate, selectionMethod, crossoverMethod, clampTimeWeight(timeWeight)};
                    currentSequence.push_back(Segment(geneType, params));
                }
                else if (geneType == "DE") {
                    string f_name, cr_name, strategy_name, tw_name;
                    double F, CR;
                    int strategy;
                    int timeWeight;
                    
                    paramsSS >> f_name >> cr_name >> strategy_name >> tw_name;
                    valuesSS >> F >> CR >> strategy;
                    if(!(valuesSS >> timeWeight)){
                        timeWeight = (TIME_WEIGHT_RANGE.first + TIME_WEIGHT_RANGE.second)/2;
                    }
                    
                    SEHH_DEParams params = {F, CR, strategy, clampTimeWeight(timeWeight)};
                    currentSequence.push_back(Segment(geneType, params));
                }
                else if (geneType == "LSHADE") {
                    string ms_name, cr_name, f_name, tw_name;
                    int memorySize;
                    double CR, F; int timeWeight;
                    
                    paramsSS >> ms_name >> cr_name >> f_name >> tw_name;
                    valuesSS >> memorySize >> CR >> F;
                    if(!(valuesSS >> timeWeight)){
                        timeWeight = (TIME_WEIGHT_RANGE.first + TIME_WEIGHT_RANGE.second)/2;
                    }
                    
                    SEHH_LSHADEParams params = {memorySize, CR, F, clampTimeWeight(timeWeight)};
                    currentSequence.push_back(Segment(geneType, params));
                }
                else if (geneType == "LSRTDE") {
                    string ms_name, sigmacr_name, sigmaf_name, tw_name;
                    int memorySize;
                    double sigmaCR, sigmaF; int timeWeight;

                    paramsSS >> ms_name >> sigmacr_name >> sigmaf_name >> tw_name;
                    valuesSS >> memorySize >> sigmaCR >> sigmaF;
                    if(!(valuesSS >> timeWeight)){
                        timeWeight = (TIME_WEIGHT_RANGE.first + TIME_WEIGHT_RANGE.second)/2;
                    }

                    SEHH_LSRTDEParams params = {memorySize, sigmaCR, sigmaF, clampTimeWeight(timeWeight)};
                    currentSequence.push_back(Segment(geneType, params));
                }
                else if (geneType == "RDE") {
                    string psize_name, gflag_name, ebflag_name, smode_name, tw_name;
                    double psizeParam, G_flag, EB_flag; int timeWeight;
                    int selectionMode;

                    paramsSS >> psize_name >> gflag_name >> ebflag_name >> smode_name >> tw_name;
                    valuesSS >> psizeParam >> G_flag >> EB_flag >> selectionMode;
                    if(!(valuesSS >> timeWeight)){
                        timeWeight = (TIME_WEIGHT_RANGE.first + TIME_WEIGHT_RANGE.second)/2;
                    }

                    SEHH_RDEParams params = {psizeParam, G_flag, EB_flag, selectionMode, clampTimeWeight(timeWeight)};
                    currentSequence.push_back(Segment(geneType, params));
                }
                else {
                    cerr << "未知的基因類型: " << geneType << endl;
                }
            }
        }
        
        if (!currentSequence.empty()) {
            sequences.push_back(currentSequence);
        }

        if (sequences.empty()) {
            cerr << "沒有從檔案中讀取到任何序列" << endl;
            return;
        }

        // 使用第一個序列作為預設
        this->sequence = sequences[0];
        this->length = (int)this->sequence.size();
        this->fitness = 0;
    }
    
    // Get the sequence
    const vector<Segment>& getSequence() const{
        return sequence;
    }

    void updateSegment(int index, const Segment& segment){
        if(index < 0 || index >= length){
            throw out_of_range("Index out of range");
        }
        sequence[index] = segment;
    }

    void setRegion(double reg){
        region = reg;
    }

    int getRegion() const{
        return region;
    }

    void setFitness(double fit){
        fitness = fit;
    }

    double getFitness() const{
        return fitness;
    }

    int getLength() const{
        return length;
    }

    vector<pair<int, double>> getMidBest() const{
        return midBest;
    }

    vector<double> getBestSolution() const{
        return this->globalBestSolution;
    }

    double calaulateDummyFitness(const Instance& instance, uint32_t seed = random_device{}()){
        midBest.clear();
        globalBestSolution.clear();
        // 防呆檢查
        if (length != (int)sequence.size()) {
            cerr << "[BUG] length != sequence.size(): " << length << " vs " << sequence.size() << endl;
            throw runtime_error("length and sequence.size() mismatch");
        }
        if(instance.getDimension() <= 0){
            cerr << "[BUG] instance.getDimension() <= 0" << endl;
            throw runtime_error("instance.getDimension() <= 0");
        }
        if(sequence.empty()){
            cerr << "[BUG] sequence is empty" << endl;
            throw runtime_error("sequence is empty");
        }

        // 加權分配每個 segment 的 evaluation 次數
        vector<int> weights; weights.reserve(length);
        for(int i=0;i<length;++i){
            const auto& seg = sequence.at(i);
            const auto& p = seg.getSegmentParams();
            int w = (TIME_WEIGHT_RANGE.first + TIME_WEIGHT_RANGE.second)/2;
            if(holds_alternative<SEHH_PSOParams>(p)) w = get<SEHH_PSOParams>(p).time_weight;
            else if(holds_alternative<SEHH_GAParams>(p)) w = get<SEHH_GAParams>(p).time_weight;
            else if(holds_alternative<SEHH_DEParams>(p)) w = get<SEHH_DEParams>(p).time_weight;
            else if(holds_alternative<SEHH_LSHADEParams>(p)) w = get<SEHH_LSHADEParams>(p).time_weight;
            else if(holds_alternative<SEHH_LSRTDEParams>(p)) w = get<SEHH_LSRTDEParams>(p).time_weight;
            else if(holds_alternative<SEHH_RDEParams>(p)) w = get<SEHH_RDEParams>(p).time_weight;
            weights.push_back(clampTimeWeight(w));
        }
        vector<int> evalCounts = allocateEvaluationsByWeights(weights, META_MAX_EVALUATIONS, true);
        const double populationStep = (double)(META_POPULATION_SIZE -  SEHH_POPULATION_FINAL) / length;

        // Deterministic per-segment seeding
        mt19937 segmentSeedGen(seed);


        // 防呆檢查: preBestSolution 維度
        vector<double> preBestSolution(instance.getDimension(), 0.0); // Store the best solutions of each gene
        if (preBestSolution.size() != (size_t)instance.getDimension()) {
            cerr << "[BUG] preBestSolution size != instance.getDimension()" << endl;
            throw runtime_error("preBestSolution size mismatch");
        }
        int accumulatedEvaluationCount = 0;
        vector<vector<double>> prevSolutions;
        vector<double> prevFitnesses;
        
        SearchState searchState;
        searchState.globalMaxEval = META_MAX_EVALUATIONS;
        searchState.globalCurrentEval = 0;
        searchState.initializeMemoryIfEmpty(5, 0.5, 0.5);
        
        for(int i = 0; i < length; ++i){
            const int segmentEvaluationNum = evalCounts[i];
            if (segmentEvaluationNum <= 0) {
                cerr << "[BUG] segmentEvaluationNum <= 0 after allocation at segment " << i << endl;
                throw runtime_error("segmentEvaluationNum <= 0 after allocation");
            }
            auto& segment = sequence.at(i);
            try {
                if (i >= (int)sequence.size()) {
                    cerr << "[BUG] segment index out of bounds: " << i << " >= " << sequence.size() << endl;
                    throw runtime_error("segment index out of bounds");
                }
                
                auto params = segment.getSegmentParams();
                int currentPopulationSize = META_POPULATION_SIZE - (int)round(populationStep * i);
                int nextPopulationSize = META_POPULATION_SIZE - (int)round(populationStep * (i + 1));
                // 這個只會在最後一個 segment 是 LSHADE 時候用到
                // 最後 Final Population 先設定為 5
                if(nextPopulationSize < 0) 
                    nextPopulationSize = 5; 
                // cerr << "length:" << length << endl;
                // cerr << "次數 : " << geneEvaluationNum.at(i) << endl;
                if(holds_alternative<SEHH_PSOParams>(params)){
                    uint32_t segmentSeed = segmentSeedGen();
                    auto psoParams = get<SEHH_PSOParams>(params);
                    
                    if (psoParams.w < 0 || psoParams.c1 < 0 || psoParams.c2 < 0) {
                        cerr << "[BUG] Invalid PSO params: w=" << psoParams.w << ", c1=" << psoParams.c1 << ", c2=" << psoParams.c2 << endl;
                        throw runtime_error("Invalid PSO params");
                    }
                    
                    PSO pso(
                        instance.getDimension(),
                        instance.getObjFunction(),
                        instance.getLowerBounds(),
                        instance.getUpperBounds(),
                        currentPopulationSize,
                        segmentEvaluationNum,
                        psoParams.w,
                        psoParams.c1,
                        psoParams.c2,
                        segmentSeed
                    );
                    // PSO 使用 prevSolutions 初始化（PSO 沒有 loadState）
                    if(i != 0) {
                        if(!prevSolutions.empty()) pso.setInitialSolution(preBestSolution, prevSolutions, prevFitnesses);
                        else pso.setInitialSolution(preBestSolution);
                    }
                    auto [bestSolution, bestFitness] = pso.run();
                    preBestSolution = bestSolution;
                    prevSolutions = pso.getFinalPopulation();
                    prevFitnesses = pso.getFinalFitnesses();
                    // 更新 SearchState（PSO 不修改 Memory/Archive）
                    searchState.population = prevSolutions;
                    searchState.fitness = prevFitnesses;
                    searchState.bestSolution = preBestSolution;
                    searchState.bestFitness = bestFitness;
                    searchState.globalCurrentEval += segmentEvaluationNum;
                    auto mB = pso.getMidBest();
                    for(auto& [iter, value] : mB) {
                        midBest.push_back({iter + accumulatedEvaluationCount, value});
                    }
                    accumulatedEvaluationCount += segmentEvaluationNum;
                    // midBest.push_back({accumulatedEvaluationCount, bestFitness});
                }else if(holds_alternative<SEHH_GAParams>(params)){
                    uint32_t segmentSeed = segmentSeedGen();
                    auto gaParams = get<SEHH_GAParams>(params);
                    GA ga(
                        instance.getDimension(),
                        instance.getObjFunction(),
                        instance.getLowerBounds(),
                        instance.getUpperBounds(),
                        currentPopulationSize,
                        gaParams.crossoverRate,
                        gaParams.mutationRate,
                        gaParams.elitismRate,
                        gaParams.selectionMethod,
                        gaParams.crossoverMethod,
                        segmentEvaluationNum,
                        segmentSeed
                    );
                    // GA 使用 prevSolutions 初始化（GA 沒有 loadState）
                    if(i != 0) {
                        if(!prevSolutions.empty()) ga.setInitialSolution(preBestSolution, prevSolutions, prevFitnesses);
                        else ga.setInitialSolution(preBestSolution);
                    }
                    auto [bestSolution, bestFitness] = ga.run();
                    preBestSolution = bestSolution;
                    prevSolutions = ga.getFinalPopulation();
                    prevFitnesses = ga.getFinalFitnesses();
                    // 更新 SearchState（GA 不修改 Memory/Archive）
                    searchState.population = prevSolutions;
                    searchState.fitness = prevFitnesses;
                    searchState.bestSolution = preBestSolution;
                    searchState.bestFitness = bestFitness;
                    searchState.globalCurrentEval += segmentEvaluationNum;
                    auto mB = ga.getMidBest();
                    for(auto& [iter, value] : mB) {
                        midBest.push_back({iter + accumulatedEvaluationCount, value});
                    }
                    accumulatedEvaluationCount += segmentEvaluationNum;
                    // midBest.push_back({accumulatedEvaluationCount, bestFitness});
                }else if(holds_alternative<SEHH_DEParams>(params)){
                    uint32_t segmentSeed = segmentSeedGen();
                    auto deParams = get<SEHH_DEParams>(params);
                    DE de(
                        instance.getDimension(),
                        instance.getObjFunction(),
                        instance.getLowerBounds(),
                        instance.getUpperBounds(),
                        deParams.strategy,
                        currentPopulationSize,
                        segmentEvaluationNum,
                        deParams.F,
                        deParams.CR,
                        segmentSeed
                    );
                    // 載入狀態
                    if (!searchState.isEmpty()) {
                        de.loadState(searchState);
                    } else if(i != 0) {
                        if(!prevSolutions.empty()) de.setInitialSolution(preBestSolution, prevSolutions, prevFitnesses);
                        else de.setInitialSolution(preBestSolution);
                    }
                    auto [bestSolution, bestFitness] = de.run();
                    preBestSolution = bestSolution;
                    prevSolutions = de.getFinalPopulation();
                    prevFitnesses = de.getFinalFitnesses();
                    de.saveState(searchState);
                    auto mB = de.getMidBest();
                    for(auto& [iter, value] : mB) {
                        midBest.push_back({iter + accumulatedEvaluationCount, value});
                    }
                    accumulatedEvaluationCount += segmentEvaluationNum;
                    // midBest.push_back({accumulatedEvaluationCount, bestFitness});
                }else if(holds_alternative<SEHH_LSHADEParams>(params)){
                    uint32_t segmentSeed = segmentSeedGen();
                    auto lshadeParams = get<SEHH_LSHADEParams>(params);
                    LSHADE lshade(
                        instance.getDimension(),
                        instance.getObjFunction(),
                        instance.getLowerBounds(),
                        instance.getUpperBounds(),
                        currentPopulationSize,  // 使用固定的族群大小
                        nextPopulationSize,
                        segmentEvaluationNum,
                        lshadeParams.memorySize,
                        lshadeParams.F,        // 使用自定義的 F 值
                        lshadeParams.CR,       // 使用自定義的 CR 值
                        segmentSeed
                    );
                    lshade.setGlobalProgress(searchState.globalCurrentEval, searchState.globalMaxEval);
                    if (!searchState.isEmpty()) {
                        lshade.loadState(searchState);
                    } else if(i != 0) {
                        if(!prevSolutions.empty()) lshade.setInitialSolution(preBestSolution, prevSolutions, prevFitnesses);
                        else lshade.setInitialSolution(preBestSolution);
                    }
                    auto [bestSolution, bestFitness] = lshade.run();
                    preBestSolution = bestSolution;
                    prevSolutions = lshade.getFinalPopulation();
                    prevFitnesses = lshade.getFinalFitnesses();
                    lshade.saveState(searchState);
                    auto mB = lshade.getMidBest();
                    for(auto& [iter, value] : mB) {
                        midBest.push_back({iter + accumulatedEvaluationCount, value});
                    }
                    accumulatedEvaluationCount += segmentEvaluationNum;
                    // midBest.push_back({accumulatedEvaluationCount, bestFitness});
                }else if(holds_alternative<SEHH_LSRTDEParams>(params)){
                    uint32_t segmentSeed = segmentSeedGen();
                    auto lsrtdeParams = get<SEHH_LSRTDEParams>(params);
                    LSRTDE lsrtde(
                        instance.getDimension(),
                        instance.getObjFunction(),
                        instance.getLowerBounds(),
                        instance.getUpperBounds(),
                        currentPopulationSize,
                        nextPopulationSize,
                        segmentEvaluationNum,
                        lsrtdeParams.memorySize,
                        lsrtdeParams.sigmaCR,
                        lsrtdeParams.sigmaF,
                        segmentSeed
                    );
                    if(i != 0) {
                        if(!prevSolutions.empty()) lsrtde.setInitialSolution(preBestSolution, prevSolutions, prevFitnesses);
                        else lsrtde.setInitialSolution(preBestSolution);
                    }
                    auto [bestSolution, bestFitness] = lsrtde.run();
                    preBestSolution = bestSolution;
                    prevSolutions = lsrtde.getFinalPopulation();
                    prevFitnesses = lsrtde.getFinalFitnesses();
                    searchState.population = prevSolutions;
                    searchState.fitness = prevFitnesses;
                    searchState.bestSolution = preBestSolution;
                    searchState.bestFitness = bestFitness;
                    searchState.globalCurrentEval += segmentEvaluationNum;
                    auto mB = lsrtde.getMidBest();
                    for(auto& [iter, value] : mB) {
                        midBest.push_back({iter + accumulatedEvaluationCount, value});
                    }
                    accumulatedEvaluationCount += segmentEvaluationNum;
                } else if(holds_alternative<SEHH_RDEParams>(params)){
                    uint32_t segmentSeed = segmentSeedGen();
                    auto rdeParams = get<SEHH_RDEParams>(params);
                    RDE rde(
                        instance.getDimension(),
                        instance.getObjFunction(),
                        instance.getLowerBounds(),
                        instance.getUpperBounds(),
                        currentPopulationSize,
                        segmentEvaluationNum,
                        rdeParams.psizeParam,
                        rdeParams.G_flag,
                        rdeParams.EB_flag,
                        rdeParams.selectionMode,
                        segmentSeed
                    );
                    rde.setGlobalProgress(searchState.globalCurrentEval, searchState.globalMaxEval);
                    if (!searchState.isEmpty()) {
                        rde.loadState(searchState);
                    } else if(i != 0) {
                        if(!prevSolutions.empty()) rde.setInitialSolution(preBestSolution, prevSolutions, prevFitnesses);
                        else rde.setInitialSolution(preBestSolution);
                    }
                    auto [bestSolution, bestFitness] = rde.run();
                    preBestSolution = bestSolution;
                    prevSolutions = rde.getFinalPopulation();
                    prevFitnesses = rde.getFinalFitnesses();
                    rde.saveState(searchState);
                    auto mB = rde.getMidBest();
                    for(auto& [iter, value] : mB) {
                        midBest.push_back({iter + accumulatedEvaluationCount, value});
                    }
                    accumulatedEvaluationCount += segmentEvaluationNum;
                } else {
                    cerr << "[BUG] Unknown sequence params type at sequence " << i << endl;
                    throw runtime_error("Unknown sequence params type");
                }
                // cerr << midBest.size() << endl;
                // cerr << "population after segment " << i << ": " << currentPopulationSize << endl;
            } catch(const exception& e) {
                cerr << "[BUG] sequence metaheuristic exception at sequence " << i << endl;
                cerr << "  Template name: " << segment.getTemplateName() << endl;
                cerr << "  Instance dimension: " << instance.getDimension() << endl;
                cerr << "  segmentEvaluationNum: " << segmentEvaluationNum << endl;
                cerr << "  preBestSolution size: " << preBestSolution.size() << endl;
                cerr << "  Exception: " << e.what() << endl;
                cerr << "  Exception type: " << typeid(e).name() << endl;
                throw;
            }
            if(i != length - 1){
                midBest.pop_back(); // 移除最後一個，以避免重複記錄
            }
            // cerr << "population after segment " << i << ": " << currentPopulationSize << endl;
        }
        // cerr << "midBest size: " << midBest.size() << endl;
        if (preBestSolution.size() != (size_t)instance.getDimension()) {
            cerr << "[BUG] preBestSolution size != instance.getDimension() before evaluate" << endl;
            throw runtime_error("preBestSolution size mismatch before evaluate");
        }
        this->globalBestSolution = preBestSolution;
        double dummyFitness = 0;
        try {
            dummyFitness = instance.evaluate(preBestSolution);
        } catch(const exception& e) {
            cerr << "[BUG] instance.evaluate(preBestSolution) exception: " << e.what() << endl;
            throw;
        }
        return dummyFitness;
    }

    pair<int, double> calaulateDummyFitnessForTraining(const Instance& instance, uint32_t seed = random_device{}()) const{
        // midBest.clear();
        // globalBestSolution.clear();
        if (length != (int)sequence.size()) {
            cerr << "[BUG] length != sequence.size(): " << length << " vs " << sequence.size() << endl;
            throw runtime_error("length and sequence.size() mismatch");
        }
        if(instance.getDimension() <= 0){
            cerr << "[BUG] instance.getDimension() <= 0" << endl;
            throw runtime_error("instance.getDimension() <= 0");
        }
        if(sequence.empty()){
            cerr << "[BUG] sequence is empty" << endl;
            throw runtime_error("sequence is empty");
        }

        vector<int> weights; weights.reserve(length);
        for(int i=0;i<length;++i){
            const auto& seg = sequence.at(i);
            const auto& p = seg.getSegmentParams();
            int w = (TIME_WEIGHT_RANGE.first + TIME_WEIGHT_RANGE.second)/2;
            if(holds_alternative<SEHH_PSOParams>(p)) w = get<SEHH_PSOParams>(p).time_weight;
            else if(holds_alternative<SEHH_GAParams>(p)) w = get<SEHH_GAParams>(p).time_weight;
            else if(holds_alternative<SEHH_DEParams>(p)) w = get<SEHH_DEParams>(p).time_weight;
            else if(holds_alternative<SEHH_LSHADEParams>(p)) w = get<SEHH_LSHADEParams>(p).time_weight;
            else if(holds_alternative<SEHH_LSRTDEParams>(p)) w = get<SEHH_LSRTDEParams>(p).time_weight;
            else if(holds_alternative<SEHH_RDEParams>(p)) w = get<SEHH_RDEParams>(p).time_weight;
            weights.push_back(clampTimeWeight(w));
        }
        vector<int> evalCounts = allocateEvaluationsByWeights(weights, META_MAX_EVALUATIONS, true);
        const double populationStep = (double)(META_POPULATION_SIZE -  SEHH_POPULATION_FINAL) / length;

        mt19937 segmentSeedGen(seed);

        vector<double> preBestSolution(instance.getDimension(), 0.0); // Store the best solutions of each gene
        if (preBestSolution.size() != (size_t)instance.getDimension()) {
            cerr << "[BUG] preBestSolution size != instance.getDimension()" << endl;
            throw runtime_error("preBestSolution size mismatch");
        }
        int accumulatedEvaluationCount = 0;
        int funcChoice = instance.getFuncNo();
        if(funcChoice < 0) throw runtime_error("Invalid function choice number");
        
        int bestIter = 0;
        bool foundBest = false;
        vector<vector<double>> prevSolutions;
        vector<double> prevFitnesses;
        
        SearchState searchState;
        searchState.globalMaxEval = META_MAX_EVALUATIONS;
        searchState.globalCurrentEval = 0;
        searchState.initializeMemoryIfEmpty(5, 0.5, 0.5);
        
        for(int i = 0; i < length; ++i){
            const int segmentEvaluationNum = evalCounts[i];
            if (segmentEvaluationNum <= 0) {
                cerr << "[BUG] segmentEvaluationNum <= 0 after allocation at segment " << i << endl;
                throw runtime_error("segmentEvaluationNum <= 0 after allocation");
            }
            auto& segment = sequence.at(i);
            try {
                if (i >= (int)sequence.size()) {
                    cerr << "[BUG] segment index out of bounds: " << i << " >= " << sequence.size() << endl;
                    throw runtime_error("segment index out of bounds");
                }
                
                auto params = segment.getSegmentParams();
                int currentPopulationSize = META_POPULATION_SIZE - (int)round(populationStep * i);
                int nextPopulationSize = META_POPULATION_SIZE - (int)round(populationStep * (i + 1));

                if(nextPopulationSize < 0) 
                    nextPopulationSize = 5; 
                // cerr << "length:" << length << endl;
                // cerr << "次數 : " << geneEvaluationNum.at(i) << endl;
                
                if(holds_alternative<SEHH_PSOParams>(params)){
                    uint32_t segmentSeed = segmentSeedGen();
                    auto psoParams = get<SEHH_PSOParams>(params);
                    
                    if (psoParams.w < 0 || psoParams.c1 < 0 || psoParams.c2 < 0) {
                        cerr << "[BUG] Invalid PSO params: w=" << psoParams.w << ", c1=" << psoParams.c1 << ", c2=" << psoParams.c2 << endl;
                        throw runtime_error("Invalid PSO params");
                    }
                    
                    PSO pso(
                        instance.getDimension(),
                        instance.getObjFunction(),
                        instance.getLowerBounds(),
                        instance.getUpperBounds(),
                        currentPopulationSize,
                        segmentEvaluationNum,
                        psoParams.w,
                        psoParams.c1,
                        psoParams.c2,
                        segmentSeed
                    );
                    if(i != 0) {
                        if(!prevSolutions.empty()) pso.setInitialSolution(preBestSolution, prevSolutions, prevFitnesses);
                        else pso.setInitialSolution(preBestSolution);
                    }
                    auto [bestSolution, bestFitness] = pso.run();
                    preBestSolution = bestSolution;
                    prevSolutions = pso.getFinalPopulation();
                    prevFitnesses = pso.getFinalFitnesses();
                    searchState.population = prevSolutions;
                    searchState.fitness = prevFitnesses;
                    searchState.bestSolution = preBestSolution;
                    searchState.bestFitness = bestFitness;
                    searchState.globalCurrentEval += segmentEvaluationNum;
                    auto mB = pso.getMidBest();
                    for(auto& [iter, value] : mB) {
                        if(!foundBest && value <= EQUAL_ZERO_THRESHOLD){
                            bestIter = iter + accumulatedEvaluationCount;
                            foundBest = true;
                        }
                    }
                    accumulatedEvaluationCount += segmentEvaluationNum;
                }else if(holds_alternative<SEHH_GAParams>(params)){
                    uint32_t segmentSeed = segmentSeedGen();
                    auto gaParams = get<SEHH_GAParams>(params);
                    GA ga(
                        instance.getDimension(),
                        instance.getObjFunction(),
                        instance.getLowerBounds(),
                        instance.getUpperBounds(),
                        currentPopulationSize,
                        gaParams.crossoverRate,
                        gaParams.mutationRate,
                        gaParams.elitismRate,
                        gaParams.selectionMethod,
                        gaParams.crossoverMethod,
                        segmentEvaluationNum,
                        segmentSeed
                    );
                    if(i != 0) {
                        if(!prevSolutions.empty()) ga.setInitialSolution(preBestSolution, prevSolutions, prevFitnesses);
                        else ga.setInitialSolution(preBestSolution);
                    }
                    auto [bestSolution, bestFitness] = ga.run();
                    preBestSolution = bestSolution;
                    prevSolutions = ga.getFinalPopulation();
                    prevFitnesses = ga.getFinalFitnesses();
                    searchState.population = prevSolutions;
                    searchState.fitness = prevFitnesses;
                    searchState.bestSolution = preBestSolution;
                    searchState.bestFitness = bestFitness;
                    searchState.globalCurrentEval += segmentEvaluationNum;
                    auto mB = ga.getMidBest();
                    for(auto& [iter, value] : mB) {
                        // midBest.push_back({iter + accumulatedEvaluationCount, value});
                        if(!foundBest && value <= EQUAL_ZERO_THRESHOLD){
                            bestIter = iter + accumulatedEvaluationCount;
                            foundBest = true;
                        }
                    }
                    accumulatedEvaluationCount += segmentEvaluationNum;
                }else if(holds_alternative<SEHH_DEParams>(params)){
                    uint32_t segmentSeed = segmentSeedGen();
                    auto deParams = get<SEHH_DEParams>(params);
                    DE de(
                        instance.getDimension(),
                        instance.getObjFunction(),
                        instance.getLowerBounds(),
                        instance.getUpperBounds(),
                        deParams.strategy,
                        currentPopulationSize,
                        segmentEvaluationNum,
                        deParams.F,
                        deParams.CR,
                        segmentSeed
                    );
                    if (!searchState.isEmpty()) {
                        de.loadState(searchState);
                    } else if(i != 0) {
                        if(!prevSolutions.empty()) de.setInitialSolution(preBestSolution, prevSolutions, prevFitnesses);
                        else de.setInitialSolution(preBestSolution);
                    }
                    auto [bestSolution, bestFitness] = de.run();
                    preBestSolution = bestSolution;
                    prevSolutions = de.getFinalPopulation();
                    prevFitnesses = de.getFinalFitnesses();
                    de.saveState(searchState);
                    auto mB = de.getMidBest();
                    for(auto& [iter, value] : mB) {
                        // midBest.push_back({iter + accumulatedEvaluationCount, value});
                        if(!foundBest && value <= EQUAL_ZERO_THRESHOLD){
                            bestIter = iter + accumulatedEvaluationCount;
                            foundBest = true;
                        }
                    }
                    accumulatedEvaluationCount += segmentEvaluationNum;
                    // midBest.push_back({accumulatedEvaluationCount, bestFitness});
                }else if(holds_alternative<SEHH_LSHADEParams>(params)){
                    uint32_t segmentSeed = segmentSeedGen();
                    auto lshadeParams = get<SEHH_LSHADEParams>(params);
                    LSHADE lshade(
                        instance.getDimension(),
                        instance.getObjFunction(),
                        instance.getLowerBounds(),
                        instance.getUpperBounds(),
                        currentPopulationSize,  
                        nextPopulationSize,
                        segmentEvaluationNum,
                        lshadeParams.memorySize,
                        lshadeParams.F,
                        lshadeParams.CR,
                        segmentSeed
                    );
                    lshade.setGlobalProgress(searchState.globalCurrentEval, searchState.globalMaxEval);
                    if (!searchState.isEmpty()) {
                        lshade.loadState(searchState);
                    } else if(i != 0) {
                        if(!prevSolutions.empty()) lshade.setInitialSolution(preBestSolution, prevSolutions, prevFitnesses);
                        else lshade.setInitialSolution(preBestSolution);
                    }
                    auto [bestSolution, bestFitness] = lshade.run();
                    preBestSolution = bestSolution;
                    prevSolutions = lshade.getFinalPopulation();
                    prevFitnesses = lshade.getFinalFitnesses();
                    lshade.saveState(searchState);
                    auto mB = lshade.getMidBest();
                    for(auto& [iter, value] : mB) {
                        if(!foundBest && value <= EQUAL_ZERO_THRESHOLD){
                            bestIter = iter + accumulatedEvaluationCount;
                            foundBest = true;
                        }
                    }
                    accumulatedEvaluationCount += segmentEvaluationNum;
                    // midBest.push_back({accumulatedEvaluationCount, bestFitness});
                }else if(holds_alternative<SEHH_LSRTDEParams>(params)){
                    uint32_t segmentSeed = segmentSeedGen();
                    auto lsrtdeParams = get<SEHH_LSRTDEParams>(params);
                    LSRTDE lsrtde(
                        instance.getDimension(),
                        instance.getObjFunction(),
                        instance.getLowerBounds(),
                        instance.getUpperBounds(),
                        currentPopulationSize,
                        nextPopulationSize,
                        segmentEvaluationNum,
                        lsrtdeParams.memorySize,
                        lsrtdeParams.sigmaCR,
                        lsrtdeParams.sigmaF,
                        segmentSeed
                    );
                    if(i != 0) {
                        if(!prevSolutions.empty()) lsrtde.setInitialSolution(preBestSolution, prevSolutions, prevFitnesses);
                        else lsrtde.setInitialSolution(preBestSolution);
                    }
                    auto [bestSolution, bestFitness] = lsrtde.run();
                    preBestSolution = bestSolution;
                    prevSolutions = lsrtde.getFinalPopulation();
                    prevFitnesses = lsrtde.getFinalFitnesses();
                    searchState.population = prevSolutions;
                    searchState.fitness = prevFitnesses;
                    searchState.bestSolution = preBestSolution;
                    searchState.bestFitness = bestFitness;
                    searchState.globalCurrentEval += segmentEvaluationNum;
                    auto mB = lsrtde.getMidBest();
                    for(auto& [iter, value] : mB) {
                        // midBest.push_back({iter + accumulatedEvaluationCount, value});
                        if(!foundBest && value <= EQUAL_ZERO_THRESHOLD){
                            bestIter = iter + accumulatedEvaluationCount;
                            foundBest = true;
                        }
                    }
                    accumulatedEvaluationCount += segmentEvaluationNum;
                } else if(holds_alternative<SEHH_RDEParams>(params)){
                    uint32_t segmentSeed = segmentSeedGen();
                    auto rdeParams = get<SEHH_RDEParams>(params);
                    RDE rde(
                        instance.getDimension(),
                        instance.getObjFunction(),
                        instance.getLowerBounds(),
                        instance.getUpperBounds(),
                        currentPopulationSize,
                        segmentEvaluationNum,
                        rdeParams.psizeParam,
                        rdeParams.G_flag,
                        rdeParams.EB_flag,
                        rdeParams.selectionMode,
                        segmentSeed
                    );
                    rde.setGlobalProgress(searchState.globalCurrentEval, searchState.globalMaxEval);
                    if (!searchState.isEmpty()) {
                        rde.loadState(searchState);
                    } else if(i != 0) {
                        if(!prevSolutions.empty()) rde.setInitialSolution(preBestSolution, prevSolutions, prevFitnesses);
                        else rde.setInitialSolution(preBestSolution);
                    }
                    auto [bestSolution, bestFitness] = rde.run();
                    preBestSolution = bestSolution;
                    prevSolutions = rde.getFinalPopulation();
                    prevFitnesses = rde.getFinalFitnesses();
                    // 保存狀態（包含 Memory、Archive、EB_hybrid_rate）
                    rde.saveState(searchState);
                    auto mB = rde.getMidBest();
                    for(auto& [iter, value] : mB) {
                        if(!foundBest && value <= EQUAL_ZERO_THRESHOLD){
                            bestIter = iter + accumulatedEvaluationCount;
                            foundBest = true;
                        }
                    }
                    accumulatedEvaluationCount += segmentEvaluationNum;
                } else {
                    cerr << "[BUG] Unknown sequence params type at sequence " << i << endl;
                    throw runtime_error("Unknown sequence params type");
                }
            } catch(const exception& e) {
                cerr << "[BUG] sequence metaheuristic exception at sequence " << i << endl;
                cerr << "  Template name: " << segment.getTemplateName() << endl;
                cerr << "  Instance dimension: " << instance.getDimension() << endl;
                cerr << "  segmentEvaluationNum: " << segmentEvaluationNum << endl;
                cerr << "  preBestSolution size: " << preBestSolution.size() << endl;
                cerr << "  Exception: " << e.what() << endl;
                cerr << "  Exception type: " << typeid(e).name() << endl;
                throw;
            }
            // if(i != length - 1){
            //     midBest.pop_back(); // 移除最後一個，以避免重複記錄
            // }
        }
        if(!foundBest){
            bestIter = META_MAX_EVALUATIONS;
        }
        // cerr << "midBest size: " << midBest.size() << endl;

        if (preBestSolution.size() != (size_t)instance.getDimension()) {
            cerr << "[BUG] preBestSolution size != instance.getDimension() before evaluate" << endl;
            throw runtime_error("preBestSolution size mismatch before evaluate");
        }
        // this->globalBestSolution = preBestSolution;
        double dummyFitness = 0;
        try {
            dummyFitness = instance.evaluate(preBestSolution);
            if(dummyFitness < EQUAL_ZERO_THRESHOLD){
                dummyFitness = 0.0;
            }
        } catch(const exception& e) {
            cerr << "[BUG] instance.evaluate(preBestSolution) exception: " << e.what() << endl;
            throw;
        }
        return {bestIter, dummyFitness};
    }

    // fitness 大的排前面
    bool operator < (const HyperSequence& other) const{
        return fitness > other.fitness;
    }

    // Equality operator for HyperSequence
    bool operator==(const HyperSequence& other) const {
        return sequence == other.sequence &&
               region == other.region &&
               fitness == other.fitness &&
               length == other.length;
    }
private:
    vector<Segment> sequence;
    int region;
    vector<pair<int, double>> midBest;
    vector<double> globalBestSolution;
    double fitness; 
    int length;
};


ostream& operator<<(ostream& os, const HyperSequence& sequence) {
    os << "Sequence:\n";
    for (const auto& segment : sequence.getSequence()) {
        os << segment << "\n";
    }
    return os;
}

#endif /* HYPERSEQUENCE_H */