#ifndef LSHADE_H
#define LSHADE_H

#include<iostream>
#include<vector>
#include<unordered_set>
#include<string>
#include<random>
#include<cmath>
#include<iomanip>
#include "utils/tqdm.h"
#include "config/para_setting.h"
#include "utils/search_state.h"
#include<cstdint>
#include "utils/sample.h"

using namespace std;
using solution_t = vector<double>;
using ans_t = pair<solution_t, double>; // Solution and its fitness

typedef struct {
    solution_t solution; 
    double fitness;
} individual_t;

class LSHADE {

public:
    LSHADE(
        int dimension,
        const function<double(vector<double>&)>& objFunction,
        const vector<double>& lowerBounds,
        const vector<double>& upperBounds,
        int populationSize,
        int finalPopulationSize,
        int maxEvaluations,
        int memorySize,
        double initF = 0.5,
        double initCR = 0.5,
        uint32_t seed = random_device{}());

    void setInitialSolution(const solution_t& initialSolution);
    void setInitialSolution(const solution_t& best, const vector<solution_t>& candidates, const vector<double>& candidateFitness);
    
    void loadState(const SearchState& state);
    void saveState(SearchState& state) const;
    void setGlobalProgress(int globalCurrentEval, int globalMaxEval);
    
    ans_t run(bool);

    ans_t getBestSolution() const;
    vector<pair<int, double>> getMidBest() const;
    const vector<solution_t>& getFinalPopulation() const { return finalPopulation; }
    const vector<double>& getFinalFitnesses() const { return finalFitnesses; }

private:
    int dimension;
    function<double(vector<double>&)> objFunction;
    vector<double> lowerBounds;
    vector<double> upperBounds;

    int populationSize;
    int finalPopulationSize;
    int maxEvaluations;
    int memorySize;
    int memoryIndex;
    int pbestSize;
    vector<double> F;
    vector<double> CR;

    vector<individual_t> population;
    vector<pair<int, double>> midBest;
    solution_t bestSolution;
    double bestFitness;
    int evaluationCount;
    vector<solution_t> finalPopulation;
    vector<double> finalFitnesses;

    mt19937 gen;
    uniform_real_distribution<double> dist;
    
    // 全域進度追蹤（用於跨 segment 的 LPSR）
    int globalCurrentEval{0};
    int globalMaxEval{0};
    bool useGlobalProgress{false};

    void initialize();
    solution_t mutate(int currentIndex, solution_t pbest, double F_value);
    solution_t crossover(const vector<double>& target, const vector<double>& mutant, double CR_value);
    void selection(int index, vector<double>& trial);

    solution_t pickPBest(int newPopulationSize);
    double generateF(int F_index);
    double generateCR(int CR_index);
};

LSHADE::LSHADE(int dimension,
                const function<double(vector<double>&)>& objFunction,
                const vector<double>& lowerBounds,
                const vector<double>& upperBounds,
                int populationSize,
                int finalPopulationSize,
                int maxEvaluations,
                int memorySize,
                double initF,
                double initCR,
                uint32_t seed) :
                dimension(dimension),
                objFunction(objFunction),
                lowerBounds(lowerBounds),
                upperBounds(upperBounds),
                populationSize(populationSize),
                finalPopulationSize(finalPopulationSize),
                maxEvaluations(maxEvaluations),
                memorySize(memorySize),
                bestFitness(numeric_limits<double>::infinity()),
                gen(seed),
                dist(0.0, 1.0) {

    // 使用自定義的初始 F 和 CR 值
    F = vector<double>(memorySize, initF);
    CR = vector<double>(memorySize, initCR);
    initialize();
}

void LSHADE::setInitialSolution(const solution_t& initialSolution) {
    if (!population.empty() && !initialSolution.empty() && initialSolution.size() == dimension) {
        // 將初始解設定為第一個個體
        population[0].solution = initialSolution;
        vector<double> tempSolution = population[0].solution; // 建立非 const 副本
        population[0].fitness = objFunction(tempSolution);
        
        // 更新全域最佳解
        if (population[0].fitness < bestFitness) {
            bestFitness = population[0].fitness;
            bestSolution = population[0].solution;
        }
    }
}

void LSHADE::setInitialSolution(const solution_t& best,const vector<solution_t>& candidates,const vector<double>& candidateFitness){
    // Build a new initial population using best + roulette from candidates
    if(population.empty()) return;
    auto clampv = [&](solution_t v){
        for(int d=0; d<dimension; ++d){ v[d] = max(lowerBounds[d], min(upperBounds[d], v[d])); }
        return v;
    };
    vector<individual_t> newPop(populationSize, individual_t{solution_t(dimension), numeric_limits<double>::infinity()});
    newPop[0].solution = clampv(best);
    {
        auto tmp = newPop[0].solution; newPop[0].fitness = objFunction(tmp);
        if(newPop[0].fitness < bestFitness){ bestFitness = newPop[0].fitness; bestSolution = newPop[0].solution; }
    }
    int need = max(0, populationSize - 1);
    int n = (int)candidates.size();
    vector<double> weights;
    if(n>0 && candidateFitness.size()==candidates.size()){
        double fmin = numeric_limits<double>::infinity();
        for(double f : candidateFitness) if(isfinite(f)) fmin = min(fmin, f);
        if(!isfinite(fmin)) fmin = 0.0;
        weights.reserve(n);
        for(double f : candidateFitness){
            double w = 1.0/(1e-12 + max(0.0, f - fmin)); if(!isfinite(w)||w<0.0) w=0.0; weights.push_back(w);
        }
    }
    mt19937 rng(gen());
    vector<int> picks;
    if(!weights.empty()){
        // Allow duplicates in roulette selection
        picks = roulette_select_indices(weights, need, false, rng);
    }else if(n>0){
        uniform_int_distribution<int> uni(0, n-1); picks.resize(need); for(int i=0;i<need;++i) picks[i]=uni(rng);
    }
    for(int i=0;i<need;++i){
        solution_t cand = (n>0 && !picks.empty()) ? candidates[picks[i % (int)picks.size()]] : solution_t(dimension,0.0);
        if((int)cand.size()!=dimension){
            cand.assign(dimension,0.0); for(int d=0; d<dimension; ++d){ uniform_real_distribution<double> dis(lowerBounds[d], upperBounds[d]); cand[d]=dis(gen);}        
        }
        cand = clampv(cand);
        newPop[i+1].solution = cand;
        auto tmp = newPop[i+1].solution; newPop[i+1].fitness = objFunction(tmp);
        if(newPop[i+1].fitness < bestFitness){ bestFitness = newPop[i+1].fitness; bestSolution = newPop[i+1].solution; }
    }
    population.swap(newPop);
}

pair<vector<double>, double> LSHADE::run(bool showProgess = false){
    
    // Main iteration loop
    tqdm bar;
    while(evaluationCount < maxEvaluations){
        if(showProgess) bar.progress(evaluationCount, maxEvaluations);

        vector<double> SF, SCR, dF;
        
        // 計算 LPSR 進度：優先使用全域進度，否則使用本地進度
        double lpsrProgress;
        if (useGlobalProgress && globalMaxEval > 0) {
            // 使用全域進度（跨所有 segments 的累計評估次數）
            lpsrProgress = static_cast<double>(globalCurrentEval + evaluationCount) / static_cast<double>(globalMaxEval);
        } else {
            // 使用本地進度（僅限當前 segment）
            lpsrProgress = static_cast<double>(evaluationCount) / static_cast<double>(maxEvaluations);
        }
        lpsrProgress = min(1.0, max(0.0, lpsrProgress)); // clamp to [0, 1]
        
        int newPopulationSize = round(populationSize - lpsrProgress * (populationSize - finalPopulationSize));
        newPopulationSize = max(finalPopulationSize, min(populationSize, newPopulationSize));
        
        sort(population.begin(), population.end(), [](auto& a, auto& b) {
            return a.fitness < b.fitness;
        });

        for(int j = 0; j < newPopulationSize; j++){
            if(showProgess) bar.progress(evaluationCount, maxEvaluations);
            
            int index = gen() % memorySize; 
            double F_value = generateF(index);       // Generate F value using Cauchy distribution
            double CR_value = generateCR(index);     // Generate CR value using Normal distribution

            // Mutation
            solution_t pbest = pickPBest(newPopulationSize);
            vector<double> mutant = mutate(j, pbest, F_value);
            
            // Crossover
            vector<double> trial = crossover(population[j].solution, mutant, CR_value);
            
            double trialFitness = objFunction(trial);
            evaluationCount++;
            
            if(evaluationCount % MID_BEST_CHECK_POINT == 0){
                midBest.push_back({evaluationCount, bestFitness});
            }
            
            // Selection
            if(trialFitness < population[j].fitness) {

                // Update global best solution
                if(trialFitness < bestFitness) {
                    bestSolution = trial;
                    bestFitness = trialFitness;
                }

                // Store F and CR values
                SF.push_back(F_value);
                SCR.push_back(CR_value);
                dF.push_back(abs(population[j].fitness - trialFitness));

                // Update the population with the trial solution
                population[j].solution = trial;
                population[j].fitness = trialFitness;
            }
            
            // Check if maximum evaluations reached
            if(evaluationCount >= maxEvaluations) break;
        }

        if(evaluationCount >= maxEvaluations) break;

        // Update F/CR
        if (!SF.empty()) {
            double d_sum = accumulate(dF.begin(), dF.end(), 0.0);
            double w_sum = 0, f_num = 0, cr_num = 0, f_den = 0;
            for (size_t i = 0; i < SF.size(); ++i) {
                double w = dF[i] / d_sum;
                f_num += w * SF[i] * SF[i];
                f_den += w * SF[i];
                cr_num += w * SCR[i];
            }
            F[memoryIndex] = f_num / f_den;
            CR[memoryIndex] = cr_num;
            memoryIndex = (memoryIndex + 1) % memorySize;
        }

        sort(population.begin(), population.end(), [](auto& a, auto& b) {
            return a.fitness < b.fitness;
        });
        population.resize(newPopulationSize);

    }
    if(showProgess) bar.finish();
    midBest.push_back({maxEvaluations, bestFitness});
    // export final population
    finalPopulation.clear(); finalFitnesses.clear();
    finalPopulation.reserve(population.size()); finalFitnesses.reserve(population.size());
    for(const auto& ind : population){ finalPopulation.push_back(ind.solution); finalFitnesses.push_back(ind.fitness); }
    return {bestSolution, bestFitness};
}
    
// Get best solution and fitness value
pair<vector<double>, double> LSHADE::getBestSolution() const {
    return {bestSolution, bestFitness};
}

// Get mid best values for every evaluation
vector<pair<int, double>> LSHADE::getMidBest() const {
    return midBest;
}

// Initialize population
void LSHADE::initialize(){
    
    midBest.clear();
    evaluationCount = 0;
    population = vector<individual_t>(populationSize, individual_t{solution_t(dimension), numeric_limits<double>::infinity()});
    memoryIndex = 0;                            // Reset memory index

    for(int i = 0; i < populationSize; i++){
        for(int j = 0; j < dimension; j++){
            // Randomly initialize position within search space
            population[i].solution[j] = lowerBounds[j] + dist(gen) * (upperBounds[j] - lowerBounds[j]);
        }
        
        // Calculate initial fitness
        population[i].fitness = objFunction(population[i].solution);
        // evaluationCount++;
        
        // Update global best value
        if(population[i].fitness < bestFitness){
            bestFitness = population[i].fitness;
            bestSolution = population[i].solution;
        }
    }
}

// Perform mutation operation
solution_t LSHADE::mutate(int currentIndex, solution_t pbest, double F_value){
    vector<double> mutant(dimension);
    
    // Randomly select 5 distinct indices from the population
    int maxNeededIndices = 5;
    vector<int> indices;
    indices.reserve(maxNeededIndices);
    
    unordered_set<int> selectedIndices;
    indices.push_back(currentIndex);
    selectedIndices.insert(currentIndex);

    uniform_int_distribution<int> indexDist(0, population.size() - 1);
    while(indices.size() < maxNeededIndices) {
        int randomIdx = indexDist(gen);
        if(!selectedIndices.count(randomIdx)) {
            indices.push_back(randomIdx);
            selectedIndices.insert(randomIdx);
        }
    }

    // DE/current-to-pbest/1 strategy
    for(int j = 0; j < dimension; j++){
        mutant[j] = population[indices[0]].solution[j] + 
                    F_value * (pbest[j] - population[indices[0]].solution[j]) +
                    F_value * (population[indices[1]].solution[j] - population[indices[2]].solution[j]);
    }
    
    // Boundary handling
    for(int j = 0; j < dimension; j++){
        mutant[j] = max(mutant[j], lowerBounds[j]);
        mutant[j] = min(mutant[j], upperBounds[j]);
    }
    
    return mutant;
}

// Perform crossover operation
solution_t LSHADE::crossover(const vector<double>& target, const vector<double>& mutant, double CR_value){
    vector<double> trial(dimension);
    
    // Ensure at least one dimension is crossed
    int jRand = uniform_int_distribution<int>{0, dimension - 1}(gen);
    
    for(int j = 0; j < dimension; j++){
        // If random number is less than crossover rate or is the forced crossover dimension
        if(dist(gen) <= CR_value || j == jRand){
            trial[j] = mutant[j];
        }
        else{
            trial[j] = target[j];
        }
    }
    
    return trial;
}

// Select the best solution from the pbest set
solution_t LSHADE::pickPBest(int newPopulationSize){

    double pmin = 2.0 / newPopulationSize;
    uniform_real_distribution<double> rate_dist(pmin, 0.2);
    double p_rate = rate_dist(gen);

    int p_num = max(2, int(p_rate * population.size()));
    uniform_int_distribution<int> idx_dist(0, p_num - 1);
    return population[idx_dist(gen)].solution;
}

// Generate F using Cauchy distribution
double LSHADE::generateF(int F_index){

    // Generate F value using Cauchy distribution
    cauchy_distribution<double> F_cauchy(F[F_index], 0.1);
    double F_value = F_cauchy(gen);

    // Ensure F_value is within (0, 1]
    F_value = min(1.0, max(0.0, F_value));

    return F_value;
}

// Generate CR using Normal distribution
double LSHADE::generateCR(int CR_index){

    // Generate F value using Cauchy distribution
    normal_distribution<double> CR_normal(CR[CR_index], 0.1);
    double CR_value = CR_normal(gen);

    // Ensure F_value is within (0, 1]
    CR_value = min(1.0, max(0.0, CR_value));

    return CR_value;
}

void LSHADE::setGlobalProgress(int gCurrentEval, int gMaxEval) {
    globalCurrentEval = gCurrentEval;
    globalMaxEval = gMaxEval;
    useGlobalProgress = (gMaxEval > 0);
}

void LSHADE::loadState(const SearchState& state) {
    if (state.isEmpty()) return;
    
    if (state.globalMaxEval > 0) {
        setGlobalProgress(state.globalCurrentEval, state.globalMaxEval);
    }
    
    if (!state.bestSolution.empty() && state.bestSolution.size() == static_cast<size_t>(dimension)) {
        bestSolution = state.bestSolution;
        bestFitness = state.bestFitness;
    }
    
    if (state.hasMemory()) {
        int stateMemSize = static_cast<int>(state.memoryCR.size());
        if (stateMemSize == memorySize) {
            for (int i = 0; i < memorySize; ++i) {
                CR[i] = state.memoryCR[i];
                F[i] = state.memoryF[i];
            }
            memoryIndex = state.memoryWriteIndex % memorySize;
        } else {
            int copySize = min(stateMemSize, memorySize);
            for (int i = 0; i < copySize; ++i) {
                CR[i] = state.memoryCR[i];
                F[i] = state.memoryF[i];
            }
            memoryIndex = state.memoryWriteIndex % memorySize;
        }
    }
    
    // 載入族群
    int statePopSize = static_cast<int>(state.population.size());
    if (statePopSize > 0) {
        // 建立新族群
        int targetSize = min(statePopSize, populationSize);
        population.resize(targetSize);
        
        for (int i = 0; i < targetSize; ++i) {
            if (state.population[i].size() == static_cast<size_t>(dimension)) {
                population[i].solution = state.population[i];
                population[i].fitness = (i < static_cast<int>(state.fitness.size())) 
                    ? state.fitness[i] 
                    : numeric_limits<double>::infinity();
            }
        }
        
        // 若需要擴充族群
        if (targetSize < populationSize) {
            population.resize(populationSize);
            uniform_real_distribution<double> perturbDist(-0.1, 0.1);
            for (int i = targetSize; i < populationSize; ++i) {
                population[i].solution.resize(dimension);
                for (int d = 0; d < dimension; ++d) {
                    double range = upperBounds[d] - lowerBounds[d];
                    double perturbation = perturbDist(gen) * range;
                    population[i].solution[d] = bestSolution[d] + perturbation;
                    population[i].solution[d] = max(lowerBounds[d], min(upperBounds[d], population[i].solution[d]));
                }
                auto tmp = population[i].solution;
                population[i].fitness = objFunction(tmp);
            }
        }
        
        // 更新 populationSize 為實際大小
        populationSize = static_cast<int>(population.size());
    }
}

void LSHADE::saveState(SearchState& state) const {
    state.population.clear();
    state.fitness.clear();
    state.population.reserve(population.size());
    state.fitness.reserve(population.size());
    for (const auto& ind : population) {
        state.population.push_back(ind.solution);
        state.fitness.push_back(ind.fitness);
    }
    
    // 保存最佳解
    state.bestSolution = bestSolution;
    state.bestFitness = bestFitness;
    
    // 保存記憶體
    state.memoryCR = CR;
    state.memoryF = F;
    state.memoryWriteIndex = memoryIndex;
    
    // 更新全域進度
    if (useGlobalProgress) {
        state.globalCurrentEval = globalCurrentEval + evaluationCount;
        state.globalMaxEval = globalMaxEval;
    } else {
        state.globalCurrentEval += evaluationCount;
    }
    
}

#endif /* LSHADE_H */