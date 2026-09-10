#ifndef DE_H
#define DE_H

#include <vector>
#include <functional>
#include <random>
#include <algorithm>
#include <limits>
#include <iostream>
#include <cmath>
#include <unordered_set>
#include "utils/tqdm.h"
#include "utils/sample.h"
#include "utils/search_state.h"

using namespace std;

class DE {
private:
    int dimension;                                  
    function<double(vector<double>&)> objFunction;  
    vector<double> lowerBounds;                     
    vector<double> upperBounds;                     
    
    int populationSize;                      // Population size
    int maxEvaluations;                      // Maximum number of evaluations
    double F;                                // Scaling factor
    double CR;                               // Crossover rate
    int strategy;                            // DE strategy
    
    // DE state
    vector<vector<double>> population;      // Current population
    vector<double> fitness;                  // Fitness values
    vector<double> bestSolution;             // Best solution
    vector<pair<int, double>> midBest; // Best solutions over iterations
    double bestFitness;                      // Best fitness value
    int evaluationCount;                     // Current evaluation count
    
    // Random number generator
    mt19937 gen;
    uniform_real_distribution<double> dist;
    // handoff
    bool hasInitialPool{false};
    vector<vector<double>> initialPool;
    vector<double> initialFitnesses;
    vector<vector<double>> finalPopulation;
    vector<double> finalFitnesses;

public:
    // Constructor
    DE(int dimension, 
        const function<double(vector<double>&)>& objFunction,
        const vector<double>& lowerBounds, 
        const vector<double>& upperBounds,
        int strategy,
        int populationSize = META_POPULATION_SIZE,
        int maxEvaluations = META_MAX_EVALUATIONS,
        double F = 0.5, 
        double CR = 0.9,
        uint32_t seed = random_device{}()) : 
        dimension(dimension), 
        objFunction(objFunction),
        lowerBounds(lowerBounds), 
        upperBounds(upperBounds),
        populationSize(populationSize), 
        maxEvaluations(maxEvaluations),
        F(F), 
        CR(CR),
        strategy(strategy),
        gen(seed), 
        dist(0.0, 1.0){
        
        // Initialize best value
        bestFitness = numeric_limits<double>::infinity();
        evaluationCount = 0;
        
        // Initialize data structures
        population.resize(populationSize, vector<double>(dimension));
        fitness.resize(populationSize);
        bestSolution.resize(dimension);

        initialize();
    }

    void setInitialSolution(const vector<double>& initialSolution) {
        hasInitialPool = true;
        initialPool.clear();
        initialFitnesses.clear();
        initialPool.push_back(initialSolution);
    }

    void setInitialSolution(const vector<double>& best,
                            const vector<vector<double>>& candidates,
                            const vector<double>& candidateFitness){
        hasInitialPool = true;
        initialPool.clear();
        initialFitnesses.clear();
        auto clampv = [&](vector<double> v){
            for(int d=0; d<dimension; ++d){ v[d] = max(lowerBounds[d], min(upperBounds[d], v[d])); }
            return v;
        };
        initialPool.push_back(clampv(best));
        int need = max(0, populationSize - 1);
        const int n = (int)candidates.size();
        vector<double> weights;
        if(n>0 && candidateFitness.size()==candidates.size()){
            double fmin = numeric_limits<double>::infinity();
            for(double f : candidateFitness) if(isfinite(f)) fmin = min(fmin, f);
            if(!isfinite(fmin)) fmin = 0.0;
            weights.reserve(n);
            for(double f : candidateFitness){
                double w = 1.0/(1e-12 + max(0.0, f - fmin));
                if(!isfinite(w) || w < 0.0) w = 0.0; weights.push_back(w);
            }
        }
        mt19937 rng(gen());
        vector<int> picks;
        if(!weights.empty()){
            // Allow duplicates in roulette selection
            picks = roulette_select_indices(weights, need, false, rng);
        }else if(n>0){
            uniform_int_distribution<int> uni(0, n-1);
            picks.resize(need); for(int i=0;i<need;++i) picks[i] = uni(rng);
        }
        for(int i=0;i<need;++i){
            vector<double> cand = n>0 && !picks.empty()? candidates[picks[i % (int)picks.size()]] : vector<double>(dimension,0.0);
            if((int)cand.size()!=dimension){
                cand.assign(dimension,0.0);
                for(int d=0; d<dimension; ++d){ uniform_real_distribution<double> dis(lowerBounds[d], upperBounds[d]); cand[d] = dis(gen);}            
            }
            initialPool.push_back(clampv(cand));
        }
    }

    // Initialize population
    void initialize(){
        midBest.clear();
        for(int i = 0; i < populationSize; i++){
            for(int j = 0; j < dimension; j++){
                // Randomly initialize position within search space
                population[i][j] = lowerBounds[j] + dist(gen) * (upperBounds[j] - lowerBounds[j]);
            }
            
            // Calculate initial fitness
            fitness[i] = objFunction(population[i]);
            
            // Update global best value
            if(fitness[i] < bestFitness){
                bestFitness = fitness[i];
                bestSolution = population[i];
            }
        }
        if(hasInitialPool && !initialPool.empty()){
            int m = min(populationSize, (int)initialPool.size());
            for(int i=0;i<m;++i){
                population[i] = initialPool[i];
                fitness[i] = objFunction(population[i]);
                if(fitness[i] < bestFitness){ bestFitness = fitness[i]; bestSolution = population[i]; }
            }
            hasInitialPool = false; initialPool.clear(); initialFitnesses.clear();
        }
    }

    // Perform mutation operation
    vector<double> mutate(int currentIndex){
        vector<double> mutant(dimension);
        
        int maxNeededIndices = 5;
        vector<int> indices;
        indices.push_back(currentIndex);
        indices.reserve(maxNeededIndices);
        
        unordered_set<int> selectedIndices;
        selectedIndices.insert(currentIndex);
        
        uniform_int_distribution<int> indexDist(0, populationSize - 1);
        while(indices.size() < maxNeededIndices) {
            int randomIdx = indexDist(gen);
            if(!selectedIndices.count(randomIdx)) {
                indices.push_back(randomIdx);
                selectedIndices.insert(randomIdx);
            }
        }
        
        // Implement different DE strategies
        switch(strategy){
            case 1: // DE/rand/1
                for(int j = 0; j < dimension; j++){
                    mutant[j] = population[indices[0]][j] + F * (population[indices[1]][j] - population[indices[2]][j]);
                }
                break;
                
            case 2: // DE/best/1
                for(int j = 0; j < dimension; j++){
                    mutant[j] = bestSolution[j] + F * (population[indices[0]][j] - population[indices[1]][j]);
                }
                break;
                
            case 3: // DE/rand/2
                for(int j = 0; j < dimension; j++){
                    mutant[j] = population[indices[0]][j] + 
                               F * (population[indices[1]][j] - population[indices[2]][j]) +
                               F * (population[indices[3]][j] - population[indices[4]][j]);
                }
                break;
                
            case 4: // DE/best/2
                for(int j = 0; j < dimension; j++){
                    mutant[j] = bestSolution[j] + 
                               F * (population[indices[0]][j] - population[indices[1]][j]) +
                               F * (population[indices[2]][j] - population[indices[3]][j]);
                }
                break;
                
            default: // Default to DE/rand/1
                for(int j = 0; j < dimension; j++){
                    mutant[j] = population[indices[0]][j] + F * (population[indices[1]][j] - population[indices[2]][j]);
                }
        }
        
        // Boundary handling
        for(int j = 0; j < dimension; j++){
            mutant[j] = max(mutant[j], lowerBounds[j]);
            mutant[j] = min(mutant[j], upperBounds[j]);
        }
        
        return mutant;
    }

    // Perform crossover operation
    vector<double> crossover(const vector<double>& target, const vector<double>& mutant){
        vector<double> trial(dimension);
        
        // Ensure at least one dimension is crossed
        int jRand = uniform_int_distribution<int>{0, dimension - 1}(gen);
        
        for(int j = 0; j < dimension; j++){
            // If random number is less than crossover rate or is the forced crossover dimension
            if(dist(gen) <= CR || j == jRand){
                trial[j] = mutant[j];
            }
            else{
                trial[j] = target[j];
            }
        }
        
        return trial;
    }

    // Selection operation
    void selection(int index, vector<double>& trial){
        double trialFitness = objFunction(trial);
        
        // Greedy selection
        if(trialFitness <= fitness[index]){
            population[index] = trial;
            fitness[index] = trialFitness;
            
            // Update global best value
            if(trialFitness < bestFitness){
                bestFitness = trialFitness;
                bestSolution = trial;
            }
        }
    }

    // Run DE algorithm
    pair<vector<double>, double> run(bool showProgess = false){
        initialize();
        
        // Main iteration loop
        tqdm bar;
        for(int i=0;i<maxEvaluations;i+=populationSize){
            if(showProgess) bar.progress(i/populationSize, maxEvaluations/populationSize);
            if(i % MID_BEST_CHECK_POINT == 0){
                midBest.push_back({i, bestFitness});
            }
            for(int j = 0; j < populationSize; j++){
                // Mutation
                vector<double> mutant = mutate(j);
                
                // Crossover
                vector<double> trial = crossover(population[j], mutant);
                
                // Selection
                selection(j, trial);
                
                // Check if maximum evaluations reached
                if(evaluationCount >= maxEvaluations){
                    break;
                }
            }
        }
        if(showProgess) bar.finish();
        midBest.push_back({maxEvaluations, bestFitness});
        finalPopulation = population;
        finalFitnesses = fitness;
        return {bestSolution, bestFitness};
    }
    
    // Get best solution and fitness value
    pair<vector<double>, double> getBestSolution() const {
        return {bestSolution, bestFitness};
    }

    // Get mid best values
    vector<pair<int, double>> getMidBest() const {
        return midBest;
    }

    const vector<vector<double>>& getFinalPopulation() const { return finalPopulation; }
    const vector<double>& getFinalFitnesses() const { return finalFitnesses; }
    
    // SearchState 
    void loadState(const SearchState& state);
    void saveState(SearchState& state) const;
};


inline void DE::loadState(const SearchState& state) {
    if (state.isEmpty()) return;
    
    if (!state.bestSolution.empty() && state.bestSolution.size() == static_cast<size_t>(dimension)) {
        bestSolution = state.bestSolution;
        bestFitness = state.bestFitness;
    }
    
    int statePopSize = static_cast<int>(state.population.size());
    if (statePopSize > 0) {
        int targetSize = min(statePopSize, populationSize);
        
        for (int i = 0; i < targetSize; ++i) {
            if (state.population[i].size() == static_cast<size_t>(dimension)) {
                population[i] = state.population[i];
                fitness[i] = (i < static_cast<int>(state.fitness.size())) 
                    ? state.fitness[i] 
                    : numeric_limits<double>::infinity();
            }
        }
        
        if (targetSize < populationSize) {
            uniform_real_distribution<double> perturbDist(-0.1, 0.1);
            for (int i = targetSize; i < populationSize; ++i) {
                for (int d = 0; d < dimension; ++d) {
                    double range = upperBounds[d] - lowerBounds[d];
                    double perturbation = perturbDist(gen) * range;
                    population[i][d] = bestSolution[d] + perturbation;
                    population[i][d] = max(lowerBounds[d], min(upperBounds[d], population[i][d]));
                }
                fitness[i] = objFunction(population[i]);
            }
        }
    }
}

inline void DE::saveState(SearchState& state) const {
    state.population.clear();
    state.fitness.clear();
    state.population.reserve(population.size());
    state.fitness.reserve(fitness.size());
    for (int i = 0; i < populationSize; ++i) {
        state.population.push_back(population[i]);
        state.fitness.push_back(fitness[i]);
    }
    
    state.bestSolution = bestSolution;
    state.bestFitness = bestFitness;
    
    state.globalCurrentEval += evaluationCount;
    
}

#endif // DE_H