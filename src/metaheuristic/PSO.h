#ifndef PSO_H
#define PSO_H

#include <vector>
#include <functional>
#include <random>
#include <algorithm>
#include <limits>
#include <iostream>
#include "utils/tqdm.h"
#include "config/para_setting.h"
#include "utils/sample.h"

using namespace std;

class PSO {
private:
    int dimension;  // Problem dimension
    function<double(vector<double>&)> objFunction;  // Objective function
    vector<double> lowerBounds; // Search space lower bounds
    vector<double> upperBounds; // Search space upper bounds
    
    // PSO parameters
    int swarmSize;  // Swarm size
    int maxEvalution;   // Maximum iterations
    double w;   // Inertia weight
    double c1;  // Cognitive parameter
    double c2;  // Social parameter
    
    // PSO state
    vector<vector<double>> positions;   // Particle positions
    vector<vector<double>> velocities;  // Particle velocities
    vector<double> fitnesses;   // Particle fitnesses
    vector<vector<double>> personalBests;   // Personal best positions
    vector<double> personalBestFitnesses;   // Personal best fitnesses
    vector<double> globalBest;  // Global best position
    vector<pair<int, double>> midBest;  // best fitness during the iteration
    double globalBestFitness;   // Global best fitness
    
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
    PSO(int dimension, 
        const function<double(vector<double>&)>& objFunction,
        const vector<double>& lowerBounds, 
        const vector<double>& upperBounds,
        int swarmSize = META_POPULATION_SIZE, 
        int maxEvalution = META_MAX_EVALUATIONS,
        double w = 0.7, 
        double c1 = 1.5, 
        double c2 = 1.5,
        uint32_t seed = std::random_device{}()) : 
        dimension(dimension), 
        objFunction(objFunction),
        lowerBounds(lowerBounds), 
        upperBounds(upperBounds),
        swarmSize(swarmSize), 
        maxEvalution(maxEvalution),
        w(w), 
        c1(c1), 
        c2(c2),
        gen(seed), 
        dist(0.0, 1.0) {
        
        // Initialize global best value
        globalBestFitness = numeric_limits<double>::infinity();
        
        // Initialize data structures
        positions.resize(swarmSize, vector<double>(dimension));
        velocities.resize(swarmSize, vector<double>(dimension));
        personalBests.resize(swarmSize, vector<double>(dimension));
        fitnesses.resize(swarmSize);
        personalBestFitnesses.resize(swarmSize, numeric_limits<double>::infinity());
        globalBest.resize(dimension);

        initialize();
    }

    void setInitialSolution(const vector<double>& initialSolution) {
        hasInitialPool = true;
        initialPool.clear();
        initialFitnesses.clear();
        initialPool.push_back(initialSolution);
    }

    void setInitialSolution(const vector<double>& best, const vector<vector<double>>& candidates, const vector<double>& candidateFitness) {
        hasInitialPool = true;
        initialPool.clear();
        initialFitnesses.clear();
        auto clampv = [&](vector<double> v){
            for(int d=0; d<dimension; ++d){
                v[d] = max(lowerBounds[d], min(upperBounds[d], v[d]));
            }
            return v;
        };
        initialPool.push_back(clampv(best));
        // sample remaining
        int need = max(0, swarmSize - 1);
        const int n = (int)candidates.size();
        vector<double> weights;
        if(n > 0 && candidateFitness.size() == candidates.size()){
            double fmin = numeric_limits<double>::infinity();
            for(double f : candidateFitness) if(isfinite(f)) fmin = min(fmin, f);
            if(!isfinite(fmin)) fmin = 0.0;
            weights.reserve(n);
            for(double f : candidateFitness){
                double w = 1.0 / (1e-12 + max(0.0, f - fmin));
                if(!isfinite(w) || w < 0.0) w = 0.0; weights.push_back(w);
            }
        }
        mt19937 rng(gen());
        vector<int> picks;
        if(!weights.empty()){
            // Allow duplicates in roulette selection
            picks = roulette_select_indices(weights, need, false, rng);
        }else{
            if(n > 0){
                uniform_int_distribution<int> uni(0, n-1);
                picks.resize(need);
                for(int i=0;i<need;++i) picks[i] = uni(rng);
            }
        }
        for(int i=0;i<need;++i){
            vector<double> cand = n>0 && !picks.empty()? candidates[picks[i % (int)picks.size()]] : vector<double>(dimension, 0.0);
            if((int)cand.size()!=dimension){
                cand.assign(dimension, 0.0);
                for(int d=0; d<dimension; ++d){
                    uniform_real_distribution<double> dis(lowerBounds[d], upperBounds[d]);
                    cand[d] = dis(gen);
                }
            }
            initialPool.push_back(clampv(cand));
        }
    }

    // Initialize particle swarm
    void initialize() {
        midBest.clear();
        for (int i = 0; i < swarmSize; ++i) {
            for (int j = 0; j < dimension; ++j) {
                positions[i][j] = lowerBounds[j] + dist(gen) * (upperBounds[j] - lowerBounds[j]);
                double vMax = 0.5 * (upperBounds[j] - lowerBounds[j]);
                velocities[i][j] = -vMax + 2 * vMax * dist(gen);
                personalBests[i][j] = positions[i][j];
            }
            fitnesses[i] = objFunction(positions[i]);
            personalBestFitnesses[i] = fitnesses[i];
            if (personalBestFitnesses[i] < globalBestFitness) {
                globalBestFitness = personalBestFitnesses[i];
                globalBest = personalBests[i];
            }
        }
        if(hasInitialPool && !initialPool.empty()){
            // overwrite the front according to pool
            int m = min(swarmSize, (int)initialPool.size());
            for(int i=0;i<m;++i){
                positions[i] = initialPool[i];
                fitnesses[i] = objFunction(positions[i]);
                personalBests[i] = positions[i];
                personalBestFitnesses[i] = fitnesses[i];
                if(fitnesses[i] < globalBestFitness){
                    globalBestFitness = fitnesses[i];
                    globalBest = positions[i];
                }
            }
            hasInitialPool = false;
            initialPool.clear();
            initialFitnesses.clear();
        }
    }

    // Run PSO algorithm

    pair<vector<double>, double> run(bool showProgress = false) {
        initialize();
        tqdm bar;
        for (int iter = 0; iter < maxEvalution; iter += swarmSize) {
            if(showProgress) bar.progress(iter / swarmSize, maxEvalution / swarmSize);
            if(iter % MID_BEST_CHECK_POINT == 0){
                midBest.push_back({iter, globalBestFitness});
            }
            for (int i = 0; i < swarmSize; ++i) {
                // Update velocities and positions
                for (int j = 0; j < dimension; ++j) {
                    // Update velocity
                    velocities[i][j] = w * velocities[i][j] + 
                                     c1 * dist(gen) * (personalBests[i][j] - positions[i][j]) +
                                     c2 * dist(gen) * (globalBest[j] - positions[i][j]);
                    
                    // Update position
                    positions[i][j] += velocities[i][j];
                    
                    // Boundary handling
                    positions[i][j] = max(positions[i][j], lowerBounds[j]);
                    positions[i][j] = min(positions[i][j], upperBounds[j]);
                }
                
                // Calculate new fitness
                fitnesses[i] = objFunction(positions[i]);
                
                // Update personal best position
                if (fitnesses[i] < personalBestFitnesses[i]) {
                    personalBestFitnesses[i] = fitnesses[i];
                    personalBests[i] = positions[i];
                    
                    // Update global best position
                    if (personalBestFitnesses[i] < globalBestFitness) {
                        globalBestFitness = personalBestFitnesses[i];
                        globalBest = personalBests[i];
                    }
                }
            }
        }
        if(showProgress) bar.finish();
        midBest.push_back({maxEvalution, globalBestFitness});
        // export final swarm
        finalPopulation = positions;
        finalFitnesses = fitnesses;
        return {globalBest, globalBestFitness};
    }
    
    // Get global best position and fitness
    pair<vector<double>, double> getBestSolution() const {
        return {globalBest, globalBestFitness};
    }

    vector<pair<int, double>> getMidBest() const {
        return midBest;
    }

    const vector<vector<double>>& getFinalPopulation() const { return finalPopulation; }
    const vector<double>& getFinalFitnesses() const { return finalFitnesses; }
};

#endif // PSO_H