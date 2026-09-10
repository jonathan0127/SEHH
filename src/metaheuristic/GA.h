// Using Genetic Algorithm to solve function optimization problem
#ifndef GA_H
#define GA_H

#include<iostream>
#include<vector>
#include<string>
#include<random>
#include<cmath>
#include<iomanip>
#include<cstdint>
#include "utils/sample.h"
#include "utils/tqdm.h"
#include "config/para_setting.h"

using namespace std;
using ans_t = pair<vector<double>, double>; // Solution and its fitness

class Chromosome{
    friend class GA;
public:
    Chromosome(){
        chromo.resize(0);
        val = numeric_limits<double>::infinity();
    }
    bool operator < (const Chromosome &x)const{
        if(this->val < x.val) return true;
        return false;
    }
private:
    vector<double> chromo;
    double val;
};

class GA{
public:
    GA(
        const int dimension,
        const function<double(vector<double>&)>& objFunction,
        const vector<double>& lowerBounds,
        const vector<double>& upperBounds,
        const int populationSize,
        const double crossoverRate,
        const double mutationRate,
        const double elitismRate,
        const int selectionMethod,
        const double crossoverMethod,
        const int evaluationMax,
        uint32_t seed = random_device{}()
    );

    ~GA(){};

    void evaluation();

    void selection();

    void crossover();

    void mutation();

    void setInitialSolution(const vector<double>& initialSolution);
    void setInitialSolution(
        const vector<double>& best,
        const vector<vector<double>>& candidates,
        const vector<double>& candidateFitness
    );

    ans_t run(bool);

    vector<pair<int, double>> getMidBest() const;
    const vector<vector<double>>& getFinalPopulation() const { return finalPopulation; }
    const vector<double>& getFinalFitnesses() const { return finalFitnesses; }

    void checkNaN(const vector<Chromosome>& pop, const string& location, int generation) {
        for (int i = 0; i < (int)pop.size(); ++i) {
            for (int j = 0; j < (int)pop[i].chromo.size(); ++j) {
                double v = pop[i].chromo[j];
                if (isnan(v) || isinf(v)) {
                    cerr << "[NaN/Inf] at " << location
                        << "  gen=" << generation
                        << "  indiv=" << i
                        << "  gene=" << j
                        << "  value=" << v
                        << "\n";
                    return;  // 找到就中斷，方便你馬上停下來看
                }
            }
        }
    }

private:
    const int populationSize = META_POPULATION_SIZE; // Population size
    int dimension, evaluationMax, evaluationCount, eliteNum, selectionMethod, crossoverMethod;
    function<double(vector<double>&)> objFunction; // Objective function
    vector<double> lowerBounds; // Search space lower bounds
    vector<double> upperBounds; // Search space upper bounds
    vector<Chromosome> chromosomes; // chromosomes[0] always stores the best solution
    vector<pair<int, double>> midBest;
    double crossoverRate, mutationRate, elitismRate;
    mt19937 gen;
    uniform_real_distribution<double> unitDist;

    // For handoff between segments
    vector<vector<double>> finalPopulation;
    vector<double> finalFitnesses;
};

GA::GA(
    const int dimension,
    const function<double(vector<double>&)>& objFunction,
    const vector<double>& lowerBounds,
    const vector<double>& upperBounds,
    const int populationSize,
    const double crossoverRate,
    const double mutationRate,
    const double elitismRate,
    const int selectionMethod,
    const double crossoverMethod,
    const int evaluationMax,
    uint32_t seed
) : dimension(dimension), objFunction(objFunction), lowerBounds(lowerBounds), upperBounds(upperBounds), populationSize(populationSize),
    crossoverRate(crossoverRate), mutationRate(mutationRate), elitismRate(elitismRate), selectionMethod(selectionMethod),
    crossoverMethod(crossoverMethod), evaluationMax(evaluationMax), gen(seed), unitDist(0.0, 1.0)
{
    evaluationCount = 0;
    chromosomes.resize(populationSize);
    midBest.clear();
    eliteNum = max(1, static_cast<int>(populationSize * elitismRate)); // Number of elite chromosomes

    for(int i = 0; i < populationSize; ++i){
        chromosomes[i].chromo.resize(dimension);
        for(int j = 0; j < dimension; ++j){
            uniform_real_distribution<double> dis(lowerBounds[j], upperBounds[j]);
            chromosomes[i].chromo[j] = dis(this->gen);
        }
    }
}

ans_t GA::run(bool showProgress = false){

    tqdm bar;
    for(int i = 0; i < evaluationMax; i += populationSize){
        if(showProgress) bar.progress(i / populationSize, evaluationMax / populationSize);
        evaluation();
        if(i % MID_BEST_CHECK_POINT == 0){
            midBest.push_back({i, chromosomes[0].val});
        }
        selection();
        crossover();
        mutation();
    }
    evaluation();
    if(showProgress) bar.finish();
    midBest.push_back({evaluationMax, chromosomes[0].val});

    // export final population
    finalPopulation.assign(populationSize, vector<double>());
    finalFitnesses.assign(populationSize, 0.0);
    for(int i=0;i<populationSize;++i){
        finalPopulation[i] = chromosomes[i].chromo;
        finalFitnesses[i] = chromosomes[i].val;
    }
    return {chromosomes[0].chromo, chromosomes[0].val};
}

void GA::evaluation(){
    for(int i = 0; i < populationSize; ++i){
        chromosomes[i].val = objFunction(chromosomes[i].chromo);
    }
    sort(chromosomes.begin(), chromosomes.end());
}

void GA::selection(){
    vector<Chromosome> temp(populationSize);

    // Get elite chromosomes
    for(int i = 0; i < eliteNum; ++i){
        temp[i] = chromosomes[i];
    }

    if(selectionMethod == 0){
        // Method 0: Roulette Wheel Selection
        vector<double> cumulativeFitness(populationSize);
        cumulativeFitness[0] = 1.0 / chromosomes[0].val;
        for(int i = 1; i < populationSize; ++i){
            cumulativeFitness[i] = cumulativeFitness[i - 1] + 1.0 / chromosomes[i].val;
        }
        uniform_real_distribution<double> dis(0.0, cumulativeFitness.back());
        for(int i = eliteNum; i < populationSize; ++i){
            double randVal = dis(this->gen);
            auto it = lower_bound(cumulativeFitness.begin(), cumulativeFitness.end(), randVal);
            int idx = it - cumulativeFitness.begin();
            if(idx == populationSize) idx = populationSize - 1;
            temp[i] = chromosomes[idx];
        }
    }else if(selectionMethod == 1){
        // Method 1: Tournament Selection
        uniform_int_distribution<int> dis(0, populationSize - 1);
        for(int i = eliteNum; i < populationSize; ++i){
            int idx1 = dis(this->gen);
            int idx2 = dis(this->gen);
            if(chromosomes[idx1].val < chromosomes[idx2].val){
                temp[i] = chromosomes[idx1];
            }else{
                temp[i] = chromosomes[idx2];
            }
        }
    }else if(selectionMethod == 2){
        // Method 2: Rank Selection
        vector<double> rankWeights(populationSize);
        rankWeights[0] = 1.0; // Assign the highest weight to the best chromosome
        for(int i = 1; i < populationSize; ++i){
            rankWeights[i] = rankWeights[i - 1] + 1.0 / (i + 1); // Cumulative weights
        }
        uniform_real_distribution<double> dis(0.0, rankWeights.back());
        for(int i = eliteNum; i < populationSize; ++i){
            double randVal = dis(this->gen);
            auto it = lower_bound(rankWeights.begin(), rankWeights.end(), randVal);
            int idx = it - rankWeights.begin();
            if(idx == populationSize) idx = populationSize - 1;
            temp[i] = chromosomes[idx];
        }
    }

    chromosomes = temp;
}

void GA::crossover(){
    if(crossoverMethod == 0){
        // Method 0:  Arithmetic Crossover
        for(int i = eliteNum; i + 1 < populationSize; i += 2){
            Chromosome offspring1 = chromosomes[i];
            Chromosome offspring2 = chromosomes[i + 1];

            double alpha = unitDist(this->gen);
            for(int j = 0; j < dimension; ++j){
                offspring1.chromo[j] = alpha * chromosomes[i].chromo[j] + (1 - alpha) * chromosomes[i + 1].chromo[j];
                offspring2.chromo[j] = alpha * chromosomes[i + 1].chromo[j] + (1 - alpha) * chromosomes[i].chromo[j];
            }

            chromosomes[i] = offspring1;
            chromosomes[i + 1] = offspring2;
        }
    }else if(crossoverMethod == 1){
        // Method 1: Blend Crossover
        for(int i = eliteNum; i + 1 < populationSize; i += 2){
            if(unitDist(this->gen) > crossoverRate) continue;

            Chromosome offspring1 = chromosomes[i];
            Chromosome offspring2 = chromosomes[i + 1];

            for(int j = 0; j < dimension; ++j){
                const double gamma = 0.3; // TODO: Set a proper value for gamma
                double d = abs(chromosomes[i].chromo[j] - chromosomes[i + 1].chromo[j]);
                double lower = min(chromosomes[i].chromo[j], chromosomes[i + 1].chromo[j]) - gamma * d;
                double upper = max(chromosomes[i].chromo[j], chromosomes[i + 1].chromo[j]) + gamma * d;

                uniform_real_distribution<double> blendDis(max(lower, lowerBounds[j]), min(upper, upperBounds[j]));
                offspring1.chromo[j] = blendDis(this->gen);
                offspring2.chromo[j] = blendDis(this->gen);
            }

            chromosomes[i] = offspring1;
            chromosomes[i + 1] = offspring2;
        }
    }
}

void GA::mutation(){
    for(int i = eliteNum; i < populationSize; ++i){
        if(unitDist(this->gen) > mutationRate) continue;

        for(int j = 0; j < dimension; ++j){
            if(unitDist(this->gen) < mutationRate){
                uniform_real_distribution<double> mutationDis(lowerBounds[j], upperBounds[j]);
                chromosomes[i].chromo[j] = mutationDis(this->gen);
            }
        }
    }
}

vector<pair<int, double>> GA::getMidBest() const{
    return midBest;
}

void GA::setInitialSolution(const vector<double>& initialSolution){
    chromosomes[0].chromo = initialSolution;
    chromosomes[0].val = objFunction(chromosomes[0].chromo);
    sort(chromosomes.begin(), chromosomes.end());
    checkNaN(chromosomes, "after set initial solution", -1);
}

void GA::setInitialSolution(
    const vector<double>& best,
    const vector<vector<double>>& candidates,
    const vector<double>& candidateFitness
){
    // Construct a full initial population from previous segment
    auto clamp_to_bounds = [&](vector<double> v){
        for(int d=0; d<dimension; ++d){
            if(d < (int)lowerBounds.size() && d < (int)upperBounds.size()){
                v[d] = max(lowerBounds[d], min(upperBounds[d], v[d]));
            }
        }
        return v;
    };

    chromosomes.clear();
    chromosomes.resize(populationSize);

    // Put best first
    chromosomes[0].chromo = clamp_to_bounds(vector<double>(best));
    chromosomes[0].val = objFunction(chromosomes[0].chromo);

    int need = max(0, populationSize - 1);
    if(need > 0){
        const int n = (int)candidates.size();
        vector<double> weights;
        weights.reserve(n);
        if(!candidates.empty() && candidateFitness.size() == candidates.size()){
            double fmin = numeric_limits<double>::infinity();
            for(double f : candidateFitness) if(isfinite(f)) fmin = min(fmin, f);
            if(!isfinite(fmin)) fmin = 0.0;
            for(double f : candidateFitness){
                double w = 1.0 / (1e-12 + max(0.0, f - fmin));
                if(!isfinite(w) || w < 0.0) w = 0.0;
                weights.push_back(w);
            }
        }
        mt19937 localRng(gen());
        vector<int> picks;
        if(!weights.empty()){
            // Allow duplicates in roulette selection
            picks = roulette_select_indices(weights, need, false, localRng);
        } else {
            // fallback uniform
            picks.resize(need);
            if(n > 0){
                uniform_int_distribution<int> uni(0, n-1);
                for(int i=0;i<need;++i) picks[i] = uni(localRng);
            }
        }
        for(int i=0;i<need;++i){
            vector<double> cand;
            if(!candidates.empty() && !picks.empty()) cand = candidates[picks[i % (int)picks.size()]];
            if((int)cand.size() != dimension){
                // random fallback
                cand.assign(dimension, 0.0);
                for(int d=0; d<dimension; ++d){
                    uniform_real_distribution<double> dis(lowerBounds[d], upperBounds[d]);
                    cand[d] = dis(gen);
                }
            }
            cand = clamp_to_bounds(cand);
            chromosomes[i+1].chromo = move(cand);
            chromosomes[i+1].val = objFunction(chromosomes[i+1].chromo);
        }
    }
    sort(chromosomes.begin(), chromosomes.end());
    checkNaN(chromosomes, "after set initial population (pool)", -1);
}

#endif