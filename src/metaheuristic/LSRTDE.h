#ifndef LSRTDE_H
#define LSRTDE_H

#include <iostream>
#include <vector>
#include <unordered_set>
#include <string>
#include <random>
#include <cmath>
#include <iomanip>
#include <functional>
#include <algorithm>
#include <numeric>
#include "utils/tqdm.h"
#include "config/para_setting.h"
#include "utils/search_state.h"
#include <cstdint>
#include "utils/sample.h"

using namespace std;
using solution_t = vector<double>;
using ans_t = pair<solution_t, double>;

typedef struct {
    solution_t solution;
    double fitness;
} lsrtde_individual_t;

class LSRTDE {
public:
    LSRTDE(
        int dimension,
        const function<double(vector<double>&)>& objFunction,
        const vector<double>& lowerBounds,
        const vector<double>& upperBounds,
        int populationSize,
        int finalPopulationSize,
        int maxEvaluations,
        int memorySize = 5,
        double sigmaCR = 0.05,
        double sigmaF = 0.02,
        uint32_t seed = random_device{}()
    );

    void setInitialSolution(const solution_t& initialSolution);
    void setInitialSolution(const solution_t& best,
                            const vector<solution_t>& candidates,
                            const vector<double>& candidateFitness);

    void loadState(const SearchState& state);
    void saveState(SearchState& state) const;
    void setGlobalProgress(int globalCurrentEval, int globalMaxEval);

    ans_t run(bool showProgress = false);

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

    int NIndsCurrent;
    int NIndsFront;
    int NIndsFrontMax;
    int PFIndex;

    double successRate;
    int successFilled;

    vector<double> memoryCR;
    vector<double> tempSuccessCr;
    vector<double> fitDelta;
    vector<double> weights;

    double sigmaCR;
    double sigmaF;

    vector<lsrtde_individual_t> population;
    vector<lsrtde_individual_t> populationFront;
    
    vector<pair<int, double>> midBest;
    solution_t bestSolution;
    double bestFitness;
    int evaluationCount;
    int generation;

    vector<solution_t> finalPopulation;
    vector<double> finalFitnesses;

    mt19937 gen;
    uniform_real_distribution<double> dist;
    normal_distribution<double> normDist;

    int globalCurrentEval{0};
    int globalMaxEval{0};
    bool useGlobalProgress{false};

    void initialize();
    
    void quickSort2(vector<double>& arr, vector<int>& indices, int low, int high);
    
    double meanWL(const vector<double>& vec, const vector<double>& weights, int count);
    
    void removeWorst(int oldSize, int newSize);
    
    void updateMemoryCr();
    
    double generateF(double meanF, double sigmaF);
    
    double generateCR(int memIdx);
};

LSRTDE::LSRTDE(
    int dimension,
    const function<double(vector<double>&)>& objFunction,
    const vector<double>& lowerBounds,
    const vector<double>& upperBounds,
    int populationSize,
    int finalPopulationSize,
    int maxEvaluations,
    int memorySize,
    double sigmaCR,
    double sigmaF,
    uint32_t seed
) : dimension(dimension),
    objFunction(objFunction),
    lowerBounds(lowerBounds),
    upperBounds(upperBounds),
    populationSize(populationSize),
    finalPopulationSize(finalPopulationSize),
    maxEvaluations(maxEvaluations),
    memorySize(memorySize),
    memoryIndex(0),
    successRate(0.5),
    successFilled(0),
    bestFitness(numeric_limits<double>::infinity()),
    evaluationCount(0),
    generation(0),
    sigmaCR(sigmaCR),
    sigmaF(sigmaF),
    gen(seed),
    dist(0.0, 1.0),
    normDist(0.0, 1.0) {
    
    NIndsFront = populationSize;
    NIndsFrontMax = populationSize;
    NIndsCurrent = populationSize;
    PFIndex = 0;
    
    memoryCR.assign(memorySize, 1.0);
    tempSuccessCr.resize(populationSize * 2);
    fitDelta.resize(populationSize * 2);
    weights.resize(populationSize * 2);
    
    initialize();
}

void LSRTDE::initialize() {
    midBest.clear();
    evaluationCount = 0;
    generation = 0;
    memoryIndex = 0;
    successRate = 0.5;
    successFilled = 0;
    PFIndex = 0;
    
    NIndsFront = populationSize;
    NIndsFrontMax = populationSize;
    NIndsCurrent = populationSize;
    
    population.resize(populationSize * 2);
    for (int i = 0; i < populationSize * 2; ++i) {
        population[i].solution.resize(dimension);
        for (int j = 0; j < dimension; ++j) {
            population[i].solution[j] = lowerBounds[j] + dist(gen) * (upperBounds[j] - lowerBounds[j]);
        }
        population[i].fitness = numeric_limits<double>::infinity();
    }
    
    populationFront.resize(populationSize);
    for (int i = 0; i < populationSize; ++i) {
        populationFront[i].solution.resize(dimension);
        populationFront[i].fitness = numeric_limits<double>::infinity();
    }
    
    memoryCR.assign(memorySize, 1.0);
    fill(tempSuccessCr.begin(), tempSuccessCr.end(), 0.0);
    
    bestFitness = numeric_limits<double>::infinity();
    bestSolution.clear();
}

void LSRTDE::setInitialSolution(const solution_t& initialSolution) {
    if (initialSolution.size() == static_cast<size_t>(dimension) && !population.empty()) {
        population[0].solution = initialSolution;
    }
}

void LSRTDE::setInitialSolution(const solution_t& best,
                                 const vector<solution_t>& candidates,
                                 const vector<double>& candidateFitness) {
    if (!best.empty() && best.size() == static_cast<size_t>(dimension)) {
        bestSolution = best;
        if (!population.empty()) {
            population[0].solution = best;
        }
    }
    
    int copyCount = min(static_cast<int>(candidates.size()), populationSize - 1);
    for (int i = 0; i < copyCount; ++i) {
        if (candidates[i].size() == static_cast<size_t>(dimension)) {
            population[i + 1].solution = candidates[i];
            if (i < static_cast<int>(candidateFitness.size())) {
                population[i + 1].fitness = candidateFitness[i];
            }
        }
    }
}

void LSRTDE::setGlobalProgress(int gCurrentEval, int gMaxEval) {
    globalCurrentEval = gCurrentEval;
    globalMaxEval = gMaxEval;
    useGlobalProgress = (gMaxEval > 0);
}

void LSRTDE::loadState(const SearchState& state) {
    if (state.isEmpty()) return;

    if (state.globalMaxEval > 0) {
        setGlobalProgress(state.globalCurrentEval, state.globalMaxEval);
    }

    if (!state.bestSolution.empty() && 
        state.bestSolution.size() == static_cast<size_t>(dimension)) {
        bestSolution = state.bestSolution;
        bestFitness = state.bestFitness;
    }

    if (!state.memoryCR.empty()) {
        int stateMemSize = static_cast<int>(state.memoryCR.size());
        int copySize = min(stateMemSize, memorySize);
        for (int i = 0; i < copySize; ++i) {
            memoryCR[i] = state.memoryCR[i];
        }
        memoryIndex = state.memoryWriteIndex % memorySize;
    }

    int statePopSize = static_cast<int>(state.population.size());
    if (statePopSize > 0) {
        int targetSize = min(statePopSize, populationSize);
        for (int i = 0; i < targetSize; ++i) {
            if (state.population[i].size() == static_cast<size_t>(dimension)) {
                population[i].solution = state.population[i];
                population[i].fitness = (i < static_cast<int>(state.fitness.size()))
                    ? state.fitness[i]
                    : numeric_limits<double>::infinity();
            }
        }
    }
}

void LSRTDE::saveState(SearchState& state) const {
    state.population.clear();
    state.fitness.clear();
    state.population.reserve(NIndsCurrent);
    state.fitness.reserve(NIndsCurrent);
    
    for (int i = 0; i < NIndsCurrent; ++i) {
        state.population.push_back(population[i].solution);
        state.fitness.push_back(population[i].fitness);
    }

    state.bestSolution = bestSolution;
    state.bestFitness = bestFitness;

    state.memoryCR = memoryCR;
    if (state.memoryF.empty()) {
        state.memoryF.resize(memoryCR.size(), 0.5);
    }
    state.memoryWriteIndex = memoryIndex;

    if (useGlobalProgress) {
        state.globalCurrentEval = globalCurrentEval + evaluationCount;
        state.globalMaxEval = globalMaxEval;
    } else {
        state.globalCurrentEval += evaluationCount;
    }
}

void LSRTDE::quickSort2(vector<double>& arr, vector<int>& indices, int low, int high) {
    if (low >= high) return;
    
    int i = low, j = high;
    double pivot = arr[(low + high) / 2];
    
    while (i <= j) {
        while (arr[i] < pivot) ++i;
        while (arr[j] > pivot) --j;
        if (i <= j) {
            swap(arr[i], arr[j]);
            swap(indices[i], indices[j]);
            ++i;
            --j;
        }
    }
    
    if (low < j) quickSort2(arr, indices, low, j);
    if (i < high) quickSort2(arr, indices, i, high);
}

double LSRTDE::meanWL(const vector<double>& vec, const vector<double>& tempWeights, int count) {
    if (count == 0) return 1.0;
    
    double sumWeight = 0;
    for (int i = 0; i < count; ++i) {
        sumWeight += tempWeights[i];
    }
    
    if (sumWeight < 1e-10) return 1.0;
    
    double sumSquare = 0, sum = 0;
    for (int i = 0; i < count; ++i) {
        double w = tempWeights[i] / sumWeight;
        sumSquare += w * vec[i] * vec[i];
        sum += w * vec[i];
    }
    
    if (fabs(sum) > 1e-8) {
        return sumSquare / sum;
    }
    return 1.0;
}

void LSRTDE::removeWorst(int oldSize, int newSize) {
    int pointsToRemove = oldSize - newSize;
    
    for (int L = 0; L < pointsToRemove; ++L) {
        double worstFit = populationFront[0].fitness;
        int worstNum = 0;
        
        for (int i = 1; i < oldSize - L; ++i) {
            if (populationFront[i].fitness > worstFit) {
                worstFit = populationFront[i].fitness;
                worstNum = i;
            }
        }
        
        for (int i = worstNum; i < oldSize - L - 1; ++i) {
            populationFront[i] = populationFront[i + 1];
        }
    }
}

void LSRTDE::updateMemoryCr() {
    if (successFilled > 0) {
        double newCr = meanWL(tempSuccessCr, fitDelta, successFilled);
        memoryCR[memoryIndex] = 0.5 * (newCr + memoryCR[memoryIndex]);
        memoryIndex = (memoryIndex + 1) % memorySize;
    }
}

double LSRTDE::generateF(double meanF, double sigmaF) {
    double F;
    do {
        F = normDist(gen) * sigmaF + meanF;
    } while (F < 0.0 || F > 1.0);
    return F;
}

double LSRTDE::generateCR(int memIdx) {
    double Cr = normDist(gen) * sigmaCR + memoryCR[memIdx];
    return max(0.0, min(1.0, Cr));
}

ans_t LSRTDE::run(bool showProgress) {
    tqdm bar;
    
    for (int i = 0; i < NIndsFront; ++i) {
        population[i].fitness = objFunction(population[i].solution);
        evaluationCount++;
        
        if (i == 0 || population[i].fitness < bestFitness) {
            bestFitness = population[i].fitness;
            bestSolution = population[i].solution;
        }
        
        if (evaluationCount % MID_BEST_CHECK_POINT == 0) {
            midBest.push_back({evaluationCount, bestFitness});
        }
    }
    
    vector<double> fitArrCopy(NIndsFront);
    vector<int> indices(NIndsFront);
    for (int i = 0; i < NIndsFront; ++i) {
        fitArrCopy[i] = population[i].fitness;
        indices[i] = i;
    }
    
    double minFit = *min_element(fitArrCopy.begin(), fitArrCopy.end());
    double maxFit = *max_element(fitArrCopy.begin(), fitArrCopy.end());
    if (minFit != maxFit) {
        quickSort2(fitArrCopy, indices, 0, NIndsFront - 1);
    }
    
    for (int i = 0; i < NIndsFront; ++i) {
        populationFront[i] = population[indices[i]];
    }
    
    PFIndex = 0;
    
    while (evaluationCount < maxEvaluations) {
        generation++;
        if (showProgress) bar.progress(evaluationCount, maxEvaluations);
        
        double meanF = 0.4 + tanh(successRate * 5.0) * 0.25;
        
        int currentSize = min(NIndsCurrent, static_cast<int>(population.size()));
        fitArrCopy.resize(currentSize);
        indices.resize(currentSize);
        for (int i = 0; i < currentSize; ++i) {
            fitArrCopy[i] = population[i].fitness;
            indices[i] = i;
        }
        minFit = *min_element(fitArrCopy.begin(), fitArrCopy.begin() + NIndsFront);
        maxFit = *max_element(fitArrCopy.begin(), fitArrCopy.begin() + NIndsFront);
        if (minFit != maxFit) {
            quickSort2(fitArrCopy, indices, 0, NIndsFront - 1);
        }
        
        vector<double> fitArrFrontCopy(NIndsFront);
        vector<int> indices2(NIndsFront);
        for (int i = 0; i < NIndsFront; ++i) {
            fitArrFrontCopy[i] = populationFront[i].fitness;
            indices2[i] = i;
        }
        minFit = *min_element(fitArrFrontCopy.begin(), fitArrFrontCopy.end());
        maxFit = *max_element(fitArrFrontCopy.begin(), fitArrFrontCopy.end());
        if (minFit != maxFit) {
            quickSort2(fitArrFrontCopy, indices2, 0, NIndsFront - 1);
        }
        
        vector<double> fitTemp2(NIndsFront);
        for (int i = 0; i < NIndsFront; ++i) {
            fitTemp2[i] = exp(-static_cast<double>(i) / NIndsFront * 3.0);
        }
        discrete_distribution<int> componentSelectorFront(fitTemp2.begin(), fitTemp2.end());
        
        int psizeval = max(2, static_cast<int>(NIndsFront * 0.7 * exp(-successRate * 7.0)));
        
        successFilled = 0;
        
        for (int indIter = 0; indIter < NIndsFront; ++indIter) {
            if (evaluationCount >= maxEvaluations) break;
            
            int theChosenOne = uniform_int_distribution<int>(0, NIndsFront - 1)(gen);
            int memCurrentIndex = uniform_int_distribution<int>(0, memorySize - 1)(gen);
            
            int prand;
            do {
                prand = indices[uniform_int_distribution<int>(0, psizeval - 1)(gen)];
            } while (prand == theChosenOne);
            
            int rand1;
            do {
                rand1 = indices2[componentSelectorFront(gen)];
            } while (rand1 == prand);
            
            int rand2;
            do {
                rand2 = indices[uniform_int_distribution<int>(0, NIndsFront - 1)(gen)];
            } while (rand2 == prand || rand2 == rand1);
            
            double F = generateF(meanF, sigmaF);
            double Cr = generateCR(memCurrentIndex);
            
            solution_t trial(dimension);
            double actualCr = 0;
            int willCrossover = uniform_int_distribution<int>(0, dimension - 1)(gen);
            
            for (int j = 0; j < dimension; ++j) {
                if (dist(gen) < Cr || j == willCrossover) {
                    trial[j] = populationFront[theChosenOne].solution[j] +
                               F * (population[prand].solution[j] - populationFront[theChosenOne].solution[j]) +
                               F * (populationFront[rand1].solution[j] - population[rand2].solution[j]);
                    
                    if (trial[j] < lowerBounds[j]) {
                        trial[j] = lowerBounds[j] + dist(gen) * (upperBounds[j] - lowerBounds[j]);
                    }
                    if (trial[j] > upperBounds[j]) {
                        trial[j] = lowerBounds[j] + dist(gen) * (upperBounds[j] - lowerBounds[j]);
                    }
                    actualCr++;
                } else {
                    trial[j] = populationFront[theChosenOne].solution[j];
                }
            }
            actualCr /= dimension;
            
            double trialFit = objFunction(trial);
            evaluationCount++;
            
            if (trialFit <= populationFront[theChosenOne].fitness) {
                int storeIdx = NIndsCurrent + successFilled;
                if (storeIdx < static_cast<int>(population.size())) {
                    population[storeIdx].solution = trial;
                    population[storeIdx].fitness = trialFit;
                }
                
                populationFront[PFIndex].solution = trial;
                populationFront[PFIndex].fitness = trialFit;
                
                if (trialFit < bestFitness) {
                    bestFitness = trialFit;
                    bestSolution = trial;
                }
                
                tempSuccessCr[successFilled] = actualCr;
                fitDelta[successFilled] = fabs(populationFront[theChosenOne].fitness - trialFit);
                successFilled++;
                
                PFIndex = (PFIndex + 1) % NIndsFront;
            }
            
            if (evaluationCount % MID_BEST_CHECK_POINT == 0) {
                midBest.push_back({evaluationCount, bestFitness});
            }
        }
        
        successRate = static_cast<double>(successFilled) / NIndsFront;
        
        int totalEval = useGlobalProgress ? (globalCurrentEval + evaluationCount) : evaluationCount;
        int totalMaxEval = useGlobalProgress ? globalMaxEval : maxEvaluations;
        
        int newNIndsFront = static_cast<int>(
            static_cast<double>(finalPopulationSize - NIndsFrontMax) / totalMaxEval * totalEval + NIndsFrontMax
        );
        newNIndsFront = max(finalPopulationSize, min(newNIndsFront, NIndsFront));
        
        if (newNIndsFront < NIndsFront) {
            removeWorst(NIndsFront, newNIndsFront);
        }
        NIndsFront = newNIndsFront;
        
        updateMemoryCr();
        
        NIndsCurrent = NIndsFront + successFilled;
        
        if (NIndsCurrent > NIndsFront) {
            fitArrCopy.resize(NIndsCurrent);
            indices.resize(NIndsCurrent);
            for (int i = 0; i < NIndsCurrent; ++i) {
                fitArrCopy[i] = population[i].fitness;
                indices[i] = i;
            }
            
            minFit = *min_element(fitArrCopy.begin(), fitArrCopy.end());
            maxFit = *max_element(fitArrCopy.begin(), fitArrCopy.end());
            if (minFit != maxFit) {
                quickSort2(fitArrCopy, indices, 0, NIndsCurrent - 1);
            }
            
            vector<lsrtde_individual_t> tempPop(NIndsFront);
            for (int i = 0; i < NIndsFront; ++i) {
                tempPop[i] = population[indices[i]];
            }
            for (int i = 0; i < NIndsFront; ++i) {
                population[i] = tempPop[i];
            }
            NIndsCurrent = NIndsFront;
        }
        
        successFilled = 0;
    }
    
    if (showProgress) bar.finish();
    
    midBest.push_back({evaluationCount, bestFitness});
    
    finalPopulation.clear();
    finalFitnesses.clear();
    finalPopulation.reserve(NIndsFront);
    finalFitnesses.reserve(NIndsFront);
    for (int i = 0; i < NIndsFront; ++i) {
        finalPopulation.push_back(populationFront[i].solution);
        finalFitnesses.push_back(populationFront[i].fitness);
    }
    
    return {bestSolution, bestFitness};
}

ans_t LSRTDE::getBestSolution() const {
    return {bestSolution, bestFitness};
}

vector<pair<int, double>> LSRTDE::getMidBest() const {
    return midBest;
}

#endif
