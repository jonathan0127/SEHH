#ifndef SEARCH_STATE_H
#define SEARCH_STATE_H

#include <vector>
#include <limits>
using namespace std;

struct SearchState {
    vector<vector<double>> population;
    vector<double> fitness;
    vector<double> bestSolution;
    double bestFitness{numeric_limits<double>::infinity()};

    vector<double> memoryCR;
    vector<double> memoryF;
    int memoryWriteIndex{0};

    vector<vector<double>> archive;
    vector<double> archiveFitness;
    int currentArchiveSize{0};

    int globalCurrentEval{0};
    int globalMaxEval{0};

    double EB_hybrid_rate{0.5};

    bool isEmpty() const {
        return population.empty();
    }

    bool hasMemory() const {
        return !memoryCR.empty() && !memoryF.empty();
    }

    bool hasArchive() const {
        return !archive.empty() && currentArchiveSize > 0;
    }

    double getGlobalProgress() const {
        if (globalMaxEval <= 0) return 0.0;
        return static_cast<double>(globalCurrentEval) / static_cast<double>(globalMaxEval);
    }

    void clear() {
        population.clear();
        fitness.clear();
        bestSolution.clear();
        bestFitness = numeric_limits<double>::infinity();
        memoryCR.clear();
        memoryF.clear();
        memoryWriteIndex = 0;
        archive.clear();
        archiveFitness.clear();
        currentArchiveSize = 0;
        globalCurrentEval = 0;
        globalMaxEval = 0;
        EB_hybrid_rate = 0.5;
    }

    void initializeMemoryIfEmpty(int size, double defaultCR = 0.5, double defaultF = 0.5) {
        if (memoryCR.empty()) {
            memoryCR.assign(size, defaultCR);
        }
        if (memoryF.empty()) {
            memoryF.assign(size, defaultF);
        }
    }

    template<typename RNG>
    void adjustPopulationSize(int targetSize, int dimension,
                               const vector<double>& lowerBounds,
                               const vector<double>& upperBounds,
                               RNG& rng) {
        int currentSize = static_cast<int>(population.size());
        
        if (currentSize == targetSize) return;
        
        if (currentSize > targetSize) {
            vector<int> indices(currentSize);
            for (int i = 0; i < currentSize; ++i) indices[i] = i;
            partial_sort(indices.begin(), indices.begin() + targetSize, indices.end(),
                [this](int a, int b) { return fitness[a] < fitness[b]; });
            
            vector<vector<double>> newPopulation(targetSize);
            vector<double> newFitness(targetSize);
            for (int i = 0; i < targetSize; ++i) {
                newPopulation[i] = population[indices[i]];
                newFitness[i] = fitness[indices[i]];
            }
            population = move(newPopulation);
            fitness = move(newFitness);
        } else {
            population.reserve(targetSize);
            fitness.reserve(targetSize);
            
            uniform_real_distribution<double> dist(-0.1, 0.1);
            
            for (int i = currentSize; i < targetSize; ++i) {
                vector<double> newInd(dimension);
                for (int d = 0; d < dimension; ++d) {
                    double range = upperBounds[d] - lowerBounds[d];
                    double perturbation = dist(rng) * range;
                    newInd[d] = bestSolution[d] + perturbation;
                    newInd[d] = max(lowerBounds[d], min(upperBounds[d], newInd[d]));
                }
                population.push_back(newInd);
                fitness.push_back(numeric_limits<double>::infinity());
            }
        }
    }
};

#endif
