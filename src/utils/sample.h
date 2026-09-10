#ifndef SAMPLE_H
#define SAMPLE_H

#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <cmath>

using namespace std;

template <typename T>
class Sample{
public:
    static T sampleOne(const vector<T>& items, const vector<double>& weights){
        if(items.size() != weights.size()){
            throw invalid_argument("Items and weights vectors must have the same size");
        }
        
        if(items.empty()){
            throw invalid_argument("Cannot sample from empty vector");
        }
        
        double sum = accumulate(weights.begin(), weights.end(), 0.0);
        if(sum <= 0){
            throw invalid_argument("Sum of weights must be positive");
        }
        
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<> dis(0, sum);
        double randomValue = dis(gen);
        
        vector<double> cumulativeWeights(weights.size());
        cumulativeWeights[0] = weights[0];
        for(size_t i = 1; i < weights.size(); ++i){
            cumulativeWeights[i] = cumulativeWeights[i-1] + weights[i];
        }
        
        auto it = lower_bound(cumulativeWeights.begin(), cumulativeWeights.end(), randomValue);
        size_t index = distance(cumulativeWeights.begin(), it);
        
        if(index >= items.size()){
            index = items.size() - 1;
        }
        
        return items[index];
    }
    
    static vector<T> sampleMultiple(
        const vector<T>& items, 
        const vector<double>& weights,
        size_t n,
        bool allowDuplicates = true
    ){
        if(items.size() != weights.size()){
            throw invalid_argument("Items and weights vectors must have the same size");
        }
        
        if(items.empty()){
            throw invalid_argument("Cannot sample from empty vector");
        }
        
        if(!allowDuplicates && n > items.size()){
            throw invalid_argument("Cannot sample more unique items than are available");
        }
        
        vector<T> result;
        result.reserve(n);
        
        if(allowDuplicates){
            // With duplicates, we can just sample n times
            for(size_t i = 0; i < n; ++i){
                result.push_back(sampleOne(items, weights));
            }
        }else{
            // Without duplicates, we need to keep track of what we've sampled
            vector<T> itemsCopy = items;
            vector<double> weightsCopy = weights;
            
            for(size_t i = 0; i < n; ++i){
                T sampled = sampleOne(itemsCopy, weightsCopy);
                result.push_back(sampled);
                
                // Find and remove the sampled item
                auto it = find(itemsCopy.begin(), itemsCopy.end(), sampled);
                if(it != itemsCopy.end()){
                    size_t index = distance(itemsCopy.begin(), it);
                    itemsCopy.erase(itemsCopy.begin() + index);
                    weightsCopy.erase(weightsCopy.begin() + index);
                }
            }
        }
        
        return result;
    }
};

inline vector<int> roulette_select_indices(
    const vector<double>& weights,
    int k,
    bool unique,
    mt19937& rng
){
    const int n = static_cast<int>(weights.size());
    vector<int> result;
    if(n <= 0 || k <= 0) return result;
    result.reserve(k);

    // Build working copies
    vector<int> idx(n);
    iota(idx.begin(), idx.end(), 0);
    vector<double> w(weights.begin(), weights.end());
    for(double& x : w){ if(!isfinite(x) || x < 0.0) x = 0.0; }

    auto total_weight = [&](){
        double s = 0.0; for(double x : w) s += x; return s;
    };

    uniform_real_distribution<double> uni01(0.0, 1.0);

    auto draw_one = [&](){
        double sum = total_weight();
        if(sum <= 0.0){
            // fallback to uniform
            uniform_int_distribution<int> uni(0, static_cast<int>(idx.size()) - 1);
            return idx[uni(rng)];
        }
        double r = uni01(rng) * sum;
        double acc = 0.0;
        for(size_t i = 0; i < idx.size(); ++i){
            acc += w[i];
            if(r <= acc){
                return idx[i];
            }
        }
        return idx.back();
    };

    if(unique){
        // draw without replacement up to n
        int firstPhase = min(k, n);
        for(int t=0; t<firstPhase && !idx.empty(); ++t){
            int sel = draw_one();
            result.push_back(sel);
            // remove selected from pools
            for(size_t i=0;i<idx.size();++i){
                if(idx[i] == sel){
                    idx.erase(idx.begin()+static_cast<long>(i));
                    w.erase(w.begin()+static_cast<long>(i));
                    break;
                }
            }
        }
        // if need more, allow duplicates from original weights
        if(k > n){
            vector<int> fullIdx(n); iota(fullIdx.begin(), fullIdx.end(), 0);
            vector<double> fullW(weights.begin(), weights.end());
            for(double& x : fullW){ if(!isfinite(x) || x < 0.0) x = 0.0; }
            auto sumFull = accumulate(fullW.begin(), fullW.end(), 0.0);
            if(sumFull <= 0.0){
                uniform_int_distribution<int> uni(0, n-1);
                for(int t=0; t<k-n; ++t){ result.push_back(fullIdx[uni(rng)]); }
            }else{
                for(int t=0; t<k-n; ++t){
                    // draw one from full
                    double r = uni01(rng) * sumFull;
                    double acc = 0.0; int pick = fullIdx.back();
                    for(int i=0;i<n;++i){ acc += fullW[i]; if(r <= acc){ pick = fullIdx[i]; break; } }
                    result.push_back(pick);
                }
            }
        }
    }else{
        for(int t=0; t<k; ++t){
            result.push_back(draw_one());
        }
    }
    return result;
}

#endif // SAMPLE_H