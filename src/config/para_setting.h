#ifndef PARA_SETTING_H
#define PARA_SETTING_H

namespace ParameterSetting {
    inline constexpr int FUNCTION_CHOICE = 0; // 0 for train, 1 for test
}

#include <vector>
#include "benchmark.h"
#include "CEC2020.h"
#include "CEC2022.h"
#include "CEC2024.h"

// using namespace CEC2020FUNCTION;
// using namespace CEC2022FUNCTION;
#ifndef BENCH_YEAR
#define BENCH_YEAR 2024
#endif
#ifndef DIMENSION
#define DIMENSION 30
#endif

#if BENCH_YEAR == 2014
    namespace BENCH = CEC2014FUNCTION;
    #if DIMENSION == 30
        namespace BenchmarkSetting { const Benchmark benchmark = CEC2014::benchmark_30D; }
    #endif
#endif

#if BENCH_YEAR == 2022
    namespace BENCH = CEC2022FUNCTION;
    #if DIMENSION == 10
        namespace BenchmarkSetting { const Benchmark benchmark = CEC2022::benchmark_10D; }
    #elif DIMENSION == 20
        namespace BenchmarkSetting { const Benchmark benchmark = CEC2022::benchmark_20D; }
    #endif
#endif

#if BENCH_YEAR == 2024
    namespace BENCH = CEC2024FUNCTION;
    #if DIMENSION == 30
        namespace BenchmarkSetting { const Benchmark benchmark = CEC2024::benchmark_30D; }
    #endif
#endif
// General settings
constexpr int RUNS = 25;

// 注意一下 MAX_GENERATIONS 要被 POPULATION_SIZE 整除
constexpr int META_POPULATION_SIZE = 600;
constexpr int SEHH_POPULATION_FINAL = 10;
constexpr int META_MAX_EVALUATIONS = 300000;
constexpr int HYPER_MAX_EVALUATIONS = 30000;
constexpr int ITERATIONS_FOR_EVALUATION = 3;
constexpr int FITNESS_TYPE = 1; // 0: ORIGINAL, 1: NO BIAS, 2: DEVIDED BY ANSWER
constexpr int FITNESS_METHOD_CHOICE = 1; // 0: average fitness, 1: fitness with iteration
constexpr double EQUAL_ZERO_THRESHOLD = 1e-8;
constexpr double kEps = 4; //iteration 影響的權重
constexpr int MID_BEST_CHECK_POINT = 100;
constexpr pair<int, int> TIME_WEIGHT_RANGE = make_pair(1, 20); // time weight 範圍

constexpr int OUTPUT_CHROMOSOME_NUM = 3;
// ========== mutation rate ===========
namespace MutationRateParameter{
    double mutationLowerBound = 0.2;
    double mutationUpperBound = 0.4;
    double mutationStep = 0.05; // 每次加多少 mutation rate
    int stepToChange = 6; // generation 幾個 iteration 沒動調整一次 mutation rate
};
// ====================================

// ========== SEHH Training Parameters ==========
namespace SEHHTrainSetting {
    constexpr int MAX_EVALUATIONS = HYPER_MAX_EVALUATIONS;
    inline double MUTATION_RATE = MutationRateParameter::mutationLowerBound;
    constexpr double CROSSOVER_RATE = 0.9;
    constexpr int SEQUENCE_LENGTH = OUTPUT_CHROMOSOME_NUM; // 序列長度
    constexpr int NUM_SEARCHERS = 10;                     // 搜尋者數量
    constexpr int INIT_NUM_SAMPLES = 5;                   // 每個區域初始樣本數
    constexpr int MIN_SAMPLE_PER_REGION = 2;              // 每個區域最小樣本數
    constexpr int MAX_SAMPLE_PER_REGION = 8;              // 每個區域最大樣本數
    constexpr int NUM_PLAYERS = 3;                        // 競賽選擇玩家數
}
// =================================================

const string BASE_ALGORITHM = "SEHH-1";
// all algorithms
// 執行時請註解掉不需要的 algorithm
const vector<string> algorithmName = {
    // "PSO",
    // "GA",
    // "DE_r1",
    // "DE_b1",
    // "DE_r2",
    // "DE_b2",
    // "LSHADE",
    "SEHH-1",
    "SEHH-2",
    // "SEHH-3",
};


#endif