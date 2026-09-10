#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <functional>
#include <cstdlib>
#include "utils/instance.h"
#include "hyper/SEHH.h"
#include "config/para_setting.h"
#include "CEC2014.h"
#include "CEC2020.h"
#include "CEC2022.h"
#include "CEC2024.h"

using namespace BENCH;

using namespace std;

int main() {
    using namespace SEHHTrainSetting;

    vector<Instance> instances = {
        // all 30D CEC2014 functions
        Instance("CEC2014 F1: Rotated High Conditioned Elliptic (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2014FUNCTION::CEC2014_F1_NO_BIAS, 1),
        Instance("CEC2014 F2: Rotated Bent Cigar (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2014FUNCTION::CEC2014_F2_NO_BIAS, 2),
        Instance("CEC2014 F3: Rotated Discus (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2014FUNCTION::CEC2014_F3_NO_BIAS, 3),
        Instance("CEC2014 F4: Shifted and Rotated Rosenbrock (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2014FUNCTION::CEC2014_F4_NO_BIAS, 4),
        Instance("CEC2014 F5: Shifted and Rotated Ackley (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2014FUNCTION::CEC2014_F5_NO_BIAS, 5),
        Instance("CEC2014 F6: Shifted and Rotated Weierstrass (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2014FUNCTION::CEC2014_F6_NO_BIAS, 6),
        Instance("CEC2014 F7: Shifted and Rotated Griewank (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2014FUNCTION::CEC2014_F7_NO_BIAS, 7),
        Instance("CEC2014 F8: Shifted Rastrigin (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2014FUNCTION::CEC2014_F8_NO_BIAS, 8),
        Instance("CEC2014 F9: Shifted and Rotated Rastrigin (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2014FUNCTION::CEC2014_F9_NO_BIAS, 9),
        Instance("CEC2014 F10: Shifted Schwefel (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2014FUNCTION::CEC2014_F10_NO_BIAS, 10),
        // Instance("CEC2014 F11: Shifted and Rotated Schwefel (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2014FUNCTION::CEC2014_F11_NO_BIAS, 11),
        // Instance("CEC2014 F12: Shifted and Rotated Katsuura (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2014FUNCTION::CEC2014_F12_NO_BIAS, 12),
        Instance("CEC2014 F13: Shifted and Rotated HappyCat (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2014FUNCTION::CEC2014_F13_NO_BIAS, 13),
        Instance("CEC2014 F14: Shifted and Rotated HGBat (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2014FUNCTION::CEC2014_F14_NO_BIAS, 14),
        Instance("CEC2014 F15: Shifted and Rotated Griewank-Rosenbrock (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2014FUNCTION::CEC2014_F15_NO_BIAS, 15),
        Instance("CEC2014 F16: Shifted and Rotated Expanded Scaffer F6 (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2014FUNCTION::CEC2014_F16_NO_BIAS, 16),
        Instance("CEC2014 F17: Hybrid Function 1 (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2014FUNCTION::CEC2014_F17_NO_BIAS, 17),
        Instance("CEC2014 F18: Hybrid Function 2 (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2014FUNCTION::CEC2014_F18_NO_BIAS, 18),
        Instance("CEC2014 F19: Hybrid Function 3 (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2014FUNCTION::CEC2014_F19_NO_BIAS, 19),
        Instance("CEC2014 F20: Hybrid Function 4 (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2014FUNCTION::CEC2014_F20_NO_BIAS, 20),
        Instance("CEC2014 F21: Hybrid Function 5 (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2014FUNCTION::CEC2014_F21_NO_BIAS, 21),
        Instance("CEC2014 F22: Hybrid Function 6 (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2014FUNCTION::CEC2014_F22_NO_BIAS, 22),
        Instance("CEC2014 F23: Composition Function 1 (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2014FUNCTION::CEC2014_F23_NO_BIAS, 23),
        Instance("CEC2014 F24: Composition Function 2 (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2014FUNCTION::CEC2014_F24_NO_BIAS, 24),
        Instance("CEC2014 F25: Composition Function 3 (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2014FUNCTION::CEC2014_F25_NO_BIAS, 25),
        // Instance("CEC2014 F26: Composition Function 4 (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2014FUNCTION::CEC2014_F26_NO_BIAS, 26),
        // Instance("CEC2014 F27: Composition Function 5 (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2014FUNCTION::CEC2014_F27_NO_BIAS, 27),
        Instance("CEC2014 F28: Composition Function 6 (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2014FUNCTION::CEC2014_F28_NO_BIAS, 28),
        Instance("CEC2014 F29: Composition Function 7 (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2014FUNCTION::CEC2014_F29_NO_BIAS, 29),
        Instance("CEC2014 F30: Composition Function 8 (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2014FUNCTION::CEC2014_F30_NO_BIAS, 30),
    };

    SEHH sehh(
        instances,
        MAX_EVALUATIONS,
        MUTATION_RATE,
        CROSSOVER_RATE,
        SEQUENCE_LENGTH,
        NUM_SEARCHERS,
        INIT_NUM_SAMPLES,
        MIN_SAMPLE_PER_REGION,
        MAX_SAMPLE_PER_REGION,
        NUM_PLAYERS
    );
    sehh.run("result/output/SEHH_output.txt");
    return 0;
}
