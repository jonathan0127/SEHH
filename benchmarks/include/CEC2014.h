#ifndef CEC2014_H
#define CEC2014_H

#include <array>
#include <mutex>
#include <stdexcept>
#include <vector>
#include "config/para_setting.h"

using namespace std;

void cec14_test_func(double *x, double *f, int nx, int mx, int func_num);
void cec14_test_func_fortrain(double *x, double *f, int nx, int mx, int func_num);

namespace CEC2014FUNCTION {

namespace detail {

// CEC2014 biases for F1-F30
constexpr array<double, 31> biases = {
    0.0,      // placeholder for index 0
    100.0,    // F1 - Ellips
    200.0,    // F2 - Bent Cigar
    300.0,    // F3 - Discus
    400.0,    // F4 - Rosenbrock
    500.0,    // F5 - Ackley
    600.0,    // F6 - Weierstrass
    700.0,    // F7 - Griewank
    800.0,    // F8 - Rastrigin (no rotation)
    900.0,    // F9 - Rastrigin
    1000.0,   // F10 - Schwefel (no rotation)
    1100.0,   // F11 - Schwefel
    1200.0,   // F12 - Katsuura
    1300.0,   // F13 - HappyCat
    1400.0,   // F14 - HGBat
    1500.0,   // F15 - Griewank-Rosenbrock
    1600.0,   // F16 - Expanded Scaffer's F6
    1700.0,   // F17 - Hybrid 1
    1800.0,   // F18 - Hybrid 2
    1900.0,   // F19 - Hybrid 3
    2000.0,   // F20 - Hybrid 4
    2100.0,   // F21 - Hybrid 5
    2200.0,   // F22 - Hybrid 6
    2300.0,   // F23 - Composition 1
    2400.0,   // F24 - Composition 2
    2500.0,   // F25 - Composition 3
    2600.0,   // F26 - Composition 4
    2700.0,   // F27 - Composition 5
    2800.0,   // F28 - Composition 6
    2900.0,   // F29 - Composition 7
    3000.0    // F30 - Composition 8
};

inline void validate_dimension(const vector<double> &x) {
    if (x.empty()) {
        throw invalid_argument("CEC2014 functions require at least one dimension.");
    }
    const size_t n = x.size();
    if (n != 2 && n != 10 && n != 20 && n != 30 && n != 50 && n != 100) {
        throw invalid_argument("CEC2014 official benchmarks are defined only for dimensions 2, 10, 20, 30, 50, or 100.");
    }
}

inline double call_raw(int func_num, const vector<double> &x) {
    
    vector<double> x_copy(x.begin(), x.end());
    array<double, 1> f{0.0};

    if (ParameterSetting::FUNCTION_CHOICE == 0) {
        cec14_test_func_fortrain(x_copy.data(), f.data(), static_cast<int>(x_copy.size()), 1, func_num);
        return f[0];
    }
    
    static mutex evaluator_mutex;
    lock_guard<mutex> lock(evaluator_mutex);
    cec14_test_func(x_copy.data(), f.data(), static_cast<int>(x_copy.size()), 1, func_num);
    return f[0];
}

inline double evaluate(const vector<double> &x, int func_num, int fitness_type) {
    if (func_num < 1 || func_num > 30) {
        throw out_of_range("CEC2014 function index must be between 1 and 30.");
    }

    validate_dimension(x);

    const double value = call_raw(func_num, x);

    switch (fitness_type) {
        case 0: // ORIGINAL
            return value;
        case 1: // NO BIAS
            return value - biases[func_num];
        case 2: // DEVIDED BY ANSWER
            return value / biases[func_num];
        default:
            throw invalid_argument("Invalid fitness type. Must be 0 (ORIGINAL), 1 (NO BIAS), or 2 (DEVIDED BY ANSWER).");
    }
}

} // namespace detail

// Original functions with bias
inline double CEC2014_F1(const vector<double> &x) { return detail::evaluate(x, 1, 0); }
inline double CEC2014_F2(const vector<double> &x) { return detail::evaluate(x, 2, 0); }
inline double CEC2014_F3(const vector<double> &x) { return detail::evaluate(x, 3, 0); }
inline double CEC2014_F4(const vector<double> &x) { return detail::evaluate(x, 4, 0); }
inline double CEC2014_F5(const vector<double> &x) { return detail::evaluate(x, 5, 0); }
inline double CEC2014_F6(const vector<double> &x) { return detail::evaluate(x, 6, 0); }
inline double CEC2014_F7(const vector<double> &x) { return detail::evaluate(x, 7, 0); }
inline double CEC2014_F8(const vector<double> &x) { return detail::evaluate(x, 8, 0); }
inline double CEC2014_F9(const vector<double> &x) { return detail::evaluate(x, 9, 0); }
inline double CEC2014_F10(const vector<double> &x) { return detail::evaluate(x, 10, 0); }
inline double CEC2014_F11(const vector<double> &x) { return detail::evaluate(x, 11, 0); }
inline double CEC2014_F12(const vector<double> &x) { return detail::evaluate(x, 12, 0); }
inline double CEC2014_F13(const vector<double> &x) { return detail::evaluate(x, 13, 0); }
inline double CEC2014_F14(const vector<double> &x) { return detail::evaluate(x, 14, 0); }
inline double CEC2014_F15(const vector<double> &x) { return detail::evaluate(x, 15, 0); }
inline double CEC2014_F16(const vector<double> &x) { return detail::evaluate(x, 16, 0); }
inline double CEC2014_F17(const vector<double> &x) { return detail::evaluate(x, 17, 0); }
inline double CEC2014_F18(const vector<double> &x) { return detail::evaluate(x, 18, 0); }
inline double CEC2014_F19(const vector<double> &x) { return detail::evaluate(x, 19, 0); }
inline double CEC2014_F20(const vector<double> &x) { return detail::evaluate(x, 20, 0); }
inline double CEC2014_F21(const vector<double> &x) { return detail::evaluate(x, 21, 0); }
inline double CEC2014_F22(const vector<double> &x) { return detail::evaluate(x, 22, 0); }
inline double CEC2014_F23(const vector<double> &x) { return detail::evaluate(x, 23, 0); }
inline double CEC2014_F24(const vector<double> &x) { return detail::evaluate(x, 24, 0); }
inline double CEC2014_F25(const vector<double> &x) { return detail::evaluate(x, 25, 0); }
inline double CEC2014_F26(const vector<double> &x) { return detail::evaluate(x, 26, 0); }
inline double CEC2014_F27(const vector<double> &x) { return detail::evaluate(x, 27, 0); }
inline double CEC2014_F28(const vector<double> &x) { return detail::evaluate(x, 28, 0); }
inline double CEC2014_F29(const vector<double> &x) { return detail::evaluate(x, 29, 0); }
inline double CEC2014_F30(const vector<double> &x) { return detail::evaluate(x, 30, 0); }

// Functions without bias
inline double CEC2014_F1_NO_BIAS(const vector<double> &x) { return detail::evaluate(x, 1, 1); }
inline double CEC2014_F2_NO_BIAS(const vector<double> &x) { return detail::evaluate(x, 2, 1); }
inline double CEC2014_F3_NO_BIAS(const vector<double> &x) { return detail::evaluate(x, 3, 1); }
inline double CEC2014_F4_NO_BIAS(const vector<double> &x) { return detail::evaluate(x, 4, 1); }
inline double CEC2014_F5_NO_BIAS(const vector<double> &x) { return detail::evaluate(x, 5, 1); }
inline double CEC2014_F6_NO_BIAS(const vector<double> &x) { return detail::evaluate(x, 6, 1); }
inline double CEC2014_F7_NO_BIAS(const vector<double> &x) { return detail::evaluate(x, 7, 1); }
inline double CEC2014_F8_NO_BIAS(const vector<double> &x) { return detail::evaluate(x, 8, 1); }
inline double CEC2014_F9_NO_BIAS(const vector<double> &x) { return detail::evaluate(x, 9, 1); }
inline double CEC2014_F10_NO_BIAS(const vector<double> &x) { return detail::evaluate(x, 10, 1); }
inline double CEC2014_F11_NO_BIAS(const vector<double> &x) { return detail::evaluate(x, 11, 1); }
inline double CEC2014_F12_NO_BIAS(const vector<double> &x) { return detail::evaluate(x, 12, 1); }
inline double CEC2014_F13_NO_BIAS(const vector<double> &x) { return detail::evaluate(x, 13, 1); }
inline double CEC2014_F14_NO_BIAS(const vector<double> &x) { return detail::evaluate(x, 14, 1); }
inline double CEC2014_F15_NO_BIAS(const vector<double> &x) { return detail::evaluate(x, 15, 1); }
inline double CEC2014_F16_NO_BIAS(const vector<double> &x) { return detail::evaluate(x, 16, 1); }
inline double CEC2014_F17_NO_BIAS(const vector<double> &x) { return detail::evaluate(x, 17, 1); }
inline double CEC2014_F18_NO_BIAS(const vector<double> &x) { return detail::evaluate(x, 18, 1); }
inline double CEC2014_F19_NO_BIAS(const vector<double> &x) { return detail::evaluate(x, 19, 1); }
inline double CEC2014_F20_NO_BIAS(const vector<double> &x) { return detail::evaluate(x, 20, 1); }
inline double CEC2014_F21_NO_BIAS(const vector<double> &x) { return detail::evaluate(x, 21, 1); }
inline double CEC2014_F22_NO_BIAS(const vector<double> &x) { return detail::evaluate(x, 22, 1); }
inline double CEC2014_F23_NO_BIAS(const vector<double> &x) { return detail::evaluate(x, 23, 1); }
inline double CEC2014_F24_NO_BIAS(const vector<double> &x) { return detail::evaluate(x, 24, 1); }
inline double CEC2014_F25_NO_BIAS(const vector<double> &x) { return detail::evaluate(x, 25, 1); }
inline double CEC2014_F26_NO_BIAS(const vector<double> &x) { return detail::evaluate(x, 26, 1); }
inline double CEC2014_F27_NO_BIAS(const vector<double> &x) { return detail::evaluate(x, 27, 1); }
inline double CEC2014_F28_NO_BIAS(const vector<double> &x) { return detail::evaluate(x, 28, 1); }
inline double CEC2014_F29_NO_BIAS(const vector<double> &x) { return detail::evaluate(x, 29, 1); }
inline double CEC2014_F30_NO_BIAS(const vector<double> &x) { return detail::evaluate(x, 30, 1); }

// Functions divided by bias
inline double CEC2014_F1_DEVIDED(const vector<double> &x) { return detail::evaluate(x, 1, 2); }
inline double CEC2014_F2_DEVIDED(const vector<double> &x) { return detail::evaluate(x, 2, 2); }
inline double CEC2014_F3_DEVIDED(const vector<double> &x) { return detail::evaluate(x, 3, 2); }
inline double CEC2014_F4_DEVIDED(const vector<double> &x) { return detail::evaluate(x, 4, 2); }
inline double CEC2014_F5_DEVIDED(const vector<double> &x) { return detail::evaluate(x, 5, 2); }
inline double CEC2014_F6_DEVIDED(const vector<double> &x) { return detail::evaluate(x, 6, 2); }
inline double CEC2014_F7_DEVIDED(const vector<double> &x) { return detail::evaluate(x, 7, 2); }
inline double CEC2014_F8_DEVIDED(const vector<double> &x) { return detail::evaluate(x, 8, 2); }
inline double CEC2014_F9_DEVIDED(const vector<double> &x) { return detail::evaluate(x, 9, 2); }
inline double CEC2014_F10_DEVIDED(const vector<double> &x) { return detail::evaluate(x, 10, 2); }
inline double CEC2014_F11_DEVIDED(const vector<double> &x) { return detail::evaluate(x, 11, 2); }
inline double CEC2014_F12_DEVIDED(const vector<double> &x) { return detail::evaluate(x, 12, 2); }
inline double CEC2014_F13_DEVIDED(const vector<double> &x) { return detail::evaluate(x, 13, 2); }
inline double CEC2014_F14_DEVIDED(const vector<double> &x) { return detail::evaluate(x, 14, 2); }
inline double CEC2014_F15_DEVIDED(const vector<double> &x) { return detail::evaluate(x, 15, 2); }
inline double CEC2014_F16_DEVIDED(const vector<double> &x) { return detail::evaluate(x, 16, 2); }
inline double CEC2014_F17_DEVIDED(const vector<double> &x) { return detail::evaluate(x, 17, 2); }
inline double CEC2014_F18_DEVIDED(const vector<double> &x) { return detail::evaluate(x, 18, 2); }
inline double CEC2014_F19_DEVIDED(const vector<double> &x) { return detail::evaluate(x, 19, 2); }
inline double CEC2014_F20_DEVIDED(const vector<double> &x) { return detail::evaluate(x, 20, 2); }
inline double CEC2014_F21_DEVIDED(const vector<double> &x) { return detail::evaluate(x, 21, 2); }
inline double CEC2014_F22_DEVIDED(const vector<double> &x) { return detail::evaluate(x, 22, 2); }
inline double CEC2014_F23_DEVIDED(const vector<double> &x) { return detail::evaluate(x, 23, 2); }
inline double CEC2014_F24_DEVIDED(const vector<double> &x) { return detail::evaluate(x, 24, 2); }
inline double CEC2014_F25_DEVIDED(const vector<double> &x) { return detail::evaluate(x, 25, 2); }
inline double CEC2014_F26_DEVIDED(const vector<double> &x) { return detail::evaluate(x, 26, 2); }
inline double CEC2014_F27_DEVIDED(const vector<double> &x) { return detail::evaluate(x, 27, 2); }
inline double CEC2014_F28_DEVIDED(const vector<double> &x) { return detail::evaluate(x, 28, 2); }
inline double CEC2014_F29_DEVIDED(const vector<double> &x) { return detail::evaluate(x, 29, 2); }
inline double CEC2014_F30_DEVIDED(const vector<double> &x) { return detail::evaluate(x, 30, 2); }

} // namespace CEC2014FUNCTION

#endif // CEC2014_H
