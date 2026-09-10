#ifndef CEC2020_H
#define CEC2020_H

#include <array>
#include <mutex>
#include <stdexcept>
#include <vector>
#include "config/para_setting.h"

using namespace std;
void cec20_test_func(double *x, double *f, int nx, int mx, int func_num);
void cec20_test_func_fortrain(double *x, double *f, int nx, int mx, int func_num);


namespace CEC2020FUNCTION {

namespace detail {

// CEC2020 biases for F1-F10
// Mapping: F1->100, F2->1100, F3->700, F4->1900, F5->1700, F6->1600, F7->2100, F8->2200, F9->2400, F10->2500
constexpr array<double, 11> biases = {
    0.0,     // placeholder for index 0
    100.0,   // F1 - Bent Cigar
    1100.0,  // F2 - Schwefel
    700.0,   // F3 - Bi-Rastrigin
    1900.0,  // F4 - Griewank-Rosenbrock
    1700.0,  // F5 - Hybrid 1
    1600.0,  // F6 - Hybrid 6
    2100.0,  // F7 - Hybrid 5
    2200.0,  // F8 - Composition 2
    2400.0,  // F9 - Composition 4
    2500.0   // F10 - Composition 5
};

inline void validate_dimension(const vector<double> &x) {
    if (x.empty()) {
        throw invalid_argument("CEC2020 functions require at least one dimension.");
    }
    const size_t n = x.size();
    if (n != 5 && n != 10 && n != 15 && n != 20) {
        throw invalid_argument("CEC2020 official benchmarks are defined only for dimensions 5, 10, 15, or 20.");
    }
}

inline double call_raw(int func_num, const vector<double> &x) {
    
    vector<double> x_copy(x.begin(), x.end());
    array<double, 1> f{0.0};

    // CEC2020 currently does not have a fortrain version
    if (ParameterSetting::FUNCTION_CHOICE == 0) {
        cec20_test_func_fortrain(x_copy.data(), f.data(), static_cast<int>(x_copy.size()), 1, func_num);
        return f[0];
    }
    
    static mutex evaluator_mutex;
    lock_guard<mutex> lock(evaluator_mutex);
    cec20_test_func(x_copy.data(), f.data(), static_cast<int>(x_copy.size()), 1, func_num);
    return f[0];
}

inline double evaluate(const vector<double> &x, int func_num, int fitness_type) {
    if (func_num < 1 || func_num > 10) {
        throw out_of_range("CEC2020 function index must be between 1 and 10.");
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

inline double CEC2020_F1(const vector<double> &x) { return detail::evaluate(x, 1, 0); }
inline double CEC2020_F2(const vector<double> &x) { return detail::evaluate(x, 2, 0); }
inline double CEC2020_F3(const vector<double> &x) { return detail::evaluate(x, 3, 0); }
inline double CEC2020_F4(const vector<double> &x) { return detail::evaluate(x, 4, 0); }
inline double CEC2020_F5(const vector<double> &x) { return detail::evaluate(x, 5, 0); }
inline double CEC2020_F6(const vector<double> &x) { return detail::evaluate(x, 6, 0); }
inline double CEC2020_F7(const vector<double> &x) { return detail::evaluate(x, 7, 0); }
inline double CEC2020_F8(const vector<double> &x) { return detail::evaluate(x, 8, 0); }
inline double CEC2020_F9(const vector<double> &x) { return detail::evaluate(x, 9, 0); }
inline double CEC2020_F10(const vector<double> &x) { return detail::evaluate(x, 10, 0); }

inline double CEC2020_F1_NO_BIAS(const vector<double> &x) { return detail::evaluate(x, 1, 1); }
inline double CEC2020_F2_NO_BIAS(const vector<double> &x) { return detail::evaluate(x, 2, 1); }
inline double CEC2020_F3_NO_BIAS(const vector<double> &x) { return detail::evaluate(x, 3, 1); }
inline double CEC2020_F4_NO_BIAS(const vector<double> &x) { return detail::evaluate(x, 4, 1); }
inline double CEC2020_F5_NO_BIAS(const vector<double> &x) { return detail::evaluate(x, 5, 1); }
inline double CEC2020_F6_NO_BIAS(const vector<double> &x) { return detail::evaluate(x, 6, 1); }
inline double CEC2020_F7_NO_BIAS(const vector<double> &x) { return detail::evaluate(x, 7, 1); }
inline double CEC2020_F8_NO_BIAS(const vector<double> &x) { return detail::evaluate(x, 8, 1); }
inline double CEC2020_F9_NO_BIAS(const vector<double> &x) { return detail::evaluate(x, 9, 1); }
inline double CEC2020_F10_NO_BIAS(const vector<double> &x) { return detail::evaluate(x, 10, 1); }

inline double CEC2020_F1_DEVIDED(const vector<double> &x) { return detail::evaluate(x, 1, 2); }
inline double CEC2020_F2_DEVIDED(const vector<double> &x) { return detail::evaluate(x, 2, 2); }
inline double CEC2020_F3_DEVIDED(const vector<double> &x) { return detail::evaluate(x, 3, 2); }
inline double CEC2020_F4_DEVIDED(const vector<double> &x) { return detail::evaluate(x, 4, 2); }
inline double CEC2020_F5_DEVIDED(const vector<double> &x) { return detail::evaluate(x, 5, 2); }
inline double CEC2020_F6_DEVIDED(const vector<double> &x) { return detail::evaluate(x, 6, 2); }
inline double CEC2020_F7_DEVIDED(const vector<double> &x) { return detail::evaluate(x, 7, 2); }
inline double CEC2020_F8_DEVIDED(const vector<double> &x) { return detail::evaluate(x, 8, 2); }
inline double CEC2020_F9_DEVIDED(const vector<double> &x) { return detail::evaluate(x, 9, 2); }
inline double CEC2020_F10_DEVIDED(const vector<double> &x) { return detail::evaluate(x, 10, 2); }

} // namespace CEC2020FUNCTION

#endif // CEC2020_H
