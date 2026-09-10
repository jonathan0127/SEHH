#ifndef CEC2022_H
#define CEC2022_H

#include <array>
#include <mutex>
#include <stdexcept>
#include <vector>
#include "config/para_setting.h"
using namespace std;
void cec22_test_func(double *x, double *f, int nx, int mx, int func_num);
void cec22_test_func_fortrain(double *x, double *f, int nx, int mx, int func_num);


namespace CEC2022FUNCTION {

namespace detail {

constexpr array<double, 13> biases = {
    0.0,
    300.0,
    400.0,
    600.0,
    800.0,
    900.0,
    1800.0,
    2000.0,
    2200.0,
    2300.0,
    2400.0,
    2600.0,
    2700.0
};

inline void validate_dimension(const vector<double> &x) {
    if (x.empty()) {
        throw invalid_argument("CEC2022 functions require at least one dimension.");
    }
    const size_t n = x.size();
    if (n != 2 && n != 10 && n != 20) {
        throw invalid_argument("CEC2022 official benchmarks are defined only for dimensions 2, 10, or 20.");
    }
}

inline double call_raw(int func_num, const vector<double> &x) {
    
    vector<double> x_copy(x.begin(), x.end());
    array<double, 1> f{0.0};


    if (ParameterSetting::FUNCTION_CHOICE == 0) {
        cec22_test_func_fortrain(x_copy.data(), f.data(), static_cast<int>(x_copy.size()), 1, func_num);
        return f[0];
    }
    
    else{
        static mutex evaluator_mutex;
        lock_guard<mutex> lock(evaluator_mutex);
        cec22_test_func(x_copy.data(), f.data(), static_cast<int>(x_copy.size()), 1, func_num);
        return f[0];
    }
}

inline double evaluate(const vector<double> &x, int func_num, int fitness_type) {
    if (func_num < 1 || func_num > 12) {
        throw out_of_range("CEC2022 function index must be between 1 and 12.");
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

inline double CEC2022_F1(const vector<double> &x) { return detail::evaluate(x, 1, 0); }
inline double CEC2022_F2(const vector<double> &x) { return detail::evaluate(x, 2, 0); }
inline double CEC2022_F3(const vector<double> &x) { return detail::evaluate(x, 3, 0); }
inline double CEC2022_F4(const vector<double> &x) { return detail::evaluate(x, 4, 0); }
inline double CEC2022_F5(const vector<double> &x) { return detail::evaluate(x, 5, 0); }
inline double CEC2022_F6(const vector<double> &x) { return detail::evaluate(x, 6, 0); }
inline double CEC2022_F7(const vector<double> &x) { return detail::evaluate(x, 7, 0); }
inline double CEC2022_F8(const vector<double> &x) { return detail::evaluate(x, 8, 0); }
inline double CEC2022_F9(const vector<double> &x) { return detail::evaluate(x, 9, 0); }
inline double CEC2022_F10(const vector<double> &x) { return detail::evaluate(x, 10, 0); }
inline double CEC2022_F11(const vector<double> &x) { return detail::evaluate(x, 11, 0); }
inline double CEC2022_F12(const vector<double> &x) { return detail::evaluate(x, 12, 0); }

inline double CEC2022_F1_NO_BIAS(const vector<double> &x) { return detail::evaluate(x, 1, 1); }
inline double CEC2022_F2_NO_BIAS(const vector<double> &x) { return detail::evaluate(x, 2, 1); }
inline double CEC2022_F3_NO_BIAS(const vector<double> &x) { return detail::evaluate(x, 3, 1); }
inline double CEC2022_F4_NO_BIAS(const vector<double> &x) { return detail::evaluate(x, 4, 1); }
inline double CEC2022_F5_NO_BIAS(const vector<double> &x) { return detail::evaluate(x, 5, 1); }
inline double CEC2022_F6_NO_BIAS(const vector<double> &x) { return detail::evaluate(x, 6, 1); }
inline double CEC2022_F7_NO_BIAS(const vector<double> &x) { return detail::evaluate(x, 7, 1); }
inline double CEC2022_F8_NO_BIAS(const vector<double> &x) { return detail::evaluate(x, 8, 1); }
inline double CEC2022_F9_NO_BIAS(const vector<double> &x) { return detail::evaluate(x, 9, 1); }
inline double CEC2022_F10_NO_BIAS(const vector<double> &x) { return detail::evaluate(x, 10, 1); }
inline double CEC2022_F11_NO_BIAS(const vector<double> &x) { return detail::evaluate(x, 11, 1); }
inline double CEC2022_F12_NO_BIAS(const vector<double> &x) { return detail::evaluate(x, 12, 1); }

inline double CEC2022_F1_DEVIDED(const vector<double> &x) { return detail::evaluate(x, 1, 2); }
inline double CEC2022_F2_DEVIDED(const vector<double> &x) { return detail::evaluate(x, 2, 2); }
inline double CEC2022_F3_DEVIDED(const vector<double> &x) { return detail::evaluate(x, 3, 2); }
inline double CEC2022_F4_DEVIDED(const vector<double> &x) { return detail::evaluate(x, 4, 2); }
inline double CEC2022_F5_DEVIDED(const vector<double> &x) { return detail::evaluate(x, 5, 2); }
inline double CEC2022_F6_DEVIDED(const vector<double> &x) { return detail::evaluate(x, 6, 2); }
inline double CEC2022_F7_DEVIDED(const vector<double> &x) { return detail::evaluate(x, 7, 2); }
inline double CEC2022_F8_DEVIDED(const vector<double> &x) { return detail::evaluate(x, 8, 2); }
inline double CEC2022_F9_DEVIDED(const vector<double> &x) { return detail::evaluate(x, 9, 2); }
inline double CEC2022_F10_DEVIDED(const vector<double> &x) { return detail::evaluate(x, 10, 2); }
inline double CEC2022_F11_DEVIDED(const vector<double> &x) { return detail::evaluate(x, 11, 2); }
inline double CEC2022_F12_DEVIDED(const vector<double> &x) { return detail::evaluate(x, 12, 2); }

} // namespace CEC2022FUNCTION

#endif // CEC2022_H