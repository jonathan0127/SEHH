#ifndef CEC2024_H
#define CEC2024_H

#include <array>
#include <mutex>
#include <stdexcept>
#include <vector>
using namespace std;
void cec17_test_func(double *x, double *f, int nx, int mx, int func_num);
void cec17_test_func_fortrain(double *x, double *f, int nx, int mx, int func_num);

namespace CEC2024FUNCTION {

namespace detail {

constexpr array<double, 31> biases = {
    0.0,
    100.0,
    200.0,
    300.0,
    400.0,
    500.0,
    600.0,
    700.0,
    800.0,
    900.0,
    1000.0,
    1100.0,
    1200.0,
    1300.0,
    1400.0,
    1500.0,
    1600.0,
    1700.0,
    1800.0,
    1900.0,
    2000.0,
    2100.0,
    2200.0,
    2300.0,
    2400.0,
    2500.0,
    2600.0,
    2700.0,
    2800.0,
    2900.0,
    3000.0
};

inline void validate_dimension(const vector<double> &x, int func_num) {
    if (x.empty()) {
        throw invalid_argument("CEC2024 functions require at least one dimension.");
    }

    const size_t n = x.size();
    switch (n) {
        case 2:
            if ((func_num >= 17 && func_num <= 22) || (func_num >= 29 && func_num <= 30)) {
                throw invalid_argument("CEC2024 hybrid/composition functions 17-22 and 29-30 are not defined for dimension 2.");
            }
            break;
        case 10:
        case 20:
        case 30:
        case 50:
        case 100:
            break;
        default:
            throw invalid_argument("CEC2024 benchmarks are defined only for dimensions 2, 10, 20, 30, 50, or 100.");
    }
}

inline double call_raw(int func_num, const vector<double> &x) {
    if (func_num < 1 || func_num > 30) {
        throw out_of_range("CEC2024 function index must be between 1 and 30.");
    }

    vector<double> x_copy(x.begin(), x.end());
    array<double, 1> f{0.0};

    if (ParameterSetting::FUNCTION_CHOICE == 0) {
        cec17_test_func_fortrain(x_copy.data(), f.data(), static_cast<int>(x_copy.size()), 1, func_num);
        return f[0];
    }
    
    else{
        static mutex evaluator_mutex;
        lock_guard<mutex> lock(evaluator_mutex);
        cec17_test_func(x_copy.data(), f.data(), static_cast<int>(x_copy.size()), 1, func_num);
        return f[0];
    }

    
}

inline double evaluate(const vector<double> &x, int func_num, int fitness_type) {
    validate_dimension(x, func_num);

    const double value = call_raw(func_num, x);

    switch (fitness_type) {
        case 0: // ORIGINAL
            return value;
        case 1: // NO BIAS
            return value - biases[func_num];
        case 2: // DEVIDED BY ANSWER
            return biases[func_num] == 0.0 ? value : value / biases[func_num];
        default:
            throw invalid_argument("Invalid fitness type. Must be 0 (ORIGINAL), 1 (NO BIAS), or 2 (DEVIDED BY ANSWER).");
    }
}

} // namespace detail

inline double CEC2024_F1(const vector<double> &x) { return detail::evaluate(x, 1, 0); }
inline double CEC2024_F2(const vector<double> &x) { return detail::evaluate(x, 2, 0); }
inline double CEC2024_F3(const vector<double> &x) { return detail::evaluate(x, 3, 0); }
inline double CEC2024_F4(const vector<double> &x) { return detail::evaluate(x, 4, 0); }
inline double CEC2024_F5(const vector<double> &x) { return detail::evaluate(x, 5, 0); }
inline double CEC2024_F6(const vector<double> &x) { return detail::evaluate(x, 6, 0); }
inline double CEC2024_F7(const vector<double> &x) { return detail::evaluate(x, 7, 0); }
inline double CEC2024_F8(const vector<double> &x) { return detail::evaluate(x, 8, 0); }
inline double CEC2024_F9(const vector<double> &x) { return detail::evaluate(x, 9, 0); }
inline double CEC2024_F10(const vector<double> &x) { return detail::evaluate(x, 10, 0); }
inline double CEC2024_F11(const vector<double> &x) { return detail::evaluate(x, 11, 0); }
inline double CEC2024_F12(const vector<double> &x) { return detail::evaluate(x, 12, 0); }
inline double CEC2024_F13(const vector<double> &x) { return detail::evaluate(x, 13, 0); }
inline double CEC2024_F14(const vector<double> &x) { return detail::evaluate(x, 14, 0); }
inline double CEC2024_F15(const vector<double> &x) { return detail::evaluate(x, 15, 0); }
inline double CEC2024_F16(const vector<double> &x) { return detail::evaluate(x, 16, 0); }
inline double CEC2024_F17(const vector<double> &x) { return detail::evaluate(x, 17, 0); }
inline double CEC2024_F18(const vector<double> &x) { return detail::evaluate(x, 18, 0); }
inline double CEC2024_F19(const vector<double> &x) { return detail::evaluate(x, 19, 0); }
inline double CEC2024_F20(const vector<double> &x) { return detail::evaluate(x, 20, 0); }
inline double CEC2024_F21(const vector<double> &x) { return detail::evaluate(x, 21, 0); }
inline double CEC2024_F22(const vector<double> &x) { return detail::evaluate(x, 22, 0); }
inline double CEC2024_F23(const vector<double> &x) { return detail::evaluate(x, 23, 0); }
inline double CEC2024_F24(const vector<double> &x) { return detail::evaluate(x, 24, 0); }
inline double CEC2024_F25(const vector<double> &x) { return detail::evaluate(x, 25, 0); }
inline double CEC2024_F26(const vector<double> &x) { return detail::evaluate(x, 26, 0); }
inline double CEC2024_F27(const vector<double> &x) { return detail::evaluate(x, 27, 0); }
inline double CEC2024_F28(const vector<double> &x) { return detail::evaluate(x, 28, 0); }
inline double CEC2024_F29(const vector<double> &x) { return detail::evaluate(x, 29, 0); }
inline double CEC2024_F30(const vector<double> &x) { return detail::evaluate(x, 30, 0); }

inline double CEC2024_F1_NO_BIAS(const vector<double> &x) { return detail::evaluate(x, 1, 1); }
inline double CEC2024_F2_NO_BIAS(const vector<double> &x) { return detail::evaluate(x, 2, 1); }
inline double CEC2024_F3_NO_BIAS(const vector<double> &x) { return detail::evaluate(x, 3, 1); }
inline double CEC2024_F4_NO_BIAS(const vector<double> &x) { return detail::evaluate(x, 4, 1); }
inline double CEC2024_F5_NO_BIAS(const vector<double> &x) { return detail::evaluate(x, 5, 1); }
inline double CEC2024_F6_NO_BIAS(const vector<double> &x) { return detail::evaluate(x, 6, 1); }
inline double CEC2024_F7_NO_BIAS(const vector<double> &x) { return detail::evaluate(x, 7, 1); }
inline double CEC2024_F8_NO_BIAS(const vector<double> &x) { return detail::evaluate(x, 8, 1); }
inline double CEC2024_F9_NO_BIAS(const vector<double> &x) { return detail::evaluate(x, 9, 1); }
inline double CEC2024_F10_NO_BIAS(const vector<double> &x) { return detail::evaluate(x, 10, 1); }
inline double CEC2024_F11_NO_BIAS(const vector<double> &x) { return detail::evaluate(x, 11, 1); }
inline double CEC2024_F12_NO_BIAS(const vector<double> &x) { return detail::evaluate(x, 12, 1); }
inline double CEC2024_F13_NO_BIAS(const vector<double> &x) { return detail::evaluate(x, 13, 1); }
inline double CEC2024_F14_NO_BIAS(const vector<double> &x) { return detail::evaluate(x, 14, 1); }
inline double CEC2024_F15_NO_BIAS(const vector<double> &x) { return detail::evaluate(x, 15, 1); }
inline double CEC2024_F16_NO_BIAS(const vector<double> &x) { return detail::evaluate(x, 16, 1); }
inline double CEC2024_F17_NO_BIAS(const vector<double> &x) { return detail::evaluate(x, 17, 1); }
inline double CEC2024_F18_NO_BIAS(const vector<double> &x) { return detail::evaluate(x, 18, 1); }
inline double CEC2024_F19_NO_BIAS(const vector<double> &x) { return detail::evaluate(x, 19, 1); }
inline double CEC2024_F20_NO_BIAS(const vector<double> &x) { return detail::evaluate(x, 20, 1); }
inline double CEC2024_F21_NO_BIAS(const vector<double> &x) { return detail::evaluate(x, 21, 1); }
inline double CEC2024_F22_NO_BIAS(const vector<double> &x) { return detail::evaluate(x, 22, 1); }
inline double CEC2024_F23_NO_BIAS(const vector<double> &x) { return detail::evaluate(x, 23, 1); }
inline double CEC2024_F24_NO_BIAS(const vector<double> &x) { return detail::evaluate(x, 24, 1); }
inline double CEC2024_F25_NO_BIAS(const vector<double> &x) { return detail::evaluate(x, 25, 1); }
inline double CEC2024_F26_NO_BIAS(const vector<double> &x) { return detail::evaluate(x, 26, 1); }
inline double CEC2024_F27_NO_BIAS(const vector<double> &x) { return detail::evaluate(x, 27, 1); }
inline double CEC2024_F28_NO_BIAS(const vector<double> &x) { return detail::evaluate(x, 28, 1); }
inline double CEC2024_F29_NO_BIAS(const vector<double> &x) { return detail::evaluate(x, 29, 1); }
inline double CEC2024_F30_NO_BIAS(const vector<double> &x) { return detail::evaluate(x, 30, 1); }

inline double CEC2024_F_NO_BIAS(int func_num, const vector<double> &x) { return detail::evaluate(x, func_num, 1); }
inline double CEC2024_F_DEVIDED(int func_num, const vector<double> &x) { return detail::evaluate(x, func_num, 2); }

} // namespace CEC2024FUNCTION

#endif // CEC2024_H