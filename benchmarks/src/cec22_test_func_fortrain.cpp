#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#ifndef CEC22_DATA_DIR
#define CEC22_DATA_DIR "benchmarks/data/data_2022"
#endif
using namespace std;

namespace {

using Matrix = vector<double>;

struct PairHash {
    size_t operator()(const pair<string, int> &key) const noexcept {
        size_t h1 = hash<string>()(key.first);
        size_t h2 = hash<int>()(key.second);
        return h1 ^ (h2 << 1);
    }
};

struct TripleHash {
    size_t operator()(const tuple<string, int, int> &key) const noexcept {
        const auto &[s, i1, i2] = key;
        size_t h1 = hash<string>()(s);
        size_t h2 = hash<int>()(i1);
        size_t h3 = hash<int>()(i2);
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

class DataCache {
public:
    static DataCache &instance() {
        static DataCache cache;
        return cache;
    }

    const vector<double> &shift(const string &file, int dim) {
        lock_guard<mutex> lock(mutex_);
        auto key = make_pair(file, dim);
        auto it = shifts_.find(key);
        if (it != shifts_.end()) {
            return it->second;
        }
        vector<double> values(dim, 0.0);
        ifstream fin(make_path(file));
        if (fin.is_open()) {
            for (int i = 0; i < dim && fin; ++i) {
                fin >> values[i];
            }
        } else {
            cerr << "[cec22] warning: failed to open shift file: " << make_path(file) << '\n';
        }
        return shifts_.emplace(key, move(values)).first->second;
    }

    const Matrix &matrix(const string &base, int dim) {
        lock_guard<mutex> lock(mutex_);
        auto key = make_pair(base, dim);
        auto it = matrices_.find(key);
        if (it != matrices_.end()) {
            return it->second;
        }
        Matrix mat(static_cast<size_t>(dim) * dim, 0.0);
        ifstream fin(make_matrix_path(base, dim));
        if (fin.is_open()) {
            for (int i = 0; i < dim * dim && fin; ++i) {
                fin >> mat[i];
            }
        } else {
            cerr << "[cec22] warning: failed to open matrix file: " << make_matrix_path(base, dim) << '\n';
            for (int i = 0; i < dim; ++i) {
                mat[i * dim + i] = 1.0;
            }
        }
        return matrices_.emplace(key, move(mat)).first->second;
    }

    const vector<int> &shuffle(const string &base, int dim) {
        lock_guard<mutex> lock(mutex_);
        auto key = make_pair(base, dim);
        auto it = shuffles_.find(key);
        if (it != shuffles_.end()) {
            return it->second;
        }
        vector<int> idx(dim, 0);
        ifstream fin(make_matrix_path(base, dim));
        if (fin.is_open()) {
            for (int i = 0; i < dim && fin; ++i) {
                fin >> idx[i];
                idx[i] = max(0, idx[i] - 1);
            }
        } else {
            cerr << "[cec22] warning: failed to open shuffle file: " << make_matrix_path(base, dim) << '\n';
            for (int i = 0; i < dim; ++i) {
                idx[i] = i;
            }
        }
        return shuffles_.emplace(key, move(idx)).first->second;
    }

    const vector<double> &multi_shift(const string &file, int dim, int count) {
        lock_guard<mutex> lock(mutex_);
        auto key = make_tuple(file, dim, count);
        auto it = multi_shifts_.find(key);
        if (it != multi_shifts_.end()) {
            return it->second;
        }
        vector<double> data(static_cast<size_t>(dim) * count, 0.0);
        ifstream fin(make_path(file));
        if (fin.is_open()) {
            string line;
            int loaded = 0;
            while (loaded < count && getline(fin, line)) {
                if (line.empty()) {
                    continue;
                }
                istringstream iss(line);
                bool ok = true;
                for (int j = 0; j < dim; ++j) {
                    if (!(iss >> data[static_cast<size_t>(loaded) * dim + j])) {
                        ok = false;
                        break;
                    }
                }
                if (ok) {
                    ++loaded;
                }
            }
            if (loaded < count) {
                fin.clear();
                fin.close();
                fin.open(make_path(file));
                if (fin.is_open()) {
                    for (int i = 0; i < dim * count && fin; ++i) {
                        fin >> data[i];
                    }
                }
            }
        } else {
            cerr << "[cec22] warning: failed to open shift file: " << make_path(file) << '\n';
        }
        return multi_shifts_.emplace(key, move(data)).first->second;
    }

    const vector<double> &multi_matrix(const string &base, int dim, int count) {
        lock_guard<mutex> lock(mutex_);
        auto key = make_tuple(base, dim, count);
        auto it = multi_matrices_.find(key);
        if (it != multi_matrices_.end()) {
            return it->second;
        }
       
        vector<double> data(static_cast<size_t>(dim) * dim * count, 0.0);
        ifstream fin(make_matrix_path(base, dim));
        if (fin.is_open()) {
            for (int i = 0; i < dim * dim * count && fin; ++i) {
                fin >> data[i];
            }
        } else {
            cerr << "[cec22] warning: failed to open matrix file: " << make_matrix_path(base, dim) << '\n';
            for (int k = 0; k < count; ++k) {
                for (int i = 0; i < dim; ++i) {
                    data[k * dim * dim + i * dim + i] = 1.0;
                }
            }
        }
        return multi_matrices_.emplace(key, move(data)).first->second;
    }

private:
    DataCache() = default;

    static string make_path(const string &file) {
        return string(CEC22_DATA_DIR) + '/' + file + ".txt";
    }

    static string make_matrix_path(const string &base, int dim) {
        return string(CEC22_DATA_DIR) + '/' + base + "_D" + to_string(dim) + ".txt";
    }

    mutex mutex_;
    unordered_map<pair<string, int>, vector<double>, PairHash> shifts_;
    unordered_map<pair<string, int>, Matrix, PairHash> matrices_;
    unordered_map<pair<string, int>, vector<int>, PairHash> shuffles_;
    unordered_map<tuple<string, int, int>, vector<double>, TripleHash> multi_shifts_;
    unordered_map<tuple<string, int, int>, vector<double>, TripleHash> multi_matrices_;
};

inline vector<double> matvec(const Matrix &M, const vector<double> &v, int dim) {
    vector<double> res(dim, 0.0);
    for (int i = 0; i < dim; ++i) {
        double sum = 0.0;
        for (int j = 0; j < dim; ++j) {
            sum += M[static_cast<size_t>(i) * dim + j] * v[j];
        }
        res[i] = sum;
    }
    return res;
}

struct SRResult {
    vector<double> y;
    vector<double> z;
};

inline SRResult sr_func(const vector<double> &x,
                        const vector<double> &Os,
                        const Matrix &Mr,
                        double sh_rate,
                        int s_flag,
                        int r_flag) {
    int nx = static_cast<int>(x.size());
    SRResult res;
    res.y.assign(nx, 0.0);
    if (s_flag == 1) {
        for (int i = 0; i < nx; ++i) {
            res.y[i] = (x[i] - Os[i]) * sh_rate;
        }
    } else {
        for (int i = 0; i < nx; ++i) {
            res.y[i] = x[i] * sh_rate;
        }
    }
    if (r_flag == 1) {
        res.z = matvec(Mr, res.y, nx);
    } else {
        res.z = res.y;
    }
    return res;
}

inline double ellips(const vector<double> &x, const vector<double> &Os, const Matrix &Mr, int s_flag, int r_flag) {
    auto sr = sr_func(x, Os, Mr, 1.0, s_flag, r_flag);
    double f = 0.0;
    int nx = static_cast<int>(x.size());
    for (int i = 0; i < nx; ++i) {
        double weight = pow(10.0, 6.0 * i / max(1, nx - 1));
        f += weight * sr.z[i] * sr.z[i];
    }
    return f;
}

inline double bent_cigar(const vector<double> &x, const vector<double> &Os, const Matrix &Mr, int s_flag, int r_flag) {
    auto sr = sr_func(x, Os, Mr, 1.0, s_flag, r_flag);
    double f = sr.z[0] * sr.z[0];
    for (size_t i = 1; i < sr.z.size(); ++i) {
        f += 1e6 * sr.z[i] * sr.z[i];
    }
    return f;
}

inline double discus(const vector<double> &x, const vector<double> &Os, const Matrix &Mr, int s_flag, int r_flag) {
    auto sr = sr_func(x, Os, Mr, 1.0, s_flag, r_flag);
    double f = 1e6 * sr.z[0] * sr.z[0];
    for (size_t i = 1; i < sr.z.size(); ++i) {
        f += sr.z[i] * sr.z[i];
    }
    return f;
}

inline double rosenbrock(const vector<double> &x, const vector<double> &Os, const Matrix &Mr, int s_flag, int r_flag) {
    auto sr = sr_func(x, Os, Mr, 2.048 / 100.0, s_flag, r_flag);
    vector<double> z = sr.z;
    if (!z.empty()) {
        z[0] += 1.0;
    }
    double f = 0.0;
    int nx = static_cast<int>(x.size());
    for (int i = 0; i < nx - 1; ++i) {
        z[i + 1] += 1.0;
        double tmp1 = z[i] * z[i] - z[i + 1];
        double tmp2 = z[i] - 1.0;
        f += 100.0 * tmp1 * tmp1 + tmp2 * tmp2;
    }
    return f;
}

inline double ackley(const vector<double> &x, const vector<double> &Os, const Matrix &Mr, int s_flag, int r_flag) {
    auto sr = sr_func(x, Os, Mr, 1.0, s_flag, r_flag);
    double sum1 = 0.0;
    double sum2 = 0.0;
    int nx = static_cast<int>(x.size());
    for (int i = 0; i < nx; ++i) {
        sum1 += sr.z[i] * sr.z[i];
        sum2 += cos(2.0 * M_PI * sr.z[i]);
    }
    sum1 = -0.2 * sqrt(sum1 / nx);
    sum2 /= nx;
    return exp(1.0) - 20.0 * exp(sum1) - exp(sum2) + 20.0;
}

inline double griewank(const vector<double> &x, const vector<double> &Os, const Matrix &Mr, int s_flag, int r_flag) {
    auto sr = sr_func(x, Os, Mr, 600.0 / 100.0, s_flag, r_flag);
    double s = 0.0;
    double p = 1.0;
    int nx = static_cast<int>(x.size());
    for (int i = 0; i < nx; ++i) {
        s += sr.z[i] * sr.z[i];
        p *= cos(sr.z[i] / sqrt(static_cast<double>(i + 1)));
    }
    return 1.0 + s / 4000.0 - p;
}

inline double rastrigin(const vector<double> &x, const vector<double> &Os, const Matrix &Mr, int s_flag, int r_flag) {
    auto sr = sr_func(x, Os, Mr, 5.12 / 100.0, s_flag, r_flag);
    double f = 0.0;
    int nx = static_cast<int>(x.size());
    for (int i = 0; i < nx; ++i) {
        f += sr.z[i] * sr.z[i] - 10.0 * cos(2.0 * M_PI * sr.z[i]) + 10.0;
    }
    return f;
}

inline double schwefel(const vector<double> &x, const vector<double> &Os, const Matrix &Mr, int s_flag, int r_flag) {
    auto sr = sr_func(x, Os, Mr, 1000.0 / 100.0, s_flag, r_flag);
    int nx = static_cast<int>(x.size());
    double f = 0.0;
    for (int i = 0; i < nx; ++i) {
        double zi = sr.z[i] + 4.209687462275036e+002;
        if (zi > 500.0) {
            double temp = 500.0 - fmod(zi, 500.0);
            f -= temp * sin(sqrt(temp));
            double tmp = (zi - 500.0) / 100.0;
            f += tmp * tmp / nx;
        } else if (zi < -500.0) {
            double temp = 500.0 - fmod(fabs(zi), 500.0);
            f -= temp * sin(sqrt(temp));
            double tmp = (zi + 500.0) / 100.0;
            f += tmp * tmp / nx;
        } else {
            f -= zi * sin(sqrt(fabs(zi)));
        }
    }
    f += 4.189828872724338e+002 * nx;
    return f;
}

inline double grie_rosen(const vector<double> &x, const vector<double> &Os, const Matrix &Mr, int s_flag, int r_flag) {
    auto sr = sr_func(x, Os, Mr, 5.0 / 100.0, s_flag, r_flag);
    vector<double> z = sr.z;
    int nx = static_cast<int>(x.size());
    double f = 0.0;
    if (!z.empty()) {
        z[0] += 1.0;
    }
    for (int i = 0; i < nx - 1; ++i) {
        z[i + 1] += 1.0;
        double tmp1 = z[i] * z[i] - z[i + 1];
        double tmp2 = z[i] - 1.0;
        double temp = 100.0 * tmp1 * tmp1 + tmp2 * tmp2;
        f += (temp * temp) / 4000.0 - cos(temp) + 1.0;
    }
    double tmp1 = z[nx - 1] * z[nx - 1] - z[0];
    double tmp2 = z[nx - 1] - 1.0;
    double temp = 100.0 * tmp1 * tmp1 + tmp2 * tmp2;
    f += (temp * temp) / 4000.0 - cos(temp) + 1.0;
    return f;
}

inline double escaffer6(const vector<double> &x, const vector<double> &Os, const Matrix &Mr, int s_flag, int r_flag) {
    auto sr = sr_func(x, Os, Mr, 1.0, s_flag, r_flag);
    int nx = static_cast<int>(x.size());
    double f = 0.0;
    for (int i = 0; i < nx - 1; ++i) {
        double t1 = sin(sqrt(sr.z[i] * sr.z[i] + sr.z[i + 1] * sr.z[i + 1]));
        t1 *= t1;
        double t2 = 1.0 + 0.001 * (sr.z[i] * sr.z[i] + sr.z[i + 1] * sr.z[i + 1]);
        f += 0.5 + (t1 - 0.5) / (t2 * t2);
    }
    double t1 = sin(sqrt(sr.z[nx - 1] * sr.z[nx - 1] + sr.z[0] * sr.z[0]));
    t1 *= t1;
    double t2 = 1.0 + 0.001 * (sr.z[nx - 1] * sr.z[nx - 1] + sr.z[0] * sr.z[0]);
    f += 0.5 + (t1 - 0.5) / (t2 * t2);
    return f;
}

inline double happycat(const vector<double> &x, const vector<double> &Os, const Matrix &Mr, int s_flag, int r_flag) {
    auto sr = sr_func(x, Os, Mr, 5.0 / 100.0, s_flag, r_flag);
    double r2 = 0.0;
    double sum = 0.0;
    int nx = static_cast<int>(x.size());
    for (int i = 0; i < nx; ++i) {
        double zi = sr.z[i] - 1.0;
        r2 += zi * zi;
        sum += zi;
    }
    double alpha = 1.0 / 8.0;
    return pow(fabs(r2 - nx), 2.0 * alpha) + (0.5 * r2 + sum) / nx + 0.5;
}

inline double hgbat(const vector<double> &x, const vector<double> &Os, const Matrix &Mr, int s_flag, int r_flag) {
    auto sr = sr_func(x, Os, Mr, 5.0 / 100.0, s_flag, r_flag);
    double r2 = 0.0;
    double sum = 0.0;
    int nx = static_cast<int>(x.size());
    for (int i = 0; i < nx; ++i) {
        double zi = sr.z[i] - 1.0;
        r2 += zi * zi;
        sum += zi;
    }
    double alpha = 1.0 / 4.0;
    return pow(fabs(r2 * r2 - sum * sum), 2.0 * alpha) + (0.5 * r2 + sum) / nx + 0.5;
}

inline double schaffer_F7(const vector<double> &x, const vector<double> &Os, const Matrix &Mr, int s_flag, int r_flag) {
    auto sr = sr_func(x, Os, Mr, 1.0, s_flag, r_flag);
    int nx = static_cast<int>(x.size());
    if (nx < 2) {
        return 0.0;
    }
    double f = 0.0;
    for (int i = 0; i < nx - 1; ++i) {
        double zi = sqrt(sr.y[i] * sr.y[i] + sr.y[i + 1] * sr.y[i + 1]);
        double tmp = sin(50.0 * pow(zi, 0.2));
        tmp *= tmp;
        double term = pow(zi, 0.5) + pow(zi, 0.5) * tmp;
        f += term;
    }
    double zi = sqrt(sr.y[nx - 1] * sr.y[nx - 1] + sr.y[0] * sr.y[0]);
    double tmp = sin(50.0 * pow(zi, 0.2));
    tmp *= tmp;
    double term = pow(zi, 0.5) + pow(zi, 0.5) * tmp;
    f += term;
    return (f * f) / ((nx - 1.0) * (nx - 1.0));
}

inline double step_rastrigin(const vector<double> &x, const vector<double> &Os, const Matrix &Mr, int s_flag, int r_flag) {
    int nx = static_cast<int>(x.size());
    vector<double> y(nx, 0.0);
    for (int i = 0; i < nx; ++i) {
        y[i] = x[i];
        if (fabs(y[i] - Os[i]) > 0.5) {
            y[i] = Os[i] + floor(2.0 * (y[i] - Os[i]) + 0.5) / 2.0;
        }
    }
    auto sr = sr_func(y, Os, Mr, 5.12 / 100.0, s_flag, r_flag);
    double f = 0.0;
    for (int i = 0; i < nx; ++i) {
        f += sr.z[i] * sr.z[i] - 10.0 * cos(2.0 * M_PI * sr.z[i]) + 10.0;
    }
    return f;
}

inline double levy(const vector<double> &x, const vector<double> &Os, const Matrix &Mr, int s_flag, int r_flag) {
    auto sr = sr_func(x, Os, Mr, 1.0, s_flag, r_flag);
    int nx = static_cast<int>(x.size());
    vector<double> w(nx, 0.0);
    for (int i = 0; i < nx; ++i) {
        w[i] = 1.0 + sr.z[i] / 4.0;
    }
    double term1 = pow(sin(M_PI * w[0]), 2.0);
    double term3 = pow(w[nx - 1] - 1.0, 2.0) * (1.0 + pow(sin(2.0 * M_PI * w[nx - 1]), 2.0));
    double sum = 0.0;
    for (int i = 0; i < nx - 1; ++i) {
        double wi = w[i];
        sum += pow(wi - 1.0, 2.0) * (1.0 + 10.0 * pow(sin(M_PI * wi + 1.0), 2.0));
    }
    return term1 + sum + term3;
}

inline double zakharov(const vector<double> &x, const vector<double> &Os, const Matrix &Mr, int s_flag, int r_flag) {
    auto sr = sr_func(x, Os, Mr, 1.0, s_flag, r_flag);
    double sum1 = 0.0;
    double sum2 = 0.0;
    int nx = static_cast<int>(x.size());
    for (int i = 0; i < nx; ++i) {
        double xi = sr.z[i];
        sum1 += xi * xi;
        sum2 += 0.5 * (i + 1) * xi;
    }
    return sum1 + sum2 * sum2 + pow(sum2, 4.0);
}

inline double katsuura(const vector<double> &x, const vector<double> &Os, const Matrix &Mr, int s_flag, int r_flag) {
    auto sr = sr_func(x, Os, Mr, 5.0 / 100.0, s_flag, r_flag);
    int nx = static_cast<int>(x.size());
    double f = 1.0;
    double tmp3 = pow(static_cast<double>(nx), 1.2);
    for (int i = 0; i < nx; ++i) {
        double temp = 0.0;
        for (int j = 1; j <= 32; ++j) {
            double tmp1 = pow(2.0, j);
            double tmp2 = tmp1 * sr.z[i];
            temp += fabs(tmp2 - floor(tmp2 + 0.5)) / tmp1;
        }
        f *= pow(1.0 + (i + 1) * temp, 10.0 / tmp3);
    }
    double tmp1 = 10.0 / (nx * nx);
    return f * tmp1 - tmp1;
}

inline vector<double> slice_vector(const vector<double> &data, int offset, int length) {
    return vector<double>(data.begin() + offset, data.begin() + offset + length);
}

inline Matrix slice_matrix(const vector<double> &data, int offset, int length) {
    return Matrix(data.begin() + offset, data.begin() + offset + length);
}

inline double hf02(const vector<double> &x, const vector<double> &Os_all, const Matrix &Mr_all, const vector<int> &S, int nx) {
    auto sr = sr_func(x, Os_all, Mr_all, 1.0, 1, 1);
    vector<double> y(nx, 0.0);
    for (int i = 0; i < nx; ++i) {
        int idx = S[i];
        y[i] = (idx >= 0 && idx < nx) ? sr.z[idx] : sr.z[i];
    }
    constexpr int cf_num = 3;
    const double Gp[cf_num] = {0.4, 0.4, 0.2};
    int G_nx[cf_num];
    int tmp = 0;
    for (int i = 0; i < cf_num - 1; ++i) {
        G_nx[i] = static_cast<int>(ceil(Gp[i] * nx));
        tmp += G_nx[i];
    }
    G_nx[cf_num - 1] = nx - tmp;
    int G[cf_num];
    G[0] = 0;
    for (int i = 1; i < cf_num; ++i) {
        G[i] = G[i - 1] + G_nx[i - 1];
    }

    auto eval_segment = [](const vector<double> &seg, auto func) {
        vector<double> Os(seg.size(), 0.0);
        Matrix I(seg.size() * seg.size(), 0.0);
        for (size_t i = 0; i < seg.size(); ++i) {
            I[i * seg.size() + i] = 1.0;
        }
        return func(seg, Os, I, 0, 0);
    };

    double fit[cf_num];
    fit[0] = eval_segment(vector<double>(y.begin() + G[0], y.begin() + G[0] + G_nx[0]), bent_cigar);
    fit[1] = eval_segment(vector<double>(y.begin() + G[1], y.begin() + G[1] + G_nx[1]), hgbat);
    fit[2] = eval_segment(vector<double>(y.begin() + G[2], y.begin() + G[2] + G_nx[2]), rastrigin);

    double f = 0.0;
    for (int i = 0; i < cf_num; ++i) {
        f += fit[i];
    }
    return f;
}

inline double hf10(const vector<double> &x, const vector<double> &Os_all, const Matrix &Mr_all, const vector<int> &S, int nx) {
    auto sr = sr_func(x, Os_all, Mr_all, 1.0, 1, 1);
    vector<double> y(nx, 0.0);
    for (int i = 0; i < nx; ++i) {
        int idx = S[i];
        y[i] = (idx >= 0 && idx < nx) ? sr.z[idx] : sr.z[i];
    }
    constexpr int cf_num = 6;
    const double Gp[cf_num] = {0.1, 0.2, 0.2, 0.2, 0.1, 0.2};
    int G_nx[cf_num];
    int tmp = 0;
    for (int i = 0; i < cf_num - 1; ++i) {
        G_nx[i] = static_cast<int>(ceil(Gp[i] * nx));
        tmp += G_nx[i];
    }
    G_nx[cf_num - 1] = nx - tmp;
    int G[cf_num];
    G[0] = 0;
    for (int i = 1; i < cf_num; ++i) {
        G[i] = G[i - 1] + G_nx[i - 1];
    }

    auto eval_segment = [](const vector<double> &seg, auto func) {
        vector<double> Os(seg.size(), 0.0);
        Matrix I(seg.size() * seg.size(), 0.0);
        for (size_t i = 0; i < seg.size(); ++i) {
            I[i * seg.size() + i] = 1.0;
        }
        return func(seg, Os, I, 0, 0);
    };

    double fit[cf_num];
    fit[0] = eval_segment(vector<double>(y.begin() + G[0], y.begin() + G[0] + G_nx[0]), hgbat);
    fit[1] = eval_segment(vector<double>(y.begin() + G[1], y.begin() + G[1] + G_nx[1]), katsuura);
    fit[2] = eval_segment(vector<double>(y.begin() + G[2], y.begin() + G[2] + G_nx[2]), ackley);
    fit[3] = eval_segment(vector<double>(y.begin() + G[3], y.begin() + G[3] + G_nx[3]), rastrigin);
    fit[4] = eval_segment(vector<double>(y.begin() + G[4], y.begin() + G[4] + G_nx[4]), schwefel);
    fit[5] = eval_segment(vector<double>(y.begin() + G[5], y.begin() + G[5] + G_nx[5]), schaffer_F7);

    double f = 0.0;
    for (int i = 0; i < cf_num; ++i) {
        f += fit[i];
    }
    return f;
}

inline double hf06(const vector<double> &x, const vector<double> &Os_all, const Matrix &Mr_all, const vector<int> &S, int nx) {
    auto sr = sr_func(x, Os_all, Mr_all, 1.0, 1, 1);
    vector<double> y(nx, 0.0);
    for (int i = 0; i < nx; ++i) {
        int idx = S[i];
        y[i] = (idx >= 0 && idx < nx) ? sr.z[idx] : sr.z[i];
    }
    constexpr int cf_num = 5;
    const double Gp[cf_num] = {0.3, 0.2, 0.2, 0.1, 0.2};
    int G_nx[cf_num];
    int tmp = 0;
    for (int i = 0; i < cf_num - 1; ++i) {
        G_nx[i] = static_cast<int>(ceil(Gp[i] * nx));
        tmp += G_nx[i];
    }
    G_nx[cf_num - 1] = nx - tmp;
    int G[cf_num];
    G[0] = 0;
    for (int i = 1; i < cf_num; ++i) {
        G[i] = G[i - 1] + G_nx[i - 1];
    }

    auto eval_segment = [](const vector<double> &seg, auto func) {
        vector<double> Os(seg.size(), 0.0);
        Matrix I(seg.size() * seg.size(), 0.0);
        for (size_t i = 0; i < seg.size(); ++i) {
            I[i * seg.size() + i] = 1.0;
        }
        return func(seg, Os, I, 0, 0);
    };

    double fit[cf_num];
    fit[0] = eval_segment(vector<double>(y.begin() + G[0], y.begin() + G[0] + G_nx[0]), katsuura);
    fit[1] = eval_segment(vector<double>(y.begin() + G[1], y.begin() + G[1] + G_nx[1]), happycat);
    fit[2] = eval_segment(vector<double>(y.begin() + G[2], y.begin() + G[2] + G_nx[2]), grie_rosen);
    fit[3] = eval_segment(vector<double>(y.begin() + G[3], y.begin() + G[3] + G_nx[3]), schwefel);
    fit[4] = eval_segment(vector<double>(y.begin() + G[4], y.begin() + G[4] + G_nx[4]), ackley);

    double f = 0.0;
    for (int i = 0; i < cf_num; ++i) {
        f += fit[i];
    }
    return f;
}

inline double cf_weighted(const vector<double> &x,
                          const vector<double> &Os,
                          const vector<double> &delta,
                          const vector<double> &bias,
                          const vector<double> &fit,
                          int cf_num,
                          int dim) {
    vector<double> w(cf_num, 0.0);
    double w_max = 0.0;
    for (int i = 0; i < cf_num; ++i) {
        double d2 = 0.0;
        for (int j = 0; j < dim; ++j) {
            double diff = x[j] - Os[static_cast<size_t>(i) * dim + j];
            d2 += diff * diff;
        }
        if (d2 > 0.0) {
            w[i] = pow(1.0 / d2, 0.5) * exp(-d2 / (2.0 * dim * delta[i] * delta[i]));
        } else {
            w[i] = numeric_limits<double>::infinity();
        }
        w_max = max(w_max, w[i]);
    }
    double w_sum = 0.0;
    for (double wi : w) {
        w_sum += wi;
    }
    if (isinf(w_max)) {
        for (int i = 0; i < cf_num; ++i) {
            w[i] = (isinf(w[i]) ? 1.0 : 0.0);
        }
        w_sum = static_cast<double>(count_if(w.begin(), w.end(), [](double v) { return v > 0.0; }));
    } else if (w_sum == 0.0) {
        for (int i = 0; i < cf_num; ++i) {
            w[i] = 1.0;
        }
        w_sum = static_cast<double>(cf_num);
    }
    double f = 0.0;
    for (int i = 0; i < cf_num; ++i) {
        f += (w[i] / w_sum) * (fit[i] + bias[i]);
    }
    return f;
}

inline double cf01_eval(const vector<double> &x, int dim) {
    constexpr int cf_num = 5;
    const auto &Os_flat = DataCache::instance().multi_shift("shift_data_9", dim, cf_num);
    const auto &Mr_flat = DataCache::instance().multi_matrix("M_9", dim, cf_num);
    vector<double> fit(cf_num, 0.0);
    const double delta_arr[cf_num] = {10, 20, 30, 40, 50};
    const double bias_arr[cf_num] = {0, 200, 300, 100, 400};

    for (int i = 0; i < cf_num; ++i) {
        auto Os = slice_vector(Os_flat, i * dim, dim);
        auto Mr = slice_matrix(Mr_flat, i * dim * dim, dim * dim);
        switch (i) {
            case 0:
                fit[i] = rosenbrock(x, Os, Mr, 1, 1);
                fit[i] = 10000.0 * fit[i] / 1e4;
                break;
            case 1:
                fit[i] = ellips(x, Os, Mr, 1, 1);
                fit[i] = 10000.0 * fit[i] / 1e10;
                break;
            case 2:
                fit[i] = bent_cigar(x, Os, Mr, 1, 1);
                fit[i] = 10000.0 * fit[i] / 1e30;
                break;
            case 3:
                fit[i] = discus(x, Os, Mr, 1, 1);
                fit[i] = 10000.0 * fit[i] / 1e10;
                break;
            case 4:
                fit[i] = ellips(x, Os, Mr, 1, 0);
                fit[i] = 10000.0 * fit[i] / 1e10;
                break;
        }
    }
    vector<double> delta(delta_arr, delta_arr + cf_num);
    vector<double> bias(bias_arr, bias_arr + cf_num);
    return cf_weighted(x, Os_flat, delta, bias, fit, cf_num, dim);
}

inline double cf02_eval(const vector<double> &x, int dim) {
    constexpr int cf_num = 3;
    const auto &Os_flat = DataCache::instance().multi_shift("shift_data_10", dim, cf_num);
    const auto &Mr_flat = DataCache::instance().multi_matrix("M_10", dim, cf_num);
    vector<double> fit(cf_num, 0.0);
    const double delta_arr[cf_num] = {20, 10, 10};
    const double bias_arr[cf_num] = {0, 200, 100};

    auto Os0 = slice_vector(Os_flat, 0 * dim, dim);
    auto Mr0 = slice_matrix(Mr_flat, 0 * dim * dim, dim * dim);
    fit[0] = schwefel(x, Os0, Mr0, 1, 0);

    auto Os1 = slice_vector(Os_flat, 1 * dim, dim);
    auto Mr1 = slice_matrix(Mr_flat, 1 * dim * dim, dim * dim);
    fit[1] = rastrigin(x, Os1, Mr1, 1, 1);

    auto Os2 = slice_vector(Os_flat, 2 * dim, dim);
    auto Mr2 = slice_matrix(Mr_flat, 2 * dim * dim, dim * dim);
    fit[2] = hgbat(x, Os2, Mr2, 1, 1);

    vector<double> delta(delta_arr, delta_arr + cf_num);
    vector<double> bias(bias_arr, bias_arr + cf_num);
    return cf_weighted(x, Os_flat, delta, bias, fit, cf_num, dim);
}

inline double cf06_eval(const vector<double> &x, int dim) {
    constexpr int cf_num = 5;
    const auto &Os_flat = DataCache::instance().multi_shift("shift_data_11", dim, cf_num);
    const auto &Mr_flat = DataCache::instance().multi_matrix("M_11", dim, cf_num);
    vector<double> fit(cf_num, 0.0);
    const double delta_arr[cf_num] = {20, 20, 30, 30, 20};
    const double bias_arr[cf_num] = {0, 200, 300, 400, 200};

    auto Os0 = slice_vector(Os_flat, 0 * dim, dim);
    auto Mr0 = slice_matrix(Mr_flat, 0 * dim * dim, dim * dim);
    fit[0] = escaffer6(x, Os0, Mr0, 1, 1);
    fit[0] = 10000.0 * fit[0] / 2e7;

    auto Os1 = slice_vector(Os_flat, 1 * dim, dim);
    auto Mr1 = slice_matrix(Mr_flat, 1 * dim * dim, dim * dim);
    fit[1] = schwefel(x, Os1, Mr1, 1, 1);

    auto Os2 = slice_vector(Os_flat, 2 * dim, dim);
    auto Mr2 = slice_matrix(Mr_flat, 2 * dim * dim, dim * dim);
    fit[2] = griewank(x, Os2, Mr2, 1, 1);
    fit[2] = 1000.0 * fit[2] / 100.0;

    auto Os3 = slice_vector(Os_flat, 3 * dim, dim);
    auto Mr3 = slice_matrix(Mr_flat, 3 * dim * dim, dim * dim);
    fit[3] = rosenbrock(x, Os3, Mr3, 1, 1);

    auto Os4 = slice_vector(Os_flat, 4 * dim, dim);
    auto Mr4 = slice_matrix(Mr_flat, 4 * dim * dim, dim * dim);
    fit[4] = rastrigin(x, Os4, Mr4, 1, 1);
    fit[4] = 10000.0 * fit[4] / 1e3;

    vector<double> delta(delta_arr, delta_arr + cf_num);
    vector<double> bias(bias_arr, bias_arr + cf_num);
    return cf_weighted(x, Os_flat, delta, bias, fit, cf_num, dim);
}

inline double cf07_eval(const vector<double> &x, int dim) {
    constexpr int cf_num = 6;
    const auto &Os_flat = DataCache::instance().multi_shift("shift_data_12", dim, cf_num);
    const auto &Mr_flat = DataCache::instance().multi_matrix("M_12", dim, cf_num);
    vector<double> fit(cf_num, 0.0);
    const double delta_arr[cf_num] = {10, 20, 30, 40, 50, 60};
    const double bias_arr[cf_num] = {0, 300, 500, 100, 400, 200};

    auto Os0 = slice_vector(Os_flat, 0 * dim, dim);
    auto Mr0 = slice_matrix(Mr_flat, 0 * dim * dim, dim * dim);
    fit[0] = hgbat(x, Os0, Mr0, 1, 1);
    fit[0] = 10000.0 * fit[0] / 1e3;

    auto Os1 = slice_vector(Os_flat, 1 * dim, dim);
    auto Mr1 = slice_matrix(Mr_flat, 1 * dim * dim, dim * dim);
    fit[1] = rastrigin(x, Os1, Mr1, 1, 1);
    fit[1] = 10000.0 * fit[1] / 1e3;

    auto Os2 = slice_vector(Os_flat, 2 * dim, dim);
    auto Mr2 = slice_matrix(Mr_flat, 2 * dim * dim, dim * dim);
    fit[2] = schwefel(x, Os2, Mr2, 1, 1);
    fit[2] = 10000.0 * fit[2] / 4e3;

    auto Os3 = slice_vector(Os_flat, 3 * dim, dim);
    auto Mr3 = slice_matrix(Mr_flat, 3 * dim * dim, dim * dim);
    fit[3] = bent_cigar(x, Os3, Mr3, 1, 1);
    fit[3] = 10000.0 * fit[3] / 1e30;

    auto Os4 = slice_vector(Os_flat, 4 * dim, dim);
    auto Mr4 = slice_matrix(Mr_flat, 4 * dim * dim, dim * dim);
    fit[4] = ellips(x, Os4, Mr4, 1, 1);
    fit[4] = 10000.0 * fit[4] / 1e10;

    auto Os5 = slice_vector(Os_flat, 5 * dim, dim);
    auto Mr5 = slice_matrix(Mr_flat, 5 * dim * dim, dim * dim);
    fit[5] = escaffer6(x, Os5, Mr5, 1, 1);
    fit[5] = 10000.0 * fit[5] / 2e7;

    vector<double> delta(delta_arr, delta_arr + cf_num);
    vector<double> bias(bias_arr, bias_arr + cf_num);
    return cf_weighted(x, Os_flat, delta, bias, fit, cf_num, dim);
}

inline double evaluate_without_bias(int func_num, const vector<double> &x) {
    int dim = static_cast<int>(x.size());
    auto &cache = DataCache::instance();
    switch (func_num) {
        case 1: {
            const auto &Os = cache.shift("shift_data_1", dim);
            const auto &Mr = cache.matrix("M_1", dim);
            return zakharov(x, Os, Mr, 1, 1);
        }
        case 2: {
            const auto &Os = cache.shift("shift_data_2", dim);
            const auto &Mr = cache.matrix("M_2", dim);
            return rosenbrock(x, Os, Mr, 1, 1);
        }
        case 3: {
            const auto &Os = cache.shift("shift_data_3", dim);
            const auto &Mr = cache.matrix("M_3", dim);
            return schaffer_F7(x, Os, Mr, 1, 1);
        }
        case 4: {
            const auto &Os = cache.shift("shift_data_4", dim);
            const auto &Mr = cache.matrix("M_4", dim);
            return step_rastrigin(x, Os, Mr, 1, 1);
        }
        case 5: {
            const auto &Os = cache.shift("shift_data_5", dim);
            const auto &Mr = cache.matrix("M_5", dim);
            return levy(x, Os, Mr, 1, 1);
        }
        case 6: {
            const auto &Os = cache.shift("shift_data_6", dim);
            const auto &Mr = cache.matrix("M_6", dim);
            const auto &S = cache.shuffle("shuffle_data_6", dim);
            return hf02(x, Os, Mr, S, dim);
        }
        case 7: {
            const auto &Os = cache.shift("shift_data_7", dim);
            const auto &Mr = cache.matrix("M_7", dim);
            const auto &S = cache.shuffle("shuffle_data_7", dim);
            return hf10(x, Os, Mr, S, dim);
        }
        case 8: {
            const auto &Os = cache.shift("shift_data_8", dim);
            const auto &Mr = cache.matrix("M_8", dim);
            const auto &S = cache.shuffle("shuffle_data_8", dim);
            return hf06(x, Os, Mr, S, dim);
        }
        case 9:
            return cf01_eval(x, dim);
        case 10:
            return cf02_eval(x, dim);
        case 11:
            return cf06_eval(x, dim);
        case 12:
            return cf07_eval(x, dim);
        default:
            throw out_of_range("CEC2022 function index must be between 1 and 12.");
    }
}

inline double evaluate_with_bias(int func_num, const vector<double> &x) {
    static constexpr array<double, 13> bias = {
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
    double base = evaluate_without_bias(func_num, x);
    return base + bias[func_num];
}

} // namespace

void cec22_test_func_fortrain(double *x, double *f, int nx, int mx, int func_num) {
    if (func_num < 1 || func_num > 12) {
        cerr << "[cec22] Error: Test function " << func_num << " is not defined." << endl;
        for (int i = 0; i < mx; ++i) {
            f[i] = numeric_limits<double>::quiet_NaN();
        }
        return;
    }
    if (!(nx == 2 || nx == 10 || nx == 20)) {
        cerr << "[cec22] Warning: dimensions other than 2, 10, 20 are not defined (requested D=" << nx << ")." << endl;
    }
    for (int i = 0; i < mx; ++i) {
        vector<double> xi(x + static_cast<size_t>(i) * nx, x + static_cast<size_t>(i + 1) * nx);
        try {
            f[i] = evaluate_with_bias(func_num, xi);
        } catch (const exception &ex) {
            cerr << "[cec22] Error: " << ex.what() << endl;
            f[i] = numeric_limits<double>::quiet_NaN();
        }
    }
}

