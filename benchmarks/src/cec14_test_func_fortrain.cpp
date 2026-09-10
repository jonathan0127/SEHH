
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

#ifndef CEC14_DATA_DIR
#define CEC14_DATA_DIR "benchmarks/data/data_2014"
#endif

#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795029
#endif

#ifndef M_E
#define M_E 2.7182818284590452353602874713526625
#endif

using namespace std;

namespace {

using Matrix = vector<double>;

struct PairHash {
    size_t operator()(const pair<int, int> &key) const noexcept {
        size_t h1 = hash<int>()(key.first);
        size_t h2 = hash<int>()(key.second);
        return h1 ^ (h2 << 1);
    }
};

struct TripleHash {
    size_t operator()(const tuple<int, int, int> &key) const noexcept {
        const auto &[i1, i2, i3] = key;
        size_t h1 = hash<int>()(i1);
        size_t h2 = hash<int>()(i2);
        size_t h3 = hash<int>()(i3);
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

class DataCache14 {
public:
    static DataCache14 &instance() {
        static DataCache14 cache;
        return cache;
    }

    const vector<double> &shift(int func_num, int dim) {
        lock_guard<mutex> lock(mutex_);
        auto key = make_pair(func_num, dim);
        auto it = shifts_.find(key);
        if (it != shifts_.end()) {
            return it->second;
        }
        vector<double> values(dim, 0.0);
        string path = make_shift_path(func_num);
        ifstream fin(path);
        if (fin.is_open()) {
            for (int i = 0; i < dim && fin; ++i) {
                fin >> values[i];
            }
        } else {
            cerr << "[cec14] warning: failed to open shift file: " << path << '\n';
        }
        return shifts_.emplace(key, move(values)).first->second;
    }

    const Matrix &matrix(int func_num, int dim) {
        lock_guard<mutex> lock(mutex_);
        auto key = make_pair(func_num, dim);
        auto it = matrices_.find(key);
        if (it != matrices_.end()) {
            return it->second;
        }
        Matrix mat(static_cast<size_t>(dim) * dim, 0.0);
        string path = make_matrix_path(func_num, dim);
        ifstream fin(path);
        if (fin.is_open()) {
            for (int i = 0; i < dim * dim && fin; ++i) {
                fin >> mat[i];
            }
        } else {
            cerr << "[cec14] warning: failed to open matrix file: " << path << '\n';
            for (int i = 0; i < dim; ++i) {
                mat[i * dim + i] = 1.0;
            }
        }
        return matrices_.emplace(key, move(mat)).first->second;
    }

    const vector<int> &shuffle(int func_num, int dim) {
        lock_guard<mutex> lock(mutex_);
        auto key = make_pair(func_num, dim);
        auto it = shuffles_.find(key);
        if (it != shuffles_.end()) {
            return it->second;
        }
        vector<int> idx(dim, 0);
        string path = make_shuffle_path(func_num, dim);
        ifstream fin(path);
        if (fin.is_open()) {
            for (int i = 0; i < dim && fin; ++i) {
                fin >> idx[i];
                idx[i] = max(0, idx[i] - 1);  // Convert 1-based to 0-based
            }
        } else {
            // Not all functions have shuffle data
            for (int i = 0; i < dim; ++i) {
                idx[i] = i;
            }
        }
        return shuffles_.emplace(key, move(idx)).first->second;
    }

    const vector<double> &multi_shift(int func_num, int dim, int count) {
        lock_guard<mutex> lock(mutex_);
        auto key = make_tuple(func_num, dim, count);
        auto it = multi_shifts_.find(key);
        if (it != multi_shifts_.end()) {
            return it->second;
        }
        vector<double> data(static_cast<size_t>(dim) * count, 0.0);
        string path = make_shift_path(func_num);
        ifstream fin(path);
        if (fin.is_open()) {
            string line;
            int loaded = 0;
            while (loaded < count && getline(fin, line)) {
                if (line.empty()) continue;
                istringstream iss(line);
                for (int j = 0; j < dim; ++j) {
                    iss >> data[static_cast<size_t>(loaded) * dim + j];
                }
                ++loaded;
            }
        } else {
            cerr << "[cec14] warning: failed to open shift file: " << path << '\n';
        }
        return multi_shifts_.emplace(key, move(data)).first->second;
    }

    const vector<double> &multi_matrix(int func_num, int dim, int count) {
        lock_guard<mutex> lock(mutex_);
        auto key = make_tuple(func_num, dim, count);
        auto it = multi_matrices_.find(key);
        if (it != multi_matrices_.end()) {
            return it->second;
        }
        vector<double> data(static_cast<size_t>(dim) * dim * count, 0.0);
        string path = make_matrix_path(func_num, dim);
        ifstream fin(path);
        if (fin.is_open()) {
            for (int i = 0; i < dim * dim * count && fin; ++i) {
                fin >> data[i];
            }
        } else {
            cerr << "[cec14] warning: failed to open matrix file: " << path << '\n';
            for (int k = 0; k < count; ++k) {
                for (int i = 0; i < dim; ++i) {
                    data[k * dim * dim + i * dim + i] = 1.0;
                }
            }
        }
        return multi_matrices_.emplace(key, move(data)).first->second;
    }

    const vector<int> &multi_shuffle(int func_num, int dim, int count) {
        lock_guard<mutex> lock(mutex_);
        auto key = make_tuple(func_num, dim, count);
        auto it = multi_shuffles_.find(key);
        if (it != multi_shuffles_.end()) {
            return it->second;
        }
        vector<int> data(static_cast<size_t>(dim) * count, 0);
        string path = make_shuffle_path(func_num, dim);
        ifstream fin(path);
        if (fin.is_open()) {
            for (int i = 0; i < dim * count && fin; ++i) {
                fin >> data[i];
                data[i] = max(0, data[i] - 1);
            }
        } else {
            for (int k = 0; k < count; ++k) {
                for (int i = 0; i < dim; ++i) {
                    data[k * dim + i] = i;
                }
            }
        }
        return multi_shuffles_.emplace(key, move(data)).first->second;
    }

private:
    DataCache14() = default;

    static string make_shift_path(int func_num) {
        return string(CEC14_DATA_DIR) + "/shift_data_" + to_string(func_num) + ".txt";
    }

    static string make_matrix_path(int func_num, int dim) {
        return string(CEC14_DATA_DIR) + "/M_" + to_string(func_num) + "_D" + to_string(dim) + ".txt";
    }

    static string make_shuffle_path(int func_num, int dim) {
        return string(CEC14_DATA_DIR) + "/shuffle_data_" + to_string(func_num) + "_D" + to_string(dim) + ".txt";
    }

    mutex mutex_;
    unordered_map<pair<int, int>, vector<double>, PairHash> shifts_;
    unordered_map<pair<int, int>, Matrix, PairHash> matrices_;
    unordered_map<pair<int, int>, vector<int>, PairHash> shuffles_;
    unordered_map<tuple<int, int, int>, vector<double>, TripleHash> multi_shifts_;
    unordered_map<tuple<int, int, int>, vector<double>, TripleHash> multi_matrices_;
    unordered_map<tuple<int, int, int>, vector<int>, TripleHash> multi_shuffles_;
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

inline double sphere(const vector<double> &x, const vector<double> &Os, const Matrix &Mr, int s_flag, int r_flag) {
    auto sr = sr_func(x, Os, Mr, 1.0, s_flag, r_flag);
    double f = 0.0;
    for (size_t i = 0; i < sr.z.size(); ++i) {
        f += sr.z[i] * sr.z[i];
    }
    return f;
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
    int nx = static_cast<int>(x.size());
    double f = 0.0;
    z[0] += 1.0;
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
    return M_E - 20.0 * exp(sum1) - exp(sum2) + 20.0;
}

inline double weierstrass(const vector<double> &x, const vector<double> &Os, const Matrix &Mr, int s_flag, int r_flag) {
    auto sr = sr_func(x, Os, Mr, 0.5 / 100.0, s_flag, r_flag);
    int nx = static_cast<int>(x.size());
    int k_max = 20;
    double a = 0.5;
    double b = 3.0;
    double f = 0.0;
    double sum2 = 0.0;
    for (int j = 0; j <= k_max; ++j) {
        sum2 += pow(a, j) * cos(2.0 * M_PI * pow(b, j) * 0.5);
    }
    for (int i = 0; i < nx; ++i) {
        double sum = 0.0;
        for (int j = 0; j <= k_max; ++j) {
            sum += pow(a, j) * cos(2.0 * M_PI * pow(b, j) * (sr.z[i] + 0.5));
        }
        f += sum;
    }
    f -= nx * sum2;
    return f;
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
            f -= temp * sin(sqrt(fabs(temp)));
            double tmp = (zi - 500.0) / 100.0;
            f += tmp * tmp / nx;
        } else if (zi < -500.0) {
            double temp = 500.0 - fmod(fabs(zi), 500.0);
            f -= temp * sin(sqrt(fabs(temp)));
            double tmp = (zi + 500.0) / 100.0;
            f += tmp * tmp / nx;
        } else {
            f -= zi * sin(sqrt(fabs(zi)));
        }
    }
    f += 4.189828872724338e+002 * nx;
    return f;
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

inline double grie_rosen(const vector<double> &x, const vector<double> &Os, const Matrix &Mr, int s_flag, int r_flag) {
    auto sr = sr_func(x, Os, Mr, 5.0 / 100.0, s_flag, r_flag);
    vector<double> z = sr.z;
    int nx = static_cast<int>(x.size());
    double f = 0.0;
    z[0] += 1.0;
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

inline vector<double> slice_vector(const vector<double> &data, int offset, int length) {
    return vector<double>(data.begin() + offset, data.begin() + offset + length);
}

inline Matrix slice_matrix(const vector<double> &data, int offset, int length) {
    return Matrix(data.begin() + offset, data.begin() + offset + length);
}

template<typename... Funcs>
inline double hybrid_eval(const vector<double> &x,
                          const vector<double> &Os,
                          const Matrix &Mr,
                          const vector<int> &S,
                          const vector<double> &Gp,
                          Funcs... funcs) {
    int nx = static_cast<int>(x.size());
    auto sr = sr_func(x, Os, Mr, 1.0, 1, 1);
    
    // Shuffle
    vector<double> y(nx, 0.0);
    for (int i = 0; i < nx; ++i) {
        int idx = S[i];
        y[i] = (idx >= 0 && idx < nx) ? sr.z[idx] : sr.z[i];
    }
    
    // Calculate group sizes
    int cf_num = static_cast<int>(Gp.size());
    vector<int> G_nx(cf_num);
    vector<int> G(cf_num);
    int tmp = 0;
    for (int i = 0; i < cf_num - 1; ++i) {
        G_nx[i] = static_cast<int>(ceil(Gp[i] * nx));
        tmp += G_nx[i];
    }
    G_nx[cf_num - 1] = nx - tmp;
    G[0] = 0;
    for (int i = 1; i < cf_num; ++i) {
        G[i] = G[i - 1] + G_nx[i - 1];
    }
    
    // Evaluate each function
    array<double(*)(const vector<double>&, const vector<double>&, const Matrix&, int, int), sizeof...(Funcs)> func_arr = {funcs...};
    double f = 0.0;
    for (int i = 0; i < cf_num && i < static_cast<int>(func_arr.size()); ++i) {
        vector<double> seg(y.begin() + G[i], y.begin() + G[i] + G_nx[i]);
        vector<double> Os_dummy(seg.size(), 0.0);
        Matrix I(seg.size() * seg.size(), 0.0);
        for (size_t j = 0; j < seg.size(); ++j) {
            I[j * seg.size() + j] = 1.0;
        }
        f += func_arr[i](seg, Os_dummy, I, 0, 0);
    }
    return f;
}

inline double hf01(const vector<double> &x, const vector<double> &Os, const Matrix &Mr, const vector<int> &S, int nx) {
    // schwefel + rastrigin + ellips
    return hybrid_eval(x, Os, Mr, S, {0.3, 0.3, 0.4}, schwefel, rastrigin, ellips);
}

inline double hf02(const vector<double> &x, const vector<double> &Os, const Matrix &Mr, const vector<int> &S, int nx) {
    // bent_cigar + hgbat + rastrigin
    return hybrid_eval(x, Os, Mr, S, {0.3, 0.3, 0.4}, bent_cigar, hgbat, rastrigin);
}

inline double hf03(const vector<double> &x, const vector<double> &Os, const Matrix &Mr, const vector<int> &S, int nx) {
    // griewank + weierstrass + rosenbrock + escaffer6
    return hybrid_eval(x, Os, Mr, S, {0.2, 0.2, 0.3, 0.3}, griewank, weierstrass, rosenbrock, escaffer6);
}

inline double hf04(const vector<double> &x, const vector<double> &Os, const Matrix &Mr, const vector<int> &S, int nx) {
    // hgbat + discus + grie_rosen + rastrigin
    return hybrid_eval(x, Os, Mr, S, {0.2, 0.2, 0.3, 0.3}, hgbat, discus, grie_rosen, rastrigin);
}

inline double hf05(const vector<double> &x, const vector<double> &Os, const Matrix &Mr, const vector<int> &S, int nx) {
    // escaffer6 + hgbat + rosenbrock + schwefel + ellips
    return hybrid_eval(x, Os, Mr, S, {0.1, 0.2, 0.2, 0.2, 0.3}, escaffer6, hgbat, rosenbrock, schwefel, ellips);
}

inline double hf06(const vector<double> &x, const vector<double> &Os, const Matrix &Mr, const vector<int> &S, int nx) {
    // katsuura + happycat + grie_rosen + schwefel + ackley
    return hybrid_eval(x, Os, Mr, S, {0.1, 0.2, 0.2, 0.2, 0.3}, katsuura, happycat, grie_rosen, schwefel, ackley);
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

inline double cf01_eval(const vector<double> &x, int dim, int r_flag) {
    // rosenbrock + ellips + bent_cigar + discus + ellips
    constexpr int cf_num = 5;
    const auto &Os_flat = DataCache14::instance().multi_shift(23, dim, cf_num);
    const auto &Mr_flat = DataCache14::instance().multi_matrix(23, dim, cf_num);
    vector<double> fit(cf_num, 0.0);
    const double delta_arr[cf_num] = {10, 20, 30, 40, 50};
    const double bias_arr[cf_num] = {0, 100, 200, 300, 400};

    for (int i = 0; i < cf_num; ++i) {
        auto Os_i = slice_vector(Os_flat, i * dim, dim);
        auto Mr_i = slice_matrix(Mr_flat, i * dim * dim, dim * dim);
        switch (i) {
            case 0: fit[i] = rosenbrock(x, Os_i, Mr_i, 1, r_flag); fit[i] = 10000 * fit[i] / 1e4; break;
            case 1: fit[i] = ellips(x, Os_i, Mr_i, 1, r_flag); fit[i] = 10000 * fit[i] / 1e10; break;
            case 2: fit[i] = bent_cigar(x, Os_i, Mr_i, 1, r_flag); fit[i] = 10000 * fit[i] / 1e30; break;
            case 3: fit[i] = discus(x, Os_i, Mr_i, 1, r_flag); fit[i] = 10000 * fit[i] / 1e10; break;
            case 4: fit[i] = ellips(x, Os_i, Mr_i, 1, 0); fit[i] = 10000 * fit[i] / 1e10; break;
        }
    }

    vector<double> delta(delta_arr, delta_arr + cf_num);
    vector<double> bias(bias_arr, bias_arr + cf_num);
    return cf_weighted(x, Os_flat, delta, bias, fit, cf_num, dim);
}

inline double cf02_eval(const vector<double> &x, int dim, int r_flag) {
    // schwefel + rastrigin + hgbat
    constexpr int cf_num = 3;
    const auto &Os_flat = DataCache14::instance().multi_shift(24, dim, cf_num);
    const auto &Mr_flat = DataCache14::instance().multi_matrix(24, dim, cf_num);
    vector<double> fit(cf_num, 0.0);
    const double delta_arr[cf_num] = {20, 20, 20};
    const double bias_arr[cf_num] = {0, 100, 200};

    auto Os0 = slice_vector(Os_flat, 0 * dim, dim);
    auto Mr0 = slice_matrix(Mr_flat, 0 * dim * dim, dim * dim);
    fit[0] = schwefel(x, Os0, Mr0, 1, 0);

    auto Os1 = slice_vector(Os_flat, 1 * dim, dim);
    auto Mr1 = slice_matrix(Mr_flat, 1 * dim * dim, dim * dim);
    fit[1] = rastrigin(x, Os1, Mr1, 1, r_flag);

    auto Os2 = slice_vector(Os_flat, 2 * dim, dim);
    auto Mr2 = slice_matrix(Mr_flat, 2 * dim * dim, dim * dim);
    fit[2] = hgbat(x, Os2, Mr2, 1, r_flag);

    vector<double> delta(delta_arr, delta_arr + cf_num);
    vector<double> bias(bias_arr, bias_arr + cf_num);
    return cf_weighted(x, Os_flat, delta, bias, fit, cf_num, dim);
}

inline double cf03_eval(const vector<double> &x, int dim, int r_flag) {
    // schwefel + rastrigin + ellips
    constexpr int cf_num = 3;
    const auto &Os_flat = DataCache14::instance().multi_shift(25, dim, cf_num);
    const auto &Mr_flat = DataCache14::instance().multi_matrix(25, dim, cf_num);
    vector<double> fit(cf_num, 0.0);
    const double delta_arr[cf_num] = {10, 30, 50};
    const double bias_arr[cf_num] = {0, 100, 200};

    auto Os0 = slice_vector(Os_flat, 0 * dim, dim);
    auto Mr0 = slice_matrix(Mr_flat, 0 * dim * dim, dim * dim);
    fit[0] = schwefel(x, Os0, Mr0, 1, r_flag); fit[0] = 1000 * fit[0] / 4e3;

    auto Os1 = slice_vector(Os_flat, 1 * dim, dim);
    auto Mr1 = slice_matrix(Mr_flat, 1 * dim * dim, dim * dim);
    fit[1] = rastrigin(x, Os1, Mr1, 1, r_flag); fit[1] = 1000 * fit[1] / 1e3;

    auto Os2 = slice_vector(Os_flat, 2 * dim, dim);
    auto Mr2 = slice_matrix(Mr_flat, 2 * dim * dim, dim * dim);
    fit[2] = ellips(x, Os2, Mr2, 1, r_flag); fit[2] = 1000 * fit[2] / 1e10;

    vector<double> delta(delta_arr, delta_arr + cf_num);
    vector<double> bias(bias_arr, bias_arr + cf_num);
    return cf_weighted(x, Os_flat, delta, bias, fit, cf_num, dim);
}

inline double cf04_eval(const vector<double> &x, int dim, int r_flag) {
    // schwefel + happycat + ellips + weierstrass + griewank
    constexpr int cf_num = 5;
    const auto &Os_flat = DataCache14::instance().multi_shift(26, dim, cf_num);
    const auto &Mr_flat = DataCache14::instance().multi_matrix(26, dim, cf_num);
    vector<double> fit(cf_num, 0.0);
    const double delta_arr[cf_num] = {10, 10, 10, 10, 10};
    const double bias_arr[cf_num] = {0, 100, 200, 300, 400};

    for (int i = 0; i < cf_num; ++i) {
        auto Os_i = slice_vector(Os_flat, i * dim, dim);
        auto Mr_i = slice_matrix(Mr_flat, i * dim * dim, dim * dim);
        switch (i) {
            case 0: fit[i] = schwefel(x, Os_i, Mr_i, 1, r_flag); fit[i] = 1000 * fit[i] / 4e3; break;
            case 1: fit[i] = happycat(x, Os_i, Mr_i, 1, r_flag); fit[i] = 1000 * fit[i] / 1e3; break;
            case 2: fit[i] = ellips(x, Os_i, Mr_i, 1, r_flag); fit[i] = 1000 * fit[i] / 1e10; break;
            case 3: fit[i] = weierstrass(x, Os_i, Mr_i, 1, r_flag); fit[i] = 1000 * fit[i] / 400; break;
            case 4: fit[i] = griewank(x, Os_i, Mr_i, 1, r_flag); fit[i] = 1000 * fit[i] / 100; break;
        }
    }

    vector<double> delta(delta_arr, delta_arr + cf_num);
    vector<double> bias(bias_arr, bias_arr + cf_num);
    return cf_weighted(x, Os_flat, delta, bias, fit, cf_num, dim);
}

inline double cf05_eval(const vector<double> &x, int dim, int r_flag) {
    // hgbat + rastrigin + schwefel + weierstrass + ellips
    constexpr int cf_num = 5;
    const auto &Os_flat = DataCache14::instance().multi_shift(27, dim, cf_num);
    const auto &Mr_flat = DataCache14::instance().multi_matrix(27, dim, cf_num);
    vector<double> fit(cf_num, 0.0);
    const double delta_arr[cf_num] = {10, 10, 10, 20, 20};
    const double bias_arr[cf_num] = {0, 100, 200, 300, 400};

    for (int i = 0; i < cf_num; ++i) {
        auto Os_i = slice_vector(Os_flat, i * dim, dim);
        auto Mr_i = slice_matrix(Mr_flat, i * dim * dim, dim * dim);
        switch (i) {
            case 0: fit[i] = hgbat(x, Os_i, Mr_i, 1, r_flag); fit[i] = 10000 * fit[i] / 1000; break;
            case 1: fit[i] = rastrigin(x, Os_i, Mr_i, 1, r_flag); fit[i] = 10000 * fit[i] / 1e3; break;
            case 2: fit[i] = schwefel(x, Os_i, Mr_i, 1, r_flag); fit[i] = 10000 * fit[i] / 4e3; break;
            case 3: fit[i] = weierstrass(x, Os_i, Mr_i, 1, r_flag); fit[i] = 10000 * fit[i] / 400; break;
            case 4: fit[i] = ellips(x, Os_i, Mr_i, 1, r_flag); fit[i] = 10000 * fit[i] / 1e10; break;
        }
    }

    vector<double> delta(delta_arr, delta_arr + cf_num);
    vector<double> bias(bias_arr, bias_arr + cf_num);
    return cf_weighted(x, Os_flat, delta, bias, fit, cf_num, dim);
}

inline double cf06_eval(const vector<double> &x, int dim, int r_flag) {
    // grie_rosen + happycat + schwefel + escaffer6 + ellips
    constexpr int cf_num = 5;
    const auto &Os_flat = DataCache14::instance().multi_shift(28, dim, cf_num);
    const auto &Mr_flat = DataCache14::instance().multi_matrix(28, dim, cf_num);
    vector<double> fit(cf_num, 0.0);
    const double delta_arr[cf_num] = {10, 20, 30, 40, 50};
    const double bias_arr[cf_num] = {0, 100, 200, 300, 400};

    for (int i = 0; i < cf_num; ++i) {
        auto Os_i = slice_vector(Os_flat, i * dim, dim);
        auto Mr_i = slice_matrix(Mr_flat, i * dim * dim, dim * dim);
        switch (i) {
            case 0: fit[i] = grie_rosen(x, Os_i, Mr_i, 1, r_flag); fit[i] = 10000 * fit[i] / 4e3; break;
            case 1: fit[i] = happycat(x, Os_i, Mr_i, 1, r_flag); fit[i] = 10000 * fit[i] / 1e3; break;
            case 2: fit[i] = schwefel(x, Os_i, Mr_i, 1, r_flag); fit[i] = 10000 * fit[i] / 4e3; break;
            case 3: fit[i] = escaffer6(x, Os_i, Mr_i, 1, r_flag); fit[i] = 10000 * fit[i] / 2e7; break;
            case 4: fit[i] = ellips(x, Os_i, Mr_i, 1, r_flag); fit[i] = 10000 * fit[i] / 1e10; break;
        }
    }

    vector<double> delta(delta_arr, delta_arr + cf_num);
    vector<double> bias(bias_arr, bias_arr + cf_num);
    return cf_weighted(x, Os_flat, delta, bias, fit, cf_num, dim);
}

inline double cf07_eval(const vector<double> &x, int dim, int r_flag) {
    // hf01 + hf02 + hf03 (composition of hybrid functions)
    constexpr int cf_num = 3;
    const auto &Os_flat = DataCache14::instance().multi_shift(29, dim, cf_num);
    const auto &Mr_flat = DataCache14::instance().multi_matrix(29, dim, cf_num);
    const auto &SS_flat = DataCache14::instance().multi_shuffle(29, dim, cf_num);
    vector<double> fit(cf_num, 0.0);
    const double delta_arr[cf_num] = {10, 30, 50};
    const double bias_arr[cf_num] = {0, 100, 200};

    for (int i = 0; i < cf_num; ++i) {
        auto Os_i = slice_vector(Os_flat, i * dim, dim);
        auto Mr_i = slice_matrix(Mr_flat, i * dim * dim, dim * dim);
        vector<int> SS_i(SS_flat.begin() + i * dim, SS_flat.begin() + (i + 1) * dim);
        switch (i) {
            case 0: fit[i] = hf01(x, Os_i, Mr_i, SS_i, dim); break;
            case 1: fit[i] = hf02(x, Os_i, Mr_i, SS_i, dim); break;
            case 2: fit[i] = hf03(x, Os_i, Mr_i, SS_i, dim); break;
        }
    }

    vector<double> delta(delta_arr, delta_arr + cf_num);
    vector<double> bias(bias_arr, bias_arr + cf_num);
    return cf_weighted(x, Os_flat, delta, bias, fit, cf_num, dim);
}

inline double cf08_eval(const vector<double> &x, int dim, int r_flag) {
    // hf04 + hf05 + hf06 (composition of hybrid functions)
    constexpr int cf_num = 3;
    const auto &Os_flat = DataCache14::instance().multi_shift(30, dim, cf_num);
    const auto &Mr_flat = DataCache14::instance().multi_matrix(30, dim, cf_num);
    const auto &SS_flat = DataCache14::instance().multi_shuffle(30, dim, cf_num);
    vector<double> fit(cf_num, 0.0);
    const double delta_arr[cf_num] = {10, 30, 50};
    const double bias_arr[cf_num] = {0, 100, 200};

    for (int i = 0; i < cf_num; ++i) {
        auto Os_i = slice_vector(Os_flat, i * dim, dim);
        auto Mr_i = slice_matrix(Mr_flat, i * dim * dim, dim * dim);
        vector<int> SS_i(SS_flat.begin() + i * dim, SS_flat.begin() + (i + 1) * dim);
        switch (i) {
            case 0: fit[i] = hf04(x, Os_i, Mr_i, SS_i, dim); break;
            case 1: fit[i] = hf05(x, Os_i, Mr_i, SS_i, dim); break;
            case 2: fit[i] = hf06(x, Os_i, Mr_i, SS_i, dim); break;
        }
    }

    vector<double> delta(delta_arr, delta_arr + cf_num);
    vector<double> bias(bias_arr, bias_arr + cf_num);
    return cf_weighted(x, Os_flat, delta, bias, fit, cf_num, dim);
}

inline double evaluate_without_bias(int func_num, const vector<double> &x) {
    int dim = static_cast<int>(x.size());
    auto &cache = DataCache14::instance();

    switch (func_num) {
        case 1: {
            const auto &Os = cache.shift(1, dim);
            const auto &Mr = cache.matrix(1, dim);
            return ellips(x, Os, Mr, 1, 1);
        }
        case 2: {
            const auto &Os = cache.shift(2, dim);
            const auto &Mr = cache.matrix(2, dim);
            return bent_cigar(x, Os, Mr, 1, 1);
        }
        case 3: {
            const auto &Os = cache.shift(3, dim);
            const auto &Mr = cache.matrix(3, dim);
            return discus(x, Os, Mr, 1, 1);
        }
        case 4: {
            const auto &Os = cache.shift(4, dim);
            const auto &Mr = cache.matrix(4, dim);
            return rosenbrock(x, Os, Mr, 1, 1);
        }
        case 5: {
            const auto &Os = cache.shift(5, dim);
            const auto &Mr = cache.matrix(5, dim);
            return ackley(x, Os, Mr, 1, 1);
        }
        case 6: {
            const auto &Os = cache.shift(6, dim);
            const auto &Mr = cache.matrix(6, dim);
            return weierstrass(x, Os, Mr, 1, 1);
        }
        case 7: {
            const auto &Os = cache.shift(7, dim);
            const auto &Mr = cache.matrix(7, dim);
            return griewank(x, Os, Mr, 1, 1);
        }
        case 8: {
            const auto &Os = cache.shift(8, dim);
            const auto &Mr = cache.matrix(8, dim);
            return rastrigin(x, Os, Mr, 1, 0);  // no rotation
        }
        case 9: {
            const auto &Os = cache.shift(9, dim);
            const auto &Mr = cache.matrix(9, dim);
            return rastrigin(x, Os, Mr, 1, 1);
        }
        case 10: {
            const auto &Os = cache.shift(10, dim);
            const auto &Mr = cache.matrix(10, dim);
            return schwefel(x, Os, Mr, 1, 0);  // no rotation
        }
        case 11: {
            const auto &Os = cache.shift(11, dim);
            const auto &Mr = cache.matrix(11, dim);
            return schwefel(x, Os, Mr, 1, 1);
        }
        case 12: {
            const auto &Os = cache.shift(12, dim);
            const auto &Mr = cache.matrix(12, dim);
            return katsuura(x, Os, Mr, 1, 1);
        }
        case 13: {
            const auto &Os = cache.shift(13, dim);
            const auto &Mr = cache.matrix(13, dim);
            return happycat(x, Os, Mr, 1, 1);
        }
        case 14: {
            const auto &Os = cache.shift(14, dim);
            const auto &Mr = cache.matrix(14, dim);
            return hgbat(x, Os, Mr, 1, 1);
        }
        case 15: {
            const auto &Os = cache.shift(15, dim);
            const auto &Mr = cache.matrix(15, dim);
            return grie_rosen(x, Os, Mr, 1, 1);
        }
        case 16: {
            const auto &Os = cache.shift(16, dim);
            const auto &Mr = cache.matrix(16, dim);
            return escaffer6(x, Os, Mr, 1, 1);
        }
        case 17: {
            const auto &Os = cache.shift(17, dim);
            const auto &Mr = cache.matrix(17, dim);
            const auto &S = cache.shuffle(17, dim);
            return hf01(x, Os, Mr, S, dim);
        }
        case 18: {
            const auto &Os = cache.shift(18, dim);
            const auto &Mr = cache.matrix(18, dim);
            const auto &S = cache.shuffle(18, dim);
            return hf02(x, Os, Mr, S, dim);
        }
        case 19: {
            const auto &Os = cache.shift(19, dim);
            const auto &Mr = cache.matrix(19, dim);
            const auto &S = cache.shuffle(19, dim);
            return hf03(x, Os, Mr, S, dim);
        }
        case 20: {
            const auto &Os = cache.shift(20, dim);
            const auto &Mr = cache.matrix(20, dim);
            const auto &S = cache.shuffle(20, dim);
            return hf04(x, Os, Mr, S, dim);
        }
        case 21: {
            const auto &Os = cache.shift(21, dim);
            const auto &Mr = cache.matrix(21, dim);
            const auto &S = cache.shuffle(21, dim);
            return hf05(x, Os, Mr, S, dim);
        }
        case 22: {
            const auto &Os = cache.shift(22, dim);
            const auto &Mr = cache.matrix(22, dim);
            const auto &S = cache.shuffle(22, dim);
            return hf06(x, Os, Mr, S, dim);
        }
        case 23:
            return cf01_eval(x, dim, 1);
        case 24:
            return cf02_eval(x, dim, 1);
        case 25:
            return cf03_eval(x, dim, 1);
        case 26:
            return cf04_eval(x, dim, 1);
        case 27:
            return cf05_eval(x, dim, 1);
        case 28:
            return cf06_eval(x, dim, 1);
        case 29:
            return cf07_eval(x, dim, 1);
        case 30:
            return cf08_eval(x, dim, 1);
        default:
            throw out_of_range("CEC2014 function index must be between 1 and 30.");
    }
}

inline double evaluate_with_bias(int func_num, const vector<double> &x) {
    // Bias values for CEC2014 functions: F_i has bias = i * 100
    static constexpr array<double, 31> bias = {
        0.0,      // placeholder for 0-index
        100.0,    // F1
        200.0,    // F2
        300.0,    // F3
        400.0,    // F4
        500.0,    // F5
        600.0,    // F6
        700.0,    // F7
        800.0,    // F8
        900.0,    // F9
        1000.0,   // F10
        1100.0,   // F11
        1200.0,   // F12
        1300.0,   // F13
        1400.0,   // F14
        1500.0,   // F15
        1600.0,   // F16
        1700.0,   // F17
        1800.0,   // F18
        1900.0,   // F19
        2000.0,   // F20
        2100.0,   // F21
        2200.0,   // F22
        2300.0,   // F23
        2400.0,   // F24
        2500.0,   // F25
        2600.0,   // F26
        2700.0,   // F27
        2800.0,   // F28
        2900.0,   // F29
        3000.0    // F30
    };
    double base = evaluate_without_bias(func_num, x);
    return base + bias[func_num];
}

} // anonymous namespace

void cec14_test_func_fortrain(double *x, double *f, int nx, int mx, int func_num) {
    if (func_num < 1 || func_num > 30) {
        cerr << "[cec14] Error: Test function " << func_num << " is not defined." << endl;
        for (int i = 0; i < mx; ++i) {
            f[i] = numeric_limits<double>::quiet_NaN();
        }
        return;
    }
    if (!(nx == 2 || nx == 10 || nx == 20 || nx == 30 || nx == 50 || nx == 100)) {
        cerr << "[cec14] Warning: dimensions other than 2,10,20,30,50,100 are not officially defined (requested D=" << nx << ")." << endl;
    }
    // Check for hybrid/composition functions that don't support D=2
    if (nx == 2 && ((func_num >= 17 && func_num <= 22) || (func_num >= 29 && func_num <= 30))) {
        cerr << "[cec14] Error: hf01-hf06, cf07, cf08 are NOT defined for D=2." << endl;
        for (int i = 0; i < mx; ++i) {
            f[i] = numeric_limits<double>::quiet_NaN();
        }
        return;
    }
    for (int i = 0; i < mx; ++i) {
        vector<double> xi(x + static_cast<size_t>(i) * nx, x + static_cast<size_t>(i + 1) * nx);
        try {
            f[i] = evaluate_with_bias(func_num, xi);
        } catch (const exception &ex) {
            cerr << "[cec14] Error: " << ex.what() << endl;
            f[i] = numeric_limits<double>::quiet_NaN();
        }
    }
}
