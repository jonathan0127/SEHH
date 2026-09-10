
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

#ifndef CEC24_DATA_DIR
#define CEC24_DATA_DIR "benchmarks/data/data_2024"
#endif

using namespace std;

namespace {

using Matrix = vector<double>; // row-major, size dim*dim

struct PairHash {
	size_t operator()(const pair<int, int> &p) const noexcept {
		// hash combine
		return (static_cast<size_t>(p.first) * 1315423911u) ^ static_cast<size_t>(p.second);
	}
};

class DataCache {
public:
	static DataCache &instance() {
		static DataCache inst;
		return inst;
	}

	const vector<double> &shift(int func, int nx, int cf_num) {
		const auto key = make_pair(func, nx);
		{
			lock_guard<mutex> lk(mu_);
			auto it = shifts_.find(key);
			if (it != shifts_.end()) return it->second;
		}
		// load
		const string path = string(CEC24_DATA_DIR) + "/shift_data_" + to_string(func) + ".txt";
		ifstream in(path);
		if (!in.is_open()) {
			throw runtime_error("Cannot open shift file: " + path);
		}
	const size_t need = (func <= 20) ? static_cast<size_t>(nx) : static_cast<size_t>(cf_num) * nx;
		vector<double> data; data.reserve(need);
		double v;
		while (in >> v) {
			data.push_back(v);
			if (data.size() == need) break;
		}
		if (data.size() != need) {
			throw runtime_error("Shift data size mismatch for func " + to_string(func) + ", nx=" + to_string(nx));
		}
		{
			lock_guard<mutex> lk(mu_);
			auto &ref = shifts_[key];
			ref = move(data);
			return ref;
		}
	}

	// Rotation matrix: for func<=20 => nx*nx; for func>20 => cf_num*nx*nx
	const Matrix &matrix(int func, int nx, int cf_num) {
		const auto key = make_pair(func, nx);
		{
			lock_guard<mutex> lk(mu_);
			auto it = mats_.find(key);
			if (it != mats_.end()) return it->second;
		}
		const string path = string(CEC24_DATA_DIR) + "/M_" + to_string(func) + "_D" + to_string(nx) + ".txt";
		ifstream in(path);
		if (!in.is_open()) {
			throw runtime_error("Cannot open matrix file: " + path);
		}
	const size_t blocks = (func <= 20) ? 1u : static_cast<size_t>(cf_num);
		const size_t need = blocks * static_cast<size_t>(nx) * static_cast<size_t>(nx);
		Matrix data; data.reserve(need);
		double v;
		while (in >> v) {
			data.push_back(v);
			if (data.size() == need) break;
		}
		if (data.size() != need) {
			throw runtime_error("Matrix data size mismatch for func " + to_string(func) + ", nx=" + to_string(nx));
		}
		{
			lock_guard<mutex> lk(mu_);
			auto &ref = mats_[key];
			ref = move(data);
			return ref;
		}
	}

	const vector<int> &shuffle(int func, int nx, int cf_num) {
		const auto key = make_pair(func, nx);
		{
			lock_guard<mutex> lk(mu_);
			auto it = shuf_.find(key);
			if (it != shuf_.end()) return it->second;
		}
		vector<int> empty;
		if (!((func >= 11 && func <= 20) || func == 29 || func == 30)) {
			lock_guard<mutex> lk(mu_);
			return shuf_[key] = move(empty);
		}
		const string path = string(CEC24_DATA_DIR) + "/shuffle_data_" + to_string(func) + "_D" + to_string(nx) + ".txt";
		ifstream in(path);
		if (!in.is_open()) {
			throw runtime_error("Cannot open shuffle file: " + path);
		}
		const size_t need = (func == 29 || func == 30) ? static_cast<size_t>(cf_num) * nx : static_cast<size_t>(nx);
		vector<int> data; data.reserve(need);
		int v;
		while (in >> v) {
			data.push_back(v);
			if (data.size() == need) break;
		}
		if (data.size() != need) {
			throw runtime_error("Shuffle data size mismatch for func " + to_string(func) + ", nx=" + to_string(nx));
		}
		{
			lock_guard<mutex> lk(mu_);
			auto &ref = shuf_[key];
			ref = move(data);
			return ref;
		}
	}

private:
	DataCache() = default;
	mutex mu_;
	unordered_map<pair<int, int>, vector<double>, PairHash> shifts_;
	unordered_map<pair<int, int>, Matrix, PairHash> mats_;
	unordered_map<pair<int, int>, vector<int>, PairHash> shuf_;
};

// ====== math helpers ======
inline vector<double> matvec_block(const Matrix &M, const vector<double> &v, int nx, size_t blockIndex) {
	vector<double> out(nx, 0.0);
	const size_t offset = blockIndex * static_cast<size_t>(nx) * static_cast<size_t>(nx);
	for (int i = 0; i < nx; ++i) {
		double sum = 0.0;
		const size_t row = offset + static_cast<size_t>(i) * nx;
		for (int j = 0; j < nx; ++j) sum += M[row + j] * v[j];
		out[i] = sum;
	}
	return out;
}

struct SRResult { vector<double> y, z; };

inline SRResult sr_func(const vector<double> &x,
						const vector<double> &Os,
						const Matrix &Mr,
						double sh_rate,
						int s_flag,
						int r_flag,
						int nx,
						size_t blockIndex = 0) {
	SRResult r; r.y.assign(nx, 0.0); r.z.assign(nx, 0.0);
	if (s_flag == 1) {
		for (int i = 0; i < nx; ++i) r.y[i] = (x[i] - Os[i]) * sh_rate;
	} else {
		for (int i = 0; i < nx; ++i) r.y[i] = x[i] * sh_rate;
	}
	if (r_flag == 1) r.z = matvec_block(Mr, r.y, nx, blockIndex); else r.z = r.y;
	return r;
}

inline vector<double> slice_vec(const vector<double> &v, size_t off, size_t len) {
	return vector<double>(v.begin() + static_cast<ptrdiff_t>(off), v.begin() + static_cast<ptrdiff_t>(off + len));
}

inline vector<int> slice_vec_int(const vector<int> &v, size_t off, size_t len) {
	return vector<int>(v.begin() + static_cast<ptrdiff_t>(off), v.begin() + static_cast<ptrdiff_t>(off + len));
}

inline Matrix slice_matrix_block(const Matrix &M, int nx, size_t blockIndex) {
	const size_t offset = blockIndex * static_cast<size_t>(nx) * nx;
	return Matrix(M.begin() + static_cast<ptrdiff_t>(offset), M.begin() + static_cast<ptrdiff_t>(offset + nx * static_cast<size_t>(nx)));
}


inline double ellips(const vector<double> &x, const vector<double> &Os, const Matrix &Mr, int s_flag, int r_flag) {
	const int nx = static_cast<int>(x.size());
	auto sr = sr_func(x, Os, Mr, 1.0, s_flag, r_flag, nx);
	double f = 0.0;
	for (int i = 0; i < nx; ++i) f += pow(10.0, 6.0 * i / max(1, nx - 1)) * sr.z[i] * sr.z[i];
	return f;
}

inline double sum_diff_pow(const vector<double> &x, const vector<double> &Os, const Matrix &Mr, int s_flag, int r_flag) {
	const int nx = static_cast<int>(x.size());
	auto sr = sr_func(x, Os, Mr, 1.0, s_flag, r_flag, nx);
	double sum = 0.0;
	for (int i = 0; i < nx; ++i) sum += pow(fabs(sr.z[i]), i + 1);
	return sum;
}

inline double zakharov(const vector<double> &x, const vector<double> &Os, const Matrix &Mr, int s_flag, int r_flag) {
	const int nx = static_cast<int>(x.size());
	auto sr = sr_func(x, Os, Mr, 1.0, s_flag, r_flag, nx);
	double sum1 = 0.0, sum2 = 0.0;
	for (int i = 0; i < nx; ++i) { sum1 += sr.z[i] * sr.z[i]; sum2 += 0.5 * (i + 1) * sr.z[i]; }
	return sum1 + sum2 * sum2 + pow(sum2, 4.0);
}

inline double levy(const vector<double> &x, const vector<double> &Os, const Matrix &Mr, int s_flag, int r_flag) {
	const int nx = static_cast<int>(x.size());
	auto sr = sr_func(x, Os, Mr, 1.0, s_flag, r_flag, nx);
	vector<double> w(nx, 0.0);
	for (int i = 0; i < nx; ++i) w[i] = 1.0 + (sr.z[i] - 1.0) / 4.0;
	double term1 = pow(sin(M_PI * w[0]), 2.0);
	double term3 = pow(w[nx - 1] - 1.0, 2.0) * (1.0 + pow(sin(2.0 * M_PI * w[nx - 1]), 2.0));
	double sum = 0.0;
	for (int i = 0; i < nx - 1; ++i) sum += pow(w[i] - 1.0, 2.0) * (1.0 + 10.0 * pow(sin(M_PI * w[i] + 1.0), 2.0));
	return term1 + sum + term3;
}

inline double dixon_price(const vector<double> &x, const vector<double> &Os, const Matrix &Mr, int s_flag, int r_flag) {
	const int nx = static_cast<int>(x.size());
	auto sr = sr_func(x, Os, Mr, 1.0, s_flag, r_flag, nx);
	double term1 = pow(sr.z[0] - 1.0, 2.0);
	double sum = 0.0;
	for (int i = 1; i < nx; ++i) sum += i * pow((2.0 * sr.z[i]) * (2.0 * sr.z[i]) - sr.z[i - 1], 2.0);
	return term1 + sum;
}

inline double bent_cigar(const vector<double> &x, const vector<double> &Os, const Matrix &Mr, int s_flag, int r_flag) {
	const int nx = static_cast<int>(x.size());
	auto sr = sr_func(x, Os, Mr, 1.0, s_flag, r_flag, nx);
	double f = sr.z[0] * sr.z[0];
	for (int i = 1; i < nx; ++i) f += pow(10.0, 6.0) * sr.z[i] * sr.z[i];
	return f;
}

inline double discus(const vector<double> &x, const vector<double> &Os, const Matrix &Mr, int s_flag, int r_flag) {
	const int nx = static_cast<int>(x.size());
	auto sr = sr_func(x, Os, Mr, 1.0, s_flag, r_flag, nx);
	double f = pow(10.0, 6.0) * sr.z[0] * sr.z[0];
	for (int i = 1; i < nx; ++i) f += sr.z[i] * sr.z[i];
	return f;
}

inline double rosenbrock(const vector<double> &x, const vector<double> &Os, const Matrix &Mr, int s_flag, int r_flag) {
	const int nx = static_cast<int>(x.size());
	auto sr = sr_func(x, Os, Mr, 2.048 / 100.0, s_flag, r_flag, nx);
	vector<double> z = sr.z;
	if (!z.empty()) {
		z[0] += 1.0;
		for (int i = 0; i < nx - 1; ++i) z[i + 1] += 1.0;
	}
	double f = 0.0;
	for (int i = 0; i < nx - 1; ++i) {
		double tmp1 = z[i] * z[i] - z[i + 1];
		double tmp2 = z[i] - 1.0;
		f += 100.0 * tmp1 * tmp1 + tmp2 * tmp2;
	}
	return f;
}

inline double schaffer_F7(const vector<double> &x, const vector<double> &Os, const Matrix &Mr, int s_flag, int r_flag) {
	const int nx = static_cast<int>(x.size());
	auto sr = sr_func(x, Os, Mr, 1.0, s_flag, r_flag, nx);
	double f = 0.0;
	for (int i = 0; i < nx - 1; ++i) {
		double zi = sqrt(sr.y[i] * sr.y[i] + sr.y[i + 1] * sr.y[i + 1]);
		double tmp = sin(50.0 * pow(zi, 0.2));
		f += pow(zi, 0.5) + pow(zi, 0.5) * tmp * tmp;
	}
	return f * f / ((nx - 1.0) * (nx - 1.0));
}

inline double ackley(const vector<double> &x, const vector<double> &Os, const Matrix &Mr, int s_flag, int r_flag) {
	const int nx = static_cast<int>(x.size());
	auto sr = sr_func(x, Os, Mr, 1.0, s_flag, r_flag, nx);
	double sum1 = 0.0, sum2 = 0.0;
	for (int i = 0; i < nx; ++i) { sum1 += sr.z[i] * sr.z[i]; sum2 += cos(2.0 * M_PI * sr.z[i]); }
	sum1 = -0.2 * sqrt(sum1 / nx);
	sum2 /= nx;
	return exp(1.0) - 20.0 * exp(sum1) - exp(sum2) + 20.0;
}

inline double weierstrass(const vector<double> &x, const vector<double> &Os, const Matrix &Mr, int s_flag, int r_flag) {
	const int nx = static_cast<int>(x.size());
	auto sr = sr_func(x, Os, Mr, 0.5 / 100.0, s_flag, r_flag, nx);
	const int k_max = 20; const double a = 0.5, b = 3.0;
	double f = 0.0;
	double sum2 = 0.0; // precompute
	for (int k = 0; k <= k_max; ++k) sum2 += pow(a, k) * cos(2.0 * M_PI * pow(b, k) * 0.5);
	for (int i = 0; i < nx; ++i) {
		double sum = 0.0;
		for (int k = 0; k <= k_max; ++k) sum += pow(a, k) * cos(2.0 * M_PI * pow(b, k) * (sr.z[i] + 0.5));
		f += sum;
	}
	return f - nx * sum2;
}

inline double griewank(const vector<double> &x, const vector<double> &Os, const Matrix &Mr, int s_flag, int r_flag) {
	const int nx = static_cast<int>(x.size());
	auto sr = sr_func(x, Os, Mr, 600.0 / 100.0, s_flag, r_flag, nx);
	double s = 0.0, p = 1.0;
	for (int i = 0; i < nx; ++i) { s += sr.z[i] * sr.z[i]; p *= cos(sr.z[i] / sqrt(1.0 + i)); }
	return 1.0 + s / 4000.0 - p;
}

inline double rastrigin(const vector<double> &x, const vector<double> &Os, const Matrix &Mr, int s_flag, int r_flag) {
	const int nx = static_cast<int>(x.size());
	auto sr = sr_func(x, Os, Mr, 5.12 / 100.0, s_flag, r_flag, nx);
	double f = 0.0;
	for (int i = 0; i < nx; ++i) f += (sr.z[i] * sr.z[i] - 10.0 * cos(2.0 * M_PI * sr.z[i]) + 10.0);
	return f;
}

inline double step_rastrigin(const vector<double> &x, const vector<double> &Os, const Matrix &Mr, int s_flag, int r_flag) {
	const int nx = static_cast<int>(x.size());
	// first do the noncontinuous step on y relative to Os
	vector<double> y(nx, 0.0);
	for (int i = 0; i < nx; ++i) {
		double yi = x[i];
		if (fabs(yi - Os[i]) > 0.5) yi = Os[i] + floor(2.0 * (yi - Os[i]) + 0.5) / 2.0;
		y[i] = yi;
	}
	auto sr = sr_func(y, Os, Mr, 5.12 / 100.0, s_flag, r_flag, nx);
	double f = 0.0;
	for (int i = 0; i < nx; ++i) f += (sr.z[i] * sr.z[i] - 10.0 * cos(2.0 * M_PI * sr.z[i]) + 10.0);
	return f;
}

inline double schwefel(const vector<double> &x, const vector<double> &Os, const Matrix &Mr, int s_flag, int r_flag) {
	const int nx = static_cast<int>(x.size());
	auto sr = sr_func(x, Os, Mr, 1000.0 / 100.0, s_flag, r_flag, nx);
	double f = 0.0;
	vector<double> z = sr.z;
	for (int i = 0; i < nx; ++i) {
		z[i] += 4.209687462275036e+002;
		if (z[i] > 500.0) {
			f -= (500.0 - fmod(z[i], 500.0)) * sin(sqrt(500.0 - fmod(z[i], 500.0)));
			double tmp = (z[i] - 500.0) / 100.0; f += tmp * tmp / nx;
		} else if (z[i] < -500.0) {
			f -= (-500.0 + fmod(fabs(z[i]), 500.0)) * sin(sqrt(500.0 - fmod(fabs(z[i]), 500.0)));
			double tmp = (z[i] + 500.0) / 100.0; f += tmp * tmp / nx;
		} else {
			f -= z[i] * sin(sqrt(fabs(z[i])));
		}
	}
	f += 4.189828872724338e+002 * nx;
	return f;
}

inline double katsuura(const vector<double> &x, const vector<double> &Os, const Matrix &Mr, int s_flag, int r_flag) {
	const int nx = static_cast<int>(x.size());
	auto sr = sr_func(x, Os, Mr, 5.0 / 100.0, s_flag, r_flag, nx);
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

inline double bi_rastrigin(const vector<double> &x, const vector<double> &Os, const Matrix &Mr, int s_flag, int r_flag) {
	// Lunacek bi-Rastrigin (ported, simplified)
	const int nx = static_cast<int>(x.size());
	const double mu0 = 2.5, d = 1.0;
	double s = 1.0 - 1.0 / (2.0 * sqrt(nx + 20.0) - 8.2);
	double mu1 = -sqrt((mu0 * mu0 - d) / s);

	vector<double> y(nx, 0.0);
	if (s_flag == 1) { for (int i = 0; i < nx; ++i) y[i] = x[i] - Os[i]; }
	else { for (int i = 0; i < nx; ++i) y[i] = x[i]; }
	for (int i = 0; i < nx; ++i) y[i] *= 10.0 / 100.0;

	vector<double> tmpx(nx, 0.0), z(nx, 0.0);
	for (int i = 0; i < nx; ++i) {
		tmpx[i] = 2.0 * y[i];
		if (Os[i] < 0.0) tmpx[i] *= -1.0;
	}
	for (int i = 0; i < nx; ++i) { z[i] = tmpx[i]; tmpx[i] += mu0; }

	double tmp1 = 0.0, tmp2 = 0.0, tmp = 0.0;
	for (int i = 0; i < nx; ++i) { double t = tmpx[i] - mu0; tmp1 += t * t; t = tmpx[i] - mu1; tmp2 += t * t; }
	tmp2 = tmp2 * s + d * nx;

	vector<double> yrot = (r_flag == 1) ? matvec_block(Mr, z, nx, 0) : z;
	for (int i = 0; i < nx; ++i) tmp += cos(2.0 * M_PI * yrot[i]);
	double f = (tmp1 < tmp2 ? tmp1 : tmp2) + 10.0 * (nx - tmp);
	return f;
}

inline double grie_rosen(const vector<double> &x, const vector<double> &Os, const Matrix &Mr, int s_flag, int r_flag) {
	const int nx = static_cast<int>(x.size());
	auto sr = sr_func(x, Os, Mr, 5.0 / 100.0, s_flag, r_flag, nx);
	vector<double> z = sr.z;
	if (!z.empty()) { z[0] += 1.0; for (int i = 0; i < nx - 1; ++i) z[i + 1] += 1.0; }
	double f = 0.0;
	for (int i = 0; i < nx - 1; ++i) {
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
	const int nx = static_cast<int>(x.size());
	auto sr = sr_func(x, Os, Mr, 1.0, s_flag, r_flag, nx);
	double f = 0.0;
	for (int i = 0; i < nx - 1; ++i) {
		double t1 = sin(sqrt(sr.z[i] * sr.z[i] + sr.z[i + 1] * sr.z[i + 1]));
		t1 *= t1;
		double t2 = 1.0 + 0.001 * (sr.z[i] * sr.z[i] + sr.z[i + 1] * sr.z[i + 1]);
		f += 0.5 + (t1 - 0.5) / (t2 * t2);
	}
	double t1 = sin(sqrt(sr.z[nx - 1] * sr.z[nx - 1] + sr.z[0] * sr.z[0])); t1 *= t1;
	double t2 = 1.0 + 0.001 * (sr.z[nx - 1] * sr.z[nx - 1] + sr.z[0] * sr.z[0]);
	f += 0.5 + (t1 - 0.5) / (t2 * t2);
	return f;
}

inline double happycat(const vector<double> &x, const vector<double> &Os, const Matrix &Mr, int s_flag, int r_flag) {
	const int nx = static_cast<int>(x.size());
	auto sr = sr_func(x, Os, Mr, 5.0 / 100.0, s_flag, r_flag, nx);
	double r2 = 0.0, sum = 0.0; vector<double> z = sr.z;
	for (int i = 0; i < nx; ++i) { z[i] = z[i] - 1.0; r2 += z[i] * z[i]; sum += z[i]; }
	double alpha = 1.0 / 8.0;
	return pow(fabs(r2 - nx), 2.0 * alpha) + (0.5 * r2 + sum) / nx + 0.5;
}

inline double hgbat(const vector<double> &x, const vector<double> &Os, const Matrix &Mr, int s_flag, int r_flag) {
	const int nx = static_cast<int>(x.size());
	auto sr = sr_func(x, Os, Mr, 5.0 / 100.0, s_flag, r_flag, nx);
	double r2 = 0.0, sum = 0.0; vector<double> z = sr.z;
	for (int i = 0; i < nx; ++i) { z[i] = z[i] - 1.0; r2 += z[i] * z[i]; sum += z[i]; }
	double alpha = 1.0 / 4.0;
	return pow(fabs(r2 * r2 - sum * sum), 2.0 * alpha) + (0.5 * r2 + sum) / nx + 0.5;
}

// ====== hybrid helpers ======
inline void reorder_by_S(vector<double> &dst, const vector<double> &src, const vector<int> &S) {
	const int nx = static_cast<int>(src.size());
	dst.resize(nx);
	for (int i = 0; i < nx; ++i) dst[i] = src[S[i] - 1]; // S is 1-based
}

inline double hf01_eval(const vector<double> &x, const vector<double> &Os, const Matrix &Mr, const vector<int> &S) {
	const int nx = static_cast<int>(x.size());
	const int cf_num = 3; const double Gp[cf_num] = {0.2, 0.4, 0.4};
	vector<int> G_nx(cf_num, 0), G(cf_num, 0);
	int tmp = 0; for (int i = 0; i < cf_num - 1; ++i) { G_nx[i] = (int)ceil(Gp[i] * nx); tmp += G_nx[i]; }
	G_nx[cf_num - 1] = nx - tmp; for (int i = 1; i < cf_num; ++i) G[i] = G[i - 1] + G_nx[i - 1];

	auto sr = sr_func(x, Os, Mr, 1.0, 1, 1, nx);
	vector<double> y; reorder_by_S(y, sr.z, S);

	double f0 = zakharov(slice_vec(y, G[0], G_nx[0]), Os, Mr, 0, 0);
	double f1 = rosenbrock(slice_vec(y, G[1], G_nx[1]), Os, Mr, 0, 0);
	double f2 = rastrigin(slice_vec(y, G[2], G_nx[2]), Os, Mr, 0, 0);
	return f0 + f1 + f2;
}

inline double hf02_eval(const vector<double> &x, const vector<double> &Os, const Matrix &Mr, const vector<int> &S) {
	const int nx = static_cast<int>(x.size());
	const int cf_num = 3; const double Gp[cf_num] = {0.3, 0.3, 0.4};
	vector<int> G_nx(cf_num, 0), G(cf_num, 0);
	int tmp = 0; for (int i = 0; i < cf_num - 1; ++i) { G_nx[i] = (int)ceil(Gp[i] * nx); tmp += G_nx[i]; }
	G_nx[cf_num - 1] = nx - tmp; for (int i = 1; i < cf_num; ++i) G[i] = G[i - 1] + G_nx[i - 1];

	auto sr = sr_func(x, Os, Mr, 1.0, 1, 1, nx);
	vector<double> y; reorder_by_S(y, sr.z, S);

	double f0 = ellips(slice_vec(y, G[0], G_nx[0]), Os, Mr, 0, 0);
	double f1 = schwefel(slice_vec(y, G[1], G_nx[1]), Os, Mr, 0, 0);
	double f2 = bent_cigar(slice_vec(y, G[2], G_nx[2]), Os, Mr, 0, 0);
	return f0 + f1 + f2;
}

inline double hf03_eval(const vector<double> &x, const vector<double> &Os, const Matrix &Mr, const vector<int> &S) {
	const int nx = static_cast<int>(x.size());
	const int cf_num = 3; const double Gp[cf_num] = {0.3, 0.3, 0.4};
	vector<int> G_nx(cf_num, 0), G(cf_num, 0);
	int tmp = 0; for (int i = 0; i < cf_num - 1; ++i) { G_nx[i] = (int)ceil(Gp[i] * nx); tmp += G_nx[i]; }
	G_nx[cf_num - 1] = nx - tmp; for (int i = 1; i < cf_num; ++i) G[i] = G[i - 1] + G_nx[i - 1];

	auto sr = sr_func(x, Os, Mr, 1.0, 1, 1, nx);
	vector<double> y; reorder_by_S(y, sr.z, S);

	double f0 = bent_cigar(slice_vec(y, G[0], G_nx[0]), Os, Mr, 0, 0);
	double f1 = rosenbrock(slice_vec(y, G[1], G_nx[1]), Os, Mr, 0, 0);
	double f2 = bi_rastrigin(slice_vec(y, G[2], G_nx[2]), Os, Mr, 0, 0);
	return f0 + f1 + f2;
}

inline double hf04_eval(const vector<double> &x, const vector<double> &Os, const Matrix &Mr, const vector<int> &S) {
	const int nx = static_cast<int>(x.size());
	const int cf_num = 4; const double Gp[cf_num] = {0.2, 0.2, 0.2, 0.4};
	vector<int> G_nx(cf_num, 0), G(cf_num, 0);
	int tmp = 0; for (int i = 0; i < cf_num - 1; ++i) { G_nx[i] = (int)ceil(Gp[i] * nx); tmp += G_nx[i]; }
	G_nx[cf_num - 1] = nx - tmp; for (int i = 1; i < cf_num; ++i) G[i] = G[i - 1] + G_nx[i - 1];

	auto sr = sr_func(x, Os, Mr, 1.0, 1, 1, nx);
	vector<double> y; reorder_by_S(y, sr.z, S);

	double f0 = ellips(slice_vec(y, G[0], G_nx[0]), Os, Mr, 0, 0);
	double f1 = ackley(slice_vec(y, G[1], G_nx[1]), Os, Mr, 0, 0);
	double f2 = schaffer_F7(slice_vec(y, G[2], G_nx[2]), Os, Mr, 0, 0);
	double f3 = rastrigin(slice_vec(y, G[3], G_nx[3]), Os, Mr, 0, 0);
	return f0 + f1 + f2 + f3;
}

inline double hf05_eval(const vector<double> &x, const vector<double> &Os, const Matrix &Mr, const vector<int> &S) {
	const int nx = static_cast<int>(x.size());
	const int cf_num = 4; const double Gp[cf_num] = {0.2, 0.2, 0.3, 0.3};
	vector<int> G_nx(cf_num, 0), G(cf_num, 0);
	int tmp = 0; for (int i = 0; i < cf_num - 1; ++i) { G_nx[i] = (int)ceil(Gp[i] * nx); tmp += G_nx[i]; }
	G_nx[cf_num - 1] = nx - tmp; for (int i = 1; i < cf_num; ++i) G[i] = G[i - 1] + G_nx[i - 1];

	auto sr = sr_func(x, Os, Mr, 1.0, 1, 1, nx);
	vector<double> y; reorder_by_S(y, sr.z, S);

	double f0 = bent_cigar(slice_vec(y, G[0], G_nx[0]), Os, Mr, 0, 0);
	double f1 = hgbat(slice_vec(y, G[1], G_nx[1]), Os, Mr, 0, 0);
	double f2 = rastrigin(slice_vec(y, G[2], G_nx[2]), Os, Mr, 0, 0);
	double f3 = rosenbrock(slice_vec(y, G[3], G_nx[3]), Os, Mr, 0, 0);
	return f0 + f1 + f2 + f3;
}

inline double hf06_eval(const vector<double> &x, const vector<double> &Os, const Matrix &Mr, const vector<int> &S) {
	const int nx = static_cast<int>(x.size());
	const int cf_num = 4; const double Gp[cf_num] = {0.2, 0.2, 0.3, 0.3};
	vector<int> G_nx(cf_num, 0), G(cf_num, 0);
	int tmp = 0; for (int i = 0; i < cf_num - 1; ++i) { G_nx[i] = (int)ceil(Gp[i] * nx); tmp += G_nx[i]; }
	G_nx[cf_num - 1] = nx - tmp; for (int i = 1; i < cf_num; ++i) G[i] = G[i - 1] + G_nx[i - 1];

	auto sr = sr_func(x, Os, Mr, 1.0, 1, 1, nx);
	vector<double> y; reorder_by_S(y, sr.z, S);

	double f0 = escaffer6(slice_vec(y, G[0], G_nx[0]), Os, Mr, 0, 0);
	double f1 = hgbat(slice_vec(y, G[1], G_nx[1]), Os, Mr, 0, 0);
	double f2 = rosenbrock(slice_vec(y, G[2], G_nx[2]), Os, Mr, 0, 0);
	double f3 = schwefel(slice_vec(y, G[3], G_nx[3]), Os, Mr, 0, 0);
	return f0 + f1 + f2 + f3;
}

inline double hf07_eval(const vector<double> &x, const vector<double> &Os, const Matrix &Mr, const vector<int> &S) {
	const int nx = static_cast<int>(x.size());
	const int cf_num = 5; const double Gp[cf_num] = {0.1, 0.2, 0.2, 0.2, 0.3};
	vector<int> G_nx(cf_num, 0), G(cf_num, 0);
	int tmp = 0; for (int i = 0; i < cf_num - 1; ++i) { G_nx[i] = (int)ceil(Gp[i] * nx); tmp += G_nx[i]; }
	G_nx[cf_num - 1] = nx - tmp; for (int i = 1; i < cf_num; ++i) G[i] = G[i - 1] + G_nx[i - 1];

	auto sr = sr_func(x, Os, Mr, 1.0, 1, 1, nx);
	vector<double> y; reorder_by_S(y, sr.z, S);

	double f0 = katsuura(slice_vec(y, G[0], G_nx[0]), Os, Mr, 0, 0);
	double f1 = ackley(slice_vec(y, G[1], G_nx[1]), Os, Mr, 0, 0);
	double f2 = grie_rosen(slice_vec(y, G[2], G_nx[2]), Os, Mr, 0, 0);
	double f3 = schwefel(slice_vec(y, G[3], G_nx[3]), Os, Mr, 0, 0);
	double f4 = rastrigin(slice_vec(y, G[4], G_nx[4]), Os, Mr, 0, 0);
	return f0 + f1 + f2 + f3 + f4;
}

inline double hf08_eval(const vector<double> &x, const vector<double> &Os, const Matrix &Mr, const vector<int> &S) {
	const int nx = static_cast<int>(x.size());
	const int cf_num = 5; const double Gp[cf_num] = {0.2, 0.2, 0.2, 0.2, 0.2};
	vector<int> G_nx(cf_num, 0), G(cf_num, 0);
	int tmp = 0; for (int i = 0; i < cf_num - 1; ++i) { G_nx[i] = (int)ceil(Gp[i] * nx); tmp += G_nx[i]; }
	G_nx[cf_num - 1] = nx - tmp; for (int i = 1; i < cf_num; ++i) G[i] = G[i - 1] + G_nx[i - 1];

	auto sr = sr_func(x, Os, Mr, 1.0, 1, 1, nx);
	vector<double> y; reorder_by_S(y, sr.z, S);

	double f0 = ellips(slice_vec(y, G[0], G_nx[0]), Os, Mr, 0, 0);
	double f1 = ackley(slice_vec(y, G[1], G_nx[1]), Os, Mr, 0, 0);
	double f2 = rastrigin(slice_vec(y, G[2], G_nx[2]), Os, Mr, 0, 0);
	double f3 = hgbat(slice_vec(y, G[3], G_nx[3]), Os, Mr, 0, 0);
	double f4 = discus(slice_vec(y, G[4], G_nx[4]), Os, Mr, 0, 0);
	return f0 + f1 + f2 + f3 + f4;
}

inline double hf09_eval(const vector<double> &x, const vector<double> &Os, const Matrix &Mr, const vector<int> &S) {
	const int nx = static_cast<int>(x.size());
	const int cf_num = 5; const double Gp[cf_num] = {0.2, 0.2, 0.2, 0.2, 0.2};
	vector<int> G_nx(cf_num, 0), G(cf_num, 0);
	int tmp = 0; for (int i = 0; i < cf_num - 1; ++i) { G_nx[i] = (int)ceil(Gp[i] * nx); tmp += G_nx[i]; }
	G_nx[cf_num - 1] = nx - tmp; for (int i = 1; i < cf_num; ++i) G[i] = G[i - 1] + G_nx[i - 1];

	auto sr = sr_func(x, Os, Mr, 1.0, 1, 1, nx);
	vector<double> y; reorder_by_S(y, sr.z, S);

	double f0 = bent_cigar(slice_vec(y, G[0], G_nx[0]), Os, Mr, 0, 0);
	double f1 = rastrigin(slice_vec(y, G[1], G_nx[1]), Os, Mr, 0, 0);
	double f2 = grie_rosen(slice_vec(y, G[2], G_nx[2]), Os, Mr, 0, 0);
	double f3 = weierstrass(slice_vec(y, G[3], G_nx[3]), Os, Mr, 0, 0);
	double f4 = escaffer6(slice_vec(y, G[4], G_nx[4]), Os, Mr, 0, 0);
	return f0 + f1 + f2 + f3 + f4;
}

inline double hf10_eval(const vector<double> &x, const vector<double> &Os, const Matrix &Mr, const vector<int> &S) {
	const int nx = static_cast<int>(x.size());
	const int cf_num = 6; const double Gp[cf_num] = {0.1, 0.1, 0.2, 0.2, 0.2, 0.2};
	vector<int> G_nx(cf_num, 0), G(cf_num, 0);
	int tmp = 0; for (int i = 0; i < cf_num - 1; ++i) { G_nx[i] = (int)ceil(Gp[i] * nx); tmp += G_nx[i]; }
	G_nx[cf_num - 1] = nx - tmp; for (int i = 1; i < cf_num; ++i) G[i] = G[i - 1] + G_nx[i - 1];

	auto sr = sr_func(x, Os, Mr, 1.0, 1, 1, nx);
	vector<double> y; reorder_by_S(y, sr.z, S);

	double f0 = hgbat(slice_vec(y, G[0], G_nx[0]), Os, Mr, 0, 0);
	double f1 = katsuura(slice_vec(y, G[1], G_nx[1]), Os, Mr, 0, 0);
	double f2 = ackley(slice_vec(y, G[2], G_nx[2]), Os, Mr, 0, 0);
	double f3 = rastrigin(slice_vec(y, G[3], G_nx[3]), Os, Mr, 0, 0);
	double f4 = schwefel(slice_vec(y, G[4], G_nx[4]), Os, Mr, 0, 0);
	double f5 = schaffer_F7(slice_vec(y, G[5], G_nx[5]), Os, Mr, 0, 0);
	return f0 + f1 + f2 + f3 + f4 + f5;
}

inline double cf_weighted(const vector<double> &x,
						  const vector<double> &Os,
						  const vector<double> &delta,
						  const vector<double> &bias,
						  const vector<double> &fit,
						  int cf_num,
						  int nx) {
	vector<double> w(cf_num, 0.0);
	double w_max = 0.0, w_sum = 0.0;
	for (int i = 0; i < cf_num; ++i) {
		double d2 = 0.0;
		for (int j = 0; j < nx; ++j) {
			const double diff = x[j] - Os[static_cast<size_t>(i) * nx + j];
			d2 += diff * diff;
		}
		double wi = (d2 != 0.0) ? pow(1.0 / d2, 0.5) * exp(-d2 / (2.0 * nx * delta[i] * delta[i])) : numeric_limits<double>::infinity();
		w[i] = wi; if (wi > w_max) w_max = wi;
	}
	for (int i = 0; i < cf_num; ++i) w_sum += w[i];
	if (w_max == 0.0) { fill(w.begin(), w.end(), 1.0); w_sum = static_cast<double>(cf_num); }
	double f = 0.0;
	for (int i = 0; i < cf_num; ++i) f += (w[i] / w_sum) * (fit[i] + bias[i]);
	return f;
}

inline double cf01_eval(const vector<double> &x, int nx, const vector<double> &Os, const Matrix &Mr) {
	const int cf_num = 3; vector<double> fit(cf_num, 0.0);
	vector<double> delta{10, 20, 30}; vector<double> bias{0, 100, 200};
	fit[0] = rosenbrock(x, slice_vec(Os, 0 * nx, nx), Mr, 1, 1);
	fit[1] = ellips(x,    slice_vec(Os, 1 * nx, nx), Mr, 1, 1); fit[1] = 10000 * fit[1] / 1e+10;
	fit[2] = rastrigin(x, slice_vec(Os, 2 * nx, nx), Mr, 1, 1);
	return cf_weighted(x, Os, delta, bias, fit, cf_num, nx);
}

inline double cf02_eval(const vector<double> &x, int nx, const vector<double> &Os, const Matrix &Mr) {
	const int cf_num = 3; vector<double> fit(cf_num, 0.0);
	vector<double> delta{10, 20, 30}; vector<double> bias{0, 100, 200};
	fit[0] = rastrigin(x, slice_vec(Os, 0 * nx, nx), Mr, 1, 1);
	fit[1] = griewank(x,  slice_vec(Os, 1 * nx, nx), Mr, 1, 1); fit[1] = 1000 * fit[1] / 100;
	fit[2] = schwefel(x,  slice_vec(Os, 2 * nx, nx), Mr, 1, 1);
	return cf_weighted(x, Os, delta, bias, fit, cf_num, nx);
}

inline double cf03_eval(const vector<double> &x, int nx, const vector<double> &Os, const Matrix &Mr) {
	const int cf_num = 4; vector<double> fit(cf_num, 0.0);
	vector<double> delta{10, 20, 30, 40}; vector<double> bias{0, 100, 200, 300};
	fit[0] = rosenbrock(x,  slice_vec(Os, 0 * nx, nx), Mr, 1, 1);
	fit[1] = ackley(x,     slice_vec(Os, 1 * nx, nx), Mr, 1, 1); fit[1] = 1000 * fit[1] / 100;
	fit[2] = schwefel(x,   slice_vec(Os, 2 * nx, nx), Mr, 1, 1);
	fit[3] = rastrigin(x,  slice_vec(Os, 3 * nx, nx), Mr, 1, 1);
	return cf_weighted(x, Os, delta, bias, fit, cf_num, nx);
}

inline double cf04_eval(const vector<double> &x, int nx, const vector<double> &Os, const Matrix &Mr) {
	const int cf_num = 4; vector<double> fit(cf_num, 0.0);
	vector<double> delta{10, 20, 30, 40}; vector<double> bias{0, 100, 200, 300};
	fit[0] = ackley(x,     slice_vec(Os, 0 * nx, nx), Mr, 1, 1); fit[0] = 1000 * fit[0] / 100;
	fit[1] = ellips(x,     slice_vec(Os, 1 * nx, nx), Mr, 1, 1); fit[1] = 10000 * fit[1] / 1e+10;
	fit[2] = griewank(x,   slice_vec(Os, 2 * nx, nx), Mr, 1, 1); fit[2] = 1000 * fit[2] / 100;
	fit[3] = rastrigin(x,  slice_vec(Os, 3 * nx, nx), Mr, 1, 1);
	return cf_weighted(x, Os, delta, bias, fit, cf_num, nx);
}

inline double cf05_eval(const vector<double> &x, int nx, const vector<double> &Os, const Matrix &Mr) {
	const int cf_num = 5; vector<double> fit(cf_num, 0.0);
	vector<double> delta{10, 20, 30, 40, 50}; vector<double> bias{0, 100, 200, 300, 400};
	fit[0] = rastrigin(x,  slice_vec(Os, 0 * nx, nx), Mr, 1, 1); fit[0] = 10000 * fit[0] / 1e+3;
	fit[1] = happycat(x,   slice_vec(Os, 1 * nx, nx), Mr, 1, 1); fit[1] = 1000 * fit[1] / 1e+3;
	fit[2] = ackley(x,     slice_vec(Os, 2 * nx, nx), Mr, 1, 1); fit[2] = 1000 * fit[2] / 100;
	fit[3] = discus(x,     slice_vec(Os, 3 * nx, nx), Mr, 1, 1); fit[3] = 10000 * fit[3] / 1e+10;
	fit[4] = rosenbrock(x, slice_vec(Os, 4 * nx, nx), Mr, 1, 1);
	return cf_weighted(x, Os, delta, bias, fit, cf_num, nx);
}

inline double cf06_eval(const vector<double> &x, int nx, const vector<double> &Os, const Matrix &Mr) {
	const int cf_num = 5; vector<double> fit(cf_num, 0.0);
	vector<double> delta{10, 20, 20, 30, 40}; vector<double> bias{0, 100, 200, 300, 400};
	fit[0] = escaffer6(x,  slice_vec(Os, 0 * nx, nx), Mr, 1, 1); fit[0] = 10000 * fit[0] / 2e+7;
	fit[1] = schwefel(x,   slice_vec(Os, 1 * nx, nx), Mr, 1, 1);
	fit[2] = griewank(x,   slice_vec(Os, 2 * nx, nx), Mr, 1, 1); fit[2] = 1000 * fit[2] / 100;
	fit[3] = rosenbrock(x, slice_vec(Os, 3 * nx, nx), Mr, 1, 1);
	fit[4] = rastrigin(x,  slice_vec(Os, 4 * nx, nx), Mr, 1, 1); fit[4] = 10000 * fit[4] / 1e+3;
	return cf_weighted(x, Os, delta, bias, fit, cf_num, nx);
}

inline double cf07_eval(const vector<double> &x, int nx, const vector<double> &Os, const Matrix &Mr) {
	const int cf_num = 6; vector<double> fit(cf_num, 0.0);
	vector<double> delta{10, 20, 30, 40, 50, 60}; vector<double> bias{0, 100, 200, 300, 400, 500};
	fit[0] = hgbat(x,      slice_vec(Os, 0 * nx, nx), Mr, 1, 1); fit[0] = 10000 * fit[0] / 1000;
	fit[1] = rastrigin(x,  slice_vec(Os, 1 * nx, nx), Mr, 1, 1); fit[1] = 10000 * fit[1] / 1e+3;
	fit[2] = schwefel(x,   slice_vec(Os, 2 * nx, nx), Mr, 1, 1); fit[2] = 10000 * fit[2] / 4e+3;
	fit[3] = bent_cigar(x, slice_vec(Os, 3 * nx, nx), Mr, 1, 1); fit[3] = 10000 * fit[3] / 1e+30;
	fit[4] = ellips(x,     slice_vec(Os, 4 * nx, nx), Mr, 1, 1); fit[4] = 10000 * fit[4] / 1e+10;
	fit[5] = escaffer6(x,  slice_vec(Os, 5 * nx, nx), Mr, 1, 1); fit[5] = 10000 * fit[5] / 2e+7;
	return cf_weighted(x, Os, delta, bias, fit, cf_num, nx);
}

inline double cf08_eval(const vector<double> &x, int nx, const vector<double> &Os, const Matrix &Mr) {
	const int cf_num = 6; vector<double> fit(cf_num, 0.0);
	vector<double> delta{10, 20, 30, 40, 50, 60}; vector<double> bias{0, 100, 200, 300, 400, 500};
	fit[0] = ackley(x,     slice_vec(Os, 0 * nx, nx), Mr, 1, 1); fit[0] = 1000 * fit[0] / 100;
	fit[1] = griewank(x,   slice_vec(Os, 1 * nx, nx), Mr, 1, 1); fit[1] = 1000 * fit[1] / 100;
	fit[2] = discus(x,     slice_vec(Os, 2 * nx, nx), Mr, 1, 1); fit[2] = 10000 * fit[2] / 1e+10;
	fit[3] = rosenbrock(x, slice_vec(Os, 3 * nx, nx), Mr, 1, 1);
	fit[4] = happycat(x,   slice_vec(Os, 4 * nx, nx), Mr, 1, 1); fit[4] = 1000 * fit[4] / 1e+3;
	fit[5] = escaffer6(x,  slice_vec(Os, 5 * nx, nx), Mr, 1, 1); fit[5] = 10000 * fit[5] / 2e+7;
	return cf_weighted(x, Os, delta, bias, fit, cf_num, nx);
}

inline double cf09_eval(const vector<double> &x, int nx, const vector<double> &OsAll, const Matrix &MrAll, const vector<int> &SSAll) {
	// composition of hybrid hf05/hf06/hf07; each component has its own shift / matrix / shuffle block
	const int cf_num = 3; vector<double> fit(cf_num, 0.0);
	vector<double> delta{10, 30, 50}; vector<double> bias{0, 100, 200};
	for (int b = 0; b < cf_num; ++b) {
		vector<double> Osb = slice_vec(OsAll, static_cast<size_t>(b) * nx, nx);
		Matrix Mb = slice_matrix_block(MrAll, nx, static_cast<size_t>(b));
		vector<int> Sb = slice_vec_int(SSAll, static_cast<size_t>(b) * nx, nx);
		switch (b) {
			case 0: fit[b] = hf05_eval(x, Osb, Mb, Sb); break;
			case 1: fit[b] = hf06_eval(x, Osb, Mb, Sb); break;
			case 2: fit[b] = hf07_eval(x, Osb, Mb, Sb); break;
		}
	}
	// reuse original OsAll for weighting (needs cf_num*nx shift data sequential)
	return cf_weighted(x, OsAll, delta, bias, fit, cf_num, nx);
}

inline double cf10_eval(const vector<double> &x, int nx, const vector<double> &OsAll, const Matrix &MrAll, const vector<int> &SSAll) {
	const int cf_num = 3; vector<double> fit(cf_num, 0.0);
	vector<double> delta{10, 30, 50}; vector<double> bias{0, 100, 200};
	for (int b = 0; b < cf_num; ++b) {
		vector<double> Osb = slice_vec(OsAll, static_cast<size_t>(b) * nx, nx);
		Matrix Mb = slice_matrix_block(MrAll, nx, static_cast<size_t>(b));
		vector<int> Sb = slice_vec_int(SSAll, static_cast<size_t>(b) * nx, nx);
		switch (b) {
			case 0: fit[b] = hf05_eval(x, Osb, Mb, Sb); break;
			case 1: fit[b] = hf08_eval(x, Osb, Mb, Sb); break;
			case 2: fit[b] = hf09_eval(x, Osb, Mb, Sb); break;
		}
	}
	return cf_weighted(x, OsAll, delta, bias, fit, cf_num, nx);
}

inline void validate_dimension_2017(int nx, int func_num) {
	if (!(nx == 2 || nx == 10 || nx == 20 || nx == 30 || nx == 50 || nx == 100)) {
		throw invalid_argument("CEC2024 benchmarks are defined only for dimensions 2, 10, 20, 30, 50, or 100.");
	}
	if (nx == 2 && ((func_num >= 17 && func_num <= 22) || (func_num == 29 || func_num == 30))) {
		throw invalid_argument("Hybrid (17-22) and Composition (29-30) are NOT defined for D=2.");
	}
}

inline double evaluate_without_bias(int func_num, const vector<double> &x) {
	const int nx = static_cast<int>(x.size());
	constexpr int cf_num = 10;
	auto &cache = DataCache::instance();
	const auto &Os_all = cache.shift(func_num, nx, cf_num);
	const auto &Mr_all = cache.matrix(func_num, nx, cf_num);
	const auto &S_all  = cache.shuffle(func_num, nx, cf_num);

	switch (func_num) {
		case 1:  return bent_cigar(x, Os_all, Mr_all, 1, 1);
		case 2:  return sum_diff_pow(x, Os_all, Mr_all, 1, 1);
		case 3:  return zakharov(x, Os_all, Mr_all, 1, 1);
		case 4:  return rosenbrock(x, Os_all, Mr_all, 1, 1);
		case 5:  return rastrigin(x, Os_all, Mr_all, 1, 1);
		case 6:  return schaffer_F7(x, Os_all, Mr_all, 1, 1);
		case 7:  return bi_rastrigin(x, Os_all, Mr_all, 1, 1);
		case 8:  return step_rastrigin(x, Os_all, Mr_all, 1, 1);
		case 9:  return levy(x, Os_all, Mr_all, 1, 1);
		case 10: return schwefel(x, Os_all, Mr_all, 1, 1);
		case 11: return hf01_eval(x, Os_all, Mr_all, S_all);
		case 12: return hf02_eval(x, Os_all, Mr_all, S_all);
		case 13: return hf03_eval(x, Os_all, Mr_all, S_all);
		case 14: return hf04_eval(x, Os_all, Mr_all, S_all);
		case 15: return hf05_eval(x, Os_all, Mr_all, S_all);
		case 16: return hf06_eval(x, Os_all, Mr_all, S_all);
		case 17: return hf07_eval(x, Os_all, Mr_all, S_all);
		case 18: return hf08_eval(x, Os_all, Mr_all, S_all);
		case 19: return hf09_eval(x, Os_all, Mr_all, S_all);
		case 20: return hf10_eval(x, Os_all, Mr_all, S_all);
		case 21: return cf01_eval(x, nx, Os_all, Mr_all);
		case 22: return cf02_eval(x, nx, Os_all, Mr_all);
		case 23: return cf03_eval(x, nx, Os_all, Mr_all);
		case 24: return cf04_eval(x, nx, Os_all, Mr_all);
		case 25: return cf05_eval(x, nx, Os_all, Mr_all);
		case 26: return cf06_eval(x, nx, Os_all, Mr_all);
		case 27: return cf07_eval(x, nx, Os_all, Mr_all);
		case 28: return cf08_eval(x, nx, Os_all, Mr_all);
	case 29: return cf09_eval(x, nx, Os_all, Mr_all, S_all);
	case 30: return cf10_eval(x, nx, Os_all, Mr_all, S_all);
		default: throw out_of_range("Function index must be 1..30");
	}
}

inline double evaluate_with_bias(int func_num, const vector<double> &x) {
	static const array<double, 31> bias = {0.0,
		100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0,
		1100.0, 1200.0, 1300.0, 1400.0, 1500.0, 1600.0, 1700.0, 1800.0, 1900.0, 2000.0,
		2100.0, 2200.0, 2300.0, 2400.0, 2500.0, 2600.0, 2700.0, 2800.0, 2900.0, 3000.0
	};
	return evaluate_without_bias(func_num, x) + bias[func_num];
}

} // namespace

void cec17_test_func_fortrain(double *x, double *f, int nx, int mx, int func_num) {
	if (func_num < 1 || func_num > 30) {
		throw invalid_argument("CEC2024 function index must be in [1, 30].");
	}
	validate_dimension_2017(nx, func_num);
	for (int i = 0; i < mx; ++i) {
		vector<double> xi(x + static_cast<size_t>(i) * nx, x + static_cast<size_t>(i + 1) * nx);
		f[i] = evaluate_with_bias(func_num, xi);
	}
}

