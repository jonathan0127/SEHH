#ifndef RDE_H
#define RDE_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

#include "config/para_setting.h"
#include "utils/tqdm.h"
#include "utils/sample.h"
#include "utils/search_state.h"
using namespace std;
class RDE {
public:
	RDE(
		int dimension,
		const function<double(vector<double>&)>& objFunction,
		const vector<double>& lowerBounds,
		const vector<double>& upperBounds,
		int populationSize = META_POPULATION_SIZE,
		int maxEvaluations = META_MAX_EVALUATIONS,
		double psizeParam = 0.25,
		double G_flag = 1.0,
		double EB_flag = 1.0,
		int selectionMode = 1,
		uint32_t seed = random_device{}()
	);

	void setInitialSolution(const vector<double>& initialSolution);
	void setInitialSolution(const vector<double>& best,
							 const vector<vector<double>>& candidates,
							 const vector<double>& candidateFitness);
	pair<vector<double>, double> run(bool showProgress = false);
	pair<vector<double>, double> getBestSolution() const;
	vector<pair<int, double>> getMidBest() const;
    const vector<vector<double>>& getFinalPopulation() const { return Popul; }
    const vector<double>& getFinalFitnesses() const { return FitMass; }
    
    void loadState(const SearchState& state);
    void saveState(SearchState& state) const;
    void setGlobalProgress(int globalCurrentEval, int globalMaxEval);

private:
	int dimension;
	function<double(vector<double>&)> objFunction;
	vector<double> lowerBounds;
	vector<double> upperBounds;

	int NIndsMax;
	int NInds;
	int NIndsMin{4};
	int maxEvaluations;
	int evalCount{0};
	double psizeParam;
	double G_flag;
	double EB_flag;
	int selectionMode;

	const double A_flag{1.0};
	const double Cs_flag{1.0};
	const double I_flag{1.0};
	double H_flag;
	double Hp_flag;
	double Ha_flag;
	double EB_hybrid_rate{0.5};
	double jumping_rate{0.2};

	double ArchiveSizeParam{1.0};
	double ArchiveProb{0.25};
	int ArchiveSize{0};
	int CurrentArchiveSize{0};
	int MemorySize{5};
	int MemoryIter{0};
	int SuccessFilled{0};
	vector<double> MemoryCr;
	vector<double> MemoryF;
	vector<double> tempSuccessCr;
	vector<double> tempSuccessF;
	vector<double> FitDelta;

	vector<vector<double>> Popul;
	vector<vector<double>> PopulTemp;
	vector<double> FitMass;
	vector<double> FitMassTemp;
	vector<double> FitMassCopy;
	vector<int> Indexes;
	vector<vector<double>> Archive;
	vector<double> FitMassArch;
	vector<double> BestInd;
	vector<double> Donor;
	vector<double> Trial;
	vector<int> Rands;

	double* ord_best_arch{nullptr};
	double* ord_medium_arch{nullptr};
	double* ord_worst_arch{nullptr};
	double* ord_best_popul{nullptr};
	double* ord_medium_popul{nullptr};
	double* ord_worst_popul{nullptr};
	vector<double> EB_hybrid_flag;

	double bestfit{numeric_limits<double>::infinity()};
	double globalbest{numeric_limits<double>::infinity()};
	bool globalbestinit{false};
	vector<double> bestSolution;
	vector<pair<int, double>> midBest;

	int globalCurrentEval_{0};
	int globalMaxEval_{0};
	bool useGlobalProgress_{false};

	mt19937 gen;
	uniform_real_distribution<double> dist{0.0, 1.0};
	uniform_int_distribution<int> uni_int{0, 32768};

	int IntRandom(int target);
	double Random(double minimal, double maximal);
	double NormRand(double mu, double sigma);
	double CachyRand(double mu, double sigma);
	void qSort2int(vector<double>& mass, vector<int>& idx, int low, int high);
	void initialize();
	double evaluate(const vector<double>& x);
	void SaveSuccessCrF(double Cr, double F, double FitD);
	void UpdateMemoryCrF();
	double MeanWL(const vector<double>& vec, const vector<double>& weights);
	void CopyToArchive(const vector<double>& parent, double parentFit);
	void FindNSaveBest(bool init, int idx);
	void RemoveWorst(int NInds, int NewNInds);
	void EB_order(int prand, int Rand1, int Rand2);
	void UpdateEB_hybrid_param(const vector<double>& FitTemp);
	void FindLimits(vector<double>& ind, const vector<double>& parent);
	pair<vector<double>, double> MainCycle();
};

inline RDE::RDE(
	int dimension,
	const function<double(vector<double>&)>& objFunction,
	const vector<double>& lowerBounds,
	const vector<double>& upperBounds,
	int populationSize,
	int maxEvaluations,
	double psizeParam,
	double G_flag,
	double EB_flag,
	int selectionMode,
	uint32_t seed
) :
	dimension(dimension),
	objFunction(objFunction),
	lowerBounds(lowerBounds),
	upperBounds(upperBounds),
	NIndsMax(max(4, populationSize)),
	NInds(NIndsMax),
	maxEvaluations(maxEvaluations),
	psizeParam(psizeParam),
	G_flag(G_flag),
	EB_flag(EB_flag),
	selectionMode(selectionMode),
	H_flag(selectionMode ? 1.0 : 0.0),
	Hp_flag(selectionMode ? 1.0 : 0.0),
	Ha_flag(selectionMode ? 1.0 : 0.0),
	gen(seed) {
	MemoryCr.assign(MemorySize, 0.8);
	MemoryF.assign(MemorySize, 0.3);
	initialize();
}

inline int RDE::IntRandom(int target) {
	if (target <= 0) return 0;
	return uni_int(gen) % target;
}

inline double RDE::Random(double minimal, double maximal) {
	return dist(gen) * (maximal - minimal) + minimal;
}

inline double RDE::NormRand(double mu, double sigma) {
	return normal_distribution<double>(mu, sigma)(gen);
}

inline double RDE::CachyRand(double mu, double sigma) {
	return cauchy_distribution<double>(mu, sigma)(gen);
}

inline void RDE::qSort2int(vector<double>& mass, vector<int>& idx, int low, int high) {
	int i = low;
	int j = high;
	double x = mass[(low + high) >> 1];
	do {
		while (mass[i] < x) ++i;
		while (mass[j] > x) --j;
		if (i <= j) {
			swap(mass[i], mass[j]);
			swap(idx[i], idx[j]);
			i++; j--;
		}
	} while (i <= j);
	if (low < j) qSort2int(mass, idx, low, j);
	if (i < high) qSort2int(mass, idx, i, high);
}

inline double RDE::evaluate(const vector<double>& x) {
	vector<double> tmp = x;
	double f = objFunction(tmp);
	++evalCount;
	return f;
}

inline void RDE::initialize() {
	evalCount = 0;
	NInds = NIndsMax;
	ArchiveSize = NIndsMax * ArchiveSizeParam;
	CurrentArchiveSize = 0;

	Popul.assign(NIndsMax, vector<double>(dimension, 0.0));
	PopulTemp.assign(NIndsMax, vector<double>(dimension, 0.0));
	FitMass.assign(NIndsMax, numeric_limits<double>::infinity());
	FitMassTemp.assign(NIndsMax, numeric_limits<double>::infinity());
	FitMassCopy.assign(NIndsMax, numeric_limits<double>::infinity());
	Indexes.resize(NIndsMax);
	Archive.assign(NIndsMax * static_cast<int>(ArchiveSizeParam), vector<double>(dimension, 0.0));
	FitMassArch.assign(NIndsMax * static_cast<int>(ArchiveSizeParam), numeric_limits<double>::infinity());
	BestInd.assign(dimension, 0.0);
	Donor.assign(dimension, 0.0);
	Trial.assign(dimension, 0.0);
	Rands.assign(NIndsMax, 0);
	EB_hybrid_flag.assign(NIndsMax, 0.0);
	MemoryCr.assign(MemorySize, 0.8);
	MemoryF.assign(MemorySize, 0.3);
	tempSuccessCr.assign(NIndsMax, 0.0);
	tempSuccessF.assign(NIndsMax, 0.0);
	FitDelta.assign(NIndsMax, 0.0);
	midBest.clear();
	bestfit = numeric_limits<double>::infinity();
	globalbest = numeric_limits<double>::infinity();
	globalbestinit = false;
	bestSolution.assign(dimension, 0.0);

	for (int i = 0; i < NIndsMax; ++i) {
		for (int j = 0; j < dimension; ++j) {
			Popul[i][j] = Random(lowerBounds[j], upperBounds[j]);
		}
		FitMass[i] = evaluate(Popul[i]);
		if (FitMass[i] < bestfit || !globalbestinit) {
			bestfit = FitMass[i];
			bestSolution = Popul[i];
			globalbest = bestfit;
			globalbestinit = true;
		}
	}
}

inline void RDE::setInitialSolution(const vector<double>& initialSolution) {
	if (initialSolution.size() != static_cast<size_t>(dimension)) return;
	if (Popul.empty()) return;
	Popul[0] = initialSolution;
	FitMass[0] = evaluate(Popul[0]);
	if (FitMass[0] < bestfit) {
		bestfit = FitMass[0];
		bestSolution = Popul[0];
		globalbest = bestfit;
	}
}

inline void RDE::setInitialSolution(const vector<double>& best,
									const vector<vector<double>>& candidates,
									const vector<double>& candidateFitness){
	if(Popul.empty()) return;
	auto clampv = [&](vector<double> v){
		for(int d=0; d<dimension; ++d){ v[d] = max(lowerBounds[d], min(upperBounds[d], v[d])); }
		return v;
	};
	Popul[0] = clampv(best);
	FitMass[0] = evaluate(Popul[0]);
	if(FitMass[0] < bestfit){ bestfit = FitMass[0]; bestSolution = Popul[0]; globalbest = bestfit; globalbestinit = true; }
	int need = max(0, NIndsMax - 1);
	int n = (int)candidates.size();
	vector<double> weights;
	if(n>0 && candidateFitness.size()==candidates.size()){
		double fmin = numeric_limits<double>::infinity();
		for(double f : candidateFitness) if(isfinite(f)) fmin = min(fmin, f);
		if(!isfinite(fmin)) fmin = 0.0;
		weights.reserve(n);
		for(double f : candidateFitness){
			double w = 1.0/(1e-12 + max(0.0, f - fmin)); if(!isfinite(w)||w<0.0) w=0.0; weights.push_back(w);
		}
	}
	mt19937 rng(gen());
	vector<int> picks;
	if(!weights.empty()){
		picks = roulette_select_indices(weights, need, false, rng);
	}else if(n>0){
		uniform_int_distribution<int> uni(0, n-1); picks.resize(need); for(int i=0;i<need;++i) picks[i]=uni(rng);
	}
	for(int i=0;i<need;++i){
		vector<double> cand = (n>0 && !picks.empty()) ? candidates[picks[i % (int)picks.size()]] : vector<double>(dimension,0.0);
		if((int)cand.size()!=dimension){
			cand.assign(dimension,0.0); for(int d=0; d<dimension; ++d){ cand[d] = Random(lowerBounds[d], upperBounds[d]); }
		}
		Popul[i+1] = clampv(cand);
		FitMass[i+1] = evaluate(Popul[i+1]);
		if(FitMass[i+1] < bestfit || !globalbestinit){ bestfit = FitMass[i+1]; bestSolution = Popul[i+1]; globalbest = bestfit; globalbestinit = true; }
	}
}

inline void RDE::SaveSuccessCrF(double Cr, double F, double FitD) {
	if (SuccessFilled < NIndsMax) {
		tempSuccessCr[SuccessFilled] = Cr;
		tempSuccessF[SuccessFilled] = F;
		FitDelta[SuccessFilled] = FitD;
		SuccessFilled++;
	}
}

inline double RDE::MeanWL(const vector<double>& vec, const vector<double>& weights) {
	double SumWeight = 0.0;
	double SumSquare = 0.0;
	double Sum = 0.0;
	for (double w : weights) SumWeight += w;
	if (SumWeight <= 0.0) return 0.5;
	vector<double> normW(weights.size(), 0.0);
	for (size_t i = 0; i < weights.size(); ++i) normW[i] = weights[i] / SumWeight;
	for (size_t i = 0; i < normW.size(); ++i) {
		SumSquare += normW[i] * vec[i] * vec[i];
		Sum += normW[i] * vec[i];
	}
	if (fabs(Sum) > 1e-9) return SumSquare / Sum;
	return 0.5;
}

inline void RDE::UpdateMemoryCrF() {
	if (SuccessFilled == 0) return;
	double tempmax = tempSuccessCr[0];
	for (int i = 0; i < SuccessFilled; ++i) tempmax = max(tempmax, tempSuccessCr[i]);
	if (MemoryCr[MemoryIter] == -1 || tempmax == 0) {
		MemoryCr[MemoryIter] = -1;
	} else {
		vector<double> weights(FitDelta.begin(), FitDelta.begin() + SuccessFilled);
		MemoryCr[MemoryIter] = 0.5 * (MeanWL(vector<double>(tempSuccessCr.begin(), tempSuccessCr.begin() + SuccessFilled), weights) + MemoryCr[MemoryIter]);
		MemoryF[MemoryIter] = MeanWL(vector<double>(tempSuccessF.begin(), tempSuccessF.begin() + SuccessFilled), weights);
	}
	MemoryIter = (MemoryIter + 1) % MemorySize;
}

inline void RDE::CopyToArchive(const vector<double>& parent, double parentFit) {
	if (CurrentArchiveSize < ArchiveSize) {
		Archive[CurrentArchiveSize] = parent;
		FitMassArch[CurrentArchiveSize] = parentFit;
		CurrentArchiveSize++;
	} else if (ArchiveSize > 0) {
		int rnd = IntRandom(ArchiveSize);
		Archive[rnd] = parent;
		FitMassArch[rnd] = parentFit;
	}
}

inline void RDE::FindNSaveBest(bool init, int idx) {
	if (FitMass[idx] <= bestfit || init) {
		bestfit = FitMass[idx];
		bestSolution = Popul[idx];
	}
	if (!globalbestinit || bestfit < globalbest) {
		globalbest = bestfit;
		globalbestinit = true;
	}
}

inline void RDE::RemoveWorst(int N, int NewN) {
	int pointsToRemove = N - NewN;
	for (int L = 0; L < pointsToRemove; ++L) {
		double worstFit = FitMass[0];
		int worstIdx = 0;
		for (int i = 1; i < N; ++i) {
			if (FitMass[i] > worstFit) {
				worstFit = FitMass[i];
				worstIdx = i;
			}
		}
		for (int i = worstIdx; i < N - 1; ++i) {
			Popul[i] = Popul[i + 1];
			FitMass[i] = FitMass[i + 1];
		}
	}
}

inline void RDE::EB_order(int prand, int Rand1, int Rand2) {
	double* pos1 = Popul[prand].data();
	double pos1Fit = FitMass[prand];
	double* pos3 = Popul[Rand1].data();
	double pos3Fit = FitMass[Rand1];
	double* pos4_arch = Archive[Rand2].data();
	double pos4Fit_arch = FitMassArch[Rand2];

	if (pos1Fit <= pos3Fit && pos1Fit <= pos4Fit_arch){
		ord_best_arch = pos1;
		if (pos3Fit <= pos4Fit_arch){ ord_medium_arch = pos3; ord_worst_arch = pos4_arch; }
		else { ord_medium_arch = pos4_arch; ord_worst_arch = pos3; }
	}
	else if (pos3Fit <= pos1Fit && pos3Fit <= pos4Fit_arch){
		ord_best_arch = pos3;
		if (pos1Fit <= pos4Fit_arch){ ord_medium_arch = pos1; ord_worst_arch = pos4_arch; }
		else { ord_medium_arch = pos4_arch; ord_worst_arch = pos1; }
	}
	else {
		ord_best_arch = pos4_arch;
		if (pos1Fit <= pos3Fit){ ord_medium_arch = pos1; ord_worst_arch = pos3; }
		else { ord_medium_arch = pos3; ord_worst_arch = pos1; }
	}

	double* pos4_popul = Popul[Rand2].data();
	double pos4Fit_popul = FitMass[Rand2];
	if (pos1Fit <= pos3Fit && pos1Fit <= pos4Fit_popul){
		ord_best_popul = pos1;
		if (pos3Fit <= pos4Fit_popul){ ord_medium_popul = pos3; ord_worst_popul = pos4_popul; }
		else { ord_medium_popul = pos4_popul; ord_worst_popul = pos3; }
	}
	else if (pos3Fit <= pos1Fit && pos3Fit <= pos4Fit_popul){
		ord_best_popul = pos3;
		if (pos1Fit <= pos4Fit_popul){ ord_medium_popul = pos1; ord_worst_popul = pos4_popul; }
		else { ord_medium_popul = pos4_popul; ord_worst_popul = pos1; }
	}
	else {
		ord_best_popul = pos4_popul;
		if (pos1Fit <= pos3Fit){ ord_medium_popul = pos1; ord_worst_popul = pos3; }
		else { ord_medium_popul = pos3; ord_worst_popul = pos1; }
	}
}

inline void RDE::UpdateEB_hybrid_param(const vector<double>& FitTemp) {
	double SumEB_DeltaFit = 0.0;
	double SumOrigin_DeltaFit = 0.0;
	for (int i = 0; i < NInds; ++i) {
		if (EB_hybrid_flag[i] == 1) {
			if (FitTemp[i] <= FitMass[i]) SumEB_DeltaFit += FitMass[i] - FitTemp[i];
		} else {
			if (FitTemp[i] <= FitMass[i]) SumOrigin_DeltaFit += FitMass[i] - FitTemp[i];
		}
	}
	if (SumEB_DeltaFit != 0 && SumOrigin_DeltaFit != 0){
		EB_hybrid_rate = SumEB_DeltaFit / (SumEB_DeltaFit + SumOrigin_DeltaFit);
		EB_hybrid_rate = clamp(EB_hybrid_rate, 0.0, 1.0);
	} else {
		EB_hybrid_rate = 0.5;
	}
}

inline void RDE::FindLimits(vector<double>& Ind, const vector<double>& Parent) {
	for (int j = 0; j < dimension; ++j) {
		if (Ind[j] < lowerBounds[j]) Ind[j] = (lowerBounds[j] + Parent[j]) / 2.0;
		if (Ind[j] > upperBounds[j]) Ind[j] = (upperBounds[j] + Parent[j]) / 2.0;
	}
}

inline pair<vector<double>, double> RDE::MainCycle() {
	while (evalCount < maxEvaluations) {
		double minfit = FitMass[0];
		double maxfit = FitMass[0];
		for (int i = 0; i < NInds; ++i) {
			FitMassCopy[i] = FitMass[i];
			Indexes[i] = i;
			maxfit = max(maxfit, FitMass[i]);
			minfit = min(minfit, FitMass[i]);
		}
		if (minfit != maxfit) qSort2int(FitMassCopy, Indexes, 0, NInds - 1);

		vector<double> FitTemp(NInds, 0.0);
		for (int i = 0; i < NInds; ++i) FitTemp[i] = 3.0 * (NInds - i);
		discrete_distribution<int> ComponentSelector(FitTemp.begin(), FitTemp.end());

		double progress;
		if (useGlobalProgress_ && globalMaxEval_ > 0) {
			progress = static_cast<double>(globalCurrentEval_ + evalCount) / static_cast<double>(globalMaxEval_);
		} else {
			progress = static_cast<double>(evalCount) / static_cast<double>(maxEvaluations);
		}
		progress = clamp(progress, 0.0, 1.0);
		
		double psize = psizeParam * (1.0 - 0.5 * progress);
		int psizeval = max(2, static_cast<int>(NInds * psize));
		discrete_distribution<int> ComponentSelector1(FitTemp.begin(), FitTemp.begin() + psizeval);

		vector<int> IndexesArch(CurrentArchiveSize);
		vector<double> FitMassArchCopy(CurrentArchiveSize);
		vector<double> FitMassArch_RSP;
		if (Ha_flag == 1 && CurrentArchiveSize > 0) {
			double minfitArch = FitMassArch[0];
			double maxfitArch = FitMassArch[0];
			for (int i = 0; i < CurrentArchiveSize; ++i) {
				FitMassArchCopy[i] = FitMassArch[i];
				IndexesArch[i] = i;
				maxfitArch = max(maxfitArch, FitMassArch[i]);
				minfitArch = min(minfitArch, FitMassArch[i]);
			}
			if (minfitArch != maxfitArch) qSort2int(FitMassArchCopy, IndexesArch, 0, CurrentArchiveSize - 1);
			FitMassArch_RSP.resize(CurrentArchiveSize);
			for (int i = 0; i < CurrentArchiveSize; ++i) FitMassArch_RSP[i] = 3.0 * (CurrentArchiveSize - i);
		} else {
			FitMassArch_RSP = {1.0};
		}
		discrete_distribution<int> ComponentSelector2(FitMassArch_RSP.begin(), FitMassArch_RSP.end());

		vector<double> FitTempEval(NInds, numeric_limits<double>::infinity());

		for (int TheChosenOne = 0; TheChosenOne < NInds; ++TheChosenOne) {
			int MemoryCurrentIndex = IntRandom(MemorySize + 1);
			int prand;
			do {
				if (Hp_flag == 1) prand = Indexes[ComponentSelector1(gen)];
				else prand = Indexes[IntRandom(psizeval)];
			} while (prand == TheChosenOne && progress < 0.5);

			int Rand1;
			int Rand2;

			if (H_flag == 1) {
				do Rand1 = Indexes[ComponentSelector(gen)]; while (Rand1 == prand);
				do Rand2 = Indexes[ComponentSelector(gen)]; while (Rand2 == prand || Rand2 == Rand1);
			} else {
				do Rand1 = IntRandom(NInds); while (Rand1 == prand);
				do Rand2 = IntRandom(NInds); while (Rand2 == prand || Rand2 == Rand1);
			}

			double F;
			if (A_flag == 1) {
				do {
					if (MemoryCurrentIndex < MemorySize) F = CachyRand(MemoryF[MemoryCurrentIndex], 0.1);
					else F = CachyRand(0.9, 0.1);
				} while (F < 0.0);
			} else {
				do { F = CachyRand(MemoryF[MemoryCurrentIndex], 0.1); } while (F < 0.0);
			}
			if (F > 1.0) F = 1.0;
			double F2 = 1.0 * F;
			if (G_flag == 1 && progress < 0.6 && F > 0.7) F = 0.7;

			double Rand_EB = Random(0, 1);
			if (EB_flag == 1) {
				if (Rand_EB < EB_hybrid_rate) {
					EB_hybrid_flag[TheChosenOne] = 1;
					bool useArch = CurrentArchiveSize > 0 && Random(0,1) < (double)CurrentArchiveSize / (double)(CurrentArchiveSize + NInds);
					if (useArch) {
						if (Ha_flag == 1) Rand2 = IndexesArch[ComponentSelector2(gen)];
						else Rand2 = IntRandom(CurrentArchiveSize);
						EB_order(prand, Rand1, Rand2);
						for (int j = 0; j < dimension; ++j) {
							Donor[j] = Popul[TheChosenOne][j] + F2 * (ord_best_arch[j] - Popul[TheChosenOne][j]) + F * (ord_medium_arch[j] - ord_worst_arch[j]);
						}
					} else {
						EB_order(prand, Rand1, Rand2);
						for (int j = 0; j < dimension; ++j) {
							Donor[j] = Popul[TheChosenOne][j] + F2 * (ord_best_popul[j] - Popul[TheChosenOne][j]) + F * (ord_medium_popul[j] - ord_worst_popul[j]);
						}
					}
				} else {
					EB_hybrid_flag[TheChosenOne] = 0;
					bool useArch = CurrentArchiveSize > 0 && Random(0,1) < (double)CurrentArchiveSize / (double)(CurrentArchiveSize + NInds);
					if (useArch) {
						if (Ha_flag == 1) Rand2 = IndexesArch[ComponentSelector2(gen)];
						else Rand2 = IntRandom(CurrentArchiveSize);
						for (int j = 0; j < dimension; ++j) {
							Donor[j] = Popul[TheChosenOne][j] + F2 * (Popul[prand][j] - Popul[TheChosenOne][j]) + F * (Popul[Rand1][j] - Archive[Rand2][j]);
						}
					} else {
						for (int j = 0; j < dimension; ++j) {
							Donor[j] = Popul[TheChosenOne][j] + F2 * (Popul[prand][j] - Popul[TheChosenOne][j]) + F * (Popul[Rand1][j] - Popul[Rand2][j]);
						}
					}
				}
			} else {
				bool useArch = CurrentArchiveSize > 0 && Random(0,1) < (double)CurrentArchiveSize / (double)(CurrentArchiveSize + NInds);
				if (useArch) {
					Rand2 = IntRandom(CurrentArchiveSize);
					for (int j = 0; j < dimension; ++j) {
						Donor[j] = Popul[TheChosenOne][j] + F2 * (Popul[prand][j] - Popul[TheChosenOne][j]) + F * (Popul[Rand1][j] - Archive[Rand2][j]);
					}
				} else {
					for (int j = 0; j < dimension; ++j) {
						Donor[j] = Popul[TheChosenOne][j] + F2 * (Popul[prand][j] - Popul[TheChosenOne][j]) + F * (Popul[Rand1][j] - Popul[Rand2][j]);
					}
				}
			}

			FindLimits(Donor, Popul[TheChosenOne]);

			int WillCrossover = IntRandom(dimension);
			double Cr;
			if (A_flag == 1) {
				if (MemoryCurrentIndex < MemorySize) {
					Cr = MemoryCr[MemoryCurrentIndex] < 0 ? 0.0 : NormRand(MemoryCr[MemoryCurrentIndex], 0.1);
				} else {
					Cr = NormRand(0.9, 0.1);
				}
			} else {
				Cr = MemoryCr[MemoryCurrentIndex] < 0 ? 0.0 : NormRand(MemoryCr[MemoryCurrentIndex], 0.1);
			}
			Cr = clamp(Cr, 0.0, 1.0);
			if (G_flag == 1) {
				if (progress < 0.25) Cr = max(Cr, 0.7);
				if (progress < 0.5) Cr = max(Cr, 0.6);
			}

			if (I_flag == 1) {
				bool perturbation = Random(0,1) < jumping_rate;
				for (int j = 0; j < dimension; ++j) {
					if (Random(0,1) < Cr || WillCrossover == j) PopulTemp[TheChosenOne][j] = Donor[j];
					else PopulTemp[TheChosenOne][j] = perturbation ? CachyRand(Popul[TheChosenOne][j], 0.1) : Popul[TheChosenOne][j];
				}
			} else {
				for (int j = 0; j < dimension; ++j) {
					if (Random(0,1) < Cr || WillCrossover == j) PopulTemp[TheChosenOne][j] = Donor[j];
				}
			}

			FitTempEval[TheChosenOne] = evaluate(PopulTemp[TheChosenOne]);
			if (FitTempEval[TheChosenOne] <= globalbest) globalbest = FitTempEval[TheChosenOne];
			if (FitTempEval[TheChosenOne] <= FitMass[TheChosenOne]) {
				SaveSuccessCrF(Cr, F, fabs(FitMass[TheChosenOne] - FitTempEval[TheChosenOne]));
			}

			if (evalCount >= maxEvaluations) break;
		}

		if (EB_flag == 1) UpdateEB_hybrid_param(FitTempEval);

		for (int i = 0; i < NInds; ++i) {
			if (FitTempEval[i] <= FitMass[i]) {
				CopyToArchive(Popul[i], FitMass[i]);
				Popul[i] = PopulTemp[i];
				FitMass[i] = FitTempEval[i];
				FindNSaveBest(false, i);
			}
		}

		int newNInds, newArchSize;
		if (useGlobalProgress_ && globalMaxEval_ > 0) {
			int globalEval = globalCurrentEval_ + evalCount;
			newNInds = static_cast<int>(double(NIndsMin - NIndsMax) / globalMaxEval_ * globalEval + NIndsMax);
			newArchSize = static_cast<int>((double)(globalMaxEval_ - globalEval) / (double)globalMaxEval_ * (ArchiveSizeParam * (NIndsMax - NIndsMin)));
		} else {
			newNInds = static_cast<int>(double(NIndsMin - NIndsMax) / maxEvaluations * evalCount + NIndsMax);
			newArchSize = static_cast<int>((double)(maxEvaluations - evalCount) / (double)maxEvaluations * (ArchiveSizeParam * (NIndsMax - NIndsMin)));
		}
		newNInds = clamp(newNInds, NIndsMin, NIndsMax);
		newArchSize = clamp(newArchSize, NIndsMin, NIndsMax);
		newArchSize = clamp(newArchSize, NIndsMin, NIndsMax);
		ArchiveSize = newArchSize;
		if (CurrentArchiveSize >= ArchiveSize) CurrentArchiveSize = ArchiveSize;
		RemoveWorst(NInds, newNInds);
		NInds = newNInds;
		UpdateMemoryCrF();
		SuccessFilled = 0;

		if (midBest.empty() || evalCount >= midBest.back().first + MID_BEST_CHECK_POINT) {
			midBest.push_back({evalCount, bestfit});
		}
		if (evalCount >= maxEvaluations) break;
	}
	midBest.push_back({maxEvaluations, bestfit});
	return {bestSolution, bestfit};
}

inline pair<vector<double>, double> RDE::run(bool showProgress) {
	if (showProgress) {
		tqdm bar;
		auto res = MainCycle();
		bar.finish();
		return res;
	}
	return MainCycle();
}

inline pair<vector<double>, double> RDE::getBestSolution() const {
	return {bestSolution, bestfit};
}

inline vector<pair<int, double>> RDE::getMidBest() const {
	return midBest;
}

inline void RDE::setGlobalProgress(int gCurrentEval, int gMaxEval) {
	globalCurrentEval_ = gCurrentEval;
	globalMaxEval_ = gMaxEval;
	useGlobalProgress_ = (gMaxEval > 0);
}

inline void RDE::loadState(const SearchState& state) {
	if (state.isEmpty()) return;
	
	if (state.globalMaxEval > 0) {
		setGlobalProgress(state.globalCurrentEval, state.globalMaxEval);
	}
	
	if (!state.bestSolution.empty() && state.bestSolution.size() == static_cast<size_t>(dimension)) {
		bestSolution = state.bestSolution;
		bestfit = state.bestFitness;
		globalbest = state.bestFitness;
		globalbestinit = true;
	}
	
	if (state.hasMemory()) {
		int stateMemSize = static_cast<int>(state.memoryCR.size());
		int copySize = min(stateMemSize, MemorySize);
		for (int i = 0; i < copySize; ++i) {
			MemoryCr[i] = state.memoryCR[i];
			MemoryF[i] = state.memoryF[i];
		}
		MemoryIter = state.memoryWriteIndex % MemorySize;
	}
	
	EB_hybrid_rate = state.EB_hybrid_rate;
	
	if (state.hasArchive()) {
		int stateArchSize = static_cast<int>(state.archive.size());
		int copySize = min(stateArchSize, static_cast<int>(Archive.size()));
		copySize = min(copySize, state.currentArchiveSize);
		for (int i = 0; i < copySize; ++i) {
			if (state.archive[i].size() == static_cast<size_t>(dimension)) {
				Archive[i] = state.archive[i];
				FitMassArch[i] = (i < static_cast<int>(state.archiveFitness.size())) 
					? state.archiveFitness[i] 
					: numeric_limits<double>::infinity();
			}
		}
		CurrentArchiveSize = copySize;
	}
	
	int statePopSize = static_cast<int>(state.population.size());
	if (statePopSize > 0) {
		int targetSize = min(statePopSize, NIndsMax);
		
		for (int i = 0; i < targetSize; ++i) {
			if (state.population[i].size() == static_cast<size_t>(dimension)) {
				Popul[i] = state.population[i];
				FitMass[i] = (i < static_cast<int>(state.fitness.size())) 
					? state.fitness[i] 
					: numeric_limits<double>::infinity();
			}
		}
		
		if (targetSize < NIndsMax) {
			for (int i = targetSize; i < NIndsMax; ++i) {
				for (int d = 0; d < dimension; ++d) {
					double perturbation = (Random(0, 1) - 0.5) * 0.2 * (upperBounds[d] - lowerBounds[d]);
					Popul[i][d] = bestSolution[d] + perturbation;
					Popul[i][d] = max(lowerBounds[d], min(upperBounds[d], Popul[i][d]));
				}
				FitMass[i] = evaluate(Popul[i]);
			}
		}
		
		NInds = min(targetSize, NIndsMax);
	}
}

inline void RDE::saveState(SearchState& state) const {
	state.population.clear();
	state.fitness.clear();
	state.population.reserve(NInds);
	state.fitness.reserve(NInds);
	for (int i = 0; i < NInds; ++i) {
		state.population.push_back(Popul[i]);
		state.fitness.push_back(FitMass[i]);
	}
	
	state.bestSolution = bestSolution;
	state.bestFitness = bestfit;
	
	state.memoryCR = MemoryCr;
	state.memoryF = MemoryF;
	state.memoryWriteIndex = MemoryIter;
	
	state.archive.clear();
	state.archiveFitness.clear();
	state.archive.reserve(CurrentArchiveSize);
	state.archiveFitness.reserve(CurrentArchiveSize);
	for (int i = 0; i < CurrentArchiveSize; ++i) {
		state.archive.push_back(Archive[i]);
		state.archiveFitness.push_back(FitMassArch[i]);
	}
	state.currentArchiveSize = CurrentArchiveSize;
	
	state.EB_hybrid_rate = EB_hybrid_rate;
	
	if (useGlobalProgress_) {
		state.globalCurrentEval = globalCurrentEval_ + evalCount;
		state.globalMaxEval = globalMaxEval_;
	} else {
		state.globalCurrentEval += evalCount;
	}
}

#endif
