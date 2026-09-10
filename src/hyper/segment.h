#ifndef SEGMENT_H
#define SEGMENT_H

#include <variant>
#include <string>
#include <random>
#include <iostream>
#include <algorithm>
#include <cmath>
#include "config/para_setting.h"

using namespace std;

struct SEHH_PSOParams{
    double w; // Inertia weight
    double c1; // Cognitive parameter
    double c2; // Social parameter
    int time_weight; // time weight for evaluation allocation
};
struct SEHH_GAParams{
    double mutationRate;
    double crossoverRate;
    double elitismRate;
    int selectionMethod;
    int crossoverMethod;
    int time_weight; // time weight for evaluation allocation
};
struct SEHH_DEParams{
    double F; // Scaling factor
    double CR; // Crossover rate
    int strategy; // DE strategy
    int time_weight; // time weight for evaluation allocation
};
struct SEHH_LSHADEParams{
    int memorySize;
    double CR;
    double F;
    int time_weight; // time weight for evaluation allocation
};
struct SEHH_LSRTDEParams{
    int memorySize;      // CR 記憶體大小 (2~20)
    double sigmaCR;      // CR 標準差 (0.02~0.15)
    double sigmaF;       // F 標準差 (0.01~0.08)
    int time_weight; // time weight for evaluation allocation
};
struct SEHH_RDEParams{
    double psizeParam;
    double G_flag;
    double EB_flag;
    int selectionMode;
    int time_weight; // time weight for evaluation allocation
};

using SegmentParams = variant<SEHH_PSOParams, SEHH_GAParams, SEHH_DEParams, SEHH_LSHADEParams, SEHH_LSRTDEParams, SEHH_RDEParams>;

class Segment{
public:
    Segment(string name, SegmentParams p)
        : template_name(move(name)), params(move(p)){}

    Segment(string name)
    : template_name(move(name)){
        RandominitP();
    }

    Segment(int meta){
        switch(meta){
            case 0: template_name = "PSO"; break;
            case 1: template_name = "GA"; break;
            case 2: template_name = "DE"; break;
            case 3: template_name = "LSHADE"; break;
            case 4: template_name = "LSRTDE"; break;
            case 5: template_name = "RDE"; break;
            default:
                throw runtime_error("Segment(int): meta out of range");
        }
        RandominitP();
    }

    Segment(){
        RandomInitName();
        RandominitP();
    }

    const SegmentParams& getSegmentParams() const{
        if(params.valueless_by_exception()){
            cerr << "[BUG] SegmentParams is valueless_by_exception for template: " << template_name << endl;
            throw runtime_error("SegmentParams is valueless_by_exception for template: " + template_name);
        }
        return params;
    }

    string getTemplateName() const{
        return template_name;
    }

    int getTimeWeight() const{
        if(holds_alternative<SEHH_PSOParams>(params)){
            return get<SEHH_PSOParams>(params).time_weight;
        }else if(holds_alternative<SEHH_GAParams>(params)){
            return get<SEHH_GAParams>(params).time_weight;
        }else if(holds_alternative<SEHH_DEParams>(params)){
            return get<SEHH_DEParams>(params).time_weight;
        }else if(holds_alternative<SEHH_LSHADEParams>(params)){
            return get<SEHH_LSHADEParams>(params).time_weight;
        }else if(holds_alternative<SEHH_LSRTDEParams>(params)){
            return get<SEHH_LSRTDEParams>(params).time_weight;
        }else if(holds_alternative<SEHH_RDEParams>(params)){
            return get<SEHH_RDEParams>(params).time_weight;
        }
        return 0;
    }

    bool operator < (const Segment& other) const{
        if(template_name != other.template_name){
            return template_name < other.template_name;
        }
        if(holds_alternative<SEHH_PSOParams>(params) && holds_alternative<SEHH_PSOParams>(other.params)){
            return get<SEHH_PSOParams>(params).w < get<SEHH_PSOParams>(other.params).w;
        }else if(holds_alternative<SEHH_GAParams>(params) && holds_alternative<SEHH_GAParams>(other.params)){
            return get<SEHH_GAParams>(params).mutationRate < get<SEHH_GAParams>(other.params).mutationRate;
        }else if(holds_alternative<SEHH_DEParams>(params) && holds_alternative<SEHH_DEParams>(other.params)){
            return get<SEHH_DEParams>(params).F < get<SEHH_DEParams>(other.params).F;
        }else if(holds_alternative<SEHH_LSHADEParams>(params) && holds_alternative<SEHH_LSHADEParams>(other.params)){
            return get<SEHH_LSHADEParams>(params).memorySize < get<SEHH_LSHADEParams>(other.params).memorySize;
        }else if(holds_alternative<SEHH_LSRTDEParams>(params) && holds_alternative<SEHH_LSRTDEParams>(other.params)){
            return get<SEHH_LSRTDEParams>(params).memorySize < get<SEHH_LSRTDEParams>(other.params).memorySize;
        }else if(holds_alternative<SEHH_RDEParams>(params) && holds_alternative<SEHH_RDEParams>(other.params)){
            return get<SEHH_RDEParams>(params).psizeParam < get<SEHH_RDEParams>(other.params).psizeParam;
        }
        return template_name < other.template_name;
    }

    bool operator == (const Segment& other) const{
        if(template_name != other.template_name){
            return false;
        }
        if(holds_alternative<SEHH_PSOParams>(params) && holds_alternative<SEHH_PSOParams>(other.params)){
            return get<SEHH_PSOParams>(params).w == get<SEHH_PSOParams>(other.params).w &&
                   get<SEHH_PSOParams>(params).c1 == get<SEHH_PSOParams>(other.params).c1 &&
                   get<SEHH_PSOParams>(params).c2 == get<SEHH_PSOParams>(other.params).c2 &&
                   get<SEHH_PSOParams>(params).time_weight == get<SEHH_PSOParams>(other.params).time_weight;
        }else if(holds_alternative<SEHH_GAParams>(params) && holds_alternative<SEHH_GAParams>(other.params)){
            return get<SEHH_GAParams>(params).mutationRate == get<SEHH_GAParams>(other.params).mutationRate &&
                   get<SEHH_GAParams>(params).crossoverRate == get<SEHH_GAParams>(other.params).crossoverRate &&
                   get<SEHH_GAParams>(params).elitismRate == get<SEHH_GAParams>(other.params).elitismRate &&
                   get<SEHH_GAParams>(params).selectionMethod == get<SEHH_GAParams>(other.params).selectionMethod &&
                   get<SEHH_GAParams>(params).crossoverMethod == get<SEHH_GAParams>(other.params).crossoverMethod &&
                   get<SEHH_GAParams>(params).time_weight == get<SEHH_GAParams>(other.params).time_weight;
        }else if(holds_alternative<SEHH_DEParams>(params) && holds_alternative<SEHH_DEParams>(other.params)){
            return get<SEHH_DEParams>(params).F == get<SEHH_DEParams>(other.params).F &&
                   get<SEHH_DEParams>(params).CR == get<SEHH_DEParams>(other.params).CR &&
                   get<SEHH_DEParams>(params).strategy == get<SEHH_DEParams>(other.params).strategy &&
                   get<SEHH_DEParams>(params).time_weight == get<SEHH_DEParams>(other.params).time_weight;
        }else if(holds_alternative<SEHH_LSHADEParams>(params) && holds_alternative<SEHH_LSHADEParams>(other.params)){
            return get<SEHH_LSHADEParams>(params).memorySize == get<SEHH_LSHADEParams>(other.params).memorySize &&
                   get<SEHH_LSHADEParams>(params).CR == get<SEHH_LSHADEParams>(other.params).CR &&
                   get<SEHH_LSHADEParams>(params).F == get<SEHH_LSHADEParams>(other.params).F &&
                   get<SEHH_LSHADEParams>(params).time_weight == get<SEHH_LSHADEParams>(other.params).time_weight;
        }else if(holds_alternative<SEHH_LSRTDEParams>(params) && holds_alternative<SEHH_LSRTDEParams>(other.params)){
            return get<SEHH_LSRTDEParams>(params).memorySize == get<SEHH_LSRTDEParams>(other.params).memorySize &&
                   get<SEHH_LSRTDEParams>(params).sigmaCR == get<SEHH_LSRTDEParams>(other.params).sigmaCR &&
                   get<SEHH_LSRTDEParams>(params).sigmaF == get<SEHH_LSRTDEParams>(other.params).sigmaF &&
                   get<SEHH_LSRTDEParams>(params).time_weight == get<SEHH_LSRTDEParams>(other.params).time_weight;
        }else if(holds_alternative<SEHH_RDEParams>(params) && holds_alternative<SEHH_RDEParams>(other.params)){
            return get<SEHH_RDEParams>(params).psizeParam == get<SEHH_RDEParams>(other.params).psizeParam &&
                   get<SEHH_RDEParams>(params).G_flag == get<SEHH_RDEParams>(other.params).G_flag &&
                   get<SEHH_RDEParams>(params).EB_flag == get<SEHH_RDEParams>(other.params).EB_flag &&
                   get<SEHH_RDEParams>(params).selectionMode == get<SEHH_RDEParams>(other.params).selectionMode &&
                   get<SEHH_RDEParams>(params).time_weight == get<SEHH_RDEParams>(other.params).time_weight;
        }
        return false;
    }

    void perturbParams(mt19937& gen, double strength = 6) {
        // strength 代表將整個 range 分為幾個標準差
        
        auto mutate_double = [&](double& val, double lower, double upper) {
            double range = upper - lower;
            // assume strength = 6
            // range = mid ± 6 sigma
            // 99.7% data within ±3 sigma
            normal_distribution<double> norm(0.0, (range / 2.0) / strength);
            double noise = norm(gen) * strength * range;
            val = clamp(val + noise, lower, upper);
        };

        auto mutate_int = [&](int& val, int lower, int upper) {
            double range = double(upper - lower);
            normal_distribution<double> norm(0.0, (range / 2.0) / strength);
            double noise = norm(gen) * strength * range;
            val = clamp((int)round(val + noise), lower, upper);
        };

        // Time weight range
        int tw_min = TIME_WEIGHT_RANGE.first;
        int tw_max = TIME_WEIGHT_RANGE.second;

        if (holds_alternative<SEHH_PSOParams>(params)) {
             auto& p = get<SEHH_PSOParams>(params);
             mutate_double(p.w, 0.0, 1.0);
             mutate_double(p.c1, 0.0, 2.0);
             mutate_double(p.c2, 0.0, 2.0);
             mutate_int(p.time_weight, tw_min, tw_max);
        } else if (holds_alternative<SEHH_GAParams>(params)) {
             auto& p = get<SEHH_GAParams>(params);
             mutate_double(p.mutationRate, 0.0, 1.0);
             mutate_double(p.crossoverRate, 0.0, 1.0);
             mutate_double(p.elitismRate, 0.0, 1.0);
             // Categorical parameters are usually not perturbed in fine-tuning, or use low prob
             mutate_int(p.time_weight, tw_min, tw_max);
        } else if (holds_alternative<SEHH_DEParams>(params)) {
             auto& p = get<SEHH_DEParams>(params);
             mutate_double(p.F, 0.0, 1.0);
             mutate_double(p.CR, 0.0, 1.0);
             // Strategy is categorical
             mutate_int(p.time_weight, tw_min, tw_max);
        } else if (holds_alternative<SEHH_LSHADEParams>(params)) {
             auto& p = get<SEHH_LSHADEParams>(params);
             mutate_int(p.memorySize, 5, 50);
             mutate_double(p.CR, 0.1, 0.9);
             mutate_double(p.F, 0.4, 0.9);
             mutate_int(p.time_weight, tw_min, tw_max);
        } else if (holds_alternative<SEHH_LSRTDEParams>(params)) {
             auto& p = get<SEHH_LSRTDEParams>(params);
             mutate_int(p.memorySize, 2, 20);
             mutate_double(p.sigmaCR, 0.02, 0.15);
             mutate_double(p.sigmaF, 0.01, 0.08);
             mutate_int(p.time_weight, tw_min, tw_max);
        } else if (holds_alternative<SEHH_RDEParams>(params)) {
             auto& p = get<SEHH_RDEParams>(params);
             mutate_double(p.psizeParam, 0.05, 0.9);
             mutate_double(p.G_flag, 0.0, 1.0);
             mutate_double(p.EB_flag, 0.0, 1.0);
             // selectionMode categorical
             mutate_int(p.time_weight, tw_min, tw_max);
        }
    }

    void RandominitP(){
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<double> r_0_1(0.0, 1.0);
        uniform_real_distribution<double> r_0_2(0.0, 2.0);
        uniform_int_distribution<int> i_1_4(1, 4);
        uniform_int_distribution<int> i_0_2(0, 2);
        uniform_int_distribution<int> i_0_1(0, 1);
        uniform_real_distribution<double> r_1e1_1e5(1e1, 1e5);
        uniform_real_distribution<double> r_1e8_1e1(1e-8, 1e-1);
        uniform_real_distribution<double> r_0_999(0.9, 0.999);
        uniform_int_distribution<int> i_5_50(5, 50);
        uniform_real_distribution<double> r_01_09(0.1, 0.9);
        uniform_real_distribution<double> r_04_09(0.4, 0.9);
        uniform_int_distribution<int> i_2_40(2, 40);
        uniform_real_distribution<double> r_03_09(0.3, 0.9);
        uniform_real_distribution<double> r_005_09(0.05,0.9);
        uniform_real_distribution<double> r_00_10(0.0, 1.0);
        uniform_int_distribution<int> i_0_1_int(0, 1);
        // LSRTDE 參數範圍
        uniform_int_distribution<int> i_2_20(2, 20);       // memorySize: 2~20
        uniform_real_distribution<double> r_002_015(0.02, 0.15);  // sigmaCR: 0.02~0.15
        uniform_real_distribution<double> r_001_008(0.01, 0.08);  // sigmaF: 0.01~0.08

        uniform_int_distribution<int> tw(TIME_WEIGHT_RANGE.first, TIME_WEIGHT_RANGE.second);
        if(template_name == "PSO"){
            params = SEHH_PSOParams{r_0_1(gen), r_0_2(gen), r_0_2(gen), tw(gen)};
        }else if(template_name == "GA"){
            params = SEHH_GAParams{r_0_1(gen), r_0_1(gen), r_0_1(gen), i_0_2(gen), i_0_1(gen), tw(gen)};
        }else if(template_name == "DE"){
            params = SEHH_DEParams{r_0_1(gen), r_0_1(gen), i_1_4(gen), tw(gen)};
        }else if(template_name == "LSHADE"){
            params = SEHH_LSHADEParams{i_5_50(gen), r_01_09(gen), r_04_09(gen), tw(gen)};
        }else if(template_name == "LSRTDE"){
            params = SEHH_LSRTDEParams{i_2_20(gen), r_002_015(gen), r_001_008(gen), tw(gen)};
        }else if(template_name == "RDE"){
            params = SEHH_RDEParams{r_005_09(gen), r_00_10(gen), r_00_10(gen), i_0_1_int(gen), tw(gen)};
        }else {
            cerr << "[BUG] Unknown template name in RandominitP: '" << template_name << "'" << endl;
            throw runtime_error("Unknown template name: " + template_name);
        }
    }

private:
    string template_name; // "PSO", "GA", "DE", "LSHADE", "LSRTDE", "RDE"
    SegmentParams params;

    void RandomInitName(){
        random_device rd;
        mt19937 gen(rd());
        uniform_int_distribution<int> dis(0, 5);
        switch(dis(gen)){
            case 0: template_name = "PSO"; break;
            case 1: template_name = "GA"; break;
            case 2: template_name = "DE"; break;
            case 3: template_name = "LSHADE"; break;
            case 4: template_name = "LSRTDE"; break;
            case 5: template_name = "RDE"; break;
            default:
                throw runtime_error("RandomInitName produced out-of-range value");
        }
    }
};

ostream& operator<<(ostream& os, const Segment& segment){
    os << segment.getTemplateName() << endl;

    const auto& params = segment.getSegmentParams();
     if (holds_alternative<SEHH_PSOParams>(params)) {
        const auto& p = get<SEHH_PSOParams>(params);
          os << "w c1 c2 time_weight\n"
              << p.w << " " << p.c1 << " " << p.c2 << " " << p.time_weight;
    } else if (holds_alternative<SEHH_GAParams>(params)) {
        const auto& p = get<SEHH_GAParams>(params);
          os << "mutationRate crossoverRate elitismRate selectionMethod crossoverMethod time_weight\n"
              << p.mutationRate << " " << p.crossoverRate << " " << p.elitismRate << " "
              << p.selectionMethod << " " << p.crossoverMethod << " " << p.time_weight;
    } else if (holds_alternative<SEHH_DEParams>(params)) {
        const auto& p = get<SEHH_DEParams>(params);
          os << "F CR strategy time_weight\n"
              << p.F << " " << p.CR << " " << p.strategy << " " << p.time_weight;
    } else if (holds_alternative<SEHH_LSHADEParams>(params)) {
        const auto& p = get<SEHH_LSHADEParams>(params);
          os << "memorySize CR F time_weight\n"
              << p.memorySize << " " << p.CR << " " << p.F << " " << p.time_weight;
    } else if (holds_alternative<SEHH_LSRTDEParams>(params)) {
        const auto& p = get<SEHH_LSRTDEParams>(params);
             os << "memorySize sigmaCR sigmaF time_weight\n"
                  << p.memorySize << " " << p.sigmaCR << " " << p.sigmaF << " " << p.time_weight;
    } else if (holds_alternative<SEHH_RDEParams>(params)) {
        const auto& p = get<SEHH_RDEParams>(params);
          os << "psizeParam G_flag EB_flag selectionMode time_weight\n"
              << p.psizeParam << " " << p.G_flag << " " << p.EB_flag << " " << p.selectionMode << " " << p.time_weight;
    }

    return os;
}

#endif /* SEGMENT_H */