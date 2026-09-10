#include<iostream>
#include<vector>
#include "config/para_setting.h"
#include "utils/instance.h"
using namespace std;

int main(){
    for(const auto& algorithm : algorithmName){
        cout << algorithm << endl;
    }

    cout << "#BASE_ALGORITHM " << BASE_ALGORITHM << endl;

    const vector<Instance>& instances = BenchmarkSetting::benchmark.getInstances();

    cout << instances.size() << "," << BenchmarkSetting::benchmark.getName() << "," << BenchmarkSetting::benchmark.getDimension() << endl;
    for(int i = 0; i < instances.size(); i++){
        cout << i+1 << "," << instances[i].getName() << "," << instances[i].getDimension() << endl;
    }

    return 0;
}