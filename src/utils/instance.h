#ifndef INSTANCE_H
#define INSTANCE_H

#include <vector>
#include <functional>
#include <string>
#define PI 3.14159265358979323846

using namespace std;

class Instance{
public:
    Instance(
        const string& name,
        int dimension,
        const vector<double>& lowerBounds,
        const vector<double>& upperBounds,
        function<double(const vector<double>&)> objFunction,
        int funcNo = -1
    ) :
        name(name), 
        dimension(dimension),
        lowerBounds(lowerBounds),
        upperBounds(upperBounds),
        objFunction(objFunction),
        funcNo(funcNo) {}

    const string getName() const { return name; }
    int getDimension() const { return dimension; }
    const vector<double>& getLowerBounds() const { return lowerBounds; }
    const vector<double>& getUpperBounds() const { return upperBounds; }
    const function<double(const vector<double>&)>& getObjFunction() const { return objFunction; }
    int getFuncNo() const { return funcNo; }
    double evaluate(const vector<double>& solution) const { return objFunction(solution); }
private:
    string name;
    function<double(const vector<double>&)> objFunction;
    int dimension;
    int funcNo;
    vector<double> lowerBounds;
    vector<double> upperBounds;
};

#endif