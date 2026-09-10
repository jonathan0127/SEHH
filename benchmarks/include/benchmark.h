#ifndef BENCHMARK_H
#define BENCHMARK_H

#include <vector>
#include "utils/instance.h"
#include "CEC2014.h"
#include "CEC2020.h"
#include "CEC2022.h"
#include "CEC2024.h"


using namespace CEC2014FUNCTION;
using namespace CEC2020FUNCTION;
using namespace CEC2022FUNCTION;
using namespace CEC2024FUNCTION;


class Benchmark {

public:
    Benchmark(
        const string& name,
        int dimension,
        const vector<Instance>& instances
    ) : name(name), dimension(dimension), instances(instances) {}

    const string getName() const { return name; }
    int getDimension() const { return dimension; }
    const vector<Instance>& getInstances() const { return instances; }

private:

    const string name;
    const int dimension;
    const vector<Instance> instances;

};

namespace CEC2014{

    const Benchmark benchmark_10D(
        "CEC2014 10D",
        10,
        {
            Instance("CEC2014 F1: Rotated High Conditioned Elliptic (10D)", 10, vector<double>(10, -100.0), vector<double>(10, 100.0), CEC2014_F1, 1),
            Instance("CEC2014 F2: Rotated Bent Cigar (10D)", 10, vector<double>(10, -100.0), vector<double>(10, 100.0), CEC2014_F2, 2),
            Instance("CEC2014 F3: Rotated Discus (10D)", 10, vector<double>(10, -100.0), vector<double>(10, 100.0), CEC2014_F3, 3),
            Instance("CEC2014 F4: Shifted and Rotated Rosenbrock (10D)", 10, vector<double>(10, -100.0), vector<double>(10, 100.0), CEC2014_F4, 4),
            Instance("CEC2014 F5: Shifted and Rotated Ackley (10D)", 10, vector<double>(10, -100.0), vector<double>(10, 100.0), CEC2014_F5, 5),
            Instance("CEC2014 F6: Shifted and Rotated Weierstrass (10D)", 10, vector<double>(10, -100.0), vector<double>(10, 100.0), CEC2014_F6, 6),
            Instance("CEC2014 F7: Shifted and Rotated Griewank (10D)", 10, vector<double>(10, -100.0), vector<double>(10, 100.0), CEC2014_F7, 7),
            Instance("CEC2014 F8: Shifted Rastrigin (10D)", 10, vector<double>(10, -100.0), vector<double>(10, 100.0), CEC2014_F8, 8),
            Instance("CEC2014 F9: Shifted and Rotated Rastrigin (10D)", 10, vector<double>(10, -100.0), vector<double>(10, 100.0), CEC2014_F9, 9),
            Instance("CEC2014 F10: Shifted Schwefel (10D)", 10, vector<double>(10, -100.0), vector<double>(10, 100.0), CEC2014_F10, 10),
            Instance("CEC2014 F11: Shifted and Rotated Schwefel (10D)", 10, vector<double>(10, -100.0), vector<double>(10, 100.0), CEC2014_F11, 11),
            Instance("CEC2014 F12: Shifted and Rotated Katsuura (10D)", 10, vector<double>(10, -100.0), vector<double>(10, 100.0), CEC2014_F12, 12),
            Instance("CEC2014 F13: Shifted and Rotated HappyCat (10D)", 10, vector<double>(10, -100.0), vector<double>(10, 100.0), CEC2014_F13, 13),
            Instance("CEC2014 F14: Shifted and Rotated HGBat (10D)", 10, vector<double>(10, -100.0), vector<double>(10, 100.0), CEC2014_F14, 14),
            Instance("CEC2014 F15: Shifted and Rotated Griewank-Rosenbrock (10D)", 10, vector<double>(10, -100.0), vector<double>(10, 100.0), CEC2014_F15, 15),
            Instance("CEC2014 F16: Shifted and Rotated Expanded Scaffer F6 (10D)", 10, vector<double>(10, -100.0), vector<double>(10, 100.0), CEC2014_F16, 16),
            Instance("CEC2014 F17: Hybrid Function 1 (10D)", 10, vector<double>(10, -100.0), vector<double>(10, 100.0), CEC2014_F17, 17),
            Instance("CEC2014 F18: Hybrid Function 2 (10D)", 10, vector<double>(10, -100.0), vector<double>(10, 100.0), CEC2014_F18, 18),
            Instance("CEC2014 F19: Hybrid Function 3 (10D)", 10, vector<double>(10, -100.0), vector<double>(10, 100.0), CEC2014_F19, 19),
            Instance("CEC2014 F20: Hybrid Function 4 (10D)", 10, vector<double>(10, -100.0), vector<double>(10, 100.0), CEC2014_F20, 20),
            Instance("CEC2014 F21: Hybrid Function 5 (10D)", 10, vector<double>(10, -100.0), vector<double>(10, 100.0), CEC2014_F21, 21),
            Instance("CEC2014 F22: Hybrid Function 6 (10D)", 10, vector<double>(10, -100.0), vector<double>(10, 100.0), CEC2014_F22, 22),
            Instance("CEC2014 F23: Composition Function 1 (10D)", 10, vector<double>(10, -100.0), vector<double>(10, 100.0), CEC2014_F23, 23),
            Instance("CEC2014 F24: Composition Function 2 (10D)", 10, vector<double>(10, -100.0), vector<double>(10, 100.0), CEC2014_F24, 24),
            Instance("CEC2014 F25: Composition Function 3 (10D)", 10, vector<double>(10, -100.0), vector<double>(10, 100.0), CEC2014_F25, 25),
            Instance("CEC2014 F26: Composition Function 4 (10D)", 10, vector<double>(10, -100.0), vector<double>(10, 100.0), CEC2014_F26, 26),
            Instance("CEC2014 F27: Composition Function 5 (10D)", 10, vector<double>(10, -100.0), vector<double>(10, 100.0), CEC2014_F27, 27),
            Instance("CEC2014 F28: Composition Function 6 (10D)", 10, vector<double>(10, -100.0), vector<double>(10, 100.0), CEC2014_F28, 28),
            Instance("CEC2014 F29: Composition Function 7 (10D)", 10, vector<double>(10, -100.0), vector<double>(10, 100.0), CEC2014_F29, 29),
            Instance("CEC2014 F30: Composition Function 8 (10D)", 10, vector<double>(10, -100.0), vector<double>(10, 100.0), CEC2014_F30, 30),
        }
    );

    const Benchmark benchmark_30D(
        "CEC2014 30D",
        30,
        {
            Instance("CEC2014 F1: Rotated High Conditioned Elliptic (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2014_F1, 1),
            Instance("CEC2014 F2: Rotated Bent Cigar (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2014_F2, 2),
            Instance("CEC2014 F3: Rotated Discus (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2014_F3, 3),
            Instance("CEC2014 F4: Shifted and Rotated Rosenbrock (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2014_F4, 4),
            Instance("CEC2014 F5: Shifted and Rotated Ackley (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2014_F5, 5),
            Instance("CEC2014 F6: Shifted and Rotated Weierstrass (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2014_F6, 6),
            Instance("CEC2014 F7: Shifted and Rotated Griewank (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2014_F7, 7),
            Instance("CEC2014 F8: Shifted Rastrigin (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2014_F8, 8),
            Instance("CEC2014 F9: Shifted and Rotated Rastrigin (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2014_F9, 9),
            Instance("CEC2014 F10: Shifted Schwefel (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2014_F10, 10),
            Instance("CEC2014 F11: Shifted and Rotated Schwefel (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2014_F11, 11),
            Instance("CEC2014 F12: Shifted and Rotated Katsuura (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2014_F12, 12),
            Instance("CEC2014 F13: Shifted and Rotated HappyCat (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2014_F13, 13),
            Instance("CEC2014 F14: Shifted and Rotated HGBat (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2014_F14, 14),
            Instance("CEC2014 F15: Shifted and Rotated Griewank-Rosenbrock (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2014_F15, 15),
            Instance("CEC2014 F16: Shifted and Rotated Expanded Scaffer F6 (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2014_F16, 16),
            Instance("CEC2014 F17: Hybrid Function 1 (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2014_F17, 17),
            Instance("CEC2014 F18: Hybrid Function 2 (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2014_F18, 18),
            Instance("CEC2014 F19: Hybrid Function 3 (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2014_F19, 19),
            Instance("CEC2014 F20: Hybrid Function 4 (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2014_F20, 20),
            Instance("CEC2014 F21: Hybrid Function 5 (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2014_F21, 21),
            Instance("CEC2014 F22: Hybrid Function 6 (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2014_F22, 22),
            Instance("CEC2014 F23: Composition Function 1 (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2014_F23, 23),
            Instance("CEC2014 F24: Composition Function 2 (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2014_F24, 24),
            Instance("CEC2014 F25: Composition Function 3 (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2014_F25, 25),
            Instance("CEC2014 F26: Composition Function 4 (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2014_F26, 26),
            Instance("CEC2014 F27: Composition Function 5 (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2014_F27, 27),
            Instance("CEC2014 F28: Composition Function 6 (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2014_F28, 28),
            Instance("CEC2014 F29: Composition Function 7 (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2014_F29, 29),
            Instance("CEC2014 F30: Composition Function 8 (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2014_F30, 30),
        }
    );

};

namespace CEC2022{

    const Benchmark benchmark_10D(
        "CEC2022 10D",
        10,
        {
            Instance("85. CEC2022 F1: Shifted and full Rotated Zakharov (10D)", 10, vector<double>(10, -100.0), vector<double>(10, 100.0), CEC2022_F1, 1),
            Instance("87. CEC2022 F2: Shifted and Rotated Rosenbrock (10D)", 10, vector<double>(10, -100.0), vector<double>(10, 100.0), CEC2022_F2, 2),
            Instance("89. CEC2022 F3: Shifted and full Rotated Expanded Schaffer's F7 (10D)", 10, vector<double>(10, -100.0), vector<double>(10, 100.0), CEC2022_F3, 3),
            Instance("91. CEC2022 F4: Shifted and Rotated Non-Continuous Rastrigin (10D)", 10, vector<double>(10, -100.0), vector<double>(10, 100.0), CEC2022_F4, 4),
            Instance("93. CEC2022 F5: Shifted and Rotated Levy Function (10D)", 10, vector<double>(10, -100.0), vector<double>(10, 100.0), CEC2022_F5, 5),
            Instance("95. CEC2022 F6: Hybrid Function 1 (10D)", 10, vector<double>(10, -100.0), vector<double>(10, 100.0), CEC2022_F6, 6),
            Instance("97. CEC2022 F7: Hybrid Function 2 (10D)", 10, vector<double>(10, -100.0), vector<double>(10, 100.0), CEC2022_F7, 7),
            Instance("99. CEC2022 F8: Hybrid Function 3 (10D)", 10, vector<double>(10, -100.0), vector<double>(10, 100.0), CEC2022_F8, 8),
            Instance("101. CEC2022 F9: Composition Function 1 (10D)", 10, vector<double>(10, -100.0), vector<double>(10, 100.0), CEC2022_F9, 9),
            Instance("103. CEC2022 F10: Composition Function 2 (10D)", 10, vector<double>(10, -100.0), vector<double>(10, 100.0), CEC2022_F10, 10),
            Instance("105. CEC2022 F11: Composition Function 3 (10D)", 10, vector<double>(10, -100.0), vector<double>(10, 100.0), CEC2022_F11, 11),
            Instance("107. CEC2022 F12: Composition Function 4 (10D)", 10, vector<double>(10, -100.0), vector<double>(10, 100.0), CEC2022_F12, 12),
        }
    );

    const Benchmark benchmark_20D(
        "CEC2022 20D",
        20,
        {
            Instance("86. CEC2022 F1: Shifted and full Rotated Zakharov (20D)", 20, vector<double>(20, -100.0), vector<double>(20, 100.0), CEC2022_F1, 1),
            Instance("88. CEC2022 F2: Shifted and Rotated Rosenbrock (20D)", 20, vector<double>(20, -100.0), vector<double>(20, 100.0), CEC2022_F2, 2),
            Instance("90. CEC2022 F3: Shifted and full Rotated Expanded Schaffer's F7 (20D)", 20, vector<double>(20, -100.0), vector<double>(20, 100.0), CEC2022_F3, 3),
            Instance("92. CEC2022 F4: Shifted and Rotated Non-Continuous Rastrigin (20D)", 20, vector<double>(20, -100.0), vector<double>(20, 100.0), CEC2022_F4, 4),
            Instance("94. CEC2022 F5: Shifted and Rotated Levy Function (20D)", 20, vector<double>(20, -100.0), vector<double>(20, 100.0), CEC2022_F5, 5),
            Instance("96. CEC2022 F6: Hybrid Function 1 (20D)", 20, vector<double>(20, -100.0), vector<double>(20, 100.0), CEC2022_F6, 6),
            Instance("98. CEC2022 F7: Hybrid Function 2 (20D)", 20, vector<double>(20, -100.0), vector<double>(20, 100.0), CEC2022_F7, 7),
            Instance("100. CEC2022 F8: Hybrid Function 3 (20D)", 20, vector<double>(20, -100.0), vector<double>(20, 100.0), CEC2022_F8, 8),
            Instance("102. CEC2022 F9: Composition Function 1 (20D)", 20, vector<double>(20, -100.0), vector<double>(20, 100.0), CEC2022_F9, 9),
            Instance("104. CEC2022 F10: Composition Function 2 (20D)", 20, vector<double>(20, -100.0), vector<double>(20, 100.0), CEC2022_F10, 10),
            Instance("106. CEC2022 F11: Composition Function 3 (20D)", 20, vector<double>(20, -100.0), vector<double>(20, 100.0), CEC2022_F11, 11),
            Instance("108. CEC2022 F12: Composition Function 4 (20D)", 20, vector<double>(20, -100.0), vector<double>(20, 100.0), CEC2022_F12, 12),
        }
    );

};

namespace CEC2020{

    const Benchmark benchmark_10D(
        "CEC2020 10D",
        10,
        {
            Instance("48. CEC2020 F1: Shifted and Rotated Bent Cigar (10D)", 10, vector<double>(10, -100.0), vector<double>(10, 100.0), CEC2020_F1, 1),
            Instance("52. CEC2020 F2: Shifted and Rotated Schwefel (10D)", 10, vector<double>(10, -100.0), vector<double>(10, 100.0), CEC2020_F2, 2),
            Instance("56. CEC2020 F3: Shifted and Rotated Lunacek bi-Rastrigin (10D)", 10, vector<double>(10, -100.0), vector<double>(10, 100.0), CEC2020_F3, 3),
            Instance("60. CEC2020 F4: Expanded Rosenbrock plus Griewangk (10D)", 10, vector<double>(10, -100.0), vector<double>(10, 100.0), CEC2020_F4, 4),
            Instance("64. CEC2020 F5: Hybrid Function 1 (10D)", 10, vector<double>(10, -100.0), vector<double>(10, 100.0), CEC2020_F5, 5),
            Instance("67. CEC2020 F6: Hybrid Function 2 (10D)", 10, vector<double>(10, -100.0), vector<double>(10, 100.0), CEC2020_F6, 6),
            Instance("70. CEC2020 F7: Hybrid Function 3 (10D)", 10, vector<double>(10, -100.0), vector<double>(10, 100.0), CEC2020_F7, 7),
            Instance("74. CEC2020 F8: Composition Function 1 (10D)", 10, vector<double>(10, -100.0), vector<double>(10, 100.0), CEC2020_F8, 8),
            Instance("78. CEC2020 F9: Composition Function 2 (10D)", 10, vector<double>(10, -100.0), vector<double>(10, 100.0), CEC2020_F9, 9),
            Instance("82. CEC2020 F10: Composition Function 3 (10D)", 10, vector<double>(10, -100.0), vector<double>(10, 100.0), CEC2020_F10, 10),
        }
    );

    const Benchmark benchmark_20D(
        "CEC2020 20D",
        20,
        {
            Instance("50. CEC2020 F1: Shifted and Rotated Bent Cigar (20D)", 20, vector<double>(20, -100.0), vector<double>(20, 100.0), CEC2020_F1, 1),
            Instance("54. CEC2020 F2: Shifted and Rotated Schwefel (20D)", 20, vector<double>(20, -100.0), vector<double>(20, 100.0), CEC2020_F2, 2),
            Instance("58. CEC2020 F3: Shifted and Rotated Lunacek bi-Rastrigin (20D)", 20, vector<double>(20, -100.0), vector<double>(20, 100.0), CEC2020_F3, 3),
            Instance("62. CEC2020 F4: Expanded Rosenbrock plus Griewangk (20D)", 20, vector<double>(20, -100.0), vector<double>(20, 100.0), CEC2020_F4, 4),
            Instance("66. CEC2020 F5: Hybrid Function 1 (20D)", 20, vector<double>(20, -100.0), vector<double>(20, 100.0), CEC2020_F5, 5),
            Instance("69. CEC2020 F6: Hybrid Function 2 (20D)", 20, vector<double>(20, -100.0), vector<double>(20, 100.0), CEC2020_F6, 6),
            Instance("72. CEC2020 F7: Hybrid Function 3 (20D)", 20, vector<double>(20, -100.0), vector<double>(20, 100.0), CEC2020_F7, 7),
            Instance("76. CEC2020 F8: Composition Function 1 (20D)", 20, vector<double>(20, -100.0), vector<double>(20, 100.0), CEC2020_F8, 8),
            Instance("80. CEC2020 F9: Composition Function 2 (20D)", 20, vector<double>(20, -100.0), vector<double>(20, 100.0), CEC2020_F9, 9),
            Instance("84. CEC2020 F10: Composition Function 3 (20D)", 20, vector<double>(20, -100.0), vector<double>(20, 100.0), CEC2020_F10, 10),
        }
    );

};

namespace CEC2024{

    const Benchmark benchmark_30D(
        "CEC2024 30D",
        30,
        {
            Instance("CEC2024 F1: Shifted and Rotated Bent Cigar (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2024_F1, 1),
            Instance("CEC2024 F2: Shifted and Rotated Schwefel (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2024_F2, 2),
            Instance("CEC2024 F3: Shifted and Rotated Lunacek bi-Rastrigin (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2024_F3, 3),
            Instance("CEC2024 F4: Expanded Rosenbrock plus Griewangk (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2024_F4, 4),
            Instance("CEC2024 F5: Hybrid Function 1 (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2024_F5, 5),
            Instance("CEC2024 F6: Hybrid Function 2 (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2024_F6, 6),
            Instance("CEC2024 F7: Hybrid Function 3 (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2024_F7, 7),
            Instance("CEC2024 F8: Composition Function 1 (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2024_F8, 8),
            Instance("CEC2024 F9: Composition Function 2 (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2024_F9, 9),
            Instance("CEC2024 F10: Composition Function 3 (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2024_F10, 10),
            Instance("CEC2024 F11: Composition Function 4 (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2024_F11, 11),
            Instance("CEC2024 F12: Composition Function 5 (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2024_F12, 12),
            Instance("CEC2024 F13: Composition Function 6 (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2024_F13, 13),
            Instance("CEC2024 F14: Composition Function 7 (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2024_F14, 14),
            Instance("CEC2024 F15: Composition Function 8 (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2024_F15, 15),
            Instance("CEC2024 F16: Composition Function 9 (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2024_F16, 16),
            Instance("CEC2024 F17: Hybrid Function 4 (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2024_F17, 17),
            Instance("CEC2024 F18: Hybrid Function 5 (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2024_F18, 18),
            Instance("CEC2024 F19: Hybrid Function 6 (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2024_F19, 19),
            Instance("CEC2024 F20: Hybrid Function 7 (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2024_F20, 20),
            Instance("CEC2024 F21: Hybrid Function 8 (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2024_F21, 21),
            Instance("CEC2024 F22: Hybrid Function 9 (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2024_F22, 22),
            Instance("CEC2024 F23: Composition Function 10 (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2024_F23, 23),
            Instance("CEC2024 F24: Composition Function 11 (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2024_F24, 24),
            Instance("CEC2024 F25: Composition Function 12 (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2024_F25, 25),
            Instance("CEC2024 F26: Composition Function 13 (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2024_F26, 26),
            Instance("CEC2024 F27: Composition Function 14 (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2024_F27, 27),
            Instance("CEC2024 F28: Composition Function 15 (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2024_F28, 28),
            Instance("CEC2024 F29: Composition Function 16 (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2024_F29, 29),
            Instance("CEC2024 F30: Composition Function 17 (30D)", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), CEC2024_F30, 30)
        }
    );

};

// // all instances
// // 執行時請註解掉不需要的 instance
// namespace Instances{
//     const vector<Instance> instances = {
//         // Instance("1. CEC Function 1: Sphere Function", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), OptimizationFunction::Sphere_Function),
//         // Instance("2. CEC Function 2: Sum and Product Function", 30, vector<double>(30, -10.0), vector<double>(30, 10.0), OptimizationFunction::Sum_And_Product_Function),
//         // Instance("3. CEC Function 3: Schwefel Function Part 2", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), OptimizationFunction::Schwefel_Function_Part2),
//         // Instance("4. CEC Function 4: Chebyshev Function", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), OptimizationFunction::Chebyshev_Function),
//         // Instance("5. CEC Function 5: Rosenbrock Function", 30, vector<double>(30, -30.0), vector<double>(30, 30.0), OptimizationFunction::Rosenbrock_Function),
//         // Instance("6. CEC Function 6: Step Function", 30, vector<double>(30, -100.0), vector<double>(30, 100.0), OptimizationFunction::Step_Function),
//         // Instance("7. CEC Function 7: Quartic Function with Noise", 30, vector<double>(30, -1.28), vector<double>(30, 1.28), OptimizationFunction::Quartic_Function),
//         // Instance("8. CEC Function 8: Schwefel Function", 30, vector<double>(30, -500.0), vector<double>(30, 500.0), OptimizationFunction::Schwefel_Function),
//         // Instance("9. CEC Function 9: Rastrigin Function", 30, vector<double>(30, -5.12), vector<double>(30, 5.12), OptimizationFunction::Rastrigin_Function),
//         // Instance("10. CEC Function 10: Ackley Function CEC", 30, vector<double>(30, -32.0), vector<double>(30, 32.0), OptimizationFunction::Ackley_Function_CEC),
//         // Instance("11. CEC Function 11: Griewank Function", 30, vector<double>(30, -600.0), vector<double>(30, 600.0), OptimizationFunction::Griewank_Function),
//         // Instance("12. CEC Function 12: Penalized Function 1", 30, vector<double>(30, -50.0), vector<double>(30, 50.0), OptimizationFunction::Penalized_Function_1),
//         // Instance("13. CEC Function 13: Penalized Function 2", 30, vector<double>(30, -50.0), vector<double>(30, 50.0), OptimizationFunction::Penalized_Function_2),
//         // Instance("14. CEC Function 14: Shekel Foxholes Function", 2, vector<double>(2, -65.536), vector<double>(2, 65.536), OptimizationFunction::Shekel_Foxholes_Function),
//         // Instance("15. CEC Function 15: Kowalik Function", 4, vector<double>(4, -5.0), vector<double>(4, 5.0), OptimizationFunction::Kowalik_Function),
//         // Instance("16. CEC Function 16: Six-Hump Camel Function", 2, vector<double>(2, -5.0), vector<double>(2, 5.0), OptimizationFunction::Six_Hump_Camel_Function),
//         // Instance("17. CEC Function 17: Branin Function", 2, vector<double>(2, -5.0), vector<double>(2, 5.0), OptimizationFunction::Branin_Function),
//         // Instance("18. CEC Function 18: Goldstein-Price Function", 2, vector<double>(2, -2.0), vector<double>(2, 2.0), OptimizationFunction::Goldstein_Price_Function),
//         // Instance("19. CEC Function 19: Hartman 3D Function", 3, vector<double>(3, 1.0), vector<double>(3, 3.0), OptimizationFunction::Hartman_3D_Function),
//         // Instance("20. CEC Function 20: Hartman 6D Function", 6, vector<double>(6, 0.0), vector<double>(6, 1.0), OptimizationFunction::Hartman_6D_Function),
//         // Instance("21. CEC Function 21: Shekel 5 Function", 4, vector<double>(4, 0.0), vector<double>(4, 10.0), OptimizationFunction::Shekel_5_Function),
//         // Instance("22. CEC Function 22: Shekel 7 Function", 4, vector<double>(4, 0.0), vector<double>(4, 10.0), OptimizationFunction::Shekel_7_Function),
//         // Instance("23. CEC Function 23: Shekel 10 Function", 4, vector<double>(4, 0.0), vector<double>(4, 10.0), OptimizationFunction::Shekel_10_Function),
//         // Instance("24. Function 1: Shubert_3_Function", 2, vector<double>(2, -10.0), vector<double>(2, 10.0), OptimizationFunction::Shubert_3_Function),
//         // Instance("25. Function 2: Keane_Function", 2, vector<double>(2, -10.0), vector<double>(2, 10.0), OptimizationFunction::Keane_Function),
//         // Instance("26. Function 3: Cross_in_Tray_Function", 2, vector<double>(2, -100.0), vector<double>(2, 100.0), OptimizationFunction::Cross_in_Tray_Function),
//         // Instance("27. Function 4: Holder_Table_Function", 2, vector<double>(2, -10.0), vector<double>(2, 10.0), OptimizationFunction::Holder_Table_Function),
//         // Instance("28. Function 5: Ackley_Function", 30, vector<double>(30, -32.768), vector<double>(30, 32.768), OptimizationFunction::Ackley_Function),
//         // Instance("29. Bowl: Bohachevsky Function", 2, vector<double>(2, -100.0), vector<double>(2, 100.0), OptimizationFunction::Bohachevsky),
//         // Instance("30. Bowl: Perm Function", 30, vector<double>(30, 0.0), vector<double>(30, 1.0), [](const vector<double>& x){ return OptimizationFunction::Perm_Function(x, 0.5); }),
//         // Instance("31. Bowl: Rotated Hyper-Ellipsoid (De Jong's f2)", 30, vector<double>(30, -5.0), vector<double>(30, 5.0), OptimizationFunction::Rotated_HyperEllipsoid),
//         // Instance("32. Bowl: Sum of Different Powers", 30, vector<double>(30, -1.0), vector<double>(30, 1.0), OptimizationFunction::Sum_of_Different_Powers),
//         // Instance("33. Bowl: Sum Squares", 30, vector<double>(30, -10.0), vector<double>(30, 10.0), OptimizationFunction::Sum_Squares),
//         // Instance("34. Bowl: Trid", 30, vector<double>(30, -900.0), vector<double>(30, 900.0), OptimizationFunction::Trid),
//         // Instance("35. Plate: Booth Function (2D)", 2, vector<double>{-1.5, -3.0}, vector<double>{4.0, 4.0}, OptimizationFunction::Booth),
//         // Instance("36. Plate: Power Sum Function (30D)", 30, vector<double>(30, -1.0), vector<double>(30, 1.0), OptimizationFunction::Power_Sum),
//         // Instance("37. Plate: Zakharov Function (30D)", 30, vector<double>(30, -5.0), vector<double>(30, 10.0), OptimizationFunction::Zakharov),
//         // Instance("38. Valley: Three-Hump Camel (2D)", 2, vector<double>(2, -5.0), vector<double>(2, 5.0), OptimizationFunction::Three_Hump_Camel),
//         // Instance("39. Valley: Dixon-Price (30D)", 30, vector<double>(30, -10.0), vector<double>(30, 10.0), OptimizationFunction::Dixon_Price),
//         // Instance("40. Steep: Easom Function (2D)", 2, vector<double>(2, -100.0), vector<double>(2, 100.0), OptimizationFunction::Easom),
//         // Instance("41. Steep: Michalewicz Function (30D, m=10)", 30, vector<double>(30, 0.0), vector<double>(30, PI), [](const vector<double>& x){ return OptimizationFunction::Michalewicz(x, 10.0); }),
//         // Instance("42. Other: Beale Function (2D)", 2, vector<double>(2, -4.5), vector<double>(2, 4.5), OptimizationFunction::Beale),
//         // Instance("43. Other: Colville Function (4D)", 4, vector<double>(4, -10.0), vector<double>(4, 10.0), OptimizationFunction::Colville),
//         // Instance("44. Other: Forrester Function (1D)", 1, vector<double>{0.0}, vector<double>{1.0}, OptimizationFunction::Forrester),
//         // Instance("45. Other: Powell Function (4D)", 4, vector<double>(4, -4.0), vector<double>(4, 5.0), OptimizationFunction::Powell),
//         // Instance("46. Other: Styblinski-Tang Function (30D)", 30, vector<double>(30, -5.0), vector<double>(30, 5.0), OptimizationFunction::Styblinski_Tang),
        
//         // // CEC2020 測試函數 - F1 (四個維度: 5, 10, 15, 20)
//         // Instance("47. CEC2020 F1: Shifted and Rotated Bent Cigar (5D)", 5, vector<double>(5, -100.0), vector<double>(5, 100.0), CEC2020_F1),
//         // Instance("48. CEC2020 F1: Shifted and Rotated Bent Cigar (10D)", 10, vector<double>(10, -100.0), vector<double>(10, 100.0), CEC2020_F1),
//         // Instance("49. CEC2020 F1: Shifted and Rotated Bent Cigar (15D)", 15, vector<double>(15, -100.0), vector<double>(15, 100.0), CEC2020_F1),
//         // Instance("50. CEC2020 F1: Shifted and Rotated Bent Cigar (20D)", 20, vector<double>(20, -100.0), vector<double>(20, 100.0), CEC2020_F1),
        
//         // // CEC2020 測試函數 - F2 (四個維度: 5, 10, 15, 20)
//         // Instance("51. CEC2020 F2: Shifted and Rotated Schwefel (5D)", 5, vector<double>(5, -100.0), vector<double>(5, 100.0), CEC2020_F2),
//         // Instance("52. CEC2020 F2: Shifted and Rotated Schwefel (10D)", 10, vector<double>(10, -100.0), vector<double>(10, 100.0), CEC2020_F2),
//         // Instance("53. CEC2020 F2: Shifted and Rotated Schwefel (15D)", 15, vector<double>(15, -100.0), vector<double>(15, 100.0), CEC2020_F2),
//         // Instance("54. CEC2020 F2: Shifted and Rotated Schwefel (20D)", 20, vector<double>(20, -100.0), vector<double>(20, 100.0), CEC2020_F2),
        
//         // // CEC2020 測試函數 - F3 (四個維度: 5, 10, 15, 20)
//         // Instance("55. CEC2020 F3: Shifted and Rotated Lunacek bi-Rastrigin (5D)", 5, vector<double>(5, -100.0), vector<double>(5, 100.0), CEC2020_F3),
//         // Instance("56. CEC2020 F3: Shifted and Rotated Lunacek bi-Rastrigin (10D)", 10, vector<double>(10, -100.0), vector<double>(10, 100.0), CEC2020_F3),
//         // Instance("57. CEC2020 F3: Shifted and Rotated Lunacek bi-Rastrigin (15D)", 15, vector<double>(15, -100.0), vector<double>(15, 100.0), CEC2020_F3),
//         // Instance("58. CEC2020 F3: Shifted and Rotated Lunacek bi-Rastrigin (20D)", 20, vector<double>(20, -100.0), vector<double>(20, 100.0), CEC2020_F3),
        
//         // // CEC2020 測試函數 - F4 (四個維度: 5, 10, 15, 20)
//         // Instance("59. CEC2020 F4: Expanded Rosenbrock plus Griewangk (5D)", 5, vector<double>(5, -100.0), vector<double>(5, 100.0), CEC2020_F4),
//         // Instance("60. CEC2020 F4: Expanded Rosenbrock plus Griewangk (10D)", 10, vector<double>(10, -100.0), vector<double>(10, 100.0), CEC2020_F4),
//         // Instance("61. CEC2020 F4: Expanded Rosenbrock plus Griewangk (15D)", 15, vector<double>(15, -100.0), vector<double>(15, 100.0), CEC2020_F4),
//         // Instance("62. CEC2020 F4: Expanded Rosenbrock plus Griewangk (20D)", 20, vector<double>(20, -100.0), vector<double>(20, 100.0), CEC2020_F4),
        
//         // // CEC2020 測試函數 - F5 (四個維度: 5, 10, 15, 20)
//         // Instance("63. CEC2020 F5: Hybrid Function 1 (5D)", 5, vector<double>(5, -100.0), vector<double>(5, 100.0), CEC2020_F5),
//         // Instance("64. CEC2020 F5: Hybrid Function 1 (10D)", 10, vector<double>(10, -100.0), vector<double>(10, 100.0), CEC2020_F5),
//         // Instance("65. CEC2020 F5: Hybrid Function 1 (15D)", 15, vector<double>(15, -100.0), vector<double>(15, 100.0), CEC2020_F5),
//         // Instance("66. CEC2020 F5: Hybrid Function 1 (20D)", 20, vector<double>(20, -100.0), vector<double>(20, 100.0), CEC2020_F5),
        
//         // // CEC2020 測試函數 - F6 (三個維度: 10, 15, 20)
//         // Instance("67. CEC2020 F6: Hybrid Function 2 (10D)", 10, vector<double>(10, -100.0), vector<double>(10, 100.0), CEC2020_F6),
//         // Instance("68. CEC2020 F6: Hybrid Function 2 (15D)", 15, vector<double>(15, -100.0), vector<double>(15, 100.0), CEC2020_F6),
//         // Instance("69. CEC2020 F6: Hybrid Function 2 (20D)", 20, vector<double>(20, -100.0), vector<double>(20, 100.0), CEC2020_F6),
        
//         // // CEC2020 測試函數 - F7 (三個維度: 10, 15, 20)
//         // Instance("70. CEC2020 F7: Hybrid Function 3 (10D)", 10, vector<double>(10, -100.0), vector<double>(10, 100.0), CEC2020_F7),
//         // Instance("71. CEC2020 F7: Hybrid Function 3 (15D)", 15, vector<double>(15, -100.0), vector<double>(15, 100.0), CEC2020_F7),
//         // Instance("72. CEC2020 F7: Hybrid Function 3 (20D)", 20, vector<double>(20, -100.0), vector<double>(20, 100.0), CEC2020_F7),
        
//         // // CEC2020 測試函數 - F8 (四個維度: 5, 10, 15, 20)
//         // Instance("73. CEC2020 F8: Composition Function 1 (5D)", 5, vector<double>(5, -100.0), vector<double>(5, 100.0), CEC2020_F8),
//         // Instance("74. CEC2020 F8: Composition Function 1 (10D)", 10, vector<double>(10, -100.0), vector<double>(10, 100.0), CEC2020_F8),
//         // Instance("75. CEC2020 F8: Composition Function 1 (15D)", 15, vector<double>(15, -100.0), vector<double>(15, 100.0), CEC2020_F8),
//         // Instance("76. CEC2020 F8: Composition Function 1 (20D)", 20, vector<double>(20, -100.0), vector<double>(20, 100.0), CEC2020_F8),
        
//         // // CEC2020 測試函數 - F9 (四個維度: 5, 10, 15, 20)
//         // Instance("77. CEC2020 F9: Composition Function 2 (5D)", 5, vector<double>(5, -100.0), vector<double>(5, 100.0), CEC2020_F9),
//         // Instance("78. CEC2020 F9: Composition Function 2 (10D)", 10, vector<double>(10, -100.0), vector<double>(10, 100.0), CEC2020_F9),
//         // Instance("79. CEC2020 F9: Composition Function 2 (15D)", 15, vector<double>(15, -100.0), vector<double>(15, 100.0), CEC2020_F9),
//         // Instance("80. CEC2020 F9: Composition Function 2 (20D)", 20, vector<double>(20, -100.0), vector<double>(20, 100.0), CEC2020_F9),
        
//         // // CEC2020 測試函數 - F10 (四個維度: 5, 10, 15, 20)
//         // Instance("81. CEC2020 F10: Composition Function 3 (5D)", 5, vector<double>(5, -100.0), vector<double>(5, 100.0), CEC2020_F10),
//         // Instance("82. CEC2020 F10: Composition Function 3 (10D)", 10, vector<double>(10, -100.0), vector<double>(10, 100.0), CEC2020_F10),
//         // Instance("83. CEC2020 F10: Composition Function 3 (15D)", 15, vector<double>(15, -100.0), vector<double>(15, 100.0), CEC2020_F10),
//         // Instance("84. CEC2020 F10: Composition Function 3 (20D)", 20, vector<double>(20, -100.0), vector<double>(20, 100.0), CEC2020_F10),

//         // CEC2022 測試函數 - F1 (兩個維度: 10, 20)
//         // Instance("85. CEC2022 F1: Shifted and full Rotated Zakharov (10D)", 10, vector<double>(10, -100.0), vector<double>(10, 100.0), CEC2022_F1),
//         // Instance("86. CEC2022 F1: Shifted and full Rotated Zakharov (20D)", 20, vector<double>(20, -100.0), vector<double>(20, 100.0), CEC2022_F1),
        
//         // // CEC2022 測試函數 - F2 (兩個維度: 10, 20)
//         // Instance("87. CEC2022 F2: Shifted and Rotated Rosenbrock (10D)", 10, vector<double>(10, -100.0), vector<double>(10, 100.0), CEC2022_F2),
//         // Instance("88. CEC2022 F2: Shifted and Rotated Rosenbrock (20D)", 20, vector<double>(20, -100.0), vector<double>(20, 100.0), CEC2022_F2),

//         // // CEC2022 測試函數 - F3 (兩個維度: 10, 20)
//         // Instance("89. CEC2022 F3: Shifted and full Rotated Expanded Schaffer's F7 (10D)", 10, vector<double>(10, -100.0), vector<double>(10, 100.0), CEC2022_F3),
//         // Instance("90. CEC2022 F3: Shifted and full Rotated Expanded Schaffer's F7 (20D)", 20, vector<double>(20, -100.0), vector<double>(20, 100.0), CEC2022_F3),

//         // // CEC2022 測試函數 - F4 (兩個維度: 10, 20)
//         // Instance("91. CEC2022 F4: Shifted and Rotated Non-Continuous Rastrigin (10D)", 10, vector<double>(10, -100.0), vector<double>(10, 100.0), CEC2022_F4),
//         // Instance("92. CEC2022 F4: Shifted and Rotated Non-Continuous Rastrigin (20D)", 20, vector<double>(20, -100.0), vector<double>(20, 100.0), CEC2022_F4),

//         // // CEC2022 測試函數 - F5 (兩個維度: 10, 20)
//         // Instance("93. CEC2022 F5: Shifted and Rotated Levy Function (10D)", 10, vector<double>(10, -100.0), vector<double>(10, 100.0), CEC2022_F5),
//         // Instance("94. CEC2022 F5: Shifted and Rotated Levy Function (20D)", 20, vector<double>(20, -100.0), vector<double>(20, 100.0), CEC2022_F5),

//         // // CEC2022 測試函數 - F6 (兩個維度: 10, 20)
//         // Instance("95. CEC2022 F6: Hybrid Function 1 (10D)", 10, vector<double>(10, -100.0), vector<double>(10, 100.0), CEC2022_F6),
//         // Instance("96. CEC2022 F6: Hybrid Function 1 (20D)", 20, vector<double>(20, -100.0), vector<double>(20, 100.0), CEC2022_F6),

//         // // CEC2022 測試函數 - F7 (兩個維度: 10, 20)
//         // Instance("97. CEC2022 F7: Hybrid Function 2 (10D)", 10, vector<double>(10, -100.0), vector<double>(10, 100.0), CEC2022_F7),
//         // Instance("98. CEC2022 F7: Hybrid Function 2 (20D)", 20, vector<double>(20, -100.0), vector<double>(20, 100.0), CEC2022_F7),

//         // // CEC2022 測試函數 - F8 (兩個維度: 10, 20)
//         // Instance("99. CEC2022 F8: Hybrid Function 3 (10D)", 10, vector<double>(10, -100.0), vector<double>(10, 100.0), CEC2022_F8),
//         // Instance("100. CEC2022 F8: Hybrid Function 3 (20D)", 20, vector<double>(20, -100.0), vector<double>(20, 100.0), CEC2022_F8),

//         // CEC2022 測試函數 - F9 (兩個維度: 10, 20)
//         Instance("101. CEC2022 F9: Composition Function 1 (10D)", 10, vector<double>(10, -100.0), vector<double>(10, 100.0), CEC2022_F9),
//         Instance("102. CEC2022 F9: Composition Function 1 (20D)", 20, vector<double>(20, -100.0), vector<double>(20, 100.0), CEC2022_F9),

//         // // CEC2022 測試函數 - F10 (兩個維度: 10, 20)
//         // Instance("103. CEC2022 F10: Composition Function 2 (10D)", 10, vector<double>(10, -100.0), vector<double>(10, 100.0), CEC2022_F10),
//         // Instance("104. CEC2022 F10: Composition Function 2 (20D)", 20, vector<double>(20, -100.0), vector<double>(20, 100.0), CEC2022_F10),

//         // // CEC2022 測試函數 - F11 (兩個維度: 10, 20)
//         // Instance("105. CEC2022 F11: Composition Function 3 (10D)", 10, vector<double>(10, -100.0), vector<double>(10, 100.0), CEC2022_F11),
//         // Instance("106. CEC2022 F11: Composition Function 3 (20D)", 20, vector<double>(20, -100.0), vector<double>(20, 100.0), CEC2022_F11),

//         // // CEC2022 測試函數 - F12 (兩個維度: 10, 20)
//         // Instance("107. CEC2022 F12: Composition Function 4 (10D)", 10, vector<double>(10, -100.0), vector<double>(10, 100.0), CEC2022_F12),
//         // Instance("108. CEC2022 F12: Composition Function 4 (20D)", 20, vector<double>(20, -100.0), vector<double>(20, 100.0), CEC2022_F12),
//     };
// }

#endif /* BENCHMARK_H */