CXX := g++
CXXSTD ?= -std=c++17
OPTFLAGS ?= -O3 -march=native -flto=auto -funroll-loops
SANFLAGS ?=
LDFLAGS ?= -lm -pthread $(SANFLAGS)

INCLUDES := -I./src -I./benchmarks/include -I./src/config -I./src/utils -I./src/metaheuristic -I./src/hyper

CEC14_SRC := benchmarks/src/cec14_test_func.cpp
CEC14_Train_SRC := benchmarks/src/cec14_test_func_fortrain.cpp
CEC20_SRC := benchmarks/src/cec20_test_func.cpp
CEC20_Train_SRC := benchmarks/src/cec20_test_func_fortrain.cpp
CEC22_SRC := benchmarks/src/cec22_test_func.cpp
CEC22_Train_SRC := benchmarks/src/cec22_test_func_fortrain.cpp
CEC24_SRC := benchmarks/src/cec17_test_func.cpp
CEC24_Train_SRC := benchmarks/src/cec17_test_func_fortrain.cpp

# 同時編譯 CEC2014, CEC2020, CEC2022 和 CEC2024
ALL_BENCH_SRC := $(CEC14_SRC) $(CEC14_Train_SRC) $(CEC20_SRC) $(CEC20_Train_SRC) $(CEC22_SRC) $(CEC22_Train_SRC) $(CEC24_SRC) $(CEC24_Train_SRC)

BIN_DIR := bin
$(shell mkdir -p $(BIN_DIR))

TARGET_MAIN := $(BIN_DIR)/main
TARGET_TRAIN := $(BIN_DIR)/train_SEHH
TARGET_EXPORT := $(BIN_DIR)/export_config

APP_MAIN := src/apps/main.cpp
APP_TRAIN := src/apps/train_SEHH.cpp
APP_EXPORT := src/apps/export_config.cpp

RELEASEFLAGS ?=

.PHONY: all debug release pgo-gen pgo-use clean asan ubsan sanitize main_cpp

all: release

main_cpp: $(APP_MAIN) $(APP_EXPORT)
	$(CXX) $(CXXSTD) $(OPTFLAGS) $(RELEASEFLAGS) $(INCLUDES) $(SANFLAGS) -o $(TARGET_MAIN) $(APP_MAIN) $(ALL_BENCH_SRC) $(LDFLAGS)
	$(CXX) $(CXXSTD) $(OPTFLAGS) $(RELEASEFLAGS) $(INCLUDES) $(SANFLAGS) -o $(TARGET_EXPORT) $(APP_EXPORT) $(ALL_BENCH_SRC) $(LDFLAGS)

release: $(APP_MAIN) $(APP_TRAIN) $(APP_EXPORT)
	$(CXX) $(CXXSTD) $(OPTFLAGS) $(RELEASEFLAGS) $(INCLUDES) $(SANFLAGS) -o $(TARGET_MAIN) $(APP_MAIN) $(ALL_BENCH_SRC) $(LDFLAGS)
	$(CXX) $(CXXSTD) $(OPTFLAGS) $(RELEASEFLAGS) $(INCLUDES) $(SANFLAGS) -o $(TARGET_TRAIN) $(APP_TRAIN) $(ALL_BENCH_SRC) $(LDFLAGS)
	$(CXX) $(CXXSTD) $(OPTFLAGS) $(RELEASEFLAGS) $(INCLUDES) $(SANFLAGS) -o $(TARGET_EXPORT) $(APP_EXPORT) $(ALL_BENCH_SRC) $(LDFLAGS)

pgo-gen: clean
	$(CXX) $(CXXSTD) $(OPTFLAGS) -fprofile-generate $(INCLUDES) $(SANFLAGS) -o $(TARGET_MAIN) $(APP_MAIN) $(ALL_BENCH_SRC) $(LDFLAGS)
	$(CXX) $(CXXSTD) $(OPTFLAGS) -fprofile-generate $(INCLUDES) $(SANFLAGS) -o $(TARGET_TRAIN) $(APP_TRAIN) $(ALL_BENCH_SRC) $(LDFLAGS)
	$(CXX) $(CXXSTD) $(OPTFLAGS) -fprofile-generate $(INCLUDES) $(SANFLAGS) -o $(TARGET_EXPORT) $(APP_EXPORT) $(ALL_BENCH_SRC) $(LDFLAGS)
	@echo "Run your representative workload to generate profiles, e.g.: ./$(TARGET_TRAIN)"

pgo-use:
	$(CXX) $(CXXSTD) $(OPTFLAGS) -fprofile-use -fprofile-correction $(INCLUDES) $(SANFLAGS) -o $(TARGET_MAIN) $(APP_MAIN) $(ALL_BENCH_SRC) $(LDFLAGS)
	$(CXX) $(CXXSTD) $(OPTFLAGS) -fprofile-use -fprofile-correction $(INCLUDES) $(SANFLAGS) -o $(TARGET_TRAIN) $(APP_TRAIN) $(ALL_BENCH_SRC) $(LDFLAGS)



clean:
	rm -rf $(BIN_DIR) main train_SEHH forpy export_config in.txt