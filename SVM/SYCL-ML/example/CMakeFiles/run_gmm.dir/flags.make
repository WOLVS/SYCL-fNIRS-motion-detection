# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.24

# compile CXX with /home/yunyi/Desktop/VSCode_Proj/SYCL-ML-OneAPI/ComputeCpp-CE-1.2.0-Ubuntu-16.04-x86_64/bin/compute++
CXX_DEFINES = -DEIGEN_EXCEPTIONS=1 -DEIGEN_SYCL_ASYNC_EXECUTION=1 -DEIGEN_SYCL_LOCAL_MEM=1 -DEIGEN_SYCL_USE_DEFAULT_SELECTOR=1 -DEIGEN_USE_SYCL=1

CXX_INCLUDES = -I/home/yunyi/Desktop/VSCode_Proj/New-sycl-ml/SYCL-ML/include -I/home/yunyi/Desktop/VSCode_Proj/New-sycl-ml/SYCL-ML/example/src -isystem /home/yunyi/Desktop/VSCode_Proj/New-sycl-ml/SYCL-ML/build/Eigen-src -isystem /home/yunyi/Desktop/VSCode_Proj/SYCL-ML-OneAPI/ComputeCpp-CE-1.2.0-Ubuntu-16.04-x86_64/include

CXX_FLAGS = -sycl-driver -O2 -mllvm -inline-threshold=1000 -intelspirmetadata -sycl-target spir64 -Wall -Wextra -Wpedantic -DCOMPUTECPP_DISABLE_SYCL_NAMESPACE_ALIAS -no-serial-memop -fsycl-ih-last -std=c++14
