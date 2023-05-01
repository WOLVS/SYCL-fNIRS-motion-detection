// main.cpp
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <CL/sycl.hpp>
#include "DecisionNode.h"

using namespace cl::sycl;

std::vector<DataPoint> read_csv(const std::string& file_path) {
    std::ifstream file(file_path);
    std::vector<DataPoint> dataset;
    std::string line;

    if (file.is_open()) {
        while (std::getline(file, line)) {
            std::stringstream line_stream(line);
            std::string cell;
            DataPoint data;
            for (size_t i = 0; i < 4; ++i) {
                std::getline(line_stream, cell, ',');
                data.features.push_back(std::stoi(cell));
            }
            std::getline(line_stream, cell, ',');
            data.label = std::stoi(cell);
            dataset.push_back(data);
        }
        file.close();
    } else {
        std::cerr << "Unable to open file: " << file_path << std::endl;
    }

    return dataset;
}

std::vector<std::vector<int>> init_confusion_matrix(int num_classes) {
    std::vector<std::vector<int>> confusion_matrix(num_classes, std::vector<int>(num_classes, 0));
    return confusion_matrix;
}

void update_confusion_matrix(std::vector<std::vector<int>>& confusion_matrix, int actual, int predicted) {
    confusion_matrix[actual][predicted]++;
}

// Forward declare the kernel name
namespace kernels {
    class simple_sycl_kernel;
}

int main() {
    try {
        // Use sycl::cpu_selector to select a CPU device explicitly
        sycl::gpu_selector selector;
        queue q(selector);
        std::cout << "Running on "
                  << q.get_device().get_info<info::device::name>()
                  << std::endl;

        // A simple SYCL kernel example
        constexpr size_t dataSize = 1024;
        std::vector<int> data(dataSize, 0);

        buffer<int, 1> dataBuffer(data.data(), range<1>(dataSize));

        q.submit([&](handler &cgh) {
            auto accessor = dataBuffer.get_access<access::mode::read_write>(cgh);

            cgh.parallel_for<kernels::simple_sycl_kernel>(
                range<1>(dataSize), [=](id<1> idx) {
                    accessor[idx] = idx[0];
                });
        });

        // Decision tree example
        std::vector<DataPoint> dataset = read_csv("/home/yunyi/Desktop/zyy-workspace/Dataset/balanced_reduced_dataset.csv");

        // Start training timer
        auto start_train_time = std::chrono::high_resolution_clock::now();

        DecisionNode tree(dataset);

        // Stop training timer
        auto end_train_time = std::chrono::high_resolution_clock::now();

        // Initialize confusion matrix
        int num_classes = 2; // Assuming a binary classification problem
        std::vector<std::vector<int>> confusion_matrix = init_confusion_matrix(num_classes);

        // Start prediction timer
        auto start_pred_time = std::chrono::high_resolution_clock::now();

        for (const auto& data : dataset) {
            int prediction = tree.predict(data.features);
            update_confusion_matrix(confusion_matrix, data.label, prediction);
        }

        // Stop prediction timer
        auto end_pred_time = std::chrono::high_resolution_clock::now();

        // Calculate durations
        auto train_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_train_time - start_train_time);
        auto pred_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_pred_time - start_pred_time);
        auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_pred_time - start_train_time);


        // Display running times
        std::cout << "Training time: " << train_duration.count() << " ms" << std::endl;
        std::cout << "Prediction time: " << pred_duration.count() << " ms" << std::endl;
        std::cout << "Total running time: " << total_duration.count() << " ms" << std::endl;

        // Display confusion matrix
        std::cout << "Confusion matrix:" << std::endl;
        for (const auto& row : confusion_matrix) {
        for (const auto& value : row) {
        std::cout << value << ' ';
        }
        std::cout << std::endl;
        }
        } catch (const exception &e) {
        std::cerr << "SYCL exception caught: " << e.what() << std::endl;
        return 1;
        }
    return 0;
}

