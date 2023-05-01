// main.cpp
#include <iostream>
#include "Data.h"
#include "DecisionNode.h"

int main() {
    std::vector<DataPoint> dataset = {
        {{0, 0, 0, 0}, 0},
        {{0, 0, 0, 1}, 1},
        {{0, 0, 1, 0}, 0},
        {{0, 0, 1, 1}, 1},
        {{0, 1, 0, 0}, 0},
        {{0, 1, 0, 1}, 1},
        {{0, 1, 1, 0}, 0},
        {{0, 1, 1, 1}, 1},
        {{1, 0, 0, 0}, 0},
        {{1, 0, 0, 1}, 1},
        {{1, 0, 1, 0}, 0},
        {{1, 0, 1, 1}, 1},
        {{1, 1, 0, 0}, 0},
        {{1, 1, 0, 1}, 1},
        {{1, 1, 1, 0}, 0},
        {{1, 1, 1, 1}, 1},
    };

    DecisionNode tree(dataset);

    for (const auto& data : dataset) {
        int prediction = tree.predict(data.features);
        std::cout << "Expected: " << data.label << ", Predicted: " << prediction << std::endl;
    }

    return 0;
}
