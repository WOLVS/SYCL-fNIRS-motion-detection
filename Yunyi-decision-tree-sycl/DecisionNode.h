// DecisionNode.h
#pragma once
#include <vector>
#include "Data.h"

class DecisionNode {
public:
    DecisionNode(std::vector<DataPoint> dataset);
    ~DecisionNode(); // Add this line
    int predict(const std::vector<float>& features);

private:
    std::vector<DataPoint> dataset_;
    bool is_leaf_;
    int feature_index_;
    int threshold_;
    int class_label_;
    DecisionNode* left_child_;
    DecisionNode* right_child_;

    void split_node(int depth, int max_depth);
    double calculate_gini_index(const std::vector<DataPoint>& dataset);
    double calculate_weighted_gini_index(const std::vector<DataPoint>& left, const std::vector<DataPoint>& right);
};
