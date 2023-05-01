// DecisionNode.cpp
#include "DecisionNode.h"
#include <cstddef>
#include <map>
#include <limits>

DecisionNode::DecisionNode(std::vector<DataPoint> dataset) : dataset_(dataset) {
    split_node(0, 4); // Set maximum tree depth to 4 as an example
}

// Add the destructor implementation here
DecisionNode::~DecisionNode() {
    if (!is_leaf_) {
        delete left_child_;
        delete right_child_;
    }
}

int DecisionNode::predict(const std::vector<int>& features) {
    if (is_leaf_) {
        return class_label_;
    }

    if (features[feature_index_] <= threshold_) {
        return left_child_->predict(features);
    } else {
        return right_child_->predict(features);
    }
}

void DecisionNode::split_node(int depth, int max_depth) {
    if (depth >= max_depth || dataset_.size() <= 1) {
        is_leaf_ = true;
        std::map<int, int> label_counts;
        for (const auto& data : dataset_) {
            label_counts[data.label]++;
        }

        int max_count = 0;
        for (const auto& label_count : label_counts) {
            if (label_count.second > max_count) {
                max_count = label_count.second;
                class_label_ = label_count.first;
            }
        }
        return;
    }

    double best_gini = std::numeric_limits<double>::max();
    int best_feature_index = -1;
    int best_threshold = -1;
    std::vector<DataPoint> best_left, best_right;

    for (size_t feature_index = 0; feature_index < 4; ++feature_index) {
        for (const auto& data : dataset_) {
            std::vector<DataPoint> left, right;
            for (const auto& other_data : dataset_) {
                if (other_data.features[feature_index] <= data.features[feature_index]) {
                    left.push_back(other_data);
                } else {
                    right.push_back(other_data);
                }
            }

            double gini = calculate_weighted_gini_index(left, right);
            if (gini < best_gini && !left.empty() && !right.empty()) {
                best_gini = gini;
                best_feature_index = feature_index;
                best_threshold = data.features[feature_index];
                best_left = left;
                best_right = right;
            }
        }
    }

    if (best_gini == std::numeric_limits<double>::max()) {
        // This is the new recursive call, making it safer by reducing max_depth
        split_node(depth, max_depth - 1);
    } else {
        is_leaf_ = false;
        feature_index_ = best_feature_index;
        threshold_ = best_threshold;
        left_child_ = new DecisionNode(best_left);
        right_child_ = new DecisionNode(best_right);
    }
}



double DecisionNode::calculate_gini_index(const std::vector<DataPoint>& dataset) {
    if (dataset.empty()) {
        return 0;
    }

    std::map<int, int> label_counts;
    for (const auto& data : dataset) {
        label_counts[data.label]++;
    }

    double gini = 1.0;
    for (const auto& label_count : label_counts) {
        double p = static_cast<double>(label_count.second) / dataset.size();
        gini -= p * p;
    }

    return gini;
}

double DecisionNode::calculate_weighted_gini_index(const std::vector<DataPoint>& left, const std::vector<DataPoint>& right) {
    double left_gini = calculate_gini_index(left);
    double right_gini = calculate_gini_index(right);

   
    double left_weight = static_cast<double>(left.size()) / (left.size() + right.size());
    double right_weight = static_cast<double>(right.size()) / (left.size() + right.size());

    return left_weight * left_gini + right_weight * right_gini;
}
