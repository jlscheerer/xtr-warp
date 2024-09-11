#pragma once

#include <torch/extension.h>

#include <vector>

template<typename datatype = float>
struct annotated_stride_view {
    int32_t *size_;
    int32_t *keys_;
    datatype *data_;
};

std::vector<annotated_stride_view<>> strided_view(
        const torch::Tensor capacities, torch::Tensor sizes,
        torch::Tensor pids, torch::Tensor scores) {
    int32_t *size_ptr = sizes.data_ptr<int32_t>();
    int32_t *pids_ptr = pids.data_ptr<int32_t>();
    float *scores_ptr = scores.data_ptr<float>();

    const int num_cells = capacities.size(0);
    const auto capacities_accessor = capacities.accessor<int64_t, 1>();
    std::vector<annotated_stride_view<>> views;
    views.reserve(num_cells);
    int64_t begin = 0;
    for (int i = 0; i < num_cells; ++i) {
        const int64_t capacity = capacities_accessor[i];
        views.push_back(annotated_stride_view<>{
            .size_ = size_ptr + i, 
            .keys_ = pids_ptr + begin,
            .data_ = scores_ptr + begin
        });
        begin += capacity;
    }
    return views;
}

// [sizes, keys, data]
template<typename datatype = float>
using torch_annotated_stride_view = std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>;