#include <torch/extension.h>

#include <vector>

#include "annotated_stride_view.hpp"

// NOTE Used to max-reduce token-level of different clusters.
struct reduce_max_combiner {
    float operator()(float lhs, float rhs) const noexcept {
        return std::max(lhs, rhs);
    }
    float lhs(float lhs) const noexcept {
        return lhs;
    }
    float rhs(float rhs) const noexcept {
        return rhs;
    }
};

// NOTE Used to combine token-level scores into document-level scores.
struct reduce_sum_mse_combiner {
    reduce_sum_mse_combiner(float lhs_mse, float rhs_mse)
        : lhs_mse_(lhs_mse), rhs_mse_(rhs_mse) {}
    float operator()(float lhs, float rhs) const noexcept {
        return lhs + rhs;
    }
    float lhs(float lhs) const noexcept {
        return lhs + rhs_mse_;
    }
    float rhs(float rhs) const noexcept {
        return lhs_mse_ + rhs;
    }
private:
    float lhs_mse_, rhs_mse_;
};

template<typename combiner_type>
void merge_candidate_strides(const annotated_stride_view<> stride1,
                             const annotated_stride_view<> stride2,
                             annotated_stride_view<> result,
                             combiner_type combiner) {
    const int32_t c1_size = *stride1.size_, c2_size = *stride2.size_;
    int32_t result_size = 0, i1 = 0, i2 = 0;
    while (i1 < c1_size && i2 < c2_size) {
        const int32_t key1 = stride1.keys_[i1];
        const int32_t key2 = stride2.keys_[i2];
        result.keys_[result_size] = std::min(key1, key2);
        if (key1 == key2) {
            result.data_[result_size] = combiner(stride1.data_[i1++], stride2.data_[i2++]);
        } else if (key1 < key2) {
            result.data_[result_size] = combiner.lhs(stride1.data_[i1++]);
        } else {
            result.data_[result_size] = combiner.rhs(stride2.data_[i2++]);
        }
        ++result_size;
    }
    if (i1 < c1_size) {
        for (; i1 < c1_size; ++i1) {
            result.keys_[result_size] = stride1.keys_[i1];
            result.data_[result_size] = combiner.lhs(stride1.data_[i1]);
            ++result_size;
        }
    }
    if (i2 < c2_size) {
        for (; i2 < c2_size; ++i2) {
            result.keys_[result_size] = stride2.keys_[i2];
            result.data_[result_size] = combiner.rhs(stride2.data_[i2]);
            ++result_size;
        }
    }
    *result.size_ = result_size;
}

void copy_candidate_stride(const annotated_stride_view<> source,
                           annotated_stride_view<> destination) {
    const int32_t size = *source.size_;
    *destination.size_ = size;
    memcpy(destination.keys_, source.keys_, size * sizeof(int32_t));
    memcpy(destination.data_, source.data_, size * sizeof(int32_t));
}

// Merge the `nprobe` candidate lists associated with a specific token index.
int merge_candidates_nprobe(std::vector<annotated_stride_view<>> &views,
                            std::vector<annotated_stride_view<>> &views_buffer,
                            const int nprobe, const int query_token_idx) {
    int num_iterations = 0;
    const int begin = query_token_idx * nprobe;
    std::vector<annotated_stride_view<>> *buf1 = &views, *buf2 = &views_buffer;
    reduce_max_combiner combiner;
    for (int step_size = 1; step_size < nprobe; step_size <<= 1, ++num_iterations) {
        for (int lhs = 0; lhs < nprobe; lhs += (step_size << 1)) {
            const int rhs = lhs + step_size;
            if (rhs < nprobe) {
                merge_candidate_strides<>((*buf1)[begin + lhs], (*buf1)[begin + rhs], 
                                          (*buf2)[begin + lhs], combiner);
            } else {
                // NOTE If rhs < nprobe we don't have a merge partner for the current index.
                // In this case, move the current view to the next stage without alteration.
                copy_candidate_stride((*buf1)[begin + lhs],
                                      (*buf2)[begin + lhs]);
            }
        }
        // NOTE change which buffer is considered a "scratch" buffer.
        std::swap(buf1, buf2);
    }
    return num_iterations;
}

// Merge the 32 strides of token-level scores into a single stride of document-level scores.
void merge_candidates_tokens(std::vector<annotated_stride_view<>> &views,
                            std::vector<annotated_stride_view<>> &views_buffer,
                            const int nprobe, const float *mse_estimates) {
    constexpr int num_tokens = 32;
    std::array<float, num_tokens + 1> mse_prefix;
    mse_prefix[0] = 0;
    for (int i = 0; i < num_tokens; ++i) {
        mse_prefix[i + 1] = mse_prefix[i] + mse_estimates[i];
    }
    for (int step_size = 1; step_size < num_tokens; step_size <<= 1) {
        for (int lhs = 0; lhs < num_tokens; lhs += (step_size << 1)) {
            const int rhs = lhs + step_size;
            if (rhs < num_tokens) {
                // NOTE We can just subtract two prefix sums for the range of MSE values!
                const float lhs_mse = mse_prefix[rhs] - mse_prefix[lhs];
                const float rhs_mse = mse_prefix[std::min(rhs + step_size, num_tokens)] - mse_prefix[rhs];

                reduce_sum_mse_combiner combiner(lhs_mse, rhs_mse);
                merge_candidate_strides<>(views[lhs * nprobe], views[rhs * nprobe],
                                          views_buffer[lhs * nprobe], combiner);
            } else {
                copy_candidate_stride(views[lhs * nprobe], views_buffer[lhs * nprobe]);
            }
        }
        std::swap(views, views_buffer);
    }
}

std::vector<int> partial_sort_results(annotated_stride_view<> stride,
                          const int num_results) {
    std::vector<int> pid_idx(*stride.size_);
    std::iota(pid_idx.begin(), pid_idx.end(), 0);

    const float *scores = stride.data_;
    std::partial_sort(pid_idx.begin(), pid_idx.begin() + num_results,
                      pid_idx.end(), [scores](const int idx1, const int idx2){
        const float score1 = scores[idx1], score2 = scores[idx2];
        return (score1 > score2) || (score1 == score2 && idx1 < idx2);
    });

    return pid_idx;
}

std::tuple<torch::Tensor, torch::Tensor> merge_candidate_scores(
        const torch::Tensor candidate_capacities,
        const torch::Tensor candidate_sizes,
        const torch::Tensor candidate_pids_strided,
        const torch::Tensor candidate_scores_strided,
        const torch::Tensor mse_estimates,
        const int nprobe, const int k) {
    torch::NoGradGuard no_grad;
    torch::InferenceMode guard;
    const int num_cells = candidate_capacities.size(0);
    const int num_candidates = candidate_pids_strided.size(0);

    std::vector<annotated_stride_view<>> views = strided_view(
        candidate_capacities, candidate_sizes, 
        candidate_pids_strided, candidate_scores_strided
    );

    // Local buffers used for merging.
    torch::Tensor size_buffer = torch::zeros({num_cells}, torch::kInt32);
    torch::Tensor pid_buffer = torch::zeros({num_candidates}, torch::kInt32);
    torch::Tensor score_buffer = torch::zeros({num_candidates}, torch::kFloat32);

    // NOTE this scheme guarantees non-overlapping partitions
    std::vector<annotated_stride_view<>> views_buffer = strided_view(
        candidate_capacities, size_buffer, pid_buffer, score_buffer
    );

    int num_iterations;
    for (int query_token_idx = 0; query_token_idx < 32; ++query_token_idx) {
        // TODO(jlscheerer) Add early stopping here. In case we don't have 32 tokens!
        num_iterations = merge_candidates_nprobe(views, views_buffer, nprobe, query_token_idx);
    }
    // NOTE If we performed an odd number of iterations the scratch buffer contains the result.
    if (num_iterations % 2 != 0) {
        std::swap(views, views_buffer);
    }

    // Finally merge the results *between* different tokens.
    merge_candidates_tokens(views, views_buffer, nprobe, mse_estimates.data_ptr<float>());

    // NOTE After all merges have occured the stride at index 0 contains the resulting scores.
    const int num_results = std::min(*(views[0].size_), k);
    std::vector<int> pid_idx = partial_sort_results(views[0], num_results);

    torch::Tensor candidate_pids = torch::zeros({num_results}, torch::kInt32);
    torch::Tensor candidate_scores = torch::zeros({num_results}, torch::kFloat32);

    const int32_t *pids_ptr = views[0].keys_;
    const float *scores_ptr = views[0].data_;

    int32_t *candidate_pids_ptr = candidate_pids.data_ptr<int32_t>();
    float *candidate_scores_ptr = candidate_scores.data_ptr<float>();
    for (int i = 0; i < num_results; ++i) {
        const int idx = pid_idx[i];
        candidate_pids_ptr[i] = pids_ptr[idx];
        candidate_scores_ptr[i] = scores_ptr[idx];
    }

    return {std::move(candidate_pids), std::move(candidate_scores)};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("merge_candidate_scores_cpp", &merge_candidate_scores,
        "Merge Strided Candidate Scores");
}