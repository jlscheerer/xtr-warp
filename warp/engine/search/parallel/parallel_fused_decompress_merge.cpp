#include <torch/extension.h>

#include <algorithm>
#include <numeric>
#include <vector>
#include <variant>

#include <ATen/Parallel.h>

#include "task_graph.hpp"
#include "../annotated_stride_view.hpp"

constexpr int dim = 128;
constexpr int max_num_tokens = 32;

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

enum class reduction_type {
    kMaxReduce, kSumReduce
};

template<int8_t nbits>
float inline __attribute__((always_inline)) decompression_kernel(
    const uint8_t *__restrict residual,
    const float *__restrict bucket_scores) {
    static_assert(nbits == 2 || nbits == 4);
    constexpr int packed_vals_per_byte = 8 / nbits;
    constexpr int packed_dim = dim / packed_vals_per_byte;
    constexpr uint8_t bucket_dim_shift = nbits;

    float score = 0;
    for (int packed_idx = 0; packed_idx < packed_dim; ++packed_idx) {
        const uint8_t packed_val = residual[packed_idx];
        if constexpr (nbits == 2) {
            // TODO(jlscheerer) Double-check that this code is still correct.
            const uint8_t unpacked_0 = (packed_val & 0xC0) >> 6;
            const uint8_t unpacked_1 = (packed_val & 0x30) >> 4;
            const uint8_t unpacked_2 = (packed_val & 0x0C) >> 2;
            const uint8_t unpacked_3 = (packed_val & 0x03);

            // NOTE These correspond to an index into the "dimension"
            const int unpacked_idx_0 = packed_idx << 2;
            const int unpacked_idx_1 = unpacked_idx_0 + 1;
            const int unpacked_idx_2 = unpacked_idx_0 + 2;
            const int unpacked_idx_3 = unpacked_idx_0 + 3;

            // NOTE Constrcut the index into the "per dimension" lookup tables
            const int idx_0 = (unpacked_idx_0 << bucket_dim_shift) | unpacked_0;
            const int idx_1 = (unpacked_idx_1 << bucket_dim_shift) | unpacked_1;
            const int idx_2 = (unpacked_idx_2 << bucket_dim_shift) | unpacked_2;
            const int idx_3 = (unpacked_idx_3 << bucket_dim_shift) | unpacked_3;

            const float score_0 = bucket_scores[idx_0];
            const float score_1 = bucket_scores[idx_1];
            const float score_2 = bucket_scores[idx_2];
            const float score_3 = bucket_scores[idx_3];

            score += score_0 + score_1 + score_2 + score_3;
        } else if constexpr (nbits == 4) {
            const uint8_t unpacked_0 = packed_val >> 4;
            const uint8_t unpacked_1 = packed_val & 0x0F;

            const int unpacked_idx_0 = packed_idx << 1;
            const int base_idx = unpacked_idx_0 << bucket_dim_shift;

            score += bucket_scores[base_idx | unpacked_0] +
                     bucket_scores[(base_idx | unpacked_1) | (1 << bucket_dim_shift)];
        }
    }
    return score;
}

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

struct decompression_context {
    // int cell_idx,
    const int64_t *begins_ptr;
    const int64_t *candidate_capacities_ptr;
    const float *centroids_ptr;
    const float *vt_bucket_scores_ptr;
    const int32_t *codes_ptr;
    const uint8_t *residuals_ptr;
    const int nprobe;
    std::vector<annotated_stride_view<>> *views; // = data
};

struct merge_context {
    int nprobe;
    std::vector<annotated_stride_view<>> *data, *buffer;
    std::array<float, max_num_tokens + 1> *mse_prefix;
};

struct fused_decompression_merge_context {
    decompression_context decompression;
    merge_context merge;
};

struct merge_task {
    using context_type = merge_context;

    reduction_type type;
    int begin_or_stepsize, lhs, rhs;

    static void max_reduce_stride(const int begin, const int lhs, const int rhs,
                                  std::vector<annotated_stride_view<>> * __restrict data,
                                  std::vector<annotated_stride_view<>> * __restrict buffer) {
        reduce_max_combiner combiner;
        annotated_stride_view<> &lhs_data = (*data)[begin + lhs];
        annotated_stride_view<> &rhs_data = (*data)[begin + rhs];
        annotated_stride_view<> &lhs_buffer = (*buffer)[begin + lhs];
        merge_candidate_strides<>(lhs_data, rhs_data, lhs_buffer, combiner);
        std::swap(lhs_data, lhs_buffer); // "promote" the result.
    }

    static void execute_task(const context_type &context, const merge_task &task) {
        if (task.type == reduction_type::kMaxReduce) {
            merge_task::max_reduce_stride(task.begin_or_stepsize, task.lhs, task.rhs, context.data, context.buffer);
        } else if (task.type == reduction_type::kSumReduce) {
            const int step_size = task.begin_or_stepsize;
            const float lhs_mse = (*context.mse_prefix)[task.rhs] - (*context.mse_prefix)[task.lhs];
            const float rhs_mse = (*context.mse_prefix)[std::min(task.rhs + step_size, max_num_tokens)] - (*context.mse_prefix)[task.rhs];

            reduce_sum_mse_combiner combiner(lhs_mse, rhs_mse);
            annotated_stride_view<> &lhs_data = (*context.data)[task.lhs * context.nprobe];
            annotated_stride_view<> &rhs_data = (*context.data)[task.rhs * context.nprobe];
            annotated_stride_view<> &lhs_buffer = (*context.buffer)[task.lhs * context.nprobe];
            merge_candidate_strides<>(lhs_data, rhs_data, lhs_buffer, combiner);
            std::swap(lhs_data, lhs_buffer); // "promote" the result.
        } else {
            __builtin_unreachable();
        }
    }

    friend std::ostream &operator<<(std::ostream &os, const merge_task &task) {
        os << "lhs=" << task.lhs << " rhs=" << task.rhs;
        return os;
    }
};

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

template<uint8_t nbits>
void decompress_centroid_stride(const decompression_context &context,
                                int cell_idx) {
    constexpr int packed_vals_per_byte = 8 / nbits;
    constexpr int packed_dim = dim / packed_vals_per_byte;
    constexpr uint8_t bucket_dim_shift = nbits;

    // NOTE This is equivalent to log2(packed_dim) as packed_dim is a power of 2.
    constexpr uint8_t packed_dim_shift = __builtin_ctz(packed_dim);
    constexpr int bucket_score_offset = 128 * (1 << nbits);

    const int begin = context.begins_ptr[cell_idx];
    const int n = context.candidate_capacities_ptr[cell_idx];

    const float centroid_score = context.centroids_ptr[cell_idx];
    const float *bucket_scores_ptr = context.vt_bucket_scores_ptr + (
        (cell_idx / context.nprobe) * bucket_score_offset
    );

    const auto view = (*context.views)[cell_idx];
    int32_t pos = -1, prev_pid = -1; float prev_score = 0;
    for (int inner_idx = 0; inner_idx < n; ++inner_idx) {
        const int32_t pid = context.codes_ptr[begin + inner_idx];
        const uint8_t *residual = context.residuals_ptr + (
            static_cast<int64_t>(begin + inner_idx) << packed_dim_shift
        );
        const float score = centroid_score + decompression_kernel<nbits>(
            residual, bucket_scores_ptr
        );
        // NOTE directly perform deduplication/max-reduction within the cluster.
        if (prev_pid != pid || score > prev_score) {
            pos += (prev_pid != pid);
            view.keys_[pos] = pid;
            view.data_[pos] = score;
            prev_pid = pid;
            prev_score = score;
        }
    }
    *view.size_ = pos + (prev_pid != -1);
}

template<uint8_t nbits>
struct decompression_task {
    using context_type = decompression_context;

    int cell_idx;

    static void execute_task(const context_type &context, const decompression_task &task) {
        decompress_centroid_stride<nbits>(context, task.cell_idx);
    }


  friend std::ostream &operator<<(std::ostream &os,
                                  const decompression_task<nbits> &task) {
    os << "cell_idx=" << task.cell_idx;
    return os;
  }
};

template<uint8_t nbits>
struct fused_decompression_merge_task {
    using context_type = fused_decompression_merge_context;

    std::variant<decompression_task<nbits>, merge_task> fused_task;

    static void execute_task(const context_type &context, const fused_decompression_merge_task<nbits> &task) {
        if (std::holds_alternative<decompression_task<nbits>>(task.fused_task)) {
            decompression_task<nbits>::execute_task(context.decompression, std::get<decompression_task<nbits>>(task.fused_task));
        } else if (std::holds_alternative<merge_task>(task.fused_task)) {
            merge_task::execute_task(context.merge, std::get<merge_task>(task.fused_task));
        } else {
            __builtin_unreachable();
        }
    }

    friend std::ostream &operator<<(std::ostream &os,
                                    const fused_decompression_merge_task<nbits> &task) {
        if (std::holds_alternative<decompression_task<nbits>>(task.fused_task)) {
             os << "[decompression] " << std::get<decompression_task<nbits>>(task.fused_task);
        } else if (std::holds_alternative<merge_task>(task.fused_task)) {
            os << "[merge] " << std::get<merge_task>(task.fused_task);
        } else {
            __builtin_unreachable();
        }
        return os;
    }
};

// Input: begins, ends, capacities, centroid_scores, self.codes_compacted, residuals_compacted, bucket_weights, Q, nprobe
// Returned: capacities, candidate_sizes, candidate_pids, candidate_scores

// Input: capacities, candidate_sizes, candidate_pids, candidate_scores, mse_estimates, k
template<int8_t nbits>
std::tuple<torch::Tensor, torch::Tensor> parallel_fused_decompress_merge(
        const torch::Tensor begins,
        const torch::Tensor ends,
        const torch::Tensor candidate_capacities,
        const torch::Tensor centroid_scores,
        const torch::Tensor codes_compacted,
        const torch::Tensor residuals_compacted,
        const torch::Tensor bucket_weights,
        const torch::Tensor Q,
        const int nprobe,
        const int32_t num_query_tokens,
        const torch::Tensor mse_estimates,
        const int k) {
    using warp::task_graph;
    using warp::task_ref;

    torch::NoGradGuard no_grad;
    torch::InferenceMode guard;

    const int num_threads = at::get_num_threads();

    // Initialization for Decompression
    static_assert(nbits == 2 || nbits == 4);

    const int ncells = begins.size(0);
    const int64_t *begins_ptr = begins.data_ptr<int64_t>();
    const int64_t *candidate_capacities_ptr = candidate_capacities.data_ptr<int64_t>();
    const float *centroids_ptr = centroid_scores.data_ptr<float>();

    const int32_t *codes_ptr = codes_compacted.data_ptr<int32_t>();
    const uint8_t *residuals_ptr = residuals_compacted.data_ptr<uint8_t>();

    const int64_t numel = candidate_capacities.sum().item<int64_t>();
    torch::Tensor candidate_sizes = torch::zeros({ncells}, torch::kInt32);
    torch::Tensor candidate_pids_strided = torch::zeros({numel}, torch::kInt32);
    torch::Tensor candidate_scores_strided = torch::zeros({numel}, torch::kFloat32);

    std::vector<annotated_stride_view<>> views = strided_view(
        candidate_capacities, candidate_sizes, candidate_pids_strided, candidate_scores_strided
    );

    // NOTE Perform a single matrix-vector multiplication for the entire decompression.
    const auto vt_bucket_scores = torch::matmul(
        Q.unsqueeze(2), bucket_weights.unsqueeze(0)
    );
    const float *vt_bucket_scores_ptr = vt_bucket_scores.data_ptr<float>();

    const int num_cells = candidate_capacities.size(0);
    const int num_candidates = candidate_pids_strided.size(0);

    // Local buffers used for merging.
    torch::Tensor size_buffer = torch::zeros({num_cells}, torch::kInt32);
    torch::Tensor pid_buffer = torch::zeros({num_candidates}, torch::kInt32);
    torch::Tensor score_buffer = torch::zeros({num_candidates}, torch::kFloat32);

    // NOTE this scheme guarantees non-overlapping partitions
    std::vector<annotated_stride_view<>> views_buffer = strided_view(
        candidate_capacities, size_buffer, pid_buffer, score_buffer
    );

    const float *mse_estimates_ptr = mse_estimates.data_ptr<float>();
    std::array<float, max_num_tokens + 1> mse_prefix;
    mse_prefix[0] = 0;
    for (int i = 0; i < max_num_tokens; ++i) {
        mse_prefix[i + 1] = mse_prefix[i] + mse_estimates_ptr[i];
    }

    decompression_context dcontext {
        .begins_ptr = begins_ptr,
        .candidate_capacities_ptr = candidate_capacities_ptr,
        .centroids_ptr = centroids_ptr,
        .vt_bucket_scores_ptr = vt_bucket_scores_ptr,
        .codes_ptr = codes_ptr,
        .residuals_ptr = residuals_ptr,
        .nprobe = nprobe,
        .views = &views
    };
    merge_context mcontext = {
        .nprobe = nprobe,
        .data = &views,
        .buffer = &views_buffer,
        .mse_prefix = &mse_prefix
    };
    fused_decompression_merge_context context {
        .decompression = std::move(dcontext),
        .merge = std::move(mcontext)
    };

#if 0
    task_graph<merge_task> graph(
        std::move(context), num_query_tokens * (2 * nprobe - 1) + (2 * num_query_tokens - 1)
    );
#endif
    task_graph<fused_decompression_merge_task<nbits>> graph(std::move(context), 
        nprobe * num_query_tokens // decompression
        + num_query_tokens * (2 * nprobe - 1) // merge (token-level)
        + (2 * num_query_tokens - 1) // merge (document-level)
    );

    // Queue Decompression Tasks
    std::vector<task_ref> cell_task_map(num_query_tokens * nprobe , -1);
    for (int cell_idx = 0; cell_idx < num_query_tokens * nprobe; ++cell_idx) {
        task_ref task_id = graph.add({decompression_task<nbits>{
            .cell_idx = cell_idx
        }});
        cell_task_map[cell_idx] = task_id;
    }

    std::vector<task_ref> token_task_map(num_query_tokens, -1);
    std::vector<task_ref> probe_task_map(nprobe);

    // Queue the Token-Level Merge Tasks: Add tasks for reducing the nprobe token score strides per query token.
    for (int query_token_idx = 0; query_token_idx < num_query_tokens; ++query_token_idx) {
        std::fill(probe_task_map.begin(), probe_task_map.end(), -1);
        const int begin = query_token_idx * nprobe;
        for (int step_size = 1; step_size < nprobe; step_size <<= 1) {
            for (int lhs = 0; lhs < nprobe; lhs += (step_size << 1)) {
                if (lhs + step_size < nprobe) {
                    const int rhs = lhs + step_size;

                    task_ref task = graph.add({merge_task{
                        .type = reduction_type::kMaxReduce,
                        .begin_or_stepsize = begin,
                        .lhs = lhs,
                        .rhs = rhs
                    }});

                    const int pred1 = probe_task_map[lhs];
                    if (pred1 != -1) {
                        graph.mark_successor(pred1, task);
                    } else {
                        const int cell_idx = begin + lhs;
                        graph.mark_successor(cell_task_map[cell_idx], task);
                    }
                    const int pred2 = probe_task_map[rhs];
                    if (pred2 != -1) {
                        graph.mark_successor(pred2, task);
                    } else {
                        const int cell_idx = begin + rhs;
                        graph.mark_successor(cell_task_map[cell_idx], task);
                    }

                    probe_task_map[lhs] = task;
                }
            }
        }
        // Mark "root" of the probe reduction as the start of the token reduction.
        token_task_map[query_token_idx] = probe_task_map[0];
    }

    // Queue Document-Level Merge Tasks: Add the token-level to document-level reduction steps
    for (int step_size = 1; step_size < num_query_tokens; step_size <<= 1) {
        for (int lhs = 0; lhs < num_query_tokens; lhs += (step_size << 1)) {
            if (lhs + step_size < num_query_tokens) {
                const int rhs = lhs + step_size;
                task_ref task = graph.add({merge_task{
                    .type = reduction_type::kSumReduce,
                    .begin_or_stepsize = step_size,
                    .lhs = lhs,
                    .rhs = rhs
                }});
                
                const int pred1 = token_task_map[lhs];
                graph.mark_successor(pred1, task);
                
                const int pred2 = token_task_map[rhs];
                graph.mark_successor(pred2, task);

                token_task_map[lhs] = task;
            }
        }
    }
    
    graph.run_all_tasks(num_threads);

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
  m.def("parallel_fused_decompress_merge_2_cpp", &parallel_fused_decompress_merge<2>, "Decompress Centroid Embeddings (nbits=2)");
  m.def("parallel_fused_decompress_merge_4_cpp", &parallel_fused_decompress_merge<4>, "Decompress Centroid Embeddings (nbits=4)");
}