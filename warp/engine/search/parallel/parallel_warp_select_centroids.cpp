#include <torch/extension.h>

#include <algorithm>
#include <numeric>
#include <thread>
#include <vector>

#include <ATen/Parallel.h>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
parallel_warp_select_centroids(torch::Tensor centroid_idx,
                               const torch::Tensor Q_mask,
                               const torch::Tensor centroid_scores,
                               const torch::Tensor sizes_compacted,
                               const int64_t nprobe,
                               const int64_t t_prime,
                               const int64_t bound) {
    torch::NoGradGuard no_grad;
    torch::InferenceMode guard;

    const int num_query_tokens = Q_mask.sum().item<int64_t>();
    const int64_t num_centroids = centroid_scores.size(1);

    // NOTE we know num_centroids is a power of two so we don't need multiplications.
    assert(__builtin_popcount(num_centroids) == 1);
    const uint8_t num_centroids_shift = __builtin_ctz(num_centroids);

    int32_t *centroid_idx_ptr = centroid_idx.data_ptr<int32_t>();
    const float *centroid_scores_ptr = centroid_scores.data_ptr<float>();
    const int64_t *sizes_ptr = sizes_compacted.data_ptr<int64_t>();

    torch::Tensor cells = torch::zeros({32, nprobe}, torch::kInt32);
    int32_t *cells_ptr = cells.data_ptr<int32_t>();

    torch::Tensor scores = torch::zeros({32, nprobe}, torch::kFloat32);
    float *scores_ptr = scores.data_ptr<float>();

    torch::Tensor mse = torch::zeros({32}, torch::kFloat32);
    float *mse_ptr = mse.data_ptr<float>();

    at::parallel_for(0, num_query_tokens, 1, [&](int64_t begin, int64_t end) {
        const int32_t thread_id = at::get_thread_num();
        int32_t *thread_centroid_idx = centroid_idx_ptr + (thread_id << num_centroids_shift);
        for (int i = begin; i < end; ++i) {
            const int qidx_offset = (i * nprobe);
            int32_t *cells_qidx_ptr = cells_ptr + qidx_offset;
            float *scores_qidx_ptr = scores_ptr + qidx_offset;

            // NOTE num_centroids is a power of two so we can just shift.
            const float *cscores_qidx_ptr = centroid_scores_ptr + (i << num_centroids_shift);

            auto sort_fn = [cscores_qidx_ptr](const int i1, const int i2) {
                if (cscores_qidx_ptr[i1] != cscores_qidx_ptr[i2]) {
                    return cscores_qidx_ptr[i1] > cscores_qidx_ptr[i2];
                }
                return i1 < i2;
            };

            // NOTE Identify the largest scoring nprobe centroids for decompression.
            std::partial_sort(thread_centroid_idx, thread_centroid_idx + bound,
                              thread_centroid_idx + num_centroids, sort_fn);
            for (int j = 0; j < nprobe; ++j) {
                const int centroid_id = thread_centroid_idx[j];
                cells_qidx_ptr[j] = centroid_id;
                scores_qidx_ptr[j] = cscores_qidx_ptr[centroid_id];
            }
            
            // NOTE Identify mse centroid scores by accumulating sizes until t_prime.
            int32_t cumsum = 0, idx = 0;
            while (cumsum < t_prime) {
                cumsum += sizes_ptr[thread_centroid_idx[idx++]];
                // NOTE We could also just break here.
                //      This would use the last (available) centroid score.
                assert(!(cumsum < t_prime && idx >= bound));
            }
            mse_ptr[i] = cscores_qidx_ptr[thread_centroid_idx[idx - 1]];
        }
    });

    return {cells, scores, mse};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("parallel_warp_select_centroids_cpp", &parallel_warp_select_centroids,
        "WARP Select Centroids");
}
