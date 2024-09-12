#include <pthread.h>
#include <torch/extension.h>

#include <algorithm>
#include <numeric>
#include <vector>

#include "annotated_stride_view.hpp"

constexpr int dim = 128;
constexpr uint8_t nbits = NBITS;

template<int8_t nbits>
float inline __attribute__((always_inline)) decompression_kernel(
    const torch::TensorAccessor<uint8_t, 1> &residual,
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
            const uint8_t unpacked_0 = (packed_val & 0xF0) >> 4;
            const uint8_t unpacked_1 = (packed_val & 0x0F);

            // NOTE These correspond to an index into the "dimension"
            const int unpacked_idx_0 = packed_idx << 1;
            const int unpacked_idx_1 = unpacked_idx_0 + 1;

            // NOTE Constrcut the index into the "per dimension" lookup tables
            const int idx_0 = (unpacked_idx_0 << bucket_dim_shift) | unpacked_0;
            const int idx_1 = (unpacked_idx_1 << bucket_dim_shift) | unpacked_1;

            const float score_0 = bucket_scores[idx_0];
            const float score_1 = bucket_scores[idx_1];
            score += score_0 + score_1;
        }
    }
    return score;
}

torch_annotated_stride_view<> decompress_centroids_dedup(
    const torch::Tensor begins,
    const torch::Tensor ends,
    const torch::Tensor sizes,
    const torch::Tensor centroid_scores,
    const torch::Tensor codes_compacted,
    const torch::Tensor residuals_compacted,
    const torch::Tensor bucket_weights,
    const torch::Tensor Q,
    const int nprobe) {
  torch::NoGradGuard no_grad;
  torch::InferenceMode guard;

  const int ncells = begins.size(0);
  const int64_t *begins_ptr = begins.data_ptr<int64_t>();
  const int64_t *sizes_ptr = sizes.data_ptr<int64_t>();
  const float *centroids_ptr = centroid_scores.data_ptr<float>();
  const int32_t *codes_ptr = codes_compacted.data_ptr<int32_t>();

  const auto residuals_accessor = residuals_compacted.accessor<uint8_t, 2>();

  const int64_t numel = sizes.sum().item<int64_t>();
  torch::Tensor stride_sizes = torch::zeros({ncells}, torch::kInt32);
  torch::Tensor pids = torch::zeros({numel}, torch::kInt32);
  torch::Tensor scores = torch::zeros({numel}, torch::kFloat32);

  std::vector<annotated_stride_view<>> views = strided_view(
    sizes, stride_sizes, pids, scores
  );

  for (int cell_idx = 0; cell_idx < ncells; ++cell_idx) {
    const int begin = begins_ptr[cell_idx];
    const int n = sizes_ptr[cell_idx];
    const float centroid_score = centroids_ptr[cell_idx];

    // NOTE we could also just do a single multiplication independent of nprobe.
    const auto bucket_scores = torch::matmul(
        Q[cell_idx / nprobe].unsqueeze(1), bucket_weights.unsqueeze(0));
    const float *bucket_scores_ptr = bucket_scores.data_ptr<float>();

    const auto view = views[cell_idx];
    int32_t pos = -1, prev_pid = -1; float prev_score = 0;
    for (int inner_idx = 0; inner_idx < n; ++inner_idx) {
      const int32_t pid = codes_ptr[begin + inner_idx];
      const auto &residual = residuals_accessor[begin + inner_idx];

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

  return {stride_sizes, pids, scores};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("decompress_centroids_dedup_cpp", &decompress_centroids_dedup, "Decompress Centroid Embeddings");
}