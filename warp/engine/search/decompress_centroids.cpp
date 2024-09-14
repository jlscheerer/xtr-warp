#include <pthread.h>
#include <torch/extension.h>

#include <algorithm>
#include <numeric>
#include <vector>

#include "annotated_stride_view.hpp"

constexpr int dim = 128;

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

template<int8_t nbits>
torch_annotated_stride_view<> decompress_centroids(
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
  static_assert(nbits == 2 || nbits == 4);
  constexpr int packed_vals_per_byte = 8 / nbits;
  constexpr int packed_dim = dim / packed_vals_per_byte;
  constexpr uint8_t bucket_dim_shift = nbits;

  const int ncells = begins.size(0);
  const int64_t *begins_ptr = begins.data_ptr<int64_t>();
  const int64_t *sizes_ptr = sizes.data_ptr<int64_t>();
  const float *centroids_ptr = centroid_scores.data_ptr<float>();

  const int32_t *codes_ptr = codes_compacted.data_ptr<int32_t>();
  const uint8_t *residuals_ptr = residuals_compacted.data_ptr<uint8_t>();

  const int64_t numel = sizes.sum().item<int64_t>();
  torch::Tensor stride_sizes = torch::zeros({ncells}, torch::kInt32);
  torch::Tensor pids = torch::zeros({numel}, torch::kInt32);
  torch::Tensor scores = torch::zeros({numel}, torch::kFloat32);

  std::vector<annotated_stride_view<>> views = strided_view(
    sizes, stride_sizes, pids, scores
  );

  // NOTE Perform a single matrix-vector multiplication for the entire decompression.
  const auto vt_bucket_scores = torch::matmul(
    Q.unsqueeze(2), bucket_weights.unsqueeze(0)
  );
  const float *vt_bucket_scores_ptr = vt_bucket_scores.data_ptr<float>();
  
  // NOTE This is equivalent to log2(packed_dim) as packed_dim is a power of 2.
  constexpr uint8_t packed_dim_shift = __builtin_ctz(packed_dim);
  constexpr int bucket_score_offset = 128 * (1 << nbits);
  for (int cell_idx = 0; cell_idx < ncells; ++cell_idx) {
    const int begin = begins_ptr[cell_idx];
    const int n = sizes_ptr[cell_idx];

    const float centroid_score = centroids_ptr[cell_idx];
    const float *bucket_scores_ptr = vt_bucket_scores_ptr + (
      (cell_idx / nprobe) * bucket_score_offset
    );

    const auto view = views[cell_idx];
    int32_t pos = -1, prev_pid = -1; float prev_score = 0;
    for (int inner_idx = 0; inner_idx < n; ++inner_idx) {
      const int32_t pid = codes_ptr[begin + inner_idx];
      const uint8_t *residual = residuals_ptr + (
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

  return {stride_sizes, pids, scores};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("decompress_centroids_2_cpp", &decompress_centroids<2>, "Decompress Centroid Embeddings (nbits=2)");
  m.def("decompress_centroids_4_cpp", &decompress_centroids<4>, "Decompress Centroid Embeddings (nbits=4)");
}