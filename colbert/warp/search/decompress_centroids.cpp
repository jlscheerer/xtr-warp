#include <pthread.h>
#include <torch/extension.h>

#include <algorithm>
#include <numeric>
#include <vector>

#include <immintrin.h>

constexpr int dim = 128;
constexpr uint8_t nbits = NBITS;

/*
decompression_table refers to "decomp2", i.e., a compacted version of the
original. Also the decompression_table will be flattened to prevent memory
layout issues.

decomp2 = decompression_lookup_table[reversed_bit_map.long()]
decomp2 = decomp2.flatten()

THE SHAPE SHOULD BE 256 * packed_vals_per_byte!
*/

/*
bucket_scores = torch.matmul(Q.unsqueeze(1),
bucket_weights.unsqueeze(0)).flatten()
*/

template<int8_t nbits>
float inline __attribute__((always_inline)) decompression_kernel(
    const torch::TensorAccessor<uint8_t, 1> &residual,
    const torch::TensorAccessor<float, 2> &bucket_scores) {
    static_assert(nbits == 2 || nbits == 4);
    constexpr int packed_vals_per_byte = 8 / nbits;
    constexpr int packed_dim = dim / packed_vals_per_byte;
    float score = 0;
    for (int packed_idx = 0; packed_idx < packed_dim; ++packed_idx) {
        const uint8_t packed_val = residual[packed_idx];
        if constexpr (nbits == 2) {
            const uint8_t unpacked_0 = (packed_val & 0xC0) >> 6;
            const uint8_t unpacked_1 = (packed_val & 0x30) >> 4;
            const uint8_t unpacked_2 = (packed_val & 0x0C) >> 2;
            const uint8_t unpacked_3 = (packed_val & 0x03);

            const int unpacked_idx_0 = 4 * packed_idx;
            const int unpacked_idx_1 = unpacked_idx_0 + 1;
            const int unpacked_idx_2 = unpacked_idx_0 + 2;
            const int unpacked_idx_3 = unpacked_idx_0 + 3;

            const float score_0 = bucket_scores[unpacked_idx_0][unpacked_0];
            const float score_1 = bucket_scores[unpacked_idx_1][unpacked_1];
            const float score_2 = bucket_scores[unpacked_idx_2][unpacked_2];
            const float score_3 = bucket_scores[unpacked_idx_3][unpacked_3];

            score += score_0 + score_1 + score_2 + score_3;
        } else if constexpr (nbits == 4) {
            const uint8_t unpacked_0 = (packed_val & 0xF0) >> 4;
            const uint8_t unpacked_1 = (packed_val & 0x0F);

            const int unpacked_idx_0 = 2 * packed_idx;
            const int unpacked_idx_1 = unpacked_idx_0 + 1;

            const float score_0 = bucket_scores[unpacked_idx_0][unpacked_0];
            const float score_1 = bucket_scores[unpacked_idx_1][unpacked_1];
            score += score_0 + score_1;
        }
    }
    return score;
}

torch::Tensor decompress_centroids(
    const torch::Tensor begins, const torch::Tensor ends,
    const torch::Tensor sizes, const torch::Tensor centroid_scores,
    const torch::Tensor residuals_compacted, const torch::Tensor bucket_weights,
    const torch::Tensor Q, const int nprobe) {
  torch::NoGradGuard no_grad;
  torch::InferenceMode guard;

  const int ncells = begins.size(0);
  const auto begins_accessor = begins.accessor<int64_t, 1>();
  const auto sizes_accessor = sizes.accessor<int64_t, 1>();
  const auto centroids_accessor = centroid_scores.accessor<float, 1>();

  const int64_t numel = sizes.sum().item<int64_t>();

  std::vector<int64_t> offsets(ncells);
  const int64_t *sizes_ptr = sizes.data_ptr<int64_t>();
  std::partial_sum(sizes_ptr, sizes_ptr + ncells - 1, offsets.begin() + 1);

  const auto residuals_accessor = residuals_compacted.accessor<uint8_t, 2>();

  torch::Tensor results = torch::zeros({numel}, torch::kFloat32);
  auto results_accessor = results.accessor<float, 1>();
  at::parallel_for(
      0, ncells, /*grain_size=*/0, [&](int64_t ncell_begin, int64_t ncell_end) {
        for (int cell_idx = ncell_begin; cell_idx < ncell_end; ++cell_idx) {
          const int begin = begins_accessor[cell_idx];
          const int n = sizes_accessor[cell_idx];
          const int64_t roffset = offsets[cell_idx];
          const float centroid_score = centroids_accessor[cell_idx];

          const auto bucket_scores = torch::matmul(
              Q[cell_idx / nprobe].unsqueeze(1), bucket_weights.unsqueeze(0));
          const auto bucket_scores_accessor =
              bucket_scores.accessor<float, 2>();
          for (int residual_idx = 0; residual_idx < n; ++residual_idx) {
            const auto &residual = residuals_accessor[begin + residual_idx];
            const float score = decompression_kernel<nbits>(residual, bucket_scores_accessor);
            results_accessor[roffset + residual_idx] = centroid_score + score;
          }
        }
      });

  return results;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("decompress_centroids_cpp", &decompress_centroids, "Decompress Centroid Embeddings");
}
