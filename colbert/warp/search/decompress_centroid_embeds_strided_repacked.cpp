#include <pthread.h>
#include <torch/extension.h>

#include <algorithm>
#include <numeric>
#include <vector>

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

torch::Tensor decompress_centroid_embeds_strided_repacked(
    const torch::Tensor begins, const torch::Tensor ends,
    const torch::Tensor sizes, const torch::Tensor centroid_scores,
    const torch::Tensor residuals_compacted, const torch::Tensor bucket_weights,
    const torch::Tensor Q, const int nprobe) {
  torch::NoGradGuard no_grad;
  torch::InferenceMode guard;
  const int dim = 128;
  const int nbits = 4; // TODO(jlscheerer) Make this customizable.

  const int packed_vals_per_byte = 8 / nbits;
  const int packed_dim = dim / packed_vals_per_byte;

  const int ncells = begins.size(0);

  // TODO(jlscheerer) Rename a... to ..._accessor.
  const auto begins_accessor = begins.accessor<int64_t, 1>();
  const auto ends_accessor = ends.accessor<int64_t, 1>();
  const auto sizes_accessor = sizes.accessor<int64_t, 1>();
  const auto centroids_accessor = centroid_scores.accessor<float, 1>();

  const int64_t numel = sizes.sum().item<int64_t>();

  std::vector<int64_t> offsets(ncells);
  const int64_t *sizes_ptr = sizes.data_ptr<int64_t>();
  std::partial_sum(sizes_ptr, sizes_ptr + ncells - 1, offsets.begin() + 1);

  const auto residuals_accessor = residuals_compacted.accessor<uint8_t, 2>();
  // TODO(jlscheerer) Assert that ncells == Q.size(0) * nprobe

  torch::Tensor results = torch::zeros({numel}, torch::kFloat32);
  auto results_accessor = results.accessor<float, 1>();
  at::parallel_for(
      0, ncells, /*grain_size=*/0, [&](int64_t ncell_begin, int64_t ncell_end) {
        for (int cell_idx = ncell_begin; cell_idx < ncell_end; ++cell_idx) {
          const int begin = begins_accessor[cell_idx];
          const int end = ends_accessor[cell_idx];
          const int n = sizes_accessor[cell_idx];
          const int64_t roffset = offsets[cell_idx];
          const float centroid_score = centroids_accessor[cell_idx];

          const auto bucket_scores = torch::matmul(
              Q[cell_idx / nprobe].unsqueeze(1), bucket_weights.unsqueeze(0));
          const auto bucket_scores_accessor =
              bucket_scores.accessor<float, 2>();
          for (int residual_idx = 0; residual_idx < n; ++residual_idx) {
            float score = 0;
            for (int packed_idx = 0; packed_idx < packed_dim; ++packed_idx) {
              // NOTE This make *serious* assumptions about nbits = 4.
              const uint8_t packed_val =
                  residuals_accessor[begin + residual_idx][packed_idx];

              const uint8_t unpacked_high = (packed_val & 0xF0) >> 4;
              const uint8_t unpacked_low = (packed_val & 0x0F);

              const int unpacked_low_idx = 2 * packed_idx;
              const int unpacked_high_idx = unpacked_low_idx + 1;

              const float low_score =
                  bucket_scores_accessor[unpacked_low_idx][unpacked_high];
              const float high_score =
                  bucket_scores_accessor[unpacked_high_idx][unpacked_low];

              score += low_score + high_score;
            }
            results_accessor[roffset + residual_idx] = centroid_score + score;
          }
        }
      });

  return results;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("decompress_centroid_embeds_strided_repacked_cpp",
        &decompress_centroid_embeds_strided_repacked,
        "Decompress Centroid Embeds Strided Repacked");
}
