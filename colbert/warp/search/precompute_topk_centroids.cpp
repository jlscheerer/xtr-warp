#include <pthread.h>
#include <torch/extension.h>

#include <algorithm>
#include <numeric>
#include <vector>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
precompute_topk_centroids(const torch::Tensor Q_mask,
                          const torch::Tensor centroid_scores,
                          const torch::Tensor sizes_compacted,
                          const int64_t nprobe, const int64_t fill_blank) {
  torch::NoGradGuard no_grad;
  torch::InferenceMode guard;

  const auto Q_mask_accessor = Q_mask.accessor<bool, 1>();
  const auto centroid_scores_accessor = centroid_scores.accessor<float, 2>();
  const auto sizes_compacted_accessor = sizes_compacted.accessor<int64_t, 1>();

  const int64_t ncentroids = centroid_scores.size(1);

  torch::Tensor cells = torch::zeros({32, nprobe}, torch::kInt32);
  auto cells_accessor = cells.accessor<int32_t, 2>();

  torch::Tensor scores = torch::zeros({32, nprobe}, torch::kFloat32);
  auto scores_accessor = scores.accessor<float, 2>();

  torch::Tensor mse = torch::zeros({32}, torch::kFloat32);
  auto mse_accessor = mse.accessor<float, 1>();

  at::parallel_for(
      0, 32, /*grain_size=*/0, [&](int64_t qidx_begin, int64_t qidx_end) {
        std::vector<int32_t> centroid_idx(ncentroids);
        std::iota(centroid_idx.begin(), centroid_idx.end(), 0);
        for (int i = qidx_begin; i < qidx_end; ++i) {
          if (!Q_mask_accessor[i])
            continue;
          std::partial_sort(centroid_idx.begin(), centroid_idx.begin() + nprobe,
                            centroid_idx.end(),
                            [&](const int i1, const int i2) {
                              if (centroid_scores_accessor[i][i1] !=
                                  centroid_scores_accessor[i][i2]) {
                                return centroid_scores_accessor[i][i1] >
                                       centroid_scores_accessor[i][i2];
                              }
                              return i1 < i2;
                            });
          for (int j = 0; j < nprobe; ++j) {
            cells_accessor[i][j] = centroid_idx[j];
            scores_accessor[i][j] =
                centroid_scores_accessor[i][centroid_idx[j]];
          }
          int32_t cumsum = 0, prev_bound = 0, curr_bound = nprobe, idx = 0;
          while (cumsum < fill_blank) {
            cumsum += sizes_compacted_accessor[centroid_idx[idx++]];
            if (cumsum < fill_blank && idx >= curr_bound) {
              prev_bound = curr_bound;
              curr_bound += nprobe;
              std::partial_sort(centroid_idx.begin() + prev_bound,
                                centroid_idx.begin() + curr_bound,
                                centroid_idx.end(),
                                [&](const int i1, const int i2) {
                                  if (centroid_scores_accessor[i][i1] !=
                                      centroid_scores_accessor[i][i2]) {
                                    return centroid_scores_accessor[i][i1] >
                                           centroid_scores_accessor[i][i2];
                                  }
                                  return i1 < i2;
                                });
            }
          }
          mse_accessor[i] = centroid_scores_accessor[i][centroid_idx[idx - 1]];
        }
      });

  return {cells, scores, mse};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("precompute_topk_centroids_cpp", &precompute_topk_centroids,
        "Precompute topK Centroids");
}