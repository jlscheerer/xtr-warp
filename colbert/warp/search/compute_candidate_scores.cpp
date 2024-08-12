#include <pthread.h>
#include <torch/extension.h>

#include <algorithm>
#include <array>
#include <numeric>
#include <tuple>
#include <vector>

#include <unordered_map>

#include "flat_hash_map.hpp"

std::tuple<torch::Tensor, torch::Tensor> compute_candidate_scores(
    const torch::Tensor candidate_pids_strided,
    const torch::Tensor decompressed_candidate_scores_strided,
    const torch::Tensor candidate_sizes, const torch::Tensor mse_estimates,
    const int nprobe, const int k) {
  torch::NoGradGuard no_grad;
  torch::InferenceMode guard;
  const uint64_t nscores = candidate_pids_strided.size(0);

  const auto candidate_pids_strided_accessor =
      candidate_pids_strided.accessor<int32_t, 1>();
  const auto decompressed_candidate_scores_strided_accessor =
      decompressed_candidate_scores_strided.accessor<float, 1>();
  const auto candidate_sizes_accessor = candidate_sizes.accessor<int64_t, 1>();
  const auto mse_estimates_accessor = mse_estimates.accessor<float, 1>();

  ska::flat_hash_map<int32_t, int32_t> index_map;
  ska::flat_hash_map<int32_t, int32_t> rev_index_map;
  for (int i = 0; i < nscores; ++i) {
    const int pid = candidate_pids_strided_accessor[i];
    if (index_map.find(pid) == index_map.end()) {
      const int32_t idx = index_map.size();
      index_map[pid] = idx;
      rev_index_map[idx] = pid;
    }
  }
  int64_t ncandidates = index_map.size();

  // TODO(jlscheerer) Make this more friendly for parallelization via cumsum.
  int64_t curr_candidate_sz_idx = 0;
  int64_t curr_candidate_sz = candidate_sizes_accessor[curr_candidate_sz_idx];

  std::vector<float> score_matrix(32 * ncandidates);

  const int64_t nresults = std::min(ncandidates, static_cast<int64_t>(k));
  torch::Tensor result_candidate_scores =
      torch::zeros({nresults}, torch::kFloat32);
  auto candidate_scores_accessor = result_candidate_scores.accessor<float, 1>();

  torch::Tensor result_candidate_pids = torch::zeros({nresults}, torch::kInt32);
  auto candidate_pids_accessor = result_candidate_pids.accessor<int32_t, 1>();

#pragma omp parallel for
  for (int i = 0; i < nscores; ++i) {
    const int32_t candidate_pid = candidate_pids_strided_accessor[i];
    const float candidate_score =
        decompressed_candidate_scores_strided_accessor[i];

    const auto it = index_map.find(candidate_pid);
    if (it == index_map.end())
      continue;

    const int32_t local_pid = it->second;
    const int32_t local_qid = curr_candidate_sz_idx / nprobe;

    if (--curr_candidate_sz == 0) {
      curr_candidate_sz = candidate_sizes_accessor[++curr_candidate_sz_idx];
    }

#pragma omp atomic
    score_matrix[32 * local_pid + local_qid] =
        std::max(score_matrix[32 * local_pid + local_qid], candidate_score);
  }

  // TODO(jlscheerer) Investigate if this is worth it.
  std::array<float, 32> mse_estimates_arr;
  for (int i = 0; i < 32; ++i) {
    mse_estimates_arr[i] = mse_estimates_accessor[i];
  }

  std::vector<std::pair<float, int32_t>> candidate_scores(ncandidates);
  at::parallel_for(0, ncandidates, 0, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      float sum = 0.0f;
      for (int j = 0; j < 32; ++j) {
        float score = score_matrix[32 * i + j];
        float mse = mse_estimates_arr[j];
        sum += (score == 0.0f) ? mse : score;
      }
      const float score = sum;
      const int32_t pid = rev_index_map[i];
      candidate_scores[i] = {score, pid};
    }
  });

  std::partial_sort(candidate_scores.begin(),
                    candidate_scores.begin() + nresults, candidate_scores.end(),
                    [](const auto &c1, const auto &c2) {
                      // Sort the results by the score first.
                      if (std::get<0>(c1) != std::get<0>(c2)) {
                        return std::get<0>(c1) > std::get<0>(c2);
                      }
                      // In case of a tie, additionally sort by pids.
                      return std::get<1>(c1) < std::get<1>(c2);
                    });

  for (int i = 0; i < nresults; ++i) {
    const auto &[score, pid] = candidate_scores[i];
    candidate_scores_accessor[i] = score;
    candidate_pids_accessor[i] = pid;
  }

  return {result_candidate_pids, result_candidate_scores};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("compute_candidate_scores_cpp", &compute_candidate_scores,
        "Compute Candidate Scores");
}
