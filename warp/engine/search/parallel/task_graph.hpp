#pragma once

#include <algorithm>
#include <atomic>
#include <bitset>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <iostream>
#include <optional>
#include <vector>
#include <string>
#include <thread>
#include <vector>

namespace warp {

using task_ref = int32_t;
template <typename data_type> struct task_graph;
template <typename data_type> struct task {
public:
  bool is_leaf() const { return pending_ == 0; }
  const data_type &data() const { return data_; }

private:
  task_ref successor_ = -1;
  std::atomic<int32_t> pending_{0};
  data_type data_;

  friend struct task_graph<data_type>;

  friend std::ostream &operator<<(std::ostream &os,
                                  const task<data_type> &task) {
    os << "successor=" << task.successor_ << " pending=" << task.pending_
       << ": " << task.data_;
    return os;
  }
};

template <typename task_type> struct task_graph {
  using context_type = typename task_type::context_type;

  task_graph(context_type &&context, const int max_num_tasks) 
    : context_(std::move(context)), tasks_(max_num_tasks) {}

  task_ref add(task_type &&data) {
    tasks_[num_tasks_].data_ = std::move(data);
    return num_tasks_++;
  }

  void mark_successor(task_ref predecessor, task_ref successor) {
    tasks_[predecessor].successor_ = successor;
    ++tasks_[successor].pending_;
  }

  int size() const { return static_cast<std::size_t>(num_tasks_); }

  task<task_type> &operator[](task_ref id) { return tasks_[id]; }

  void run_all_tasks(const int32_t num_threads) {
    for (task_ref task_id = 0; task_id < num_tasks_; ++task_id) {
      if (tasks_[task_id].is_leaf()) {
        queue_.push_back(task_id);
        ++queue_size_;
      }
    }
    for (int32_t i = 0; i < num_threads; ++i) {
      threads_.emplace_back(
          std::thread(&task_graph<task_type>::thread_loop, this));
    }
    for (int32_t i = 0; i < num_threads; ++i) {
      threads_[i].join();
    }
  }

private:
  std::optional<task_ref> dequeue_task() {
    task_ref task_id;
    {
      int32_t task_pos = queue_pos_.fetch_add(1);
      // NOTE we can immediately kill a thread once the queue is empty as the number of tasks only ever decreases.
      //      i.e., we will never add to the queue again. 
      if (task_pos >= queue_size_) {
        return std::nullopt;
      }
      task_id = queue_[task_pos];
    }
    return task_id;
  }

  void thread_loop() {
    std::optional<task_ref> task_id_opt;
    while (true) {
      auto task_id_opt = dequeue_task();
      if (!task_id_opt.has_value()) {
        return;
      }
      while (task_id_opt.has_value()) {
        auto &task = tasks_[*task_id_opt];
        task_type::execute_task(context_, task.data());

        const task_ref successor = task.successor_;
        if (successor == -1) {
          return;
        }
        if (tasks_[successor].pending_.fetch_sub(1) == 1) {
          task_id_opt = successor;
        } else {
          task_id_opt = std::nullopt;
        }
      }
    }
  }

  task_ref num_tasks_ = 0;
  std::vector<task<task_type>> tasks_;
  const context_type context_;

  std::vector<task_ref> queue_;
  int32_t queue_size_ = 0;
  std::atomic<int32_t> queue_pos_{0};

  std::vector<std::thread> threads_;
  bool should_terminate_ = false;

  friend std::ostream &operator<<(std::ostream &os,
                                  const task_graph<task_type> &graph) {
    for (task_ref task_id = 0; task_id < graph.num_tasks_; ++task_id) {
      os << "task_id=" << task_id << " " << graph.tasks_[task_id] << std::endl;
    }
    return os;
  }
};

} // namespace warp