#ifndef TRANSFORMED_HOST_PAGED_KV_UTILS_H_
#define TRANSFORMED_HOST_PAGED_KV_UTILS_H_

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <future>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "host_paged_kv_worker_view.h"

namespace batchgen::kv::transformed_detail {

inline std::size_t ResolveLength(const SequenceLengths& sequence_lengths,
                                 std::size_t batch_idx,
                                 std::int64_t sequence_id,
                                 std::string_view op_name) {
    return std::visit(
        [&](const auto& container) -> std::size_t {
            using Container = std::decay_t<decltype(container)>;
            if constexpr (std::is_same_v<Container, SequenceLengthMap>) {
                const auto it = container.find(sequence_id);
                if (it == container.end()) {
                    std::ostringstream oss;
                    oss << op_name
                        << ": missing sequence length for sequence "
                        << sequence_id;
                    throw std::out_of_range(oss.str());
                }
                return it->second;
            } else {
                if (batch_idx >= container.size()) {
                    std::ostringstream oss;
                    oss << op_name
                        << ": sequence_lengths vector is missing batch index "
                        << batch_idx;
                    throw std::out_of_range(oss.str());
                }
                return container[batch_idx];
            }
        },
        sequence_lengths);
}

inline torch::Tensor SelectRows(const torch::Tensor& tensor,
                                const std::vector<std::int64_t>& rows) {
    if (rows.size() == static_cast<std::size_t>(tensor.size(0))) {
        return tensor;
    }
    auto indices =
        torch::tensor(rows, torch::TensorOptions()
                                .dtype(torch::kLong)
                                .device(tensor.device()));
    return tensor.index_select(0, indices).contiguous();
}

inline std::atomic<std::uint64_t>& AsyncTaskIdCounter() {
    static std::atomic<std::uint64_t> counter{0};
    return counter;
}

template <typename Fn>
KVAsyncTask MakeAsyncTask(Fn&& fn) {
    auto future = std::async(std::launch::async, std::forward<Fn>(fn)).share();
    const std::uint64_t id =
        AsyncTaskIdCounter().fetch_add(1, std::memory_order_relaxed) + 1;
    return KVAsyncTask{id, std::move(future)};
}

inline KVAsyncTask MakeCombinedTask(std::vector<KVAsyncTask> tasks) {
    return MakeAsyncTask([tasks = std::move(tasks)] {
        for (const auto& task : tasks) {
            task.wait();
        }
    });
}

class PendingHostWriteTasks {
   public:
    void Drain() {
        for (const auto& task : tasks_) {
            task.wait();
        }
        tasks_.clear();
    }

    void Track(const KVAsyncTask& task) {
        PruneDone();
        tasks_.push_back(task);
    }

   private:
    void PruneDone() {
        tasks_.erase(std::remove_if(tasks_.begin(), tasks_.end(),
                                    [](const KVAsyncTask& task) {
                                        return task.done();
                                    }),
                     tasks_.end());
    }

    std::vector<KVAsyncTask> tasks_;
};

}  // namespace batchgen::kv::transformed_detail

#endif  // TRANSFORMED_HOST_PAGED_KV_UTILS_H_
