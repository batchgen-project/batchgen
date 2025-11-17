#ifndef UTIL_MEASURE_TIME_H_
#define UTIL_MEASURE_TIME_H_

#include <chrono>
#include <functional>
#include <type_traits>
#include <utility>

namespace util {

template <typename F, typename... Args>
auto MeasureTime(F&& f, Args&&... args) {
  using Clock = std::chrono::steady_clock;
  const auto start = Clock::now();

  using ResultT = std::invoke_result_t<F, Args...>;

  if constexpr (std::is_void_v<ResultT>) {
    std::invoke(std::forward<F>(f), std::forward<Args>(args)...);
    const auto end = Clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  } else {
    ResultT result =
        std::invoke(std::forward<F>(f), std::forward<Args>(args)...);
    const auto end = Clock::now();
    const auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    return std::make_pair(std::move(result), duration);
  }
}

}  // namespace util

#endif  // UTIL_MEASURE_TIME_H_