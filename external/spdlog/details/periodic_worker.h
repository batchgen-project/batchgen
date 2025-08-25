/* ---------------------------------------------------------------------------- *
 *  BatchGen                                                                      *
 *  copyright (c) EfficientMoE team 2025                                             *
 *                                                                               *
 *  licensed under the apache license, version 2.0 (the "license");              *
 *  you may not use this file except in compliance with the license.             *
 *                                                                               *
 *  you may obtain a copy of the license at                                      *
 *                                                                               *
 *                  http://www.apache.org/licenses/license-2.0                   *
 *                                                                               *
 *  unless required by applicable law or agreed to in writing, software          *
 *  distributed under the license is distributed on an "as is" basis,            *
 *  without warranties or conditions of any kind, either express or implied.     *
 *  see the license for the specific language governing permissions and          *
 *  limitations under the license.                                               *
 * ---------------------------------------------------------------------------- */

// Copyright(c) 2015-present, Gabi Melman & spdlog contributors.
// Distributed under the MIT License (http://opensource.org/licenses/MIT)

#pragma once

// periodic worker thread - periodically executes the given callback function.
//
// RAII over the owned thread:
//    creates the thread on construction.
//    stops and joins the thread on destruction (if the thread is executing a callback, wait for it
//    to finish first).

#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
namespace spdlog {
namespace details {

class SPDLOG_API periodic_worker {
public:
    template <typename Rep, typename Period>
    periodic_worker(const std::function<void()> &callback_fun,
                    std::chrono::duration<Rep, Period> interval) {
        active_ = (interval > std::chrono::duration<Rep, Period>::zero());
        if (!active_) {
            return;
        }

        worker_thread_ = std::thread([this, callback_fun, interval]() {
            for (;;) {
                std::unique_lock<std::mutex> lock(this->mutex_);
                if (this->cv_.wait_for(lock, interval, [this] { return !this->active_; })) {
                    return;  // active_ == false, so exit this thread
                }
                callback_fun();
            }
        });
    }
    std::thread &get_thread() { return worker_thread_; }
    periodic_worker(const periodic_worker &) = delete;
    periodic_worker &operator=(const periodic_worker &) = delete;
    // stop the worker thread and join it
    ~periodic_worker();

private:
    bool active_;
    std::thread worker_thread_;
    std::mutex mutex_;
    std::condition_variable cv_;
};
}  // namespace details
}  // namespace spdlog

#ifdef SPDLOG_HEADER_ONLY
    #include "periodic_worker-inl.h"
#endif
