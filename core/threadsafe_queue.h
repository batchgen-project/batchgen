// clang-format off
/* ----------------------------------------------------------------------------  *
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
// clang-format on

#pragma once

#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>

template <typename T>
class threadsafe_queue {
   private:
    mutable std::mutex mut;
    std::queue<T> data_queue;
    std::condition_variable data_cond;

   public:
    threadsafe_queue() {};
    threadsafe_queue(threadsafe_queue const& other) {
        std::lock_guard<std::mutex> lk(other.mut);
        data_queue = other.data_queue;
    };  // Copy constructor

    void push(T new_value) {
        std::lock_guard<std::mutex> lk(mut);
        // data_queue.push(new_value);
        data_queue.push(std::move(new_value));
        data_cond.notify_one();
    };

    void wait_and_pop(T& value) {
        std::unique_lock<std::mutex> lk(mut);
        data_cond.wait(lk, [this] { return !data_queue.empty(); });
        // value = data_queue.front();
        value = std::move(data_queue.front());
        data_queue.pop();
    };

    std::shared_ptr<T> wait_and_pop() {
        std::unique_lock<std::mutex> lk(mut);
        data_cond.wait(lk, [this] { return !data_queue.empty(); });
        std::shared_ptr<T> res(std::make_shared<T>(data_queue.front()));
        data_queue.pop();
        return res;
    };

    bool try_pop(T& value) {
        std::lock_guard<std::mutex> lk(mut);
        if (data_queue.empty()) return false;
        // value = data_queue.front();
        value = std::move(data_queue.front());
        data_queue.pop();
        return true;
    };

    std::shared_ptr<T> try_pop() {
        std::lock_guard<std::mutex> lk(mut);
        if (data_queue.empty()) return std::shared_ptr<T>();
        std::shared_ptr<T> res(std::make_shared<T>(data_queue.front()));
        data_queue.pop();
        return res;
    };

    bool empty() const {
        std::lock_guard<std::mutex> lk(mut);
        return data_queue.empty();
    };

    int64_t size() const {
        std::lock_guard<std::mutex> lk(mut);
        return data_queue.size();
    };

    void clear() {
        std::lock_guard<std::mutex> lk(mut);
        std::queue<T> empty;
        std::swap(data_queue, empty);
    };
};
