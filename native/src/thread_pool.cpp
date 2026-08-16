#include "tensorforge/thread_pool.hpp"

#include <algorithm>
#include <stdexcept>

namespace tensorforge {

size_t ThreadPool::default_num_threads() noexcept {
    unsigned int hw = std::thread::hardware_concurrency();
    if (hw == 0) {
        return 2;
    }
    // Set a sensible default: min(hardware_concurrency, 8)
    return std::min(static_cast<size_t>(hw), static_cast<size_t>(8));
}

ThreadPool::ThreadPool(size_t num_threads) {
    if (num_threads == 0) {
        num_threads = default_num_threads();
    }
    start_workers(num_threads);
}

ThreadPool::~ThreadPool() {
    shutdown();
}

void ThreadPool::start_workers(size_t count) {
    stop_ = false;
    num_threads_ = count;
    // We create count - 1 worker threads because the calling thread also participates in work
    size_t bg_threads = count > 1 ? count - 1 : 0;
    workers_.reserve(bg_threads);

    for (size_t i = 0; i < bg_threads; ++i) {
        workers_.emplace_back(&ThreadPool::worker_loop, this);
    }
}

void ThreadPool::shutdown() {
    stop_ = true;
    cv_task_.notify_all();

    for (auto& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    workers_.clear();

    std::lock_guard<std::mutex> lock(queue_mutex_);
    while (!tasks_.empty()) {
        tasks_.pop();
    }
}

void ThreadPool::set_num_threads(size_t num_threads) {
    if (num_threads == 0) {
        throw std::invalid_argument("num_threads must be at least 1.");
    }
    if (num_threads == num_threads_.load()) {
        return;
    }
    shutdown();
    start_workers(num_threads);
}

size_t ThreadPool::num_threads() const noexcept {
    return num_threads_.load();
}

void ThreadPool::worker_loop() {
    while (true) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            cv_task_.wait(lock, [this]() {
                return stop_.load() || !tasks_.empty();
            });

            if (stop_.load() && tasks_.empty()) {
                return;
            }

            task = std::move(tasks_.front());
            tasks_.pop();
        }
        task();
    }
}

void ThreadPool::parallel_for(
    size_t start,
    size_t end,
    const std::function<void(size_t chunk_start, size_t chunk_end)>& task,
    size_t min_chunk_size
) {
    if (start >= end) {
        return;
    }

    size_t total_items = end - start;
    size_t current_threads = num_threads_.load();

    if (current_threads <= 1 || total_items <= min_chunk_size) {
        task(start, end);
        return;
    }

    size_t num_chunks = std::min(current_threads, (total_items + min_chunk_size - 1) / min_chunk_size);
    if (num_chunks <= 1) {
        task(start, end);
        return;
    }

    size_t chunk_size = (total_items + num_chunks - 1) / num_chunks;
    std::atomic<size_t> remaining{num_chunks};
    std::mutex completion_mutex;
    std::condition_variable cv_completion;

    // Enqueue background chunks
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        for (size_t c = 0; c < num_chunks - 1; ++c) {
            size_t c_start = start + c * chunk_size;
            size_t c_end = std::min(c_start + chunk_size, end);

            tasks_.push([&task, c_start, c_end, &remaining, &completion_mutex, &cv_completion]() {
                task(c_start, c_end);
                if (--remaining == 0) {
                    std::lock_guard<std::mutex> lk(completion_mutex);
                    cv_completion.notify_one();
                }
            });
        }
    }
    cv_task_.notify_all();

    // Execute last chunk on the calling thread
    size_t last_start = start + (num_chunks - 1) * chunk_size;
    task(last_start, end);

    if (--remaining > 0) {
        std::unique_lock<std::mutex> lk(completion_mutex);
        cv_completion.wait(lk, [&remaining]() { return remaining.load() == 0; });
    }
}

ThreadPool& get_global_thread_pool() {
    static ThreadPool global_pool;
    return global_pool;
}

void set_global_num_threads(size_t num_threads) {
    get_global_thread_pool().set_num_threads(num_threads);
}

size_t get_global_num_threads() {
    return get_global_thread_pool().num_threads();
}

} // namespace tensorforge
