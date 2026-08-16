#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include <condition_variable>
#include <atomic>
#include <queue>

namespace tensorforge {

/**
 * @brief Workload threshold in FLOPs/elements below which operations execute single-threaded.
 */
inline constexpr size_t PARALLEL_WORKLOAD_THRESHOLD = 8192;

/**
 * @brief High-performance, lightweight C++17 ThreadPool for parallel CPU inference execution.
 * 
 * Provides reusable worker threads across predictions, dynamic thread count adjustment,
 * and chunked range parallel execution (`parallel_for`) with zero allocation overhead.
 */
class ThreadPool {
public:
    explicit ThreadPool(size_t num_threads = 0);
    ~ThreadPool();

    // Non-copyable and non-movable for safe thread management
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    ThreadPool(ThreadPool&&) = delete;
    ThreadPool& operator=(ThreadPool&&) = delete;

    /**
     * @brief Execute a loop range [start, end) in parallel across worker threads.
     * 
     * If the total range is smaller than `min_chunk_size` or `num_threads <= 1`,
     * executes synchronously on the calling thread with zero synchronization overhead.
     * 
     * @param start Starting index (inclusive).
     * @param end Ending index (exclusive).
     * @param task Callable taking `(size_t chunk_start, size_t chunk_end)`.
     * @param min_chunk_size Minimum number of items per chunk.
     */
    void parallel_for(
        size_t start,
        size_t end,
        const std::function<void(size_t chunk_start, size_t chunk_end)>& task,
        size_t min_chunk_size = 1
    );

    /**
     * @brief Set the number of active worker threads in the pool.
     * 
     * @param num_threads Number of worker threads (clamped to hardware concurrency).
     */
    void set_num_threads(size_t num_threads);

    /**
     * @brief Get the current number of worker threads in the pool.
     */
    size_t num_threads() const noexcept;

    /**
     * @brief Return the default hardware concurrency recommended for the system.
     */
    static size_t default_num_threads() noexcept;

private:
    void worker_loop();
    void shutdown();
    void start_workers(size_t count);

    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex queue_mutex_;
    std::mutex pool_mutex_;
    std::condition_variable cv_task_;
    std::atomic<bool> stop_{false};
    std::atomic<size_t> num_threads_{0};
};

/**
 * @brief Get the global inference thread pool instance.
 */
ThreadPool& get_global_thread_pool();

/**
 * @brief Set the global thread pool size.
 */
void set_global_num_threads(size_t num_threads);

/**
 * @brief Get the global thread pool size.
 */
size_t get_global_num_threads();

} // namespace tensorforge
