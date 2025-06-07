// File: thread_pool.h
// Description: Simple thread pool class to dispatch threads continuously on input

#ifndef HPTS_UTIL_THREAD_POOL_H_
#define HPTS_UTIL_THREAD_POOL_H_

#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <queue>
#include <thread>
#include <vector>

namespace hpts {

// Create a thread pool object.
template <typename InputT, typename OutputT>
class ThreadPool {
public:
    ThreadPool() = delete;

    /**
     * Create a thread pool object.
     * @param num_threads Number of threads the pool should run
     */
    ThreadPool(std::size_t num_threads)
        : num_threads_(num_threads) {
        if (num_threads_ == 0) {
            throw std::invalid_argument("Expected at least one thread count");
        }
    }

    /**
     * Run the given function on the thread pool.
     * @param func Function to run in parallel, which should match templated arguments for input and output
     * @param inputs Input items for each job, gets passed to the given function
     * @return Vector of results, in order of given jobs during construction
     */
    [[nodiscard]] auto
        run(std::function<OutputT(InputT)> func, const std::vector<InputT> &inputs) noexcept -> std::vector<OutputT> {
        // Populate queue
        int id = -1;
        {
            std::queue<QueueItemInput> empty;
            std::swap(queue_input_, empty);
        }
        for (auto const &job : inputs) {
            queue_input_.emplace(job, ++id);
        }

        // Start N threads
        threads_.clear();
        threads_.reserve(num_threads_);
        for (std::size_t i = 0; i < num_threads_; ++i) {
            threads_.emplace_back([this, func]() { this->thread_runner(func); });
        }

        // Wait for all to complete
        for (auto &t : threads_) {
            t.join();
        }
        threads_.clear();

        // Compile results, such that the id is in order to match passed order
        std::vector<OutputT> results;
        results.reserve(inputs.size());
        std::map<int, OutputT> result_map;
        while (!queue_output_.empty()) {
            const auto result = queue_output_.front();
            queue_output_.pop();
            result_map.emplace(result.id, std::move(result.output));
        }
        for (auto const &result : result_map) {
            results.push_back(std::move(result.second));
        }

        return results;
    }

    /**
     * Run the given function on the thread pool.
     * @param func Function to run in parallel, which should match templated arguments for input and output
     * @param inputs Input items for each job, gets passed to the given function
     * @param workers Number of threads to use
     * @return Vector of results, in order of given jobs during construction
     */
    [[nodiscard]] auto
        run(std::function<OutputT(InputT)> func, const std::vector<InputT> &inputs, std::size_t workers
        ) noexcept -> std::vector<OutputT> {
        std::size_t old_count = workers;
        num_threads_ = workers;
        const auto results = run(func, inputs);
        num_threads_ = old_count;
        return results;
    }

private:
    struct QueueItemInput {    // Wrapper for input type with id
        InputT input;
        int id;
    };

    struct QueueItemOutput {    // Wrapper for output type with id
        OutputT output;
        int id;
    };

    // Runner for each thread, runs given function and pulls next item from input jobs if available
    void thread_runner(std::function<OutputT(InputT)> func) noexcept {
        while (true) {
            std::optional<QueueItemInput> item;
            {
                std::lock_guard<std::mutex> lock(queue_input_m_);

                // Jobs are done, thread can stop
                if (queue_input_.empty()) {
                    break;
                }

                item = queue_input_.front();
                queue_input_.pop();
            }

            // Run job
            OutputT result = func(item->input);

            // Store result
            {
                std::lock_guard<std::mutex> lock(queue_output_m_);
                queue_output_.emplace(std::move(result), item->id);
            }
        }
    }

    std::size_t num_threads_;             // How many threads in the pool
    std::vector<std::thread> threads_;    // Threads in the pool
    std::queue<QueueItemInput> queue_input_;
    std::queue<QueueItemOutput> queue_output_;
    std::mutex queue_input_m_;
    std::mutex queue_output_m_;
};

}    // namespace hpts

#endif    // HPTS_UTIL_THREAD_POOL_H_
