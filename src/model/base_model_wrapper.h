// File: base_model_wrapper.h
// Description: Holds model + optimizer to directly interface with nn::Module for inference + learning

#ifndef HPTS_MODEL_WRAPPER_H_
#define HPTS_MODEL_WRAPPER_H_

#include "util/concepts.h"

#include <string>

// NOLINTBEGIN
#include <torch/torch.h>
// NOLINTEND

namespace hpts::model {

class BaseModelWrapper;

// Concept which model wrappers must implement
template <typename T>
concept IsModelWrapper = std::is_base_of_v<BaseModelWrapper, T> && requires(T t) {
    typename T::InferenceInput;
    typename T::InferenceOutput;
    typename T::LearningInput;
    {
        t.name
    } -> std::same_as<const std::string &>;
    {
        t.Inference(makeval<std::vector<typename T::InferenceInput> &>())
    } -> std::same_as<std::vector<typename T::InferenceOutput>>;
    {
        t.Learn(makeval<std::vector<typename T::LearningInput> &>())
    } -> std::same_as<double>;
};

// Base model wrapper which specific models will inherit and implement specific learn/inference methods
class BaseModelWrapper {
public:
    BaseModelWrapper(
        const std::string &device,
        const std::string &output_path,
        const std::string &checkpoint_base_name = ""
    );
    virtual ~BaseModelWrapper() = default;

    // Doesn't make sense to allow copyies
    BaseModelWrapper(const BaseModelWrapper &) = delete;
    BaseModelWrapper(BaseModelWrapper &&) = delete;
    BaseModelWrapper &operator=(const BaseModelWrapper &) = delete;
    BaseModelWrapper &operator=(BaseModelWrapper &&) = delete;

    /**
     * Log model pretty print to log file
     */
    virtual void print() const = 0;

    /**
     * Checkpoint model to file
     * @param step Step number of checkpoint (only used if wanting to differentiate checkpoint steps)
     * @return Checkpoint path, used for loading models back to sync if using multiple models
     */
    virtual auto SaveCheckpoint(long long int step = -1) -> std::string = 0;
    virtual auto SaveCheckpointWithoutOptimizer(long long int step = -1) -> std::string = 0;

    /**
     * Load model from checkpoint step
     * @param step Step number of checkpoint (only used if wanting to differentiate checkpoint steps)
     */
    virtual void LoadCheckpoint(long long int step);
    virtual void LoadCheckpointWithoutOptimizer(long long int step);

    /**
     * Load model from checkpoint path
     * @param path Base path directory of model + optimizer
     */
    virtual void LoadCheckpoint(const std::string &path) = 0;
    virtual void LoadCheckpointWithoutOptimizer(const std::string &path) = 0;

protected:
    std::string device_;
    std::string path_;
    std::string checkpoint_base_name_;
    torch::Device torch_device_;
};

}    // namespace hpts::model

#endif    // HPTS_MODEL_WRAPPER_H_
