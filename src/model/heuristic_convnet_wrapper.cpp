// File: heuristic_convnet_wrapper.cpp
// Description: Convnet wrapper for heuristic net

#include "model/heuristic_convnet_wrapper.h"

// NOLINTBEGIN
#include <absl/strings/str_cat.h>
#include <absl/strings/str_format.h>
// NOLINTEND

#include "model/torch_util.h"
#include "util/zip.h"

#include <spdlog/spdlog.h>

#include <filesystem>
#include <ostream>
#include <sstream>

namespace hpts::model::wrapper {

HeuristicConvNetWrapper::HeuristicConvNetWrapper(
    HeuristicConvNetConfig config_,
    double learning_rate,
    double l2_weight_decay,
    const std::string &device,
    const std::string &output_path,
    const std::string &checkpoint_base_name
)
    : BaseModelWrapper(device, output_path, checkpoint_base_name),
      config(std::move(config_)),
      model_(
          config.observation_shape,
          config.resnet_channels,
          config.resnet_blocks,
          config.reduce_channels,
          config.mlp_layers,
          config.use_batchnorm
      ),
      model_optimizer_(model_->parameters(), torch::optim::AdamOptions(learning_rate).weight_decay(l2_weight_decay)),
      input_flat_size(config.observation_shape.flat_size()) {
    model_->to(torch_device_);
};

void HeuristicConvNetWrapper::print() const {
    std::ostringstream oss;
    std::ostream &os = oss;
    os << *model_;
    SPDLOG_INFO("{:s}", oss.str());
    std::size_t num_params = 0;
    for (const auto &p : model_->parameters()) {
        num_params += static_cast<std::size_t>(p.numel());
    }
    SPDLOG_INFO("Number of parameters: {:d}", num_params);
}

auto HeuristicConvNetWrapper::SaveCheckpoint(long long int step) -> std::string {
    // create directory for model
    std::filesystem::create_directories(path_);
    std::string full_path = absl::StrCat(path_, checkpoint_base_name_, "checkpoint-", step);
    SPDLOG_INFO("Checkpointing model to {:s}.pt", full_path);
    torch::save(model_, absl::StrCat(full_path, ".pt"));
    torch::save(model_optimizer_, absl::StrCat(full_path, "-optimizer.pt"));
    return full_path;
}
auto HeuristicConvNetWrapper::SaveCheckpointWithoutOptimizer(long long int step) -> std::string {
    // create directory for model
    std::filesystem::create_directories(path_);
    std::string full_path = absl::StrCat(path_, checkpoint_base_name_, "checkpoint-", step);
    SPDLOG_INFO("Checkpointing model to {:s}.pt", full_path);
    torch::save(model_, absl::StrCat(full_path, ".pt"));
    return full_path;
}

void HeuristicConvNetWrapper::LoadCheckpoint(const std::string &path) {
    if (!std::filesystem::exists(absl::StrCat(path, ".pt"))
        || !std::filesystem::exists(absl::StrCat(path, "-optimizer.pt")))
    {
        SPDLOG_ERROR("path {:s} does not contain model and/or optimizer", path);
        throw std::filesystem::filesystem_error(
            absl::StrFormat("path %s does not contain model and/or optimizer", path),
            std::error_code()
        );
    }
    torch::load(model_, absl::StrCat(path, ".pt"), torch_device_);
    torch::load(model_optimizer_, absl::StrCat(path, "-optimizer.pt"), torch_device_);
}
void HeuristicConvNetWrapper::LoadCheckpointWithoutOptimizer(const std::string &path) {
    if (!std::filesystem::exists(absl::StrCat(path, ".pt"))) {
        SPDLOG_ERROR("path {:s} does not contain model", path);
        throw std::filesystem::filesystem_error(
            absl::StrFormat("path %s does not contain model", path),
            std::error_code()
        );
    }
    torch::load(model_, absl::StrCat(path, ".pt"), torch_device_);
}

template <typename InferenceInputT>
    requires IsAny<InferenceInputT, HeuristicConvNetWrapper::InferenceInput, HeuristicConvNetWrapper::InferenceRefInput>
[[nodiscard]] auto HeuristicConvNetWrapper::inference(std::vector<InferenceInputT> &batch)
    -> std::vector<InferenceOutput> {
    const int batch_size = static_cast<int>(batch.size());

    // Create tensor from raw flat array
    // torch::from_blob requires a pointer to non-const and doesn't take ownership
    auto options = torch::TensorOptions().dtype(torch::kFloat);
    torch::Tensor input_observations = torch::empty({batch_size, input_flat_size}, options);
    for (auto &&[idx, batch_item] : enumerate(batch)) {
        const auto i = static_cast<int>(idx);    // stop torch from complaining about narrowing conversions
        input_observations[i] = torch::from_blob(batch_item.observation.data(), {input_flat_size}, options);
    }

    // Reshape to expected size for network (batch_size, flat) -> (batch_size, c, h, w)
    input_observations = input_observations.to(torch_device_);
    input_observations = input_observations.reshape(
        {batch_size, config.observation_shape.c, config.observation_shape.h, config.observation_shape.w}
    );

    // Put model in eval mode for inference + scoped no_grad
    model_->eval();
    const torch::NoGradGuard no_grad;

    // Run inference
    const auto model_output = model_->forward(input_observations);
    const auto heuristic_output = model_output.heuristic.to(torch::kCPU);
    std::vector<InferenceOutput> inference_output;
    for (int i = 0; i < batch_size; ++i) {
        inference_output.emplace_back(heuristic_output[i].item<double>());
    }
    return inference_output;
}

auto HeuristicConvNetWrapper::Inference(std::vector<InferenceInput> &batch) -> std::vector<InferenceOutput> {
    return inference(batch);
}

auto HeuristicConvNetWrapper::Inference(std::vector<InferenceRefInput> &batch) -> std::vector<InferenceOutput> {
    return inference(batch);
}

auto HeuristicConvNetWrapper::Learn(std::vector<LearningInput> &batch) -> double {
    const int batch_size = static_cast<int>(batch.size());
    const auto options_float = torch::TensorOptions().dtype(torch::kFloat);
    const auto options_long = torch::TensorOptions().dtype(torch::kLong);

    // Create tensor from raw flat array
    // torch::from_blob requires a pointer to non-const and doesn't take ownership
    torch::Tensor input_observations = torch::empty({batch_size, input_flat_size}, options_float);
    torch::Tensor target_costs = torch::empty({batch_size, 1}, options_float);

    for (auto &&[idx, batch_item] : enumerate(batch)) {
        const auto i = static_cast<int>(idx);    // stop torch from complaining about narrowing conversions
        input_observations[i] = torch::from_blob(batch_item.observation.data(), {input_flat_size}, options_float);
        target_costs[i] = static_cast<float>(batch_item.target_cost_to_goal);
    }

    // Reshape to expected size for network (batch_size, flat) -> (batch_size, c, h, w)
    input_observations = input_observations.to(torch_device_);
    target_costs = target_costs.to(torch_device_);
    input_observations = input_observations.reshape(
        {batch_size, config.observation_shape.c, config.observation_shape.h, config.observation_shape.w}
    );

    // Put model in train mode for learning
    model_->train();
    model_->zero_grad();

    // Get model output
    auto model_output = model_->forward(input_observations);

    const torch::Tensor loss = mean_squared_error_loss(model_output.heuristic, target_costs, true);
    auto loss_value = loss.item<double>();

    // Optimize model
    loss.backward();
    model_optimizer_.step();

    return loss_value;
}

}    // namespace hpts::model::wrapper
