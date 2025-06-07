// File: vsc_wrapper.cpp
// Description: A wrapper over a VQVAE, Subgoal, and conditional low model

#include "model/vsc_wrapper.h"

// NOLINTBEGIN
#include <absl/strings/str_cat.h>
#include <absl/strings/str_format.h>
// NOLINTEND

#include "model/torch_util.h"
#include "util/assert.h"

#include <spdlog/spdlog.h>

#include <filesystem>
#include <format>
#include <ostream>
#include <ranges>
#include <sstream>

namespace hpts::model::wrapper {

VSCFlatWrapper::VSCFlatWrapper(
    VSCConfig config_,
    double learning_rate,
    double l2_weight_decay,
    const std::string &device,
    const std::string &output_path,
    const std::string &checkpoint_base_name
)
    : BaseModelWrapper(device, output_path, checkpoint_base_name),
      config(std::move(config_)),
      model_(config),
      vqvae_optimizer_(
          model_->vqvae_->parameters(),
          torch::optim::AdamOptions(learning_rate).weight_decay(l2_weight_decay)
      ),
      subgoal_optimizer_(
          model_->subgoal_net_->parameters(),
          torch::optim::AdamOptions(learning_rate).weight_decay(l2_weight_decay)
      ),
      conditional_low_optimizer_(
          model_->conditional_low_net_->parameters(),
          torch::optim::AdamOptions(learning_rate).weight_decay(l2_weight_decay)
      ),
      beta(config.vqvae_config.beta),
      use_ema(config.vqvae_config.use_ema) {
    model_->to(torch_device_);
};

void VSCFlatWrapper::print() const {
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

auto VSCFlatWrapper::SaveCheckpoint(long long int step) -> std::string {
    // create directory for model
    std::filesystem::create_directories(path_);
    std::string full_path = absl::StrCat(path_, checkpoint_base_name_, "checkpoint-", step);
    SPDLOG_INFO("Checkpointing model to {:s}.pt", full_path);
    torch::save(model_, absl::StrCat(full_path, ".pt"));
    torch::save(vqvae_optimizer_, absl::StrCat(full_path, "-optimizer_vqvae.pt"));
    torch::save(subgoal_optimizer_, absl::StrCat(full_path, "-optimizer_subgoal.pt"));
    torch::save(conditional_low_optimizer_, absl::StrCat(full_path, "-optimizer_conditional_low.pt"));
    return full_path;
}
auto VSCFlatWrapper::SaveCheckpointWithoutOptimizer(long long int step) -> std::string {
    // create directory for model
    std::filesystem::create_directories(path_);
    std::string full_path = absl::StrCat(path_, checkpoint_base_name_, "checkpoint-", step);
    SPDLOG_INFO("Checkpointing model to {:s}.pt", full_path);
    torch::save(model_, absl::StrCat(full_path, ".pt"));
    return full_path;
}

void VSCFlatWrapper::LoadCheckpoint(const std::string &path) {
    if (!std::filesystem::exists(absl::StrCat(path, ".pt"))
        || !std::filesystem::exists(absl::StrCat(path, "-optimizer.pt")))
    {
        SPDLOG_ERROR("path {:s} does not contain model and/or optimizer", path);
        throw std::filesystem::filesystem_error(
            std::format("path {:s} does not contain model and/or optimizer", path),
            std::error_code()
        );
    }
    torch::load(model_, absl::StrCat(path, ".pt"), torch_device_);
    torch::load(vqvae_optimizer_, absl::StrCat(path, "-optimizer_vqvae.pt"), torch_device_);
    torch::load(subgoal_optimizer_, absl::StrCat(path, "-optimizer_subgoal.pt"), torch_device_);
    torch::load(conditional_low_optimizer_, absl::StrCat(path, "-optimizer_conditional_low.pt"), torch_device_);
}
void VSCFlatWrapper::LoadCheckpointWithoutOptimizer(const std::string &path) {
    if (!std::filesystem::exists(absl::StrCat(path, ".pt"))) {
        SPDLOG_ERROR("path {:s} does not contain model", path);
        throw std::filesystem::filesystem_error(
            std::format("path {:s} does not contain model", path),
            std::error_code()
        );
    }
    torch::load(model_, absl::StrCat(path, ".pt"), torch_device_);
}

// VSC Inference
template <typename InferenceInputT>
    requires IsAny<InferenceInputT, VSCFlatWrapper::InferenceInput, VSCFlatWrapper::InferenceRefInput>
auto VSCFlatWrapper::inference(std::vector<InferenceInputT> &batch) -> std::vector<InferenceOutput> {
    const int batch_size = static_cast<int>(batch.size());
    const auto options_float = torch::TensorOptions().dtype(torch::kFloat);
    const auto options_long = torch::TensorOptions().dtype(torch::kLong);
    const int input_flat_size = config.observation_shape.flat_size();

    // Create tensor from raw flat array
    // torch::from_blob requires a pointer to non-const and doesn't take ownership
    torch::Tensor input_observations = torch::empty({batch_size, input_flat_size}, options_float);
    for (const auto &[idx, batch_item] : std::views::enumerate(batch)) {
        const auto i = static_cast<int>(idx);    // stop torch from complaining about narrowing conversions
        input_observations[i] = torch::from_blob(batch_item.obs.data(), {input_flat_size}, options_float);
    }

    // Reshape to expected size for network (batch_size, flat) -> (batch_size, c, h, w)
    input_observations = input_observations.reshape(
        {batch_size, config.observation_shape.c, config.observation_shape.h, config.observation_shape.w}
    );
    input_observations = input_observations.to(torch_device_);

    // Put model in eval mode for inference + scoped no_grad
    model_->eval();
    const torch::NoGradGuard no_grad;

    // Run inference
    auto [subgoal_logits, conditional_low_logits] = model_->inference(input_observations);
    subgoal_logits = subgoal_logits.reshape({batch_size * config.vqvae_config.num_embeddings, 1}).to(torch::kCPU);
    conditional_low_logits = conditional_low_logits.to(torch::kCPU);

    std::vector<InferenceOutput> inference_output;
    inference_output.reserve(batch_size);
    for (int i = 0; i < batch_size; ++i) {
        // Get subgoal slice and view as a single policy
        const auto idx_start = i * config.vqvae_config.num_embeddings;
        const auto idx_end = idx_start + config.vqvae_config.num_embeddings;
        const auto subgoal_slice = subgoal_logits.index({torch::indexing::Slice(idx_start, idx_end)}).flatten();
        // Conditional low policy for each embedding
        std::vector<std::vector<double>> conditional_low_policies;
        conditional_low_policies.reserve(batch_size);
        for (int j = 0; j < config.vqvae_config.num_embeddings; ++j) {
            const auto conditional_low_slice = conditional_low_logits[idx_start + j];
            conditional_low_policies.push_back(tensor_to_vec<double, float>(torch::softmax(conditional_low_slice, 0)));
            assert(conditional_low_policies.back().size() == config.num_actions);
        }
        inference_output.emplace_back(
            tensor_to_vec<double, float>(torch::softmax(subgoal_slice, 0)),
            std::move(conditional_low_policies)
        );
    }
    return inference_output;
}

auto VSCFlatWrapper::Inference(std::vector<InferenceInput> &batch) -> std::vector<InferenceOutput> {
    return inference(batch);
}

auto VSCFlatWrapper::Inference(std::vector<InferenceRefInput> &batch) -> std::vector<InferenceOutput> {
    return inference(batch);
}

// VQVAE Quantize
template <typename QuantizeInputT>
    requires IsAny<QuantizeInputT, VSCFlatWrapper::VQVAEQuantizeInput, VSCFlatWrapper::VQVAEQuantizeRefInput>
auto VSCFlatWrapper::quantize(std::vector<QuantizeInputT> &batch) -> std::vector<VQVAEQuantizeOutput> {
    const int batch_size = static_cast<int>(batch.size());
    const auto options_float = torch::TensorOptions().dtype(torch::kFloat);
    const int input_flat_size = config.observation_shape.flat_size();

    // Create tensor from raw flat array
    // torch::from_blob requires a pointer to non-const and doesn't take ownership
    torch::Tensor encoder_input = torch::empty({2 * batch_size, input_flat_size}, options_float);

    for (const auto &[idx, batch_item] : std::views::enumerate(batch)) {
        const auto i = static_cast<int>(idx);    // stop torch from complaining about narrowing conversions
        encoder_input[2 * i + 0] = torch::from_blob(batch_item.obs_input.data(), {input_flat_size}, options_float);
        encoder_input[2 * i + 1] = torch::from_blob(batch_item.obs_target.data(), {input_flat_size}, options_float);
    }

    // Reshape to expected size for network (batch_size, flat) -> (batch_size, c, h, w)
    encoder_input = encoder_input.to(torch_device_);
    encoder_input = encoder_input.reshape(
        {batch_size, 2 * config.observation_shape.c, config.observation_shape.h, config.observation_shape.w}
    );

    // Put model in eval mode for inference + scoped no_grad
    model_->eval();
    const torch::NoGradGuard no_grad;

    torch::Tensor quantize_obs = model_->quantize(encoder_input).quantized_x.to(torch::kCPU);
    std::vector<VQVAEQuantizeOutput> quantized_output;
    for (int i = 0; i < batch_size; ++i) {
        quantized_output.emplace_back(tensor_to_vec<float, float>(quantize_obs[i]));
    }
    return quantized_output;
}

auto VSCFlatWrapper::Quantize(std::vector<VQVAEQuantizeInput> &batch) -> std::vector<VQVAEQuantizeOutput> {
    return quantize(batch);
}

auto VSCFlatWrapper::Quantize(std::vector<VQVAEQuantizeRefInput> &batch) -> std::vector<VQVAEQuantizeOutput> {
    return quantize(batch);
}

// Learning
auto VSCFlatWrapper::LearnVQVAE(std::vector<VQVAELearningInput> &batch) -> double {
    const int batch_size = static_cast<int>(batch.size());
    const auto options_float = torch::TensorOptions().dtype(torch::kFloat);
    const auto options_long = torch::TensorOptions().dtype(torch::kLong);
    const int input_flat_size = config.observation_shape.flat_size();

    // Create tensor from raw flat array
    // torch::from_blob requires a pointer to non-const and doesn't take ownership
    torch::Tensor encoder_input = torch::empty({2 * batch_size, input_flat_size}, options_float);
    torch::Tensor decoder_input = torch::empty({batch_size, input_flat_size}, options_float);
    torch::Tensor target = torch::empty({batch_size, input_flat_size}, options_float);

    for (const auto &[idx, batch_item] : std::views::enumerate(batch)) {
        const auto i = static_cast<int>(idx);    // stop torch from complaining about narrowing conversions
        encoder_input[2 * i + 0] = torch::from_blob(batch_item.obs_input.data(), {input_flat_size}, options_float);
        encoder_input[2 * i + 1] = torch::from_blob(batch_item.obs_target.data(), {input_flat_size}, options_float);
        decoder_input[i] = torch::from_blob(batch_item.obs_input.data(), {input_flat_size}, options_float);
        target[i] = torch::from_blob(batch_item.obs_target.data(), {input_flat_size}, options_float);
    }

    // Reshape to expected size for network (batch_size, flat) -> (batch_size, c, h, w)
    encoder_input = encoder_input.to(torch_device_);
    decoder_input = decoder_input.to(torch_device_);
    target = target.to(torch_device_);
    encoder_input = encoder_input.reshape(
        {batch_size, 2 * config.observation_shape.c, config.observation_shape.h, config.observation_shape.w}
    );
    decoder_input = decoder_input.reshape(
        {batch_size, config.observation_shape.c, config.observation_shape.h, config.observation_shape.w}
    );
    target =
        target.reshape({batch_size, config.observation_shape.c, config.observation_shape.h, config.observation_shape.w}
        );

    // Put model in train mode for learning
    model_->vqvae_->train();
    model_->vqvae_->zero_grad();

    // Get model output
    auto model_output = model_->forward_vqvae(encoder_input, decoder_input);

    // Losses
    torch::Tensor recon_loss = torch::cross_entropy_loss(model_output.obs_recon, target);
    torch::Tensor loss = recon_loss + beta * model_output.commitment_loss;
    if (!use_ema) {
        loss = loss + model_output.dictionary_loss.value();
    }
    auto loss_value = loss.item<double>();

    // Optimize model
    loss.backward();
    vqvae_optimizer_.step();

    return loss_value;
}

auto VSCFlatWrapper::LearnSubgoal(std::vector<SubgoalLearningInput> &batch) -> double {
    const int batch_size = static_cast<int>(batch.size());
    const auto options_float = torch::TensorOptions().dtype(torch::kFloat);
    const auto options_long = torch::TensorOptions().dtype(torch::kLong);
    const int input_flat_size = config.observation_shape.flat_size();

    // Create tensor from raw flat array
    // torch::from_blob requires a pointer to non-const and doesn't take ownership
    torch::Tensor encoder_input = torch::empty({2 * batch_size, input_flat_size}, options_float);
    torch::Tensor obs_input = torch::empty({batch_size, input_flat_size}, options_float);
    torch::Tensor expandeds = torch::empty({batch_size, 1}, options_float);

    for (const auto &[idx, batch_item] : std::views::enumerate(batch)) {
        const auto i = static_cast<int>(idx);    // stop torch from complaining about narrowing conversions
        encoder_input[2 * i + 0] = torch::from_blob(batch_item.obs_input.data(), {input_flat_size}, options_float);
        encoder_input[2 * i + 1] = torch::from_blob(batch_item.obs_target.data(), {input_flat_size}, options_float);
        obs_input[i] = torch::from_blob(batch_item.obs_input.data(), {input_flat_size}, options_float);
        expandeds[i] = static_cast<float>(batch_item.solution_expanded);
    }

    // Reshape to expected size for network (batch_size, flat) -> (batch_size, c, h, w)
    encoder_input = encoder_input.to(torch_device_);
    obs_input = obs_input.to(torch_device_);
    expandeds = expandeds.to(torch_device_);
    encoder_input = encoder_input.reshape(
        {batch_size, 2 * config.observation_shape.c, config.observation_shape.h, config.observation_shape.w}
    );
    obs_input = obs_input.reshape(
        {batch_size, config.observation_shape.c, config.observation_shape.h, config.observation_shape.w}
    );

    // Get the encoding idx from vqvae and use as subgoal action
    torch::Tensor target_actions;
    {
        model_->vqvae_->eval();
        const torch::NoGradGuard no_grad;
        target_actions = model_->quantize(encoder_input).encoding_indices;
    }

    // Put model in train mode for learning
    model_->subgoal_net_->train();
    model_->subgoal_net_->zero_grad();

    torch::Tensor subgoal_logits =
        model_->inference_subgoal(obs_input).reshape({batch_size, config.vqvae_config.num_embeddings});
    torch::Tensor loss = (expandeds * cross_entropy_loss(subgoal_logits, target_actions, false)).mean();
    auto loss_value = loss.item<double>();
    loss.backward();
    subgoal_optimizer_.step();

    return loss_value;
}

auto VSCFlatWrapper::LearnConditionalLow(std::vector<ConditionalLowLearningInput> &batch) -> double {
    const int batch_size = static_cast<int>(batch.size());
    const auto options_float = torch::TensorOptions().dtype(torch::kFloat);
    const auto options_long = torch::TensorOptions().dtype(torch::kLong);
    const int input_flat_size = config.observation_shape.flat_size();

    // Create tensor from raw flat array
    // torch::from_blob requires a pointer to non-const and doesn't take ownership
    torch::Tensor encoder_input = torch::empty({2 * batch_size, input_flat_size}, options_float);
    torch::Tensor obs_input = torch::empty({batch_size, input_flat_size}, options_float);
    torch::Tensor target_actions = torch::empty({batch_size, 1}, options_long);
    torch::Tensor expandeds = torch::empty({batch_size, 1}, options_float);

    for (const auto &[idx, batch_item] : std::views::enumerate(batch)) {
        const auto i = static_cast<int>(idx);    // stop torch from complaining about narrowing conversions
        encoder_input[2 * i + 0] = torch::from_blob(batch_item.obs_input.data(), {input_flat_size}, options_float);
        encoder_input[2 * i + 1] = torch::from_blob(batch_item.obs_target.data(), {input_flat_size}, options_float);
        obs_input[i] = torch::from_blob(batch_item.obs_input.data(), {input_flat_size}, options_float);
        target_actions[i] = batch_item.target_action;
        expandeds[i] = static_cast<float>(batch_item.solution_expanded);
    }

    // Reshape to expected size for network (batch_size, flat) -> (batch_size, c, h, w)
    encoder_input = encoder_input.to(torch_device_);
    obs_input = obs_input.to(torch_device_);
    target_actions = target_actions.to(torch_device_);
    expandeds = expandeds.to(torch_device_);
    encoder_input = encoder_input.reshape(
        {batch_size, 2 * config.observation_shape.c, config.observation_shape.h, config.observation_shape.w}
    );
    obs_input = obs_input.reshape(
        {batch_size, config.observation_shape.c, config.observation_shape.h, config.observation_shape.w}
    );

    // Get the encoding idx from vqvae and use as subgoal action
    torch::Tensor quantized_x;
    {
        model_->vqvae_->eval();
        const torch::NoGradGuard no_grad;
        quantized_x = model_->quantize(encoder_input).quantized_x.squeeze(3).squeeze(2);
    }

    // Put model in train mode for learning
    model_->conditional_low_net_->train();
    model_->conditional_low_net_->zero_grad();

    torch::Tensor logits = model_->inference_conditional_low(obs_input, quantized_x);
    torch::Tensor loss = (expandeds * cross_entropy_loss(logits, target_actions, false)).mean();
    auto loss_value = loss.item<double>();
    loss.backward();
    conditional_low_optimizer_.step();

    return loss_value;
}

// ----------------------------------------------

VSCWrapper::VSCWrapper(
    VSCConfig config_,
    double learning_rate,
    double l2_weight_decay,
    const std::string &device,
    const std::string &output_path,
    const std::string &checkpoint_base_name
)
    : BaseModelWrapper(device, output_path, checkpoint_base_name),
      config(std::move(config_)),
      model_(config),
      vqvae_optimizer_(
          model_->vqvae_->parameters(),
          torch::optim::AdamOptions(learning_rate).weight_decay(l2_weight_decay)
      ),
      subgoal_optimizer_(
          model_->subgoal_net_->parameters(),
          torch::optim::AdamOptions(learning_rate).weight_decay(l2_weight_decay)
      ),
      conditional_low_optimizer_(
          model_->conditional_low_net_->parameters(),
          torch::optim::AdamOptions(learning_rate).weight_decay(l2_weight_decay)
      ),
      beta(config.vqvae_config.beta),
      use_ema(config.vqvae_config.use_ema) {
    model_->to(torch_device_);
};

void VSCWrapper::print() const {
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

auto VSCWrapper::SaveCheckpoint(long long int step) -> std::string {
    // create directory for model
    std::filesystem::create_directories(path_);
    std::string full_path = absl::StrCat(path_, checkpoint_base_name_, "checkpoint-", step);
    SPDLOG_INFO("Checkpointing model to {:s}.pt", full_path);
    torch::save(model_, absl::StrCat(full_path, ".pt"));
    torch::save(vqvae_optimizer_, absl::StrCat(full_path, "-optimizer_vqvae.pt"));
    torch::save(subgoal_optimizer_, absl::StrCat(full_path, "-optimizer_subgoal.pt"));
    torch::save(conditional_low_optimizer_, absl::StrCat(full_path, "-optimizer_conditional_low.pt"));
    return full_path;
}
auto VSCWrapper::SaveCheckpointWithoutOptimizer(long long int step) -> std::string {
    // create directory for model
    std::filesystem::create_directories(path_);
    std::string full_path = absl::StrCat(path_, checkpoint_base_name_, "checkpoint-", step);
    SPDLOG_INFO("Checkpointing model to {:s}.pt", full_path);
    torch::save(model_, absl::StrCat(full_path, ".pt"));
    return full_path;
}

void VSCWrapper::LoadCheckpoint(const std::string &path) {
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
    torch::load(vqvae_optimizer_, absl::StrCat(path, "-optimizer_vqvae.pt"), torch_device_);
    torch::load(subgoal_optimizer_, absl::StrCat(path, "-optimizer_subgoal.pt"), torch_device_);
    torch::load(conditional_low_optimizer_, absl::StrCat(path, "-optimizer_conditional_low.pt"), torch_device_);
}
void VSCWrapper::LoadCheckpointWithoutOptimizer(const std::string &path) {
    if (!std::filesystem::exists(absl::StrCat(path, ".pt"))) {
        SPDLOG_ERROR("path {:s} does not contain model", path);
        throw std::filesystem::filesystem_error(
            absl::StrFormat("path %s does not contain model", path),
            std::error_code()
        );
    }
    torch::load(model_, absl::StrCat(path, ".pt"), torch_device_);
}

// VSC Inference
template <typename InferenceInputT>
    requires IsAny<InferenceInputT, VSCWrapper::InferenceInput, VSCWrapper::InferenceRefInput>
auto VSCWrapper::inference(std::vector<InferenceInputT> &batch) -> std::vector<InferenceOutput> {
    const int batch_size = static_cast<int>(batch.size());
    const auto options_float = torch::TensorOptions().dtype(torch::kFloat);
    const auto options_long = torch::TensorOptions().dtype(torch::kLong);
    const int input_flat_size = config.observation_shape.flat_size();

    // Create tensor from raw flat array
    // torch::from_blob requires a pointer to non-const and doesn't take ownership
    torch::Tensor input_observations = torch::empty({batch_size, input_flat_size}, options_float);
    for (const auto &[idx, batch_item] : std::views::enumerate(batch)) {
        const auto i = static_cast<int>(idx);    // stop torch from complaining about narrowing conversions
        input_observations[i] = torch::from_blob(batch_item.obs.data(), {input_flat_size}, options_float);
    }

    // Reshape to expected size for network (batch_size, flat) -> (batch_size, c, h, w)
    input_observations = input_observations.reshape(
        {batch_size, config.observation_shape.c, config.observation_shape.h, config.observation_shape.w}
    );
    input_observations = input_observations.to(torch_device_);

    // Put model in eval mode for inference + scoped no_grad
    model_->eval();
    const torch::NoGradGuard no_grad;

    torch::Tensor repeated_input_observations =
        input_observations.repeat_interleave(config.vqvae_config.num_embeddings, 0);

    // ------------------------
    // Get VQVAE subgoals
    torch::Tensor indices = torch::arange(config.vqvae_config.num_embeddings, options_long)
                                .reshape({config.vqvae_config.num_embeddings, 1})
                                .repeat({batch_size, 1});
    indices = indices.to(torch_device_);
    torch::Tensor decoded_output = model_->vqvae_->decode_from_indices(repeated_input_observations, indices);
    torch::Tensor obs_recon = torch::softmax(decoded_output, 1);
    // ------------------------

    // Run inference
    torch::Tensor input = torch::cat({repeated_input_observations, obs_recon}, 1);
    auto [subgoal_logits, conditional_low_logits] = model_->inference(input);
    subgoal_logits = subgoal_logits.to(torch::kCPU);
    conditional_low_logits = conditional_low_logits.to(torch::kCPU);

    std::vector<InferenceOutput> inference_output;
    inference_output.reserve(batch_size);
    for (int i = 0; i < batch_size; ++i) {
        // Get subgoal slice and view as a single policy
        const auto idx_start = i * config.vqvae_config.num_embeddings;
        const auto idx_end = idx_start + config.vqvae_config.num_embeddings;
        const auto subgoal_slice = subgoal_logits.index({torch::indexing::Slice(idx_start, idx_end)}).flatten();
        // Conditional low policy for each embedding
        std::vector<std::vector<double>> conditional_low_policies;
        conditional_low_policies.reserve(batch_size);
        for (int j = 0; j < config.vqvae_config.num_embeddings; ++j) {
            const auto conditional_low_slice = conditional_low_logits[idx_start + j];
            conditional_low_policies.push_back(tensor_to_vec<double, float>(torch::softmax(conditional_low_slice, 0)));
            assert(conditional_low_policies.back().size() == config.num_actions);
        }
        inference_output.emplace_back(
            tensor_to_vec<double, float>(torch::softmax(subgoal_slice, 0)),
            std::move(conditional_low_policies)
        );
    }
    return inference_output;
}

auto VSCWrapper::Inference(std::vector<InferenceInput> &batch) -> std::vector<InferenceOutput> {
    return inference(batch);
}

auto VSCWrapper::Inference(std::vector<InferenceRefInput> &batch) -> std::vector<InferenceOutput> {
    return inference(batch);
}

// Learning
auto VSCWrapper::LearnVQVAE(std::vector<VQVAELearningInput> &batch) -> double {
    const int batch_size = static_cast<int>(batch.size());
    const auto options_float = torch::TensorOptions().dtype(torch::kFloat);
    const auto options_long = torch::TensorOptions().dtype(torch::kLong);
    const int input_flat_size = config.observation_shape.flat_size();

    // Create tensor from raw flat array
    // torch::from_blob requires a pointer to non-const and doesn't take ownership
    torch::Tensor encoder_input = torch::empty({2 * batch_size, input_flat_size}, options_float);
    torch::Tensor decoder_input = torch::empty({batch_size, input_flat_size}, options_float);
    torch::Tensor target = torch::empty({batch_size, input_flat_size}, options_float);

    for (const auto &[idx, batch_item] : std::views::enumerate(batch)) {
        const auto i = static_cast<int>(idx);    // stop torch from complaining about narrowing conversions
        encoder_input[2 * i + 0] = torch::from_blob(batch_item.obs_input.data(), {input_flat_size}, options_float);
        encoder_input[2 * i + 1] = torch::from_blob(batch_item.obs_target.data(), {input_flat_size}, options_float);
        decoder_input[i] = torch::from_blob(batch_item.obs_input.data(), {input_flat_size}, options_float);
        target[i] = torch::from_blob(batch_item.obs_target.data(), {input_flat_size}, options_float);
    }

    // Reshape to expected size for network (batch_size, flat) -> (batch_size, c, h, w)
    encoder_input = encoder_input.to(torch_device_);
    decoder_input = decoder_input.to(torch_device_);
    target = target.to(torch_device_);
    encoder_input = encoder_input.reshape(
        {batch_size, 2 * config.observation_shape.c, config.observation_shape.h, config.observation_shape.w}
    );
    decoder_input = decoder_input.reshape(
        {batch_size, config.observation_shape.c, config.observation_shape.h, config.observation_shape.w}
    );
    target =
        target.reshape({batch_size, config.observation_shape.c, config.observation_shape.h, config.observation_shape.w}
        );

    // Put model in train mode for learning
    model_->vqvae_->train();
    model_->vqvae_->zero_grad();

    // Get model output
    auto model_output = model_->forward_vqvae(encoder_input, decoder_input);

    // Losses
    torch::Tensor recon_loss = torch::cross_entropy_loss(model_output.obs_recon, target);
    torch::Tensor loss = recon_loss + beta * model_output.commitment_loss;
    if (!use_ema) {
        loss = loss + model_output.dictionary_loss.value();
    }
    auto loss_value = loss.item<double>();

    // Optimize model
    loss.backward();
    vqvae_optimizer_.step();

    return loss_value;
}

auto VSCWrapper::LearnSubgoal(std::vector<SubgoalLearningInput> &batch) -> double {
    const int batch_size = static_cast<int>(batch.size());
    const auto options_float = torch::TensorOptions().dtype(torch::kFloat);
    const auto options_long = torch::TensorOptions().dtype(torch::kLong);
    const int input_flat_size = config.observation_shape.flat_size();

    // Create tensor from raw flat array
    // torch::from_blob requires a pointer to non-const and doesn't take ownership
    torch::Tensor encoder_input = torch::empty({2 * batch_size, input_flat_size}, options_float);
    torch::Tensor obs_input = torch::empty({batch_size, input_flat_size}, options_float);
    torch::Tensor expandeds = torch::empty({batch_size, 1}, options_float);

    for (const auto &[idx, batch_item] : std::views::enumerate(batch)) {
        const auto i = static_cast<int>(idx);    // stop torch from complaining about narrowing conversions
        encoder_input[2 * i + 0] = torch::from_blob(batch_item.obs_input.data(), {input_flat_size}, options_float);
        encoder_input[2 * i + 1] = torch::from_blob(batch_item.obs_target.data(), {input_flat_size}, options_float);
        obs_input[i] = torch::from_blob(batch_item.obs_input.data(), {input_flat_size}, options_float);
        expandeds[i] = static_cast<float>(batch_item.solution_expanded);
    }

    // Reshape to expected size for network (batch_size, flat) -> (batch_size, c, h, w)
    encoder_input = encoder_input.to(torch_device_);
    obs_input = obs_input.to(torch_device_);
    expandeds = expandeds.to(torch_device_);
    encoder_input = encoder_input.reshape(
        {batch_size, 2 * config.observation_shape.c, config.observation_shape.h, config.observation_shape.w}
    );
    obs_input = obs_input.reshape(
        {batch_size, config.observation_shape.c, config.observation_shape.h, config.observation_shape.w}
    );

    torch::Tensor repeated_input_observations = obs_input.repeat_interleave(config.vqvae_config.num_embeddings, 0);

    // Necessary VQVAE inference
    torch::Tensor input = obs_input;
    torch::Tensor target_actions;
    {
        model_->vqvae_->eval();
        const torch::NoGradGuard no_grad;
        // Get the encoding idx from vqvae and use as subgoal action
        target_actions = model_->quantize(encoder_input).encoding_indices;

        // Get the predicted subgoal states
        torch::Tensor indices = torch::arange(config.vqvae_config.num_embeddings, options_long)
                                    .reshape({config.vqvae_config.num_embeddings, 1})
                                    .repeat({batch_size, 1})
                                    .to(torch_device_);
        torch::Tensor decoded_output = model_->vqvae_->decode_from_indices(repeated_input_observations, indices);
        torch::Tensor obs_recon = torch::sigmoid(decoded_output);
        input = torch::concat({repeated_input_observations, obs_recon}, 1);
    }

    // Put model in train mode for learning
    model_->subgoal_net_->train();
    model_->subgoal_net_->zero_grad();

    torch::Tensor subgoal_logits =
        model_->inference_subgoal(input).reshape({batch_size, config.vqvae_config.num_embeddings});
    torch::Tensor loss = (expandeds * cross_entropy_loss(subgoal_logits, target_actions, false)).mean();
    auto loss_value = loss.item<double>();
    loss.backward();
    subgoal_optimizer_.step();

    return loss_value;
}

auto VSCWrapper::LearnConditionalLow(std::vector<ConditionalLowLearningInput> &batch) -> double {
    const int batch_size = static_cast<int>(batch.size());
    const auto options_float = torch::TensorOptions().dtype(torch::kFloat);
    const auto options_long = torch::TensorOptions().dtype(torch::kLong);
    const int input_flat_size = config.observation_shape.flat_size();

    // Create tensor from raw flat array
    // torch::from_blob requires a pointer to non-const and doesn't take ownership
    torch::Tensor encoder_input = torch::empty({2 * batch_size, input_flat_size}, options_float);
    torch::Tensor decoder_input = torch::empty({batch_size, input_flat_size}, options_float);
    torch::Tensor target_actions = torch::empty({batch_size, 1}, options_long);
    torch::Tensor expandeds = torch::empty({batch_size, 1}, options_float);

    for (const auto &[idx, batch_item] : std::views::enumerate(batch)) {
        const auto i = static_cast<int>(idx);    // stop torch from complaining about narrowing conversions
        encoder_input[2 * i + 0] = torch::from_blob(batch_item.obs_input.data(), {input_flat_size}, options_float);
        encoder_input[2 * i + 1] = torch::from_blob(batch_item.obs_target.data(), {input_flat_size}, options_float);
        decoder_input[i] = torch::from_blob(batch_item.obs_input.data(), {input_flat_size}, options_float);
        target_actions[i] = batch_item.target_action;
        expandeds[i] = static_cast<float>(batch_item.solution_expanded);
    }

    // Reshape to expected size for network (batch_size, flat) -> (batch_size, c, h, w)
    encoder_input = encoder_input.to(torch_device_);
    decoder_input = decoder_input.to(torch_device_);
    target_actions = target_actions.to(torch_device_);
    expandeds = expandeds.to(torch_device_);
    encoder_input = encoder_input.reshape(
        {batch_size, 2 * config.observation_shape.c, config.observation_shape.h, config.observation_shape.w}
    );
    decoder_input = decoder_input.reshape(
        {batch_size, config.observation_shape.c, config.observation_shape.h, config.observation_shape.w}
    );

    // Get decoded target from input pair
    torch::Tensor obs_recon;
    {
        model_->vqvae_->eval();
        const torch::NoGradGuard no_grad;
        obs_recon = model_->forward_vqvae(encoder_input, decoder_input).obs_recon;
        obs_recon = torch::softmax(obs_recon, 1);

        // Combine recon with current observation
        obs_recon =
            torch::concat({decoder_input, obs_recon}, 1)
                .reshape(
                    {batch_size, 2 * config.observation_shape.c, config.observation_shape.h, config.observation_shape.w}
                );
    }

    // Put model in train mode for learning
    model_->conditional_low_net_->train();
    model_->conditional_low_net_->zero_grad();

    torch::Tensor logits = model_->inference_conditional_low(obs_recon);
    torch::Tensor loss = (expandeds * cross_entropy_loss(logits, target_actions, false)).mean();
    auto loss_value = loss.item<double>();
    loss.backward();
    conditional_low_optimizer_.step();

    return loss_value;
}

}    // namespace hpts::model::wrapper
