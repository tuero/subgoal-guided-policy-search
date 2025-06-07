// File: layers.cpp
// Description: Model layers/subnets

// NOLINTBEGIN
#include <torch/torch.h>
// NOLINTEND

#include "model/layers.h"

#include <cassert>

namespace hpts::model {

// Create a conv1x1 layer using pytorch defaults
torch::nn::Conv2dOptions conv1x1(int in_channels, int out_channels, int groups) {
    return torch::nn::Conv2dOptions(in_channels, out_channels, 1)
        .stride(1)
        .padding(0)
        .bias(true)
        .dilation(1)
        .groups(groups)
        .padding_mode(torch::kZeros);
}

torch::nn::Conv1dOptions conv1x1_1d(int in_channels, int out_channels, int groups) {
    return torch::nn::Conv1dOptions(in_channels, out_channels, 1)
        .stride(1)
        .padding(0)
        .bias(true)
        .dilation(1)
        .groups(groups)
        .padding_mode(torch::kZeros);
}

// Create a conv3x3 layer using pytorch defaults
torch::nn::Conv2dOptions conv3x3(int in_channels, int out_channels, int stride, int padding, bool bias, int groups) {
    return torch::nn::Conv2dOptions(in_channels, out_channels, 3)
        .stride(stride)
        .padding(padding)
        .bias(bias)
        .dilation(1)
        .groups(groups)
        .padding_mode(torch::kZeros);
}

// Create a avgerage pooling layer using pytorch defaults
torch::nn::AvgPool2dOptions avg_pool3x3(int kernel_size, int stride, int padding) {
    return torch::nn::AvgPool2dOptions(kernel_size).stride(stride).padding(padding);
}

// Create a batchnorm2d layer using pytorch defaults
torch::nn::BatchNorm2dOptions bn(int num_filters) {
    return {num_filters};
}

// ------------------------------- MLP Network ------------------------------
// MLP
MLPImpl::MLPImpl(int input_size, const std::vector<int> &layer_sizes, int output_size, const std::string &name) {
    std::vector<int> sizes = layer_sizes;
    sizes.insert(sizes.begin(), input_size);
    sizes.push_back(output_size);

    // Walk through adding layers
    for (std::size_t i = 0; i < sizes.size() - 1; ++i) {
        layers->push_back("linear_" + std::to_string(i), torch::nn::Linear(sizes[i], sizes[i + 1]));
        if (i < sizes.size() - 2) {
            layers->push_back("activation_" + std::to_string(i), torch::nn::ReLU());
        }
    }
    register_module(name + "mlp", layers);
}

auto MLPImpl::forward(torch::Tensor x) -> torch::Tensor {
    torch::Tensor output = layers->forward(x);
    return output;
}
// ------------------------------- MLP Network ------------------------------

// ------------------------------ ResNet Block ------------------------------
// Main ResNet style residual block
ResidualBlockImpl::ResidualBlockImpl(int num_channels, int layer_num, bool use_batchnorm_, int groups)
    : conv1(conv3x3(num_channels, num_channels, 1, 1, true, groups)),
      conv2(conv3x3(num_channels, num_channels, 1, 1, true, groups)),
      batch_norm1(bn(num_channels)),
      batch_norm2(bn(num_channels)),
      use_batchnorm(use_batchnorm_) {
    register_module("resnet_" + std::to_string(layer_num) + "_conv1", conv1);
    register_module("resnet_" + std::to_string(layer_num) + "_conv2", conv2);
    if (use_batchnorm) {
        register_module("resnet_" + std::to_string(layer_num) + "_bn1", batch_norm1);
        register_module("resnet_" + std::to_string(layer_num) + "_bn2", batch_norm2);
    }
}

auto ResidualBlockImpl::forward(torch::Tensor x) -> torch::Tensor {
    const torch::Tensor residual = x;
    torch::Tensor output = conv1(x);
    if (use_batchnorm) {
        output = batch_norm1(output);
    }
    output = torch::relu(output);
    output = conv2(output);
    if (use_batchnorm) {
        output = batch_norm2(output);
    }
    output += residual;
    output = torch::relu(output);
    return output;
}
// ------------------------------ ResNet Block ------------------------------

// ------------------------------ ResNet Head -------------------------------
// Initial input convolutional before ResNet residual blocks
// Primary use is to take N channels and set to the expected number
//   of channels for the rest of the resnet body
ResidualHeadImpl::ResidualHeadImpl(
    int input_channels,
    int output_channels,
    bool use_batchnorm_,
    const std::string &name_prefix
)
    : conv(conv3x3(input_channels, output_channels)), batch_norm(bn(output_channels)), use_batchnorm(use_batchnorm_) {
    register_module(name_prefix + "resnet_head_conv", conv);
    if (use_batchnorm) {
        register_module(name_prefix + "resnet_head_bn", batch_norm);
    }
}

auto ResidualHeadImpl::forward(torch::Tensor x) -> torch::Tensor {
    torch::Tensor output = conv(x);
    if (use_batchnorm) {
        output = batch_norm(output);
    }
    output = torch::relu(output);
    return output;
}

// Shape doesn't change
ObservationShape ResidualHeadImpl::encoded_state_shape(ObservationShape observation_shape) {
    return observation_shape;
}
// ------------------------------ ResNet Head -------------------------------

// ------------------------------ ResNet Body -------------------------------
ResnetBodyImpl::ResnetBodyImpl(int input_channels, int resnet_channels, int resnet_blocks, bool use_batchnorm)
    : resnet_head_(ResidualHead(input_channels, resnet_channels, use_batchnorm, "head")) {
    // ResNet body
    for (int i = 0; i < resnet_blocks; ++i) {
        resnet_layers_->push_back(ResidualBlock(resnet_channels, i, use_batchnorm));
    }
    register_module("representation_head", resnet_head_);
    register_module("representation_layers", resnet_layers_);
}

auto ResnetBodyImpl::forward(torch::Tensor x) -> torch::Tensor {
    auto output = resnet_head_->forward(x);
    for (int i = 0; i < static_cast<int>(resnet_layers_->size()); ++i) {
        output = resnet_layers_[i]->as<ResidualBlock>()->forward(output);
    }
    return output;
}

// ------------------------------ ResNet Body -------------------------------

// ------------------------------ ResNet Neck -------------------------------
ResnetNeckImpl::ResnetNeckImpl(
    const ObservationShape &obs_shape,
    int num_actions,
    int resnet_channels,
    int policy_channels,
    int heuristic_channels,
    const std::vector<int> &policy_mlp_layers,
    const std::vector<int> &heuristic_mlp_layers
)
    : policy_mlp_input_size_(policy_channels * obs_shape.h * obs_shape.w),
      heuristic_mlp_input_size_(heuristic_channels * obs_shape.h * obs_shape.w),
      conv1x1_policy_(conv1x1(resnet_channels, policy_channels)),
      conv1x1_heuristic_(conv1x1(resnet_channels, heuristic_channels)),
      policy_mlp_(policy_mlp_input_size_, policy_mlp_layers, num_actions, "policy_head_"),
      heuristic_mlp_(heuristic_mlp_input_size_, heuristic_mlp_layers, 1, "heuristic_head_") {
    register_module("policy_1x1", conv1x1_policy_);
    register_module("heuristic_1x1", conv1x1_heuristic_);
    register_module("policy_mlp", policy_mlp_);
    register_module("heuristic_mlp", heuristic_mlp_);
}

auto ResnetNeckImpl::forward(torch::Tensor x) -> ResnetNeckOutput {
    // Reduce and mlp for policy
    torch::Tensor logits = conv1x1_policy_->forward(x);
    logits = logits.view({-1, policy_mlp_input_size_});

    logits = policy_mlp_->forward(logits);
    torch::Tensor policy = torch::softmax(logits, 1);
    torch::Tensor log_policy = torch::log_softmax(logits, 1);

    torch::Tensor heuristic = conv1x1_heuristic_->forward(x);
    heuristic = heuristic.view({-1, heuristic_mlp_input_size_});
    heuristic = heuristic_mlp_->forward(heuristic);

    return {logits, policy, log_policy, heuristic};
}

ResnetNeckPolicyImpl::ResnetNeckPolicyImpl(
    const ObservationShape &obs_shape,
    int num_actions,
    int resnet_channels,
    int policy_channels,
    const std::vector<int> &policy_mlp_layers
)
    : policy_mlp_input_size_(policy_channels * obs_shape.h * obs_shape.w),
      conv1x1_policy_(conv1x1(resnet_channels, policy_channels)),
      policy_mlp_(policy_mlp_input_size_, policy_mlp_layers, num_actions, "policy_head_") {
    register_module("policy_1x1", conv1x1_policy_);
    register_module("policy_mlp", policy_mlp_);
}

auto ResnetNeckPolicyImpl::forward(torch::Tensor x) -> ResnetNeckPolicyOutput {
    // Reduce and mlp for policy
    torch::Tensor logits = conv1x1_policy_->forward(x);
    logits = logits.view({-1, policy_mlp_input_size_});

    logits = policy_mlp_->forward(logits);
    torch::Tensor policy = torch::softmax(logits, 1);
    torch::Tensor log_policy = torch::log_softmax(logits, 1);

    return {.logits = logits, .policy = policy, .log_policy = log_policy};
}

// ------------------------------ ResNet Neck -------------------------------

// ------------------------------ Group of Blocks -------------------------------
GroupOfBlocksImpl::GroupOfBlocksImpl(int num_channels, int num_blocks, bool use_batchnorm) {
    for (int i = 0; i < num_blocks; ++i) {
        blocks->push_back(ResidualBlock(num_channels, i, use_batchnorm));
    }
    register_module("blocks", blocks);
}

auto GroupOfBlocksImpl::forward(torch::Tensor x) -> torch::Tensor {
    auto output = x;
    for (int i = 0; i < static_cast<int>(blocks->size()); ++i) {
        output = blocks[i]->as<ResidualBlock>()->forward(output);
    }
    return output;
}

// ------------------------------ Group of Blocks -------------------------------

// ------------------------------ FilmRes Block -------------------------------
FilmResBlockImpl::FilmResBlockImpl(int num_channels, int latent_width)
    : num_channels(num_channels),
      conv1(conv1x1(num_channels, num_channels)),
      conv2(conv3x3(num_channels, num_channels)),
      bn(num_channels),
      lin1(latent_width, 2 * num_channels) {
    register_module("conv1", conv1);
    register_module("conv2", conv2);
    register_module("lin1", lin1);
}

auto FilmResBlockImpl::forward(torch::Tensor x, torch::Tensor z) -> torch::Tensor {
    x = torch::relu(conv1(x));
    torch::Tensor y = bn(conv2(x));
    z = lin1(z);

    torch::Tensor gamma =
        z.index({torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, num_channels)})
            .unsqueeze(2)
            .unsqueeze(2);
    torch::Tensor beta =
        z.index({torch::indexing::Slice(), torch::indexing::Slice(num_channels, torch::indexing::None)})
            .unsqueeze(2)
            .unsqueeze(2);

    y = gamma * y + beta;
    return x + torch::relu(y);
}

// ------------------------------ FilmRes Block -------------------------------

// ------------------------------ FilmRes Block -------------------------------
SonnetExponentialMovingAverageImpl::SonnetExponentialMovingAverageImpl(double decay, torch::IntArrayRef shape)
    : decay(decay), counter(0) {
    // These are buffers, but we need them to be returned in model->named_parameters() for serialization
    hidden = this->register_parameter("hidden", torch::zeros(shape), false);
    average = this->register_parameter("average", torch::zeros(shape), false);
}

void SonnetExponentialMovingAverageImpl::update(torch::Tensor x) {
    counter += 1;
    const torch::NoGradGuard no_grad;
    hidden -= (hidden - x) * (1 - decay);
    average = hidden / (1 - std::pow(decay, counter));
}

auto SonnetExponentialMovingAverageImpl::forward(torch::Tensor x) -> torch::Tensor {
    update(x);
    return average;
}

// ------------------------------ FilmRes Block -------------------------------

// ------------------------------ Vector Quantizer -------------------------------
constexpr double e_i_ts_limit = 1.73205080756;    // sqrt(3)
VectorQuantizerImpl::VectorQuantizerImpl(
    int embedding_dim,
    int num_embeddings,
    bool use_ema,
    double decay,
    double epsilon
)
    : embedding_dim(embedding_dim),
      num_embeddings(num_embeddings),
      use_ema(use_ema),
      decay(decay),
      epsilon(epsilon),
      N_i_ts(decay, torch::IntArrayRef{num_embeddings}),
      m_i_ts(decay, torch::IntArrayRef{embedding_dim, num_embeddings}) {
    const auto float_options = torch::TensorOptions().dtype(torch::kFloat);
    // These are buffers, but we need them to be returned in model->named_parameters() for serialization
    e_i_ts = this->register_parameter(
        "e_i_ts",
        torch::empty({embedding_dim, num_embeddings}, float_options).uniform_(-e_i_ts_limit, e_i_ts_limit),
        !use_ema
    );
    register_module("N_i_ts", N_i_ts);
    register_module("m_i_ts", m_i_ts);
}

auto VectorQuantizerImpl::get_codebook() const -> torch::Tensor {
    return e_i_ts;
}

auto VectorQuantizerImpl::forward(torch::Tensor x) -> VectorQuantizerOutput {
    torch::Tensor flat_x = x.permute({0, 2, 3, 1}).reshape({-1, embedding_dim});
    torch::Tensor distances =
        torch::pow(flat_x, 2).sum(1, true) - (2 * torch::matmul(flat_x, e_i_ts)) + torch::pow(e_i_ts, 2).sum(0, true);
    torch::Tensor encoding_indices = distances.argmin(1);
    torch::Tensor quantized_x =
        torch::embedding(e_i_ts.transpose(0, 1), encoding_indices.view({x.size(0), x.size(2), x.size(3)}))
            .permute({0, 3, 1, 2});

    // Second term of Equation (3)
    std::optional<torch::Tensor> dictionary_loss;
    if (!use_ema) {
        dictionary_loss = torch::pow(x.detach() - quantized_x, 2).mean();
    }
    torch::Tensor commitment_loss = torch::pow(x - quantized_x.detach(), 2).mean();
    quantized_x = x + (quantized_x - x).detach();    // straight-through gradient

    if (use_ema && this->is_training()) {
        const torch::NoGradGuard no_grad;
        // Appendix A.1 of "Neural Discrete Representation Learning"

        // Cluster counts
        torch::Tensor encoding_one_hots = torch::one_hot(encoding_indices, num_embeddings).to(flat_x.options());
        torch::Tensor n_i_ts = encoding_one_hots.sum(0);

        // Updated exponential moving average of cluster counts
        // Equation (6)
        N_i_ts->update(n_i_ts);

        // Exponential moving afterage of the embeddings
        // Equation (7)
        torch::Tensor embed_sums = torch::matmul(flat_x.transpose(0, 1), encoding_one_hots);
        m_i_ts->update(embed_sums);

        // This is kind of weird
        // Compare: https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py#L270
        // See Equation (8).
        torch::Tensor N_i_ts_sum = N_i_ts->average.sum();
        torch::Tensor N_i_ts_stable =
            (N_i_ts->average + epsilon) / (N_i_ts_sum + num_embeddings * epsilon) * N_i_ts_sum;
        e_i_ts = m_i_ts->average / N_i_ts_stable.unsqueeze(0);
    }
    return {
        .quantized_x = quantized_x,
        .dictionary_loss = dictionary_loss,
        .commitment_loss = commitment_loss,
        .encoding_indices = encoding_indices.view({x.size(0), -1})
    };
}

auto VectorQuantizerImpl::quantize(torch::Tensor x) -> QuantizeAndIndices {
    const torch::NoGradGuard no_grad;
    torch::Tensor flat_x = x.permute({0, 2, 3, 1}).reshape({-1, embedding_dim});
    torch::Tensor distances =
        torch::pow(flat_x, 2).sum(1, true) - (2 * torch::matmul(flat_x, e_i_ts)) + torch::pow(e_i_ts, 2).sum(0, true);
    torch::Tensor encoding_indices = distances.argmin(1);
    torch::Tensor quantized_x =
        torch::embedding(e_i_ts.transpose(0, 1), encoding_indices.view({x.size(0), x.size(2), x.size(3)}))
            .permute({0, 3, 1, 2});
    return {.quantized_x = quantized_x, .encoding_indices = encoding_indices.view({x.size(0), -1})};
}

auto VectorQuantizerImpl::forward_from_index(torch::Tensor encoding_indices) -> torch::Tensor {
    const torch::NoGradGuard no_grad;
    torch::Tensor quantized_x =
        torch::embedding(e_i_ts.transpose(0, 1), encoding_indices.view({encoding_indices.size(0), 1, 1}))
            .permute({0, 3, 1, 2});
    return quantized_x;
}

// ------------------------------ Vector Quantizer -------------------------------

}    // namespace hpts::model
