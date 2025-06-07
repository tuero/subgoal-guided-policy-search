// File: layers.h
// Description: Model layers/subnets

#ifndef HPTS_MODEL_LAYERS_H_
#define HPTS_MODEL_LAYERS_H_

// NOLINTBEGIN
#include <torch/torch.h>
// NOLINTEND

#include "common/observation.h"

#include <optional>
#include <string>
#include <vector>

namespace hpts::model {

// Conv and pooling layer defaults
torch::nn::Conv2dOptions conv1x1(int in_channels, int out_channels, int groups = 1);
torch::nn::Conv1dOptions conv1x1_1d(int in_channels, int out_channels, int groups = 1);
torch::nn::Conv2dOptions
    conv3x3(int in_channels, int out_channels, int stride = 1, int padding = 1, bool bias = true, int groups = 1);
torch::nn::AvgPool2dOptions avg_pool3x3(int kernel_size, int stride, int padding);

// MLP
class MLPImpl : public torch::nn::Module {
public:
    /**
     * @param input_size Size of the input layer
     * @param layer_sizes Vector of sizes for each hidden layer
     * @param output_size Size of the output layer
     */
    MLPImpl(int input_size, const std::vector<int> &layer_sizes, int output_size, const std::string &name);
    [[nodiscard]] auto forward(torch::Tensor x) -> torch::Tensor;

private:
    torch::nn::Sequential layers;
};
TORCH_MODULE(MLP);

// Main ResNet style residual block
class ResidualBlockImpl : public torch::nn::Module {
public:
    /**
     * @param num_channels Number of channels for the resnet block
     * @param layer_num Layer number id, used for pretty printing
     * @param use_batchnorm Flag to use batch normalization
     */
    ResidualBlockImpl(int num_channels, int layer_num, bool use_batchnorm, int groups = 1);
    [[nodiscard]] auto forward(torch::Tensor x) -> torch::Tensor;

private:
    torch::nn::Conv2d conv1;
    torch::nn::Conv2d conv2;
    torch::nn::BatchNorm2d batch_norm1;
    torch::nn::BatchNorm2d batch_norm2;
    bool use_batchnorm;
};
TORCH_MODULE(ResidualBlock);

/**
 * Initial input convolutional before ResNet residual blocks
 * Primary use is to take N channels and set to the expected number
 *  of channels for the rest of the resnet body
 */
class ResidualHeadImpl : public torch::nn::Module {
public:
    /**
     * @param input_channels Number of channels the head of the network receives
     * @param output_channels Number of output channels, should match the number of
     *                        channels used for the resnet body
     * @param use_batchnorm Flag to use batch normalization
     * @param name_prefix Used to ID the sub-module for pretty printing
     */
    ResidualHeadImpl(int input_channels, int output_channels, bool use_batchnorm, const std::string &name_prefix = "");
    [[nodiscard]] auto forward(torch::Tensor x) -> torch::Tensor;
    // Get the observation shape the network outputs given the input
    static ObservationShape encoded_state_shape(ObservationShape observation_shape);

private:
    torch::nn::Conv2d conv;
    torch::nn::BatchNorm2d batch_norm;
    bool use_batchnorm;
};
TORCH_MODULE(ResidualHead);

// Resnet style head and body
class ResnetBodyImpl : public torch::nn::Module {
public:
    /**
     * @param input_channels Number of input channels for the resnet
     * @param num_channels Number of channels for the resnet
     * @param resenet_blocks Number of blocks for the resnet
     * @param use_batchnorm Flag to use batch normalization
     */
    ResnetBodyImpl(int input_channels, int resnet_channels, int resnet_blocks, bool use_batchnorm);
    [[nodiscard]] auto forward(torch::Tensor x) -> torch::Tensor;

private:
    ResidualHead resnet_head_;
    torch::nn::ModuleList resnet_layers_;
};
TORCH_MODULE(ResnetBody);

// Necks comming off of resnets
struct ResnetNeckOutput {
    torch::Tensor logits;
    torch::Tensor policy;
    torch::Tensor log_policy;
    torch::Tensor heuristic;
};
struct ResnetNeckPolicyOutput {
    torch::Tensor logits;
    torch::Tensor policy;
    torch::Tensor log_policy;
};

class ResnetNeckImpl : public torch::nn::Module {
public:
    /**
     * @param obs_shape Observation shape, used to calculcate flat size
     * @param num_channels Number of channels for the resnet
     * @param policy_channels Number of channels for reduce sub-head for policy
     * @param heuristic_channels Number of channels for reduce sub-head for heuristic
     * @param policy_mlp_layers Layers for policy MLP sub-head
     * @param heuristic_mlp_layers Layers for heuristic MLP sub-head
     */
    ResnetNeckImpl(
        const ObservationShape &obs_shape,
        int num_actions,
        int resnet_channels,
        int policy_channels,
        int heuristic_channels,
        const std::vector<int> &policy_mlp_layers,
        const std::vector<int> &heuristic_mlp_layers
    );
    [[nodiscard]] auto forward(torch::Tensor x) -> ResnetNeckOutput;

private:
    int policy_mlp_input_size_;
    int heuristic_mlp_input_size_;
    torch::nn::Conv2d conv1x1_policy_;       // Conv pass before passing to policy mlp
    torch::nn::Conv2d conv1x1_heuristic_;    // Conv pass before passing to heuristic mlp
    MLP policy_mlp_;
    MLP heuristic_mlp_;
};
TORCH_MODULE(ResnetNeck);

class ResnetNeckPolicyImpl : public torch::nn::Module {
public:
    /**
     * @param obs_shape Observation shape, used to calculcate flat size
     * @param num_channels Number of channels for the resnet
     * @param policy_channels Number of channels for reduce sub-head for policy
     * @param policy_mlp_layers Layers for policy MLP sub-head
     */
    ResnetNeckPolicyImpl(
        const ObservationShape &obs_shape,
        int num_actions,
        int resnet_channels,
        int policy_channels,
        const std::vector<int> &policy_mlp_layers
    );
    [[nodiscard]] auto forward(torch::Tensor x) -> ResnetNeckPolicyOutput;

private:
    int policy_mlp_input_size_;
    torch::nn::Conv2d conv1x1_policy_;    // Conv pass before passing to policy mlp
    MLP policy_mlp_;
};
TORCH_MODULE(ResnetNeckPolicy);

// Group of Resnet blocks
class GroupOfBlocksImpl : public torch::nn::Module {
public:
    /**
     * @param num_channels Number of channels per Resent block
     * @param num_blocks Number of Resnet blocks
     * @param use_batchnorm Flag to use batch normalization
     */
    GroupOfBlocksImpl(int num_channels, int num_blocks, bool use_batchnorm);
    [[nodiscard]] auto forward(torch::Tensor x) -> torch::Tensor;

private:
    torch::nn::ModuleList blocks;
};
TORCH_MODULE(GroupOfBlocks);

// FilmRes Block
class FilmResBlockImpl : public torch::nn::Module {
public:
    /**
     * @param num_channels Number of channels for the FilmResBlock
     * @param latent_width Fully connected size (should be same size as channels, i.e. width of z)
     */
    FilmResBlockImpl(int num_channels, int latent_width);
    [[nodiscard]] auto forward(torch::Tensor x, torch::Tensor z) -> torch::Tensor;

private:
    int num_channels;
    torch::nn::Conv2d conv1;
    torch::nn::Conv2d conv2;
    torch::nn::InstanceNorm2d bn;
    torch::nn::Linear lin1;
};
TORCH_MODULE(FilmResBlock);

// FilmRes Block
class SonnetExponentialMovingAverageImpl : public torch::nn::Module {
public:
    /**
     * @param num_channels Number of channels for the FilmResBlock
     * @param latent_width Fully connected size (should be same size as channels, i.e. width of z)
     */
    SonnetExponentialMovingAverageImpl(double decay, torch::IntArrayRef shape);
    void update(torch::Tensor x);
    [[nodiscard]] auto forward(torch::Tensor x) -> torch::Tensor;

    double decay;
    double counter;
    torch::Tensor hidden;
    torch::Tensor average;
};
TORCH_MODULE(SonnetExponentialMovingAverage);

// Vector Quantizer
struct VectorQuantizerOutput {
    torch::Tensor quantized_x;
    std::optional<torch::Tensor> dictionary_loss;
    torch::Tensor commitment_loss;
    torch::Tensor encoding_indices;
};
struct QuantizeAndIndices {
    torch::Tensor quantized_x;
    torch::Tensor encoding_indices;
};

class VectorQuantizerImpl : public torch::nn::Module {
public:
    VectorQuantizerImpl(int embedding_dim, int num_embeddings, bool use_ema, double decay, double epsilon);

    // Used for getting target codebook entry to train conditional net
    [[nodiscard]] auto quantize(torch::Tensor x) -> QuantizeAndIndices;
    [[nodiscard]] auto get_codebook() const -> torch::Tensor;
    [[nodiscard]] auto forward(torch::Tensor x) -> VectorQuantizerOutput;
    [[nodiscard]] auto forward_from_index(torch::Tensor encoding_indices) -> torch::Tensor;

private:
    int embedding_dim;
    int num_embeddings;
    bool use_ema;
    double decay;
    double epsilon;
    SonnetExponentialMovingAverage N_i_ts;
    SonnetExponentialMovingAverage m_i_ts;
    torch::Tensor e_i_ts;
};
TORCH_MODULE(VectorQuantizer);

}    // namespace hpts::model

#endif    // HPTS_MODEL_LAYERS_H_
