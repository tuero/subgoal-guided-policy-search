// File: vsc.h
// Description: Bundled VQVAE, Subgoal, and Conditional-Low nets

#ifndef HPTS_MODEL_VSC_H_
#define HPTS_MODEL_VSC_H_

#include "common/observation.h"
#include "model/layers.h"
#include "model/vqvae.h"

#include <vector>

// NOLINTBEGIN
#include <torch/torch.h>
// NOLINTEND

namespace hpts::model::network {

struct VCSInferenceOutput {
    torch::Tensor subgoal_logits;
    torch::Tensor conditional_low_logits;
};

struct VSCSubNetConfig {
    int resnet_channels;
    int resnet_blocks;
    int reduce_channels;
    std::vector<int> mlp_layers;
    bool use_batchnorm;
};

class VSCFlatSubNetImpl : public torch::nn::Module {
public:
    VSCFlatSubNetImpl(
        const VSCSubNetConfig &config,
        int input_channels,
        int input_height,
        int input_width,
        int embedding_dim,
        int output_size
    );
    [[nodiscard]] auto forward(torch::Tensor obs, torch::Tensor codebook) -> torch::Tensor;

private:
    VSCSubNetConfig config_;
    int conv_flat_size_;
    int mlp_input_size_;
    ResidualHead resnet_head_;
    torch::nn::Conv2d conv1x1_;
    MLP mlp_;
    torch::nn::ModuleList resnet_layers_;
};
TORCH_MODULE(VSCFlatSubNet);

class VSCSubNetImpl : public torch::nn::Module {
public:
    VSCSubNetImpl(
        const VSCSubNetConfig &config,
        int input_channels,
        int input_height,
        int input_width,
        int output_size
    );
    [[nodiscard]] auto forward(torch::Tensor obs) -> torch::Tensor;

private:
    VSCSubNetConfig config_;
    int mlp_input_size_;
    ResidualHead resnet_head_;
    torch::nn::Conv2d conv1x1_;
    MLP mlp_;
    torch::nn::ModuleList resnet_layers_;
};
TORCH_MODULE(VSCSubNet);

// ---------------------------

struct VSCConfig {
    ObservationShape observation_shape;
    int num_actions;
    VSCSubNetConfig subgoal_config;
    VSCSubNetConfig conditional_low_config;
    VQVAEConfig vqvae_config;
};

class VSCFlatImpl : public torch::nn::Module {
public:
    VSCFlatImpl(const VSCConfig &config);

    [[nodiscard]] auto inference(torch::Tensor obs) -> VCSInferenceOutput;
    [[nodiscard]] auto inference_subgoal(torch::Tensor obs) -> torch::Tensor;
    [[nodiscard]] auto inference_conditional_low(torch::Tensor obs) -> torch::Tensor;
    [[nodiscard]] auto inference_conditional_low(torch::Tensor obs, torch::Tensor quantized_x) -> torch::Tensor;
    [[nodiscard]] auto quantize(torch::Tensor encoder_input) -> QuantizeAndIndices;
    [[nodiscard]] auto forward_vqvae(torch::Tensor encoder_input, torch::Tensor decoder_input) -> VQVAEOutput;

    VSCConfig config_;
    VSCFlatSubNet subgoal_net_;
    VSCFlatSubNet conditional_low_net_;
    VQVAE vqvae_;

private:
    [[nodiscard]] auto get_repeated_codebook(int batch_size) -> torch::Tensor;
};
TORCH_MODULE(VSCFlat);

// ---------------------------

class VSCImpl : public torch::nn::Module {
public:
    VSCImpl(const VSCConfig &config);

    [[nodiscard]] auto inference(torch::Tensor obs) -> VCSInferenceOutput;
    [[nodiscard]] auto inference_subgoal(torch::Tensor obs) -> torch::Tensor;
    [[nodiscard]] auto inference_conditional_low(torch::Tensor obs) -> torch::Tensor;
    [[nodiscard]] auto quantize(torch::Tensor encoder_input) -> QuantizeAndIndices;
    [[nodiscard]] auto forward_vqvae(torch::Tensor encoder_input, torch::Tensor decoder_input) -> VQVAEOutput;

    VSCConfig config_;
    VSCSubNet subgoal_net_;
    VSCSubNet conditional_low_net_;
    VQVAE vqvae_;
};
TORCH_MODULE(VSC);

}    // namespace hpts::model::network

#endif    // HPTS_MODEL_VSC_H_
