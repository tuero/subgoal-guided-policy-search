// Description: VQVAE network

#ifndef HPTS_MODEL_VQVAE_H_
#define HPTS_MODEL_VQVAE_H_

#include "common/observation.h"
#include "model/layers.h"

// NOLINTBEGIN
#include <torch/torch.h>
// NOLINTEND

namespace hpts::model::network {

struct VQVAEConfig {
    int resnet_channels;
    int embedding_dim;
    int num_embeddings;
    bool use_ema;
    double decay;
    double epsilon;
    double beta;
};

struct VQVAEOutput {
    torch::Tensor obs_recon;
    torch::Tensor encoding_indices;
    std::optional<torch::Tensor> dictionary_loss;
    torch::Tensor commitment_loss;
};

// Encoder
class VQVAEEncoderImpl : public torch::nn::Module {
public:
    VQVAEEncoderImpl(int in_channels, int resenet_channels, int num_blocks, int output_size);
    [[nodiscard]] auto forward(torch::Tensor observation) -> torch::Tensor;

private:
    int resnet_channels;
    ResidualHead head;
    GroupOfBlocks body;
    torch::nn::AdaptiveAvgPool2d pool;
    torch::nn::Linear tail;
};
TORCH_MODULE(VQVAEEncoder);

// Decoder
class VQVAEDecoderImpl : public torch::nn::Module {
public:
    VQVAEDecoderImpl(const ObservationShape obs_shape, int codebook_dim, int resnet_channels, int num_blocks);
    [[nodiscard]] auto forward(torch::Tensor observation, torch::Tensor latents) -> torch::Tensor;

private:
    ObservationShape obs_shape;
    torch::nn::Conv2d obs_head;
    torch::nn::Linear codebook_head;
    ResidualHead head;
    GroupOfBlocks body;
    torch::nn::Conv2d tail;
};
TORCH_MODULE(VQVAEDecoder);

class VQVAEImpl : public torch::nn::Module {
public:
    VQVAEImpl(
        const ObservationShape &obs_shape,
        int resnet_channels,
        int embedding_dim,
        int num_embeddings,
        bool use_ema,
        double decay,
        double epsilon
    );

    [[nodiscard]] auto get_codebook() const -> torch::Tensor;
    [[nodiscard]] auto quantize(torch::Tensor encoder_input) -> QuantizeAndIndices;
    [[nodiscard]] auto forward(torch::Tensor encoder_input, torch::Tensor decoder_input) -> VQVAEOutput;
    [[nodiscard]] auto decode_from_indices(torch::Tensor decoder_input, torch::Tensor indices) -> torch::Tensor;

private:
    VQVAEEncoder encoder;
    VectorQuantizer vq;
    VQVAEDecoder decoder;
};
TORCH_MODULE(VQVAE);

}    // namespace hpts::model::network

#endif    // HPTS_MODEL_VQVAE_H_
