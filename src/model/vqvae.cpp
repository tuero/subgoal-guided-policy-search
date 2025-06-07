// File: vqvae.cpp
// Description: VQVAE

#include "model/vqvae.h"

#include <cassert>

namespace hpts::model::network {

// VQVAE Encoder
VQVAEEncoderImpl::VQVAEEncoderImpl(int in_channels, int resnet_channels, int num_blocks, int output_size)
    : resnet_channels(resnet_channels),
      head(in_channels, resnet_channels, false),
      body(resnet_channels, num_blocks, false),
      pool(1),
      tail(resnet_channels, output_size) {
    register_module("head", head);
    register_module("body", body);
    register_module("pool", pool);
    register_module("tail", tail);
}

auto VQVAEEncoderImpl::forward(torch::Tensor observation) -> torch::Tensor {
    int batch_size = static_cast<int>(observation.size(0));
    torch::Tensor output = head->forward(observation);
    output = body->forward(output);
    output = pool->forward(output);
    output = output.reshape({batch_size, resnet_channels});
    output = tail->forward(output);
    return output;
}

// VQVAE Decoder
VQVAEDecoderImpl::VQVAEDecoderImpl(
    const ObservationShape obs_shape,
    int codebook_dim,
    int resnet_channels,
    int num_blocks
)
    : obs_shape(obs_shape),
      obs_head(conv3x3(obs_shape.c, obs_shape.c)),
      codebook_head(codebook_dim, obs_shape.h * obs_shape.w),
      head(obs_shape.c + 1, resnet_channels, false),
      body(resnet_channels, num_blocks, false),
      tail(conv3x3(resnet_channels, obs_shape.c)) {
    register_module("obs_head", obs_head);
    register_module("codebook_head", codebook_head);
    register_module("head", head);
    register_module("body", body);
    register_module("tail", tail);
}

auto VQVAEDecoderImpl::forward(torch::Tensor observation, torch::Tensor latents) -> torch::Tensor {
    int batch_size = static_cast<int>(observation.size(0));
    torch::Tensor x = obs_head->forward(observation);
    torch::Tensor z = codebook_head->forward(latents);
    z = z.reshape({batch_size, 1, obs_shape.h, obs_shape.w});
    torch::Tensor output = torch::cat({x, z}, 1);

    output = head->forward(output);
    output = body->forward(output);
    output = tail->forward(output);
    return output;
}

// VQVAE
VQVAEImpl::VQVAEImpl(
    const ObservationShape &obs_shape,
    int resnet_channels,
    int embedding_dim,
    int num_embeddings,
    bool use_ema,
    double decay,
    double epsilon
)
    : encoder(2 * obs_shape.c, resnet_channels, 4, embedding_dim),
      vq(embedding_dim, num_embeddings, use_ema, decay, epsilon),
      decoder(obs_shape, embedding_dim, resnet_channels, 4) {
    register_module("encoder", encoder);
    register_module("vq", vq);
    register_module("decoder", decoder);
}

auto VQVAEImpl::get_codebook() const -> torch::Tensor {
    return vq->get_codebook();
}

auto VQVAEImpl::decode_from_indices(torch::Tensor decoder_input, torch::Tensor indices) -> torch::Tensor {
    torch::Tensor z_quantized = vq->forward_from_index(indices);
    z_quantized = z_quantized.squeeze(-1).squeeze(-1);
    return decoder->forward(decoder_input, z_quantized);
}

auto VQVAEImpl::quantize(torch::Tensor encoder_input) -> QuantizeAndIndices {
    torch::Tensor z = encoder->forward(encoder_input);
    z = z.unsqueeze(-1).unsqueeze(-1);
    return vq->quantize(z);
}

auto VQVAEImpl::forward(torch::Tensor encoder_input, torch::Tensor decoder_input) -> VQVAEOutput {
    torch::Tensor z = encoder->forward(encoder_input);
    z = z.unsqueeze(-1).unsqueeze(-1);
    VectorQuantizerOutput vq_output = vq->forward(z);
    torch::Tensor x_quantized = vq_output.quantized_x.squeeze(-1).squeeze(-1);
    torch::Tensor obs_recon = decoder->forward(decoder_input, x_quantized);
    return {
        .obs_recon = std::move(obs_recon),
        .encoding_indices = std::move(vq_output).encoding_indices,
        .dictionary_loss = std::move(vq_output).dictionary_loss,
        .commitment_loss = std::move(vq_output).commitment_loss
    };
}

}    // namespace hpts::model::network
