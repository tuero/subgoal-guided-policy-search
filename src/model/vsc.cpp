// File: vsc.cpp
// Description: Bundled VQVAE, Subgoal, and Conditional-Low nets

#include "model/vsc.h"

namespace hpts::model::network {

VSCFlatSubNetImpl::VSCFlatSubNetImpl(
    const VSCSubNetConfig &config,
    int input_channels,
    int input_height,
    int input_width,
    int embedding_dim,
    int output_size
)
    : config_(config),
      conv_flat_size_(config.reduce_channels * input_height * input_width),
      mlp_input_size_(conv_flat_size_ + embedding_dim),
      resnet_head_(ResidualHead(input_channels, config.resnet_channels, config.use_batchnorm)),
      conv1x1_(conv1x1(config.resnet_channels, config.reduce_channels)),
      mlp_(mlp_input_size_, config_.mlp_layers, output_size, "policy_head_") {
    // ResNet bodies
    for (int i = 0; i < config.resnet_blocks; ++i) {
        resnet_layers_->push_back(ResidualBlock(config.resnet_channels, i, config.use_batchnorm));
    }
    register_module("representation_head", resnet_head_);
    register_module("representation_layers", resnet_layers_);
    register_module("policy_1x1", conv1x1_);
    register_module("mlp", mlp_);
}

auto VSCFlatSubNetImpl::forward(torch::Tensor obs, torch::Tensor codebook) -> torch::Tensor {
    int batch_size = static_cast<int>(obs.size(0));
    int codebook_batch_size = static_cast<int>(codebook.size(0));
    torch::Tensor output = resnet_head_->forward(obs);
    for (int i = 0; i < config_.resnet_blocks; ++i) {
        output = resnet_layers_[i]->as<ResidualBlock>()->forward(output);
    }

    // conv1x1 and reshape to flat
    output = torch::relu(conv1x1_->forward(output).reshape({-1, conv_flat_size_}));

    // Repeating logic
    //        [A,              [1,           [A 1,
    // out =   B,   codebook =  2]  concat =  A 2,
    //         C]                             B 1,
    //                                        B 2,
    //                                        C 1,
    //                                        C 2]

    // Repeat entire batch num_embeddings times
    // if codebook_batch_size = (batch_size * num_embeddings), this will repeat num_embeddings times
    // if codebook_batch_size = batch_size, then this wont repeat (in event we are given quantized entries)
    // (batch_size * num_embeddings, conv_flat_size_)
    output = output.repeat_interleave(codebook_batch_size / batch_size, 0);

    // Concat and send to mlps
    // (batch_size * num_embeddings, conv_flat_size_ + embedding_dim)
    output = torch::concat({output, codebook}, 1);

    output = mlp_->forward(output);
    return output;
}

VSCSubNetImpl::VSCSubNetImpl(
    const VSCSubNetConfig &config,
    int input_channels,
    int input_height,
    int input_width,
    int output_size
)
    : config_(config),
      mlp_input_size_(config.reduce_channels * input_height * input_width),
      resnet_head_(ResidualHead(input_channels, config.resnet_channels, config.use_batchnorm)),
      conv1x1_(conv1x1(config.resnet_channels, config.reduce_channels)),
      mlp_(mlp_input_size_, config_.mlp_layers, output_size, "policy_head_") {
    // ResNet bodies
    for (int i = 0; i < config.resnet_blocks; ++i) {
        resnet_layers_->push_back(ResidualBlock(config.resnet_channels, i, config.use_batchnorm));
    }
    register_module("representation_head", resnet_head_);
    register_module("representation_layers", resnet_layers_);
    register_module("policy_1x1", conv1x1_);
    register_module("mlp", mlp_);
}

auto VSCSubNetImpl::forward(torch::Tensor obs) -> torch::Tensor {
    torch::Tensor output = resnet_head_->forward(obs);
    for (int i = 0; i < config_.resnet_blocks; ++i) {
        output = resnet_layers_[i]->as<ResidualBlock>()->forward(output);
    }

    // conv1x1 and reshape to flat
    output = conv1x1_->forward(output).reshape({-1, mlp_input_size_});
    output = mlp_->forward(output);
    return output;
}

// -----------------------------------------

VSCFlatImpl::VSCFlatImpl(const VSCConfig &config)
    : config_(config),
      subgoal_net_(
          config.subgoal_config,
          config.observation_shape.c,
          config.observation_shape.h,
          config.observation_shape.w,
          config.vqvae_config.embedding_dim,
          1
      ),
      conditional_low_net_(
          config.conditional_low_config,
          config.observation_shape.c,
          config.observation_shape.h,
          config.observation_shape.w,
          config.vqvae_config.embedding_dim,
          config.num_actions
      ),
      vqvae_(
          config.observation_shape,
          config.vqvae_config.resnet_channels,
          config.vqvae_config.embedding_dim,
          config.vqvae_config.num_embeddings,
          config.vqvae_config.use_ema,
          config.vqvae_config.decay,
          config.vqvae_config.epsilon
      ) {
    register_module("subgoal", subgoal_net_);
    register_module("conditional_low", conditional_low_net_);
    register_module("vqvae", vqvae_);
}

auto VSCFlatImpl::forward_vqvae(torch::Tensor encoder_input, torch::Tensor decoder_input) -> VQVAEOutput {
    return vqvae_->forward(encoder_input, decoder_input);
}

auto VSCFlatImpl::quantize(torch::Tensor encoder_input) -> QuantizeAndIndices {
    return vqvae_->quantize(encoder_input);
}

auto VSCFlatImpl::get_repeated_codebook(int batch_size) -> torch::Tensor {
    // Get VQVAE codebook and repeat to match repeated conv output
    torch::Tensor codebook = vqvae_->get_codebook();    // (embedding_dim, num_embeddings)
    codebook = codebook.transpose(0, 1);                // (num_embeddings, embedding_dim)
    codebook = codebook.repeat({batch_size, 1});        // (batch_size * num_embeddings, embedding_dim)
    return codebook;
}

auto VSCFlatImpl::inference(torch::Tensor obs) -> VCSInferenceOutput {
    int batch_size = static_cast<int>(obs.size(0));
    torch::Tensor codebook = get_repeated_codebook(batch_size);
    return {
        .subgoal_logits = subgoal_net_->forward(obs, codebook),
        .conditional_low_logits = conditional_low_net_->forward(obs, codebook)
    };
}

auto VSCFlatImpl::inference_subgoal(torch::Tensor obs) -> torch::Tensor {
    int batch_size = static_cast<int>(obs.size(0));
    torch::Tensor codebook = get_repeated_codebook(batch_size);
    return subgoal_net_->forward(obs, codebook);
}

auto VSCFlatImpl::inference_conditional_low(torch::Tensor obs) -> torch::Tensor {
    int batch_size = static_cast<int>(obs.size(0));
    torch::Tensor codebook = get_repeated_codebook(batch_size);
    return conditional_low_net_->forward(obs, codebook);
}

auto VSCFlatImpl::inference_conditional_low(torch::Tensor obs, torch::Tensor quantized_x) -> torch::Tensor {
    return conditional_low_net_->forward(obs, quantized_x);
}

// -----------------------------------------

VSCImpl::VSCImpl(const VSCConfig &config)
    : config_(config),
      subgoal_net_(
          config.subgoal_config,
          2 * config.observation_shape.c,
          config.observation_shape.h,
          config.observation_shape.w,
          1
      ),
      conditional_low_net_(
          config.conditional_low_config,
          2 * config.observation_shape.c,
          config.observation_shape.h,
          config.observation_shape.w,
          config.num_actions
      ),
      vqvae_(
          config.observation_shape,
          config.vqvae_config.resnet_channels,
          config.vqvae_config.embedding_dim,
          config.vqvae_config.num_embeddings,
          config.vqvae_config.use_ema,
          config.vqvae_config.decay,
          config.vqvae_config.epsilon
      ) {
    register_module("subgoal", subgoal_net_);
    register_module("conditional_low", conditional_low_net_);
    register_module("vqvae", vqvae_);
}
auto VSCImpl::quantize(torch::Tensor encoder_input) -> QuantizeAndIndices {
    return vqvae_->quantize(encoder_input);
}

auto VSCImpl::forward_vqvae(torch::Tensor encoder_input, torch::Tensor decoder_input) -> VQVAEOutput {
    return vqvae_->forward(encoder_input, decoder_input);
}

auto VSCImpl::inference(torch::Tensor obs) -> VCSInferenceOutput {
    return {.subgoal_logits = subgoal_net_->forward(obs), .conditional_low_logits = conditional_low_net_->forward(obs)};
}

auto VSCImpl::inference_subgoal(torch::Tensor obs) -> torch::Tensor {
    return subgoal_net_->forward(obs);
}

auto VSCImpl::inference_conditional_low(torch::Tensor obs) -> torch::Tensor {
    return conditional_low_net_->forward(obs);
}

}    // namespace hpts::model::network
