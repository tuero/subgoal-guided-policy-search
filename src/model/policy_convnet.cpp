// File: policy_convent.cpp
// Description: Convnet for policy predictions

#include "model/policy_convnet.h"

namespace hpts::model::network {

PolicyConvNetImpl::PolicyConvNetImpl(
    const ObservationShape &observation_shape,
    int num_actions,
    int resnet_channels,
    int resnet_blocks,
    int reduce_channels,
    const std::vector<int> &mlp_layers,
    bool use_batchnorm
)
    : input_channels_(observation_shape.c),
      input_height_(observation_shape.h),
      input_width_(observation_shape.w),
      resnet_channels_(resnet_channels),
      reduce_channels_(reduce_channels),
      mlp_input_size_(reduce_channels_ * input_height_ * input_width_),
      resnet_head_(ResidualHead(input_channels_, resnet_channels_, use_batchnorm, "representation_")),
      conv1x1_(conv1x1(resnet_channels_, reduce_channels_)),
      mlp_(mlp_input_size_, mlp_layers, num_actions, "policy_head_") {
    // ResNet body
    for (int i = 0; i < resnet_blocks; ++i) {
        resnet_layers_->push_back(ResidualBlock(resnet_channels_, i, use_batchnorm));
    }
    register_module("representation_head", resnet_head_);
    register_module("representation_layers", resnet_layers_);
    register_module("policy_1x1", conv1x1_);
    register_module("mlp", mlp_);
}

auto PolicyConvNetImpl::forward(torch::Tensor x) -> PolicyConvNetOutput {
    torch::Tensor output = resnet_head_->forward(x);
    // ResNet body
    for (int i = 0; i < static_cast<int>(resnet_layers_->size()); ++i) {
        output = resnet_layers_[i]->as<ResidualBlock>()->forward(output);
    }

    // Reduce and mlp for policy
    torch::Tensor logits = conv1x1_->forward(output);
    logits = logits.view({-1, mlp_input_size_});

    logits = mlp_->forward(logits);
    const torch::Tensor policy = torch::softmax(logits, 1);
    const torch::Tensor log_policy = torch::log_softmax(logits, 1);

    return {logits, policy, log_policy};
}

}    // namespace hpts::model::network
