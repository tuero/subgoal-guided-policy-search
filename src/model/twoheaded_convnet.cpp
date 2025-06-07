// File: twoheaded_convent.cpp
// Description: Convnet for combined policy and heuristic predictions

#include "model/twoheaded_convnet.h"

namespace hpts::model::network {

TwoHeadedConvNetImpl::TwoHeadedConvNetImpl(
    const ObservationShape &observation_shape,
    int num_actions,
    int resnet_channels,
    int resnet_blocks,
    int policy_channels,
    int heuristic_channels,
    const std::vector<int> &policy_mlp_layers,
    const std::vector<int> &heuristic_mlp_layers,
    bool use_batchnorm
)
    : input_channels_(observation_shape.c),
      input_height_(observation_shape.h),
      input_width_(observation_shape.w),
      resnet_channels_(resnet_channels),
      policy_channels_(policy_channels),
      heuristic_channels_(heuristic_channels),
      policy_mlp_input_size_(policy_channels_ * input_height_ * input_width_),
      heuristic_mlp_input_size_(heuristic_channels_ * input_height_ * input_width_),
      resnet_head_(ResidualHead(input_channels_, resnet_channels_, use_batchnorm, "representation_")),
      conv1x1_policy_(conv1x1(resnet_channels_, policy_channels_)),
      conv1x1_heuristic_(conv1x1(resnet_channels_, heuristic_channels_)),
      policy_mlp_(policy_mlp_input_size_, policy_mlp_layers, num_actions, "policy_head_"),
      heuristic_mlp_(heuristic_mlp_input_size_, heuristic_mlp_layers, 1, "heuristic_head_") {
    // ResNet body
    for (int i = 0; i < resnet_blocks; ++i) {
        resnet_layers_->push_back(ResidualBlock(resnet_channels_, i, use_batchnorm));
    }
    register_module("representation_head", resnet_head_);
    register_module("representation_layers", resnet_layers_);
    register_module("policy_1x1", conv1x1_policy_);
    register_module("heuristic_1x1", conv1x1_heuristic_);
    register_module("policy_mlp", policy_mlp_);
    register_module("heuristic_mlp", heuristic_mlp_);
}

auto TwoHeadedConvNetImpl::forward(torch::Tensor x) -> TwoHeadedConvNetOutput {
    torch::Tensor output = resnet_head_->forward(x);
    // ResNet body
    for (int i = 0; i < static_cast<int>(resnet_layers_->size()); ++i) {
        output = resnet_layers_[i]->as<ResidualBlock>()->forward(output);
    }

    // Reduce and mlp for policy
    torch::Tensor logits = conv1x1_policy_->forward(output);
    torch::Tensor heuristic = conv1x1_heuristic_->forward(output);
    logits = logits.view({-1, policy_mlp_input_size_});
    heuristic = heuristic.view({-1, heuristic_mlp_input_size_});

    logits = policy_mlp_->forward(logits);
    const torch::Tensor policy = torch::softmax(logits, 1);
    const torch::Tensor log_policy = torch::log_softmax(logits, 1);
    heuristic = heuristic_mlp_->forward(heuristic);
    // heuristic = torch::softplus(heuristic);

    return {logits, policy, log_policy, heuristic};
}

}    // namespace hpts::model::network
