// File: heuristic_convent.cpp
// Description: Convnet for heuristic predictions

#include "model/heuristic_convnet.h"

namespace hpts::model::network {

HeuristicConvNetImpl::HeuristicConvNetImpl(
    const ObservationShape &observation_shape,
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
      mlp_(mlp_input_size_, mlp_layers, 1, "heuristic_head_") {
    // ResNet body
    for (int i = 0; i < resnet_blocks; ++i) {
        resnet_layers_->push_back(ResidualBlock(resnet_channels_, i, use_batchnorm));
    }
    register_module("representation_head", resnet_head_);
    register_module("representation_layers", resnet_layers_);
    register_module("heuristic_1x1", conv1x1_);
    register_module("mlp", mlp_);
}

auto HeuristicConvNetImpl::forward(torch::Tensor x) -> HeuristicConvNetOutput {
    torch::Tensor output = resnet_head_->forward(x);
    // ResNet body
    for (int i = 0; i < static_cast<int>(resnet_layers_->size()); ++i) {
        output = resnet_layers_[i]->as<ResidualBlock>()->forward(output);
    }

    torch::Tensor heuristic = conv1x1_->forward(output);
    heuristic = heuristic.view({-1, mlp_input_size_});
    heuristic = mlp_->forward(heuristic);
    return {heuristic};
}

}    // namespace hpts::model::network
