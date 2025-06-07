// torch_util.h
// Utility functions for libtorch c++

#ifndef HPTS_TORCH_UTIL_H_
#define HPTS_TORCH_UTIL_H_

#include <vector>

// NOLINTBEGIN
#include <torch/torch.h>
// NOLINTEND

namespace hpts::model {

/**
 * Get tensor from vector
 * @param x The input tensor
 * @return std vector of tensor values
 */
template <typename OutT, typename InT>
auto tensor_to_vec(torch::Tensor x) -> std::vector<OutT> {
    const auto cx = x.contiguous();
    auto ptr = cx.data_ptr<InT>();
    std::vector<OutT> data;
    data.reserve(static_cast<std::size_t>(cx.numel()));
    for (int i = 0; i < cx.numel(); ++i) {
        data.push_back(static_cast<OutT>(*(ptr++)));    // NOLINT (*-pointer-arithmetic)
    }
    return data;
}

/**
 * Cross entropy loss
 * @param logits (B, num_actions)
 * @param target_actions (B, 1)
 * @param reduce Flag to mean reduce
 * @return Tensor loss
 */
inline auto cross_entropy_loss(torch::Tensor logits, torch::Tensor target_actions, bool reduce = true)
    -> torch::Tensor {
    if (target_actions.dim() > 1) {
        target_actions = target_actions.flatten();
    }
    const torch::Tensor loss = torch::cross_entropy_loss(logits, target_actions, {}, at::Reduction::None);
    return reduce ? loss.mean() : loss;
}

/**
 * Mean Squared Error loss
 * @param output (*)
 * @param target (*)
 * @param reduce Flag to mean reduce
 * @return Tensor loss
 */
inline auto mean_squared_error_loss(torch::Tensor output, torch::Tensor target, bool reduce = true) -> torch::Tensor {
    return torch::mse_loss(output, target, reduce ? at::Reduction::Mean : at::Reduction::None);
}

inline void init_model(torch::nn::Module &module) {
    torch::NoGradGuard no_grad;
    if (auto *linear = module.as<torch::nn::Linear>()) {
        torch::nn::init::kaiming_normal_(linear->weight, 0, torch::kFanIn, torch::kReLU);
        if (linear->bias.defined()) {
            torch::nn::init::constant_(linear->bias, 0);
        }
    } else if (auto *conv = module.as<torch::nn::Conv2d>()) {
        torch::nn::init::kaiming_normal_(conv->weight, 0, torch::kFanIn, torch::kReLU);
        if (conv->bias.defined()) {
            torch::nn::init::constant_(conv->bias, 0);
        }
    }
}

}    // namespace hpts::model

#endif    // HPTS_TORCH_UTIL_H_
