//
// Created by Nabil Omi on 8/11/23.
//

#ifndef POLY_CONVMODEL_H
#define POLY_CONVMODEL_H

#include "torch/torch.h"

namespace nomi {

struct ViT : torch::nn::Module {
  ViT();
  torch::Tensor forward();

 private:
  torch::nn::ModuleHolder<Transformer> transformer{nullptr};
}

// Need to implement the following:
// SELayer
// ConvBlock
// WeightlessCenteredBatchNorm
// Residual
// PreNorm
// FeedForward
// Attention
// MaiaNetEncoder
// PolicyMap

}  // namespace nomi

#endif  // POLY_CONVMODEL_H