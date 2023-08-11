//
// Created by Nabil Omi on 8/11/23.
//

#ifndef POLY_CONVMODEL_H
#define POLY_CONVMODEL_H

#include "torch/torch.h"

namespace nomi {

struct ConvModel : torch::nn::Module {
  ConvModel(uint in_h, uint in_w, uint in_ch, uint out_ch, uint n_classes);

  torch::Tensor forward(torch::Tensor x);

  torch::nn::Linear fc1{nullptr};
  torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
  torch::nn::MaxPool2d pool{nullptr};
  uint in_h, in_w, in_ch, out_ch, n_classes, dense_in_shape;
};

} // namespace nomi

#endif // POLY_CONVMODEL_H
