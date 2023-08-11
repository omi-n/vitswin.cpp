//
// Created by Nabil Omi on 8/11/23.
//

#include "ConvModel.h"

namespace nomi {

ConvModel::ConvModel(uint in_h, uint in_w, uint in_ch, uint out_ch,
                     uint n_classes) {
  this->in_h = in_h;
  this->in_w = in_w;
  this->in_ch = in_ch;
  this->out_ch = out_ch;
  this->n_classes = n_classes;

  conv1 = torch::nn::Conv2d(
      torch::nn::Conv2dOptions(in_ch, out_ch, {3, 3}).padding(1));
  conv2 = torch::nn::Conv2d(
      torch::nn::Conv2dOptions(out_ch, out_ch, {3, 3}).padding(1));
  conv3 = torch::nn::Conv2d(
      torch::nn::Conv2dOptions(out_ch, out_ch, {3, 3}).padding(1));

  pool = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions({2, 2}).stride(2));

  dense_in_shape = (in_h / 4) * (in_w / 4) * out_ch;
  fc1 = torch::nn::Linear(torch::nn::LinearOptions(dense_in_shape, n_classes));
}

torch::Tensor ConvModel::forward(torch::Tensor x) {
  x = torch::nn::functional::relu(conv1->forward(x));
  x = pool->forward(x);
  x = torch::nn::functional::relu(conv2->forward(x));
  x = pool->forward(x);
  x = torch::nn::functional::relu(conv3->forward(x));

  x = x.view({-1, dense_in_shape});
  x = fc1->forward(x);
  return x;
}

} // namespace nomi