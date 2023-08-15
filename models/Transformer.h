//
// Created by Nabil Omi on 8/11/23.
//
#ifndef POLY_TRANSFORMER_H
#define POLY_TRANSFORMER_H

#include "torch/torch.h"
#include <iostream>
#include <limits>
#include <optional>

namespace nomi {

class PositionalEncoding : torch::nn::Module {
public:
  PositionalEncoding(int d_model, int max_len);

  torch::Tensor forward(torch::Tensor x);

private:
  torch::Tensor pe{nullptr}, position{nullptr}, div_term{nullptr};
};

class ScaledDotProductAttention : torch::nn::Module {
public:
  ScaledDotProductAttention(int dim);

  torch::Tensor forward(torch::Tensor query, torch::Tensor key,
                        torch::Tensor value,
                        torch::optional<torch::Tensor> mask = torch::nullopt);

private:
  int sqrt_dim = 0;
};

class MultiHeadAttention : torch::nn::Module {
public:
  MultiHeadAttention(int d_model, int num_heads);
  torch::Tensor forward(torch::Tensor query, torch::Tensor key,
                        torch::Tensor value,
                        torch::optional<torch::Tensor> mask = torch::nullopt);

private:
  torch::nn::Linear query_proj{nullptr}, key_proj{nullptr}, value_proj{nullptr};
  torch::nn::ModuleHolder<ScaledDotProductAttention> scaled_dot_attn{nullptr};
  int d_head, num_heads;
};

// class SelfAttention : torch::nn::Module {
// public:
//   SelfAttention(int d_model, int num_heads);
//   torch::Tensor forward(torch::Tensor x,
//                         torch::optional<torch::Tensor> mask =
//                         torch::nullopt);

// private:
//   torch::nn::ModuleHolder<MultiHeadAttention> multi_head;
// };

} // namespace nomi

#endif // POLY_TRANSFORMER_H