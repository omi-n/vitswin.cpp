//
// Created by Nabil Omi on 8/11/23.
//
#ifndef POLY_TRANSFORMER_H
#define POLY_TRANSFORMER_H

#include <iostream>
#include <limits>
#include <optional>

#include "torch/torch.h"

namespace nomi {

class PositionalEncoding : torch::nn::Module {
 public:
  PositionalEncoding(int d_model, int max_len);

  torch::Tensor forward(torch::Tensor x);

 private:
  torch::Tensor pe, position, div_term;
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

class SelfAttention : torch::nn::Module {
 public:
  SelfAttention(int d_model, int num_heads);
  torch::Tensor forward(torch::Tensor x,
                        torch::optional<torch::Tensor> mask = torch::nullopt);

 private:
  torch::nn::ModuleHolder<MultiHeadAttention> multi_head{nullptr};
};

template <class TORCH_MODULE>
class PreNorm : torch::nn::Module {
 public:
  PreNorm(int dim, torch::nn::ModuleHolder<TORCH_MODULE> sublayer);
  torch::Tensor forward(torch::Tensor x);

 private:
  torch::nn::ModuleHolder<TORCH_MODULE> sublayer{nullptr};
  torch::nn::LayerNorm norm{nullptr};
};

template <class TORCH_MODULE>
class Residual : torch::nn::Module {
 public:
  Residual();
  torch::Tensor forward(torch::Tensor x);

 private:
  torch::nn::ModuleHolder<TORCH_MODULE> sublayer{nullptr};
};

class FeedForward : torch::nn::Module {

};

class PreNorm : torch::nn::Module {

};

}  // namespace nomi

#endif  // POLY_TRANSFORMER_H