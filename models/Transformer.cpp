//
// Created by Nabil Omi on 8/11/23.
//

#include "Transformer.h"

namespace nomi {

PositionalEncoding::PositionalEncoding(int d_model, int max_len) {
  pe = register_buffer("pe", torch::zeros({max_len, d_model}));
  this->position =
      torch::arange(0, torch::Scalar(max_len), torch::kFloat32).unsqueeze(1);

  auto tmp = (-std::log(10000.0) / d_model);
  this->div_term = torch::exp(
      torch::arange(0, torch::Scalar(d_model), 2, torch::kFloat32) * tmp);
  pe.slice(1, 0, -1, 2) = torch::sin(position * div_term);
  pe.slice(1, 1, -1, 2) = torch::cos(position * div_term);
  this->pe = pe.unsqueeze(0);
}

torch::Tensor PositionalEncoding::forward(torch::Tensor x) {
  return x + pe.slice(1, 0, x.size(1));
}

ScaledDotProductAttention::ScaledDotProductAttention(int dim) {
  this->sqrt_dim = std::sqrt(static_cast<float>(dim));
}

torch::Tensor ScaledDotProductAttention::forward(
    torch::Tensor query, torch::Tensor key, torch::Tensor value,
    torch::optional<torch::Tensor> mask) {
  auto score = torch::bmm(query, key.transpose(1, 2)) / sqrt_dim;

  if (mask) {
    score = score.masked_fill_(mask.value().view(score.sizes()),
                               -1 * std::numeric_limits<double>::min());
  }

  auto attention = torch::nn::functional::softmax(
      score, torch::nn::functional::SoftmaxFuncOptions(-1));
  auto context = torch::bmm(attention, value);
  return context;
}

MultiHeadAttention::MultiHeadAttention(int d_model, int num_heads) {
  this->d_head = d_model / num_heads;

  auto linear_options = torch::nn::LinearOptions(d_model, d_head * num_heads);
  this->num_heads = num_heads;
  this->query_proj = torch::nn::Linear(linear_options);
  this->value_proj = torch::nn::Linear(linear_options);
  this->key_proj = torch::nn::Linear(linear_options);
  this->scaled_dot_attn = torch::nn::ModuleHolder(
      std::make_shared<ScaledDotProductAttention>(d_head));
}

torch::Tensor MultiHeadAttention::forward(torch::Tensor query,
                                          torch::Tensor key,
                                          torch::Tensor value,
                                          torch::optional<torch::Tensor> mask) {
  int batch_size = value.sizes()[0];
  auto out_query = query_proj(query).view({batch_size, -1, num_heads, d_head});
  auto out_key = key_proj(key).view({batch_size, -1, num_heads, d_head});
  auto out_value = value_proj(value).view({batch_size, -1, num_heads, d_head});

  out_query = out_query.permute({2, 0, 1, 3})
                  .contiguous()
                  .view({batch_size * num_heads, -1, d_head});

  out_key = out_key.permute({2, 0, 1, 3})
                .contiguous()
                .view({batch_size * num_heads, -1, d_head});

  out_value = out_value.permute({2, 0, 1, 3})
                  .contiguous()
                  .view({batch_size * num_heads, -1, d_head});

  torch::Tensor context;
  if (mask) {
    auto def_mask = mask.value().unsqueeze(1).repeat({1, num_heads, 1, 1});
    context = scaled_dot_attn->forward(out_query, out_key, out_value, def_mask);
  } else {
    context = scaled_dot_attn->forward(out_query, out_key, out_value);
  }

  context = context.view({num_heads, batch_size, -1, d_head});
  context = context.permute({1, 2, 0, 3})
                .contiguous()
                .view({batch_size, -1, num_heads * d_head});
  return context;
}

SelfAttention::SelfAttention(int d_model, int num_heads) {
  this->multi_head = torch::nn::ModuleHolder<MultiHeadAttention>(
      std::make_shared<MultiHeadAttention>(d_model, num_heads));
}

torch::Tensor SelfAttention::forward(torch::Tensor x,
                                     torch::optional<torch::Tensor> mask) {
  if (mask)
    return this->multi_head->forward(x, x, x, mask.value());
  else
    return this->multi_head->forward(x, x, x);
}

}  // namespace nomi