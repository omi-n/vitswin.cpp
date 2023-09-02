#include "models/Transformer.h"
#include "torch/torch.h"

#include <iostream>

int main() {
  auto model = std::make_unique<nomi::SelfAttention>(4, 4);
  auto query = torch::zeros({128, 4, 4, 4});
  auto key = torch::zeros({128, 4, 4, 4});
  auto value = torch::zeros({128, 4, 4, 4});

  auto ret = model->forward(query);
  std::cout << ret.sizes() << std::endl;
  return 0;
}
