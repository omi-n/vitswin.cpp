#include "models/ConvModel.h"
#include "torch/torch.h"

#include <iostream>

int main(int argc, const char *argv[]) {
  torch::Tensor in_tensor = torch::zeros({128, 3, 28, 28});

  auto model = std::make_shared<nomi::ConvModel>(28, 28, 3, 16, 10);

  torch::Tensor out;
  for(int i=0; i<10000;i++) {
    std::cout << "\r" << (i / 10000.) * 100 << "%";
    out = model->forward(in_tensor);
  }

  std::cout << std::endl << out.sizes() << std::endl;
  return 0;
}
