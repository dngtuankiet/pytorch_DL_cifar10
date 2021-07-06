#pragma once
#include "torch/torch.h"

namespace model {
	namespace alexnet {
		class AlexNetImpl : public torch::nn::Module {
		public:
			std::string name = "AlexNet";
			AlexNetImpl();
			torch::Tensor forward(torch::Tensor x);
		private:
		/*	torch::nn::Conv2d conv1{ nullptr };
			torch::nn::Conv2d conv2{ nullptr };
			torch::nn::Conv2d conv3{ nullptr };
			torch::nn::Conv2d conv4{ nullptr };
			torch::nn::Conv2d conv5{ nullptr };

			torch::nn::MaxPool2d pool1{ nullptr };
			torch::nn::MaxPool2d pool2{ nullptr };
			torch::nn::MaxPool2d pool3{ nullptr };

			torch::nn::AdaptiveAvgPool2d adapt{ nullptr };

			torch::nn::Linear linear1{ nullptr };
			torch::nn::Linear linear2{ nullptr };
			torch::nn::Linear linear3{ nullptr };

			torch::nn::Dropout dropout{ nullptr };*/

			torch::nn::Sequential feature{ nullptr };
			torch::nn::Sequential classifier{ nullptr };

			
		};

		TORCH_MODULE(AlexNet);
	}
}