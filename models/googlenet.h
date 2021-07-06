#pragma once
#include "torch/torch.h"

namespace model {
	namespace googlenet {
		class InceptionImpl : public torch::nn::Module {
		public:
			InceptionImpl(int64_t in_planes, int64_t n1x1, int64_t n3x3red, int64_t n3x3, int64_t n5x5red, int64_t n5x5, int64_t pool_planes);
			torch::Tensor forward(torch::Tensor x);
		private:
			torch::nn::Sequential b1{ nullptr };
			torch::nn::Sequential b2{ nullptr };
			torch::nn::Sequential b3{ nullptr };
			torch::nn::Sequential b4{ nullptr };
		};

		TORCH_MODULE(Inception);

		class GoogLeNetImpl : public torch::nn::Module {
		public:
			std::string name = "GoogLeNet";
			GoogLeNetImpl();
			torch::Tensor forward(torch::Tensor x);
		private:
			torch::nn::Sequential pre_layers{ nullptr };
			Inception a3{ nullptr };
			Inception b3{ nullptr };

			torch::nn::MaxPool2d maxpool{ nullptr };

			Inception a4{ nullptr };
			Inception b4{ nullptr };
			Inception c4{ nullptr };
			Inception d4{ nullptr };
			Inception e4{ nullptr };

			Inception a5{ nullptr };
			Inception b5{ nullptr };

			torch::nn::AvgPool2d avgpool{ nullptr };
			torch::nn::Linear Linear{ nullptr };
		};

		TORCH_MODULE(GoogLeNet);
	}
}