#pragma once
#include "torch\torch.h"

//#define DEBUG
//#define DEBUG_MAKE_DENSE
//#define DEBUG_MAKE_TRANS

namespace model {
	namespace densenet {
		class DenseNetImpl : public torch::nn::Module {
		public:
			std::string name = "DenseNet";
			DenseNetImpl(const char* block, std::vector<int64_t> nblocks, int64_t growth_rate = 12, double reduction = 0.5, int64_t num_classes = 10);
			torch::Tensor forward(torch::Tensor);
		private:
			torch::nn::Sequential _make_dense_layer(const char* block, int64_t in_planes, int64_t nblock);

			int64_t growth_rate;
			torch::nn::Conv2d conv1{ nullptr };
			torch::nn::Sequential dense1_{ nullptr };
			torch::nn::Sequential trans1_{ nullptr };

			torch::nn::Sequential dense2_{ nullptr };
			torch::nn::Sequential trans2_{ nullptr };

			torch::nn::Sequential dense3_{ nullptr };
			torch::nn::Sequential trans3_{ nullptr };

			torch::nn::Sequential dense4_{ nullptr };

			torch::nn::BatchNorm2d bn{ nullptr };
			torch::nn::Linear linear{ nullptr };
		};
		TORCH_MODULE(DenseNet);

		class BottleNeckImpl : public torch::nn::Module {
		public:
			BottleNeckImpl(int64_t in_planes, int64_t growth_rate);
			torch::Tensor forward(torch::Tensor);
		private:
			torch::nn::BatchNorm2d bn1{ nullptr };
			torch::nn::Conv2d conv1{ nullptr };
			torch::nn::BatchNorm2d bn2{ nullptr };
			torch::nn::Conv2d conv2{ nullptr };
		};
		TORCH_MODULE(BottleNeck);

		class TransitionImpl : public torch::nn::Module {
		public:
			TransitionImpl(int64_t in_planes, int64_t outplanes);
			torch::Tensor forward(torch::Tensor);
		private:
			torch::nn::BatchNorm2d bn{ nullptr };
			torch::nn::Conv2d conv{ nullptr };
		};
		TORCH_MODULE(Transition);


		class DenseNet121Impl : public DenseNetImpl {
		public:
			DenseNet121Impl() : DenseNetImpl("BottleNeck", { 6, 12, 24, 16 }, 12) {};
		};
		TORCH_MODULE(DenseNet121);

	} //namepsace densenet
} //namespace model
