#include "densenet.h"
#include <cmath>

namespace model {
	namespace densenet {
		BottleNeckImpl::BottleNeckImpl(int64_t in_planes, int64_t growth_rate) {
			bn1 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(in_planes));
			conv1 = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_planes, 4 * growth_rate, 1).bias(false));
			bn2 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(4 * growth_rate));
			conv2 = torch::nn::Conv2d(torch::nn::Conv2dOptions(4 * growth_rate, growth_rate, 3).padding(1).bias(false));

			bn1 = register_module("bn1", bn1);
			conv1 = register_module("conv1", conv1);
			bn2 = register_module("bn2", bn2);
			conv2 = register_module("conv2", conv2);
		}

		torch::Tensor BottleNeckImpl::forward(torch::Tensor x) {
			
			torch::Tensor out = bn1->forward(x);
			out = torch::relu(out);
			out = conv1->forward(out);

			out = bn2->forward(out);
			out = torch::relu(out);
			out = conv2->forward(out);

			out = torch::cat({ out, x }, 1);
			return out;
		}

		TransitionImpl::TransitionImpl(int64_t in_planes, int64_t out_planes) {
			bn = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(in_planes));
			conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_planes, out_planes, 1).bias(false));

			bn = register_module("bn", bn);
			conv = register_module("conv", conv);
		}

		torch::Tensor TransitionImpl::forward(torch::Tensor x) {
			x = bn->forward(x);
			x = torch::relu(x);
			x = conv->forward(x);
			x = torch::avg_pool2d(x, 2);
			return x;
		}

		torch::nn::Sequential DenseNetImpl::_make_dense_layer(const char* block, int64_t in_planes, int64_t nblock) {
			torch::nn::Sequential layers;
#ifdef DEBUG_MAKE_DENSE
			std::cout << "make dense loop - in_planes: " << in_planes << std::endl;
			std::cout << "make dense loop - nblock: " << nblock << std::endl;
#endif
			for (int64_t i = 0; i < nblock; i++)
			{
				layers->push_back(BottleNeck(in_planes, this->growth_rate));
				in_planes += this->growth_rate;
#ifdef DEBUG_MAKE_DENSE
				std::cout << "make dense loop: " << i << std::endl;
#endif
			}
			return layers;
		}

		DenseNetImpl::DenseNetImpl(const char* block, std::vector<int64_t> nblocks, int64_t growth_rate, double reduction, int64_t num_classes) {
			this->growth_rate = growth_rate;
			double num_planes = 2 * growth_rate;

			conv1 = torch::nn::Conv2d(torch::nn::Conv2dOptions(3, num_planes, 3).padding(1).bias(false));
#ifdef DEBUG
			std::cout << "conv1\n";
#endif
			/* Dense 1*/
			torch::nn::Sequential dense1;
			dense1 = _make_dense_layer("BottleNeck", num_planes, nblocks[0]);
			/*int64_t in_planes = num_planes;
			for (int64_t i = 0; i < nblocks[0]; i++)
			{
				dense1->push_back(BottleNeck(in_planes, this->growth_rate));
				in_planes += this->growth_rate;
			}*/
			num_planes += nblocks[0] * growth_rate;
			int64_t out_planes = int64_t(floor(num_planes * reduction));
			torch::nn::Sequential trans1;
			trans1->push_back(Transition(num_planes, out_planes));
			num_planes = out_planes;
#ifdef DEBUG
			std::cout << "dense 1\n";
#endif
			/* Dense 2*/
			torch::nn::Sequential dense2;
			dense2 = _make_dense_layer("BottleNeck", num_planes, nblocks[1]);
			/*in_planes = num_planes;
			for (int64_t i = 0; i < nblocks[1]; i++)
			{
				dense1->push_back(BottleNeck(in_planes, this->growth_rate));
				in_planes += this->growth_rate;
			}*/
			num_planes += nblocks[1] * growth_rate;
			out_planes = int64_t(floor(num_planes * reduction));
			torch::nn::Sequential trans2;
			trans2->push_back(Transition(num_planes, out_planes));
			num_planes = out_planes;
#ifdef DEBUG
			std::cout << "dense 2\n";
#endif
			/* Dense 3*/
			torch::nn::Sequential dense3;
			dense3 = _make_dense_layer("BottleNeck", num_planes, nblocks[2]);
			/*in_planes = num_planes;
			for (int64_t i = 0; i < nblocks[2]; i++)
			{
				dense1->push_back(BottleNeck(in_planes, this->growth_rate));
				in_planes += this->growth_rate;
			}*/
			num_planes += nblocks[2] * growth_rate;
			out_planes = int64_t(floor(num_planes * reduction));
			torch::nn::Sequential trans3;
			trans3->push_back(Transition(num_planes, out_planes));
			num_planes = out_planes;
#ifdef DEBUG
			std::cout << "dense 3\n";
#endif
			/* Dense 4*/
			torch::nn::Sequential dense4;
			dense4 = _make_dense_layer("BottleNeck", num_planes, nblocks[3]);
			/*in_planes = num_planes;
			for (int64_t i = 0; i < nblocks[3]; i++)
			{
				dense1->push_back(BottleNeck(in_planes, this->growth_rate));
				in_planes += this->growth_rate;
			}*/
			num_planes += nblocks[3] * growth_rate;
#ifdef DEBUG
			std::cout << "dense 4n";
#endif
			bn = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(num_planes));
			linear = torch::nn::Linear(num_planes, num_classes);
#ifdef DEBUG
			std::cout << "bn - linear\n";
#endif
			conv1 = register_module("conv1", conv1);

			dense1_ = register_module("dense1_", dense1);
			trans1_ = register_module("trans1_", trans1);

			dense2_ = register_module("dense2_", dense2);
			trans2_ = register_module("trans2_", trans2);

			dense3_ = register_module("dense3_", dense3);
			trans3_ = register_module("trans3_", trans3);

			dense4_ = register_module("dense4_", dense4);

			bn = register_module("bn", bn);
			linear = register_module("linear", linear);
#ifdef DEBUG
			std::cout << "registered\n";
#endif
		}

		torch::Tensor DenseNetImpl::forward(torch::Tensor x) {
			x = conv1->forward(x);
			x = dense1_->forward(x);
			x = trans1_->forward(x);

			x = dense2_->forward(x);
			x = trans2_->forward(x);

			x = dense3_->forward(x);
			x = trans3_->forward(x);

			x = dense4_->forward(x);

			x = bn->forward(x);
			x = torch::relu(x);
			x = torch::avg_pool2d(x, 4);

			x = torch::flatten(x, 1);
			x = linear->forward(x);
			return x;
		}


	}
}