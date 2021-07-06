#include "googlenet.h"

namespace model {
	namespace googlenet {
		InceptionImpl::InceptionImpl(int64_t in_planes, int64_t n1x1, int64_t n3x3red, int64_t n3x3, int64_t n5x5red, int64_t n5x5, int64_t pool_planes) {
			b1 = torch::nn::Sequential(
				torch::nn::Conv2d(torch::nn::Conv2dOptions(in_planes, n1x1, 1)),
				torch::nn::BatchNorm2d(n1x1),
				torch::nn::ReLU(true)
			);

			b2 = torch::nn::Sequential(
				torch::nn::Conv2d(torch::nn::Conv2dOptions(in_planes, n3x3red, 1)),
				torch::nn::BatchNorm2d(n3x3red),
				torch::nn::ReLU(true),
				torch::nn::Conv2d(torch::nn::Conv2dOptions(n3x3red, n3x3, 3).padding(1)),
				torch::nn::BatchNorm2d(n3x3),
				torch::nn::ReLU(true)
			);

			b3 = torch::nn::Sequential(
				torch::nn::Conv2d(torch::nn::Conv2dOptions(in_planes, n5x5red, 1)),
				torch::nn::BatchNorm2d(n5x5red),
				torch::nn::ReLU(true),
				torch::nn::Conv2d(torch::nn::Conv2dOptions(n5x5red, n5x5, 3).padding(1)),
				torch::nn::BatchNorm2d(n5x5),
				torch::nn::ReLU(true),
				torch::nn::Conv2d(torch::nn::Conv2dOptions(n5x5, n5x5, 3).padding(1)),
				torch::nn::BatchNorm2d(n5x5),
				torch::nn::ReLU(true)
			);

			b4 = torch::nn::Sequential(
				torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(1).padding(1)),
				torch::nn::Conv2d(torch::nn::Conv2dOptions(in_planes, pool_planes, 1)),
				torch::nn::BatchNorm2d(pool_planes),
				torch::nn::ReLU(true)
			);

			b1 = register_module("b1", b1);
			b2 = register_module("b2", b2);
			b3 = register_module("b3", b3);
			b4 = register_module("b4", b4);
		
		}

		torch::Tensor InceptionImpl::forward(torch::Tensor x) {
			auto y1 = b1->forward(x);
			auto y2 = b2->forward(x);
			auto y3 = b3->forward(x);
			auto y4 = b4->forward(x);
			auto out = torch::cat({y1, y2, y3 ,y4}, 1);
			return out;
		}

		GoogLeNetImpl::GoogLeNetImpl() {
			pre_layers = torch::nn::Sequential(
				torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 192, 3).padding(1)),
				torch::nn::BatchNorm2d(192),
				torch::nn::ReLU(true)
			);

			a3 = Inception(192, 64, 96, 128, 16, 32, 32);
			b3 = Inception(256, 128, 128, 192, 32, 96, 64);

			maxpool = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2).padding(1));

			a4 = Inception(480, 192, 96, 208, 16, 48, 64);
			b4 = Inception(512, 160, 112, 224, 24, 64, 64);
			c4 = Inception(512, 128, 128, 256, 24, 64, 64);
			d4 = Inception(512, 112, 144, 288, 32, 64, 64);
			e4 = Inception(528, 256, 160, 320, 32, 128, 128);

			a5 = Inception(832, 256, 160, 320, 32, 128, 128);
			b5 = Inception(832, 384, 192, 384, 48, 128, 128);

			avgpool = torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions(8).stride(1));
			Linear = torch::nn::Linear(1024, 10);

			register_module("a3", a3);
			register_module("b3", b3);
			register_module("a4", a4);
			register_module("b4", b4);
			register_module("c4", c4);
			register_module("d4", d4);
			register_module("e4", e4);
			register_module("a5", a5);
			register_module("b5", b5);
			register_module("pre_layers", pre_layers);
			register_module("maxpool", maxpool);
			register_module("avgpool", avgpool);
			register_module("Linear", Linear);

		}

		torch::Tensor GoogLeNetImpl::forward(torch::Tensor x) {
			auto out = pre_layers->forward(x);
			out = a3->forward(out);
			out = b3->forward(out);
			out = maxpool->forward(out);
			out = a4->forward(out);
			out = b4->forward(out);
			out = c4->forward(out);
			out = d4->forward(out);
			out = e4->forward(out);
			out = maxpool->forward(out);
			out = a5->forward(out);
			out = b5->forward(out);
			out = avgpool->forward(out);
			out = torch::flatten(out, 1);
			out = Linear->forward(out);
			return out;

		}
	}
}