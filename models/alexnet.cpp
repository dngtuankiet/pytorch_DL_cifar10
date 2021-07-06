#include "alexnet.h"

namespace model {
	namespace alexnet {
		AlexNetImpl::AlexNetImpl() {
			//conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 64, 11).stride(4).padding(2)));
			////conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 64, 3).stride(1).padding(2)));
			//pool1 = register_module("pool1", torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2)));
			////pool1 = register_module("pool1", torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2)));

			//conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 192, 5).padding(2)));
			////conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 192, 3).padding(2)));
			//pool2 = register_module("pool2", torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2)));
			////pool2 = register_module("pool2", torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2)));

			//conv3 = register_module("conv3", torch::nn::Conv2d(torch::nn::Conv2dOptions(192, 384, 3).padding(1)));
			////conv3 = register_module("conv3", torch::nn::Conv2d(torch::nn::Conv2dOptions(192, 384, 3).padding(1)));
			//pool3 = register_module("pool3", torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2)));

			//conv4 = register_module("conv4", torch::nn::Conv2d(torch::nn::Conv2dOptions(384, 256, 3).padding(1)));
			//conv5 = register_module("conv5", torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).padding(1)));

			//adapt = register_module("adapt", torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions(6)));

			//linear1 = register_module("linear1", torch::nn::Linear(9216, 4096)); //4096 2048
			////modifyhttps://analyticsindiamag.com/implementing-alexnet-using-pytorch-as-a-transfer-learning-model-in-multi-class-classification/
			//linear2 = register_module("linear2", torch::nn::Linear(4096, 4096));
			//linear3 = register_module("linear3", torch::nn::Linear(4096, 10));
			//dropout = register_module("dropout", torch::nn::Dropout(torch::nn::DropoutOptions(0.5)));


			//feature = torch::nn::Sequential(
			//	torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 64, 11).stride(4).padding(2)),
			//	torch::nn::ReLU(true),
			//	torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2)),
			//	torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 192, 5).padding(2)),
			//	torch::nn::ReLU(true),
			//	torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2)),
			//	torch::nn::Conv2d(torch::nn::Conv2dOptions(192, 384, 3).padding(1)),
			//	torch::nn::ReLU(true),
			//	torch::nn::Conv2d(torch::nn::Conv2dOptions(384, 256, 3).padding(1)),
			//	torch::nn::ReLU(true),
			//	torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).padding(1)),
			//	torch::nn::ReLU(true),
			//	torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2))
			//);

			//classifier = torch::nn::Sequential(
			//	torch::nn::Dropout(0.5),
			//	torch::nn::Linear(256 * 6 * 6, 4096),
			//	torch::nn::ReLU(true),
			//	torch::nn::Dropout(0.5),
			//	torch::nn::Linear(4096, 4096),
			//	torch::nn::ReLU(true),
			//	torch::nn::Linear(4096, 10)
			//);


			feature = torch::nn::Sequential(
				torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 64, 3).stride(1).padding(2)),
				torch::nn::ReLU(true),
				torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2)),
				torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 192, 3).padding(2)),
				torch::nn::ReLU(true),
				torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2)),
				torch::nn::Conv2d(torch::nn::Conv2dOptions(192, 384, 3).stride(1).padding(1)),
				torch::nn::ReLU(true),
				torch::nn::Conv2d(torch::nn::Conv2dOptions(384, 256, 3).stride(1).padding(1)),
				torch::nn::ReLU(true),
				torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).stride(1).padding(1)),
				torch::nn::ReLU(true),
				torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2))
			);

			classifier = torch::nn::Sequential(
				torch::nn::Dropout(0.5),
				torch::nn::Linear(4096, 2048),
				torch::nn::ReLU(true),
				torch::nn::Dropout(0.5),
				torch::nn::Linear(2048, 2048),
				torch::nn::ReLU(true),
				torch::nn::Linear(2048, 10)
			);

			feature = register_module("feature", feature);
			classifier = register_module("classifier", classifier);
		}

		torch::Tensor AlexNetImpl::forward(torch::Tensor x) {

			x = feature->forward(x);
			x = torch::flatten(x, 1);
			x = classifier->forward(x);


			//x = conv1->forward(x);
			//x = torch::relu(x);
			//x = pool1->forward(x);

			//x = conv2->forward(x);
			//x = torch::relu(x);
			//x = pool2->forward(x);

			//x = conv3->forward(x);
			//x = torch::relu(x);

			//x = conv4->forward(x);
			//x = torch::relu(x);

			//x = conv5->forward(x);
			//x = torch::relu(x);
			//x = pool3->forward(x);
			//	
			//// Classifier, 256 * 6 * 6 = 9216
			//x = adapt->forward(x);
			//x = torch::flatten(x, 1);

			//x = dropout(x);
			//x = linear1(x);
			//x = torch::relu(x);

			//x = dropout(x);
			//x = linear2(x);
			//x = torch::relu(x);

			//x = linear3(x);
			return x;
		}
		
	}
}