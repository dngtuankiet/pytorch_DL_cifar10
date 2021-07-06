#include "vgg.h"

namespace model {
	namespace vgg {
        Vgg16Impl::Vgg16Impl() {
            // Initialize VGG-16
            conv1_1 = register_module("conv1_1", torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 64, 3).padding(1)));
            batch1_1 = register_module("batch1_1", torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(64)));
            conv1_2 = register_module("conv1_2", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).padding(1)));
            batch1_2 = register_module("batch1_2", torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(64)));
            // Insert pool layer
            conv2_1 = register_module("conv2_1", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).padding(1)));
            batch2_1 = register_module("batch2_1", torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(128)));
            conv2_2 = register_module("conv2_2", torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, 3).padding(1)));
            batch2_2 = register_module("batch2_2", torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(128)));
            // Insert pool layer
            conv3_1 = register_module("conv3_1", torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 3).padding(1)));
            batch3_1 = register_module("batch3_1", torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(256)));
            conv3_2 = register_module("conv3_2", torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).padding(1)));
            batch3_2 = register_module("batch3_2", torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(256)));
            conv3_3 = register_module("conv3_3", torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).padding(1)));
            batch3_3 = register_module("batch3_3", torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(256)));
            // Insert pool layer
            conv4_1 = register_module("conv4_1", torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 512, 3).padding(1)));
            batch4_1 = register_module("batch4_1", torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(512)));
            conv4_2 = register_module("conv4_2", torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).padding(1)));
            batch4_2 = register_module("batch4_2", torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(512)));
            conv4_3 = register_module("conv4_3", torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).padding(1)));
            batch4_3 = register_module("batch4_3", torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(512)));
            // Insert pool layer
            conv5_1 = register_module("conv5_1", torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).padding(1)));
            batch5_1 = register_module("batch5_1", torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(512)));
            conv5_2 = register_module("conv5_2", torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).padding(1)));
            batch5_2 = register_module("batch5_2", torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(512)));
            conv5_3 = register_module("conv5_3", torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).padding(1)));
            batch5_3 = register_module("batch5_3", torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(512)));
            // Insert pool layer
            //classifier = register_module("classifier", torch::nn::Linear(512, 10));
            fc1 = register_module("fc1", torch::nn::Linear(512, 4096));
            fc2 = register_module("fc2", torch::nn::Linear(4096, 4096));
            fc3 = register_module("fc3", torch::nn::Linear(4096, 10));
        }

        torch::Tensor Vgg16Impl::forward(torch::Tensor x) {
            x = conv1_1->forward(x);
            x = batch1_1->forward(x);
            x = torch::relu(x);
            x = conv1_2->forward(x);
            x = batch1_2->forward(x);
            x = torch::relu(x);
            x = torch::max_pool2d(x, 2);
            x = conv2_1->forward(x);
            x = batch2_1->forward(x);
            x = torch::relu(x);
            x = conv2_2->forward(x);
            x = batch2_2->forward(x);
            x = torch::relu(x);
            x = torch::max_pool2d(x, 2);
            x = conv3_1->forward(x);
            x = batch3_1->forward(x);
            x = torch::relu(x);
            x = conv3_2->forward(x);
            x = batch3_2->forward(x);
            x = torch::relu(x);
            x = conv3_3->forward(x);
            x = batch3_3->forward(x);
            x = torch::relu(x);
            x = torch::max_pool2d(x, 2);
            x = conv4_1->forward(x);
            x = batch4_1->forward(x);
            x = torch::relu(x);
            x = conv4_2->forward(x);
            x = batch4_2->forward(x);
            x = torch::relu(x);
            x = conv4_3->forward(x);
            x = batch4_3->forward(x);
            x = torch::relu(x);
            x = torch::max_pool2d(x, 2);
            x = conv5_1->forward(x);
            x = batch5_1->forward(x);
            x = torch::relu(x);
            x = conv5_2->forward(x);
            x = batch5_2->forward(x);
            x = torch::relu(x);
            x = conv5_3->forward(x);
            x = batch5_3->forward(x);
            x = torch::relu(x);
            x = torch::adaptive_avg_pool2d(x, { 1, 1 });
            x = torch::flatten(x, 1);
            //x = classifier->forward(x);
            x = fc1->forward(x);
            x = fc2->forward(x);
            x = fc3->forward(x);
            //std::cout << "x size: " << x.sizes() << std::endl;
            //x = torch::log_softmax(x, 1);
            return x;
        }

        Vgg19Impl::Vgg19Impl() {
            // Initialize VGG-16
            conv1_1 = register_module("conv1_1", torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 64, 3).padding(1)));
            batch1_1 = register_module("batch1_1", torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(64)));
            conv1_2 = register_module("conv1_2", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).padding(1)));
            batch1_2 = register_module("batch1_2", torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(64)));
            // Insert pool layer
            conv2_1 = register_module("conv2_1", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).padding(1)));
            batch2_1 = register_module("batch2_1", torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(128)));
            conv2_2 = register_module("conv2_2", torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, 3).padding(1)));
            batch2_2 = register_module("batch2_2", torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(128)));
            // Insert pool layer
            conv3_1 = register_module("conv3_1", torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 3).padding(1)));
            batch3_1 = register_module("batch3_1", torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(256)));
            conv3_2 = register_module("conv3_2", torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).padding(1)));
            batch3_2 = register_module("batch3_2", torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(256)));
            conv3_3 = register_module("conv3_3", torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).padding(1)));
            batch3_3 = register_module("batch3_3", torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(256)));
            conv3_4 = register_module("conv3_4", torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).padding(1)));
            batch3_4 = register_module("batch3_4", torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(256)));

            // Insert pool layer
            conv4_1 = register_module("conv4_1", torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 512, 3).padding(1)));
            batch4_1 = register_module("batch4_1", torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(512)));
            conv4_2 = register_module("conv4_2", torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).padding(1)));
            batch4_2 = register_module("batch4_2", torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(512)));
            conv4_3 = register_module("conv4_3", torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).padding(1)));
            batch4_3 = register_module("batch4_3", torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(512)));
            conv4_4 = register_module("conv4_4", torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).padding(1)));
            batch4_4 = register_module("batch4_4", torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(512)));
            // Insert pool layer
            conv5_1 = register_module("conv5_1", torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).padding(1)));
            batch5_1 = register_module("batch5_1", torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(512)));
            conv5_2 = register_module("conv5_2", torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).padding(1)));
            batch5_2 = register_module("batch5_2", torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(512)));
            conv5_3 = register_module("conv5_3", torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).padding(1)));
            batch5_3 = register_module("batch5_3", torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(512)));
            conv5_4 = register_module("conv5_4", torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).padding(1)));
            batch5_4 = register_module("batch5_4", torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(512)));
            // Insert pool layer
            //classifier = register_module("classifier", torch::nn::Linear(512, 10));
            fc1 = register_module("fc1", torch::nn::Linear(512, 4096));
            fc2 = register_module("fc2", torch::nn::Linear(4096, 4096));
            fc3 = register_module("fc3", torch::nn::Linear(4096, 10));
        }

        torch::Tensor Vgg19Impl::forward(torch::Tensor x) {
            x = conv1_1->forward(x);
            x = batch1_1->forward(x);
            x = torch::relu(x);
            x = conv1_2->forward(x);
            x = batch1_2->forward(x);
            x = torch::relu(x);
            x = torch::max_pool2d(x, 2);
            //std::cout << "DONE1\n";
            x = conv2_1->forward(x);
            x = batch2_1->forward(x);
            x = torch::relu(x);
            x = conv2_2->forward(x);
            x = batch2_2->forward(x);
            x = torch::relu(x);
            x = torch::max_pool2d(x, 2);
            //std::cout << "DONE2\n";
            x = conv3_1->forward(x);
            x = batch3_1->forward(x);
            x = torch::relu(x);
            x = conv3_2->forward(x);
            x = batch3_2->forward(x);
            x = torch::relu(x);
            x = conv3_3->forward(x);
            x = batch3_3->forward(x);
            x = torch::relu(x);
            x = conv3_4->forward(x);
            x = batch3_4->forward(x);
            x = torch::relu(x);
            x = torch::max_pool2d(x, 2);
            // std::cout << "DONE3\n";
            x = conv4_1->forward(x);
            x = batch4_1->forward(x);
            x = torch::relu(x);
            x = conv4_2->forward(x);
            x = batch4_2->forward(x);
            x = torch::relu(x);
            x = conv4_3->forward(x);
            x = batch4_3->forward(x);
            x = torch::relu(x);
            x = conv4_4->forward(x);
            x = batch4_4->forward(x);
            x = torch::relu(x);
            x = torch::max_pool2d(x, 2);
            //std::cout << "DONE4\n";
            x = conv5_1->forward(x);
            x = batch5_1->forward(x);
            x = torch::relu(x);
            x = conv5_2->forward(x);
            x = batch5_2->forward(x);
            x = torch::relu(x);
            x = conv5_3->forward(x);
            x = batch5_3->forward(x);
            x = torch::relu(x);
            x = conv5_4->forward(x);
            x = batch5_4->forward(x);
            x = torch::relu(x);
            x = torch::adaptive_avg_pool2d(x, { 1, 1 });
            //std::cout << "DONE5\n";
            x = torch::flatten(x, 1);
            //x = classifier->forward(x);
            x = fc1->forward(x);
            x = fc2->forward(x);
            x = fc3->forward(x);
            //x = torch::log_softmax(x, 1);
            return x;
        }
	}
}