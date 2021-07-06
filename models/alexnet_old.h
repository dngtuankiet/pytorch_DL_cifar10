#pragma once
#include "torch/torch.h"

struct AlexNetImpl : torch::nn::Module {

    AlexNetImpl() {
        conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 64, 11).stride(4).padding(2)));
        conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 192, 5).padding(2)));
        conv3 = register_module("conv3", torch::nn::Conv2d(torch::nn::Conv2dOptions(192, 384, 3).padding(1)));
        conv4 = register_module("conv4", torch::nn::Conv2d(torch::nn::Conv2dOptions(384, 256, 3).padding(1)));
        conv5 = register_module("conv5", torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).padding(1)));
        linear1= register_module("linear1", torch::nn::Linear(9216, 4096));
            //modifyhttps://analyticsindiamag.com/implementing-alexnet-using-pytorch-as-a-transfer-learning-model-in-multi-class-classification/
        linear2 = register_module("linear2", torch::nn::Linear(4096, 1024));
        linear3 = register_module("linear3", torch::nn::Linear(1024, 10));
        dropout = register_module("dropout", torch::nn::Dropout(torch::nn::DropoutOptions(0.5)));
    }
        

    torch::Tensor forward(const torch::Tensor& input) {
        auto x = torch::relu(conv1(input));
        x = torch::max_pool2d(x, 3, 2);

        x = torch::relu(conv2(x));
        x = max_pool2d(x, 3, 2);

        x = torch::relu(conv3(x));
        x = torch::relu(conv4(x));
        x = torch::relu(conv5(x));
        x = max_pool2d(x, 3, 2);
        // Classifier, 256 * 6 * 6 = 9216
        x = x.view({ x.size(0), 9216 });
        x = dropout(x);
        x = torch::relu(linear1(x));

        x = dropout(x);
        x = torch::relu(linear2(x));

        x = linear3(x);
        return x;
    }
    torch::nn::Linear linear1{ nullptr }, linear2{ nullptr }, linear3{ nullptr };
    torch::nn::Dropout dropout{ nullptr };
    torch::nn::Conv2d conv1{ nullptr }, conv2{ nullptr }, conv3{ nullptr }, conv4{ nullptr }, conv5{ nullptr };
};

//TORCH_MODULE_IMPL(AlexNet, AlexNetImpl);