#pragma once
#include "torch/torch.h"
#include <string>

namespace model {
	namespace vgg {
        class Vgg16Impl : public torch::nn::Module {
        public:
            std::string name = "VGG16";
            Vgg16Impl();
            torch::Tensor forward(torch::Tensor x);
        private:
            torch::nn::Conv2d conv1_1{ nullptr }, conv1_2{ nullptr };
            torch::nn::Conv2d conv2_1{ nullptr }, conv2_2{ nullptr };
            torch::nn::Conv2d conv3_1{ nullptr }, conv3_2{ nullptr }, conv3_3{ nullptr };
            torch::nn::Conv2d conv4_1{ nullptr }, conv4_2{ nullptr }, conv4_3{ nullptr };
            torch::nn::Conv2d conv5_1{ nullptr }, conv5_2{ nullptr }, conv5_3{ nullptr };

            //torch::nn::Linear classifier{ nullptr };
            torch::nn::Linear fc1{ nullptr };
            torch::nn::Linear fc2{ nullptr };
            torch::nn::Linear fc3{ nullptr };

            torch::nn::BatchNorm2d batch1_1{ nullptr }, batch1_2{ nullptr };
            torch::nn::BatchNorm2d batch2_1{ nullptr }, batch2_2{ nullptr };
            torch::nn::BatchNorm2d batch3_1{ nullptr }, batch3_2{ nullptr }, batch3_3{ nullptr };
            torch::nn::BatchNorm2d batch4_1{ nullptr }, batch4_2{ nullptr }, batch4_3{ nullptr };
            torch::nn::BatchNorm2d batch5_1{ nullptr }, batch5_2{ nullptr }, batch5_3{ nullptr };
        };

        TORCH_MODULE(Vgg16);

        class Vgg19Impl : public torch::nn::Module {
        public:
            std::string name = "VGG19";
            Vgg19Impl();
            torch::Tensor forward(torch::Tensor x);
        private:
            torch::nn::Conv2d conv1_1{ nullptr }, conv1_2{ nullptr };
            torch::nn::Conv2d conv2_1{ nullptr }, conv2_2{ nullptr };
            torch::nn::Conv2d conv3_1{ nullptr }, conv3_2{ nullptr }, conv3_3{ nullptr }, conv3_4{ nullptr };
            torch::nn::Conv2d conv4_1{ nullptr }, conv4_2{ nullptr }, conv4_3{ nullptr }, conv4_4{ nullptr };
            torch::nn::Conv2d conv5_1{ nullptr }, conv5_2{ nullptr }, conv5_3{ nullptr }, conv5_4{ nullptr };

            //torch::nn::Linear classifier{ nullptr };
            torch::nn::Linear fc1{ nullptr };
            torch::nn::Linear fc2{ nullptr };
            torch::nn::Linear fc3{ nullptr };

            torch::nn::BatchNorm2d batch1_1{ nullptr }, batch1_2{ nullptr };
            torch::nn::BatchNorm2d batch2_1{ nullptr }, batch2_2{ nullptr };
            torch::nn::BatchNorm2d batch3_1{ nullptr }, batch3_2{ nullptr }, batch3_3{ nullptr }, batch3_4{ nullptr };
            torch::nn::BatchNorm2d batch4_1{ nullptr }, batch4_2{ nullptr }, batch4_3{ nullptr }, batch4_4{ nullptr };
            torch::nn::BatchNorm2d batch5_1{ nullptr }, batch5_2{ nullptr }, batch5_3{ nullptr }, batch5_4{ nullptr };
        };

        TORCH_MODULE(Vgg19);
	}
}