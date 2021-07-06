//After training with main.cpp, use this for the inference

// Include libraries
#include <ATen/ATen.h>
#include "torch/script.h"
#include "torch/torch.h"
#include <opencv2/opencv.hpp>

#include <iostream>
#include <vector>
#include <tuple>
#include <string>
#include <windows.h>
#include <fstream>

// include Model 
#include "alexnet.h"
#include "resnet.h"
#include "vgg.h"
#include "mobilenet.h"
#include "googlenet.h"

#include "pcie_mem.h"

std::string labels_map[] = {"plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck" };
std::string img_path = "E:\\Workspace\\Software\\cifar-10\\test\\3.png";
std::string model_path = "E:\\Workspace\\Software\\vs2019\\ObjectDetection\\trained_models\\vgg16_og.pt";


auto ToCvImage(at::Tensor tensor)
{
	int width = tensor.sizes()[0];
	int height = tensor.sizes()[1];
	try
	{
		cv::Mat output_mat(cv::Size{ height, width }, CV_8UC3, tensor.data_ptr<uchar>());
		return output_mat.clone();
	}
	catch (const c10::Error& e)
	{
		std::cout << "an error has occured : " << e.msg() << std::endl;
	}
	return cv::Mat(height, width, CV_8UC3);
};

struct Normalize : public torch::data::transforms::TensorTransform<> {
	Normalize(const std::initializer_list<float>& means, const std::initializer_list<float>& stddevs)
		: means_(insertValues(means)), stddevs_(insertValues(stddevs)) {}
	std::list<torch::Tensor> insertValues(const std::initializer_list<float>& values) {
		std::list<torch::Tensor> tensorList;
		for (auto val : values) {
			tensorList.push_back(torch::tensor(val));
		}
		return tensorList;
	}
	torch::Tensor operator()(torch::Tensor input) {
		std::list<torch::Tensor>::iterator meanIter = means_.begin();
		std::list<torch::Tensor>::iterator stddevIter = stddevs_.begin();
		//  Substract each channel's mean and divide by stddev in place
		for (int i{ 0 }; meanIter != means_.end() && stddevIter != stddevs_.end(); ++i, ++meanIter, ++stddevIter) {
			//std::cout << "Mean: " << *meanIter << " Stddev: " << *stddevIter << std::endl;
			//std::cout << input[0][i] << std::endl;
			input[0][i].sub_(*meanIter).div_(*stddevIter);
		}
		return input;
	}

	std::list<torch::Tensor> means_, stddevs_;
};

torch::Tensor imageToTensor(cv::Mat& image) {
	// BGR to RGB, which is what our network was trained on
	cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

	// Convert Mat image to tensor 1 x C x H x W
	torch::Tensor tensorImage = torch::from_blob(image.data, {1, image.rows, image.cols, 3 }, torch::kByte);

	// Normalize tensor values from [0, 255] to [0, 1]
	tensorImage = tensorImage.toType(at::kFloat);
	tensorImage = tensorImage.div_(255);

	// Transpose the image for [channels, rows, columns] format of torch tensor
	//tensorImage = at::transpose(tensorImage, 1, 2);
	//tensorImage = at::transpose(tensorImage, 1, 3);
	std::cout << "Test\n";
	tensorImage = tensorImage.permute({0, 3, 1, 2});
	std::cout << "tensorImage size: " << tensorImage.sizes() << std::endl;
	//std::cout << tensorImage << std::endl;
	return tensorImage.clone(); // 1 x C x H x W
};

std::string predict(torch::jit::script::Module& module, cv::Mat& image) {
	at::Tensor tensorImage{ imageToTensor(image) };
	struct Normalize normalizeChannels({ 0.4914, 0.4822, 0.4465 }, { 0.2023, 0.1994, 0.2010 });
	tensorImage = normalizeChannels(tensorImage);
	std::cout << " Test Image tensor shape: " << tensorImage.sizes() << std::endl;

	// Move tensor to CUDA memory
	//tensorImage = tensorImage.to(at::kCUDA);
	// Forward pass
	at::Tensor result = module.forward({ tensorImage }).toTensor();
	std::cout << "-----PRINT TENSOR-----\n";
	std::cout << result << std::endl;
	std::cout << "-----TENSOR CONTENT----\n";
	auto maxResult = result.max(1);
	auto maxIndex = std::get<1>(maxResult).item<float>();
	auto maxOut = std::get<0>(maxResult).item<float>();
	std::cout << "Predicted: " << labels_map[(int)maxIndex] << " | " << maxOut << std::endl;
	//std::cout << result.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
	return labels_map[(int)maxIndex];
};

//std::string predict_customModel(std::shared_ptr<Net>& module, cv::Mat& image) {
//	torch::Tensor tensorImage = imageToTensor(image);
//	struct Normalize normalizeChannels({ 0.4914, 0.4822, 0.4465 }, { 0.2023, 0.1994, 0.2010 });
//	tensorImage = normalizeChannels(tensorImage);
//	tensorImage = tensorImage.to(torch::kF32);
//
//	at::Tensor result = module->forward(tensorImage);
//	std::cout << "-----PRINT TENSOR-----\n";
//	std::cout << result << std::endl;
//	std::cout << "-----TENSOR CONTENT----\n";
//	auto maxResult = result.max(1);
//	auto maxIndex = std::get<1>(maxResult).item<float>();
//	auto maxOut = std::get<0>(maxResult).item<float>();
//	std::cout << "Predicted: " << labels_map[(int)maxIndex] << " | " << maxOut << std::endl;
//	return labels_map[(int)maxIndex];
//};

int main(int argc, char** argv) {
	DWORD dwStatus;
	OpenLib();

	//torch::Device device(torch::kFPGA);
	torch::Tensor a = torch::ones({ 2,2 }, torch::kCPU);
	std::cout << "\n a = " << a << " \n";
	// Load model
	std::cout << "Loading trained model...\n";
	//torch::jit::script::Module model = torch::jit::load(model_path);


	auto model = model::vgg::Vgg16();
	torch::load(model, model_path);
	model->eval();
	std::cout << "Model loaded\n";

	std::cout << "--------TEST MODEL----------\n";

	//Read an image
    cv::Mat image;
    image = cv::imread(img_path, cv::IMREAD_COLOR);
    if (!image.data) {
        std::cout << "could not open or find the image" << std::endl;
        return -1;
    }
	
	//Convert image to Tensor and Implement preprocessing
	torch::Tensor tensorImage = imageToTensor(image);
	struct Normalize normalizeChannels({ 0.4914, 0.4822, 0.4465 }, { 0.2023, 0.1994, 0.2010 });
	tensorImage = normalizeChannels(tensorImage);
	tensorImage = tensorImage.to(torch::kF32);

	at::Tensor result = model->forward(tensorImage);
	result = torch::softmax(result, 1);
	//std::cout << "-----PRINT TENSOR-----\n";
	//std::cout << result << std::endl;
	//std::cout << "-----TENSOR CONTENT----\n";
	
	auto maxResult = result.max(1);
	auto maxIndex = std::get<1>(maxResult).item<float>();
	auto maxOut = std::get<0>(maxResult).item<float>();
	std::cout << "Predicted: " << labels_map[(int)maxIndex] << " | " << maxOut << std::endl;
	

    //display image
    std::string windowname = labels_map[(int)maxIndex]; //name of the window
    cv::namedWindow(windowname); // create a window
    cv::imshow(windowname, image); // show our image inside the created window.
    cv::waitKey(0); // wait for any keystroke in the window
    cv::destroyWindow(windowname); //destroy the created window

	dwStatus = CloseLib();

	return dwStatus;
}