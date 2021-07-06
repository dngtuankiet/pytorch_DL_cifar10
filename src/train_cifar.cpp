// Include libraries
#include <ATen/ATen.h>
#include <torch/script.h>
#include <torch/torch.h>
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


int create_data_file = 1; // To create custom data, use number != 0
						  // To load custom data from file, use number == 0

// Labels of CIFAR 10
std::string labels_map[] = { "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck" };
// Training folder contains 50k images of training image of CIFAR 10 dataset
std::string train_folder = "E:\\Workspace\\Software\\cifar-10\\train";
// txt files
std::string train_img_path_file = "E:\\Workspace\\Software\\cifar-10\\train_image_path.txt";
std::string train_labels_file = "E:\\Workspace\\Software\\cifar-10\\train_labels.txt";


//std::string test_img_path = "D:\\UEC\\DL_mem\\Software\\cifar-10\\test\\17.png";
// Torch script model exported from Python
/*
	Available models: *_raw.pt
		vgg16
		vgg19
		resnet50
*/

// Torch Jit Model path
//std::string model_path = "D:\\UEC\\DL_mem\\Software\\CIFA_models\\save_model\\resnet50.pt";


auto ToCvImage(at::Tensor tensor)
{
	int width = tensor.sizes()[0];
	int height = tensor.sizes()[1];
	try
	{
		cv::Mat output_mat(cv::Size{ height, width }, CV_8UC3, tensor.data_ptr<uchar>());
		return output_mat.clone();
	}
	catch (const c10::Error & e)
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

//Image resize for Alexnet, resize to 256 and center crop to 224
/*cv::resize(img, img, cv::Size(256, 256), cv::INTER_CUBIC);
const int cropSize = 224;
const int offsetW = (img.cols - cropSize) / 2;
const int offsetH = (img.rows - cropSize) / 2;
const cv::Rect roi(offsetW, offsetH, cropSize, cropSize);
img = img(roi).clone();*/

// Read image and convert to Tensor
torch::Tensor read_data(std::string loc) {
	cv::Mat img = cv::imread(loc, cv::IMREAD_COLOR);
	cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
	//cv::resize(img, img, cv::Size(32, 32), cv::INTER_CUBIC);
	//std::cout << "Sizes: " << img.size() << std::endl;

	// Image -> tensor
	torch::Tensor img_tensor = torch::from_blob(img.data, { img.rows, img.cols, 3 }, torch::kByte);
	img_tensor = img_tensor.toType(at::kFloat);
	img_tensor = img_tensor.div_(255);
	img_tensor = img_tensor.permute({ 2, 0, 1 }); // Channels x Height x Width

	// Normalization
	struct Normalize normalizeChannels({ 0.4914, 0.4822, 0.4465 }, { 0.2023, 0.1994, 0.2010 });
	img_tensor = normalizeChannels(img_tensor);
	//std::cout << "Image tensor shape: " << img_tensor.sizes() << std::endl;

	return img_tensor.clone();
}

// Read Label (int) and convert to torch::Tensor type
torch::Tensor read_label(int label) {
	// Read label here
	// Convert to tensor and return
	torch::Tensor label_tensor = torch::scalar_to_tensor(label);

	//std::cout << "label tensor\n";
	return label_tensor.clone();
}

/* Loads images to tensor type in the string argument */
std::vector<torch::Tensor> process_images(std::vector<std::string> list_images) {
	std::cout << "Reading Images..." << std::endl;
	// Return vector of Tensor form of all the images
	std::vector<torch::Tensor> states;
	for (std::vector<std::string>::iterator it = list_images.begin(); it != list_images.end(); ++it) {
		torch::Tensor img = read_data(*it);
		states.push_back(img);
	}
	return states;
}

/* Loads labels to tensor type in the string argument */
std::vector<torch::Tensor> process_labels(std::vector<int> list_labels) {
	std::cout << "Reading Labels..." << std::endl;
	// Return vector of Tensor form of all the labels
	std::vector<torch::Tensor> labels;
	for (std::vector<int>::iterator it = list_labels.begin(); it != list_labels.end(); ++it) {
		//std::cout << "label: " << *it << std::endl;
		torch::Tensor label = read_label(*it);
		labels.push_back(label);
	}
	return labels;
}


at::Tensor imageToTensor(cv::Mat& image) {
	// BGR to RGB, which is what our network was trained on
	cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

	// Convert Mat image to tensor 1 x C x H x W
	at::Tensor tensorImage = torch::from_blob(image.data, { 1, image.rows, image.cols, image.channels() }, at::kByte);

	// Normalize tensor values from [0, 255] to [0, 1]
	tensorImage = tensorImage.toType(at::kFloat);
	tensorImage = tensorImage.div_(255);

	// Transpose the image for [channels, rows, columns] format of torch tensor
	tensorImage = at::transpose(tensorImage, 1, 2);
	tensorImage = at::transpose(tensorImage, 1, 3);
	return tensorImage; // 1 x C x H x W
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

	auto maxResult = result.max(1);
	auto maxIndex = std::get<1>(maxResult).item<float>();
	auto maxOut = std::get<0>(maxResult).item<float>();
	std::cout << "Predicted: " << labels_map[(int)maxIndex] << " | " << maxOut << std::endl;
	//std::cout << result.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
	return labels_map[(int)maxIndex];
};

//// Load from files
//CustomDataset(std::vector<torch::Tensor> file_images, std::vector<torch::Tensor> file_labels) {
//	images = file_images;
//	labels = file_labels;
//};

class CustomDataset : public torch::data::Dataset<CustomDataset> {
private:
	// Declare 2 vectors of tensors for images and labels
	std::vector<torch::Tensor> images, labels;
public:
	// Constructor
	CustomDataset(std::vector<std::string> list_images, std::vector<int> list_labels) {
		images = process_images(list_images);
		labels = process_labels(list_labels);
		// Save 2 Tensor vectors
		torch::save(images, "images.pt");
		torch::save(labels, "labels.pt");
	};

	CustomDataset() {};

	void get_data(std::vector<std::string> list_images, std::vector<int> list_labels) {
		images = process_images(list_images);
		labels = process_labels(list_labels);
		// Save 2 Tensor vectors
		torch::save(images, "images.pt");
		torch::save(labels, "labels.pt");
	};

	void get_data(std::vector<torch::Tensor> file_images, std::vector<torch::Tensor> file_labels) {
		images = file_images;
		labels = file_labels;
	};

	// Override get() function to return tensor at location index
	torch::data::Example<> get(size_t index) override {
		torch::Tensor sample_img = images.at(index);
		torch::Tensor sample_label = labels.at(index);
		return { sample_img.clone(), sample_label.clone() };
	};

	// Return the length of data
	torch::optional<size_t> size() const override {
		return labels.size();
	};
};

// Create a txt file containing the absolute path of all training images
void create_image_path(std::string train_folder) {
	std::ofstream myfile;
	myfile.open(train_img_path_file);
	for (int i = 1; i <= 50000; i++) {
		myfile << train_folder + "\\" + std::to_string(i) + "." + "png";
		if (i != 50000) myfile << std::endl;
	}
	myfile.close();
}

int main(int argc, char** argv) {
	std::vector<std::string> list_images; // list of path of images
	std::vector<int> list_labels; // list of integer labels

	std::vector<torch::Tensor> images;
	std::vector<torch::Tensor> labels;

	// Create an empty CustomDataset
	CustomDataset custom_dataset_construct;

	// This part read 2 .txt files and create image and label Tensor in CustomDataset class
	if (create_data_file != 0) {
		int label = 0;
		// Create image path txt file
		create_image_path(train_folder);

		// Open image path txt file and load it into a vector of string
		std::ifstream imgPathFile;
		imgPathFile.open(train_img_path_file);
		if (imgPathFile.is_open()) {
			std::cout << "Create Image Path Vector...\n";
			std::string img_path;
			while (!imgPathFile.eof()) {
				getline(imgPathFile, img_path);
				list_images.push_back(img_path);
			}
		}
		else {
			std::cout << "Error open Image Path File\n";
			return 0;
		}
		imgPathFile.close();
		std::cout << "Image Path Vector created SUCCESS with size " << list_images.size() << "\n";

		// Open label txt file and load it into a vector of int
		std::ifstream labelFile;
		labelFile.open(train_labels_file);
		if (labelFile.is_open()) {
			std::cout << "Create Label Vector...\n";
			int label = 0;
			int i = 0;
			while (i < list_images.size()) {
				labelFile >> label;
				list_labels.push_back(label);
				i++;
			}
		}
		else {
			std::cout << "Error open Label File\n";
			return 0;
		}
		labelFile.close();
		std::cout << "Label Vector created SUCCESS with size " << list_labels.size() << "\n";
		custom_dataset_construct.get_data(list_images, list_labels);
	}
	// Load Image and Labels Tensor Vectors from files
	else {
		std::cout << "Load Tensor Images and Tensor Labels from files..." << std::endl;
		torch::load(images, "images.pt");
		std::cout << "Load Tensor Images from file DONE!\n";
		torch::load(labels, "labels.pt");
		std::cout << "Load Tensor Labels from file DONE!\n";
		custom_dataset_construct.get_data(images, labels);
	}

	// Generate data set. At this point you can add transforms to you data set, e.g. stack your
	// batches into a single tensor.
	std::cout << "Create Custom Dataset..." << "\n";
	//auto custom_dataset = CustomDataset(list_images, list_labels).map(torch::data::transforms::Stack<>());
	auto custom_dataset = custom_dataset_construct.map(torch::data::transforms::Stack<>());
	// Generate a data loader
	int batch_size = 64;
	auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
		std::move(custom_dataset),
		batch_size
		);
	std::cout << "Custom Dataset Created SUCCESS" << "\n";

	//for (torch::data::Example<>& batch : *data_loader) {
	//	std::cout << "Batch size: " << batch.data.size(0) << " | Labels: ";
	//	for (int64_t i = 0; i < batch.data.size(0); ++i) {
	//		std::cout << batch.target[i].item<int64_t>() << " ";
	//	}
	//	std::cout << std::endl;
	//}


	// Load model
	std::cout << "Creating model...\n";

	//---------------------USING TORCH SCRIPT MODEL-----------------------
	// Model is exported by Python and loaded in C++
	//torch::jit::script::Module model = torch::jit::load(model_path);
	//model.train();

	//---------------------USING CUSTOM MODEL BUILD BY TORCH NN MODULE---------------------
	/*		Available model:
			model::vgg::Vgg16()
			model::vgg::Vgg19()
			model::resnet::ResNet18(num_class)
			model::resnet::ResNet34(num_class)
			model::resnet::ResNet50(num_class)
			model::resnet::ResNet101(num_class)
			model::resnet::ResNet152(num_class)
			model::googlenet::GoogLeNet()
	*/
	auto model = model::googlenet::GoogLeNet();
	model->train();

	std::cout << "Model Created\n";

	//-----------------------------------------------------------------------------------------
	//-----------------------------------------------------------------------------------------
	//------------------------------######TRAINING PROCESS#####--------------------------------
	//-----------------------------------------------------------------------------------------
	//-----------------------------------------------------------------------------------------

	std::cout << "\n----------TRAINING PROCESS----------\n";
	// Set parameter
	/*
		Set parameter for Torch script model

	//std::vector<torch::tensor> trainable_params;
	//auto params = model->named_parameters(true);
	//for (auto& param : params)
	//{
	//	auto layer_name = param.key();
	//	if ("fc.weight" == layer_name || "fc.bias" == layer_name)
	//	{
	//		param.value().set_requires_grad(true);
	//		trainable_params.push_back(param.value());
	//	}
	//	else
	//	{
	//		param.value().set_requires_grad(false);
	//	}
	//}
	//std::cout << "trainable parameters size: " << parameters.size() << std::endl;

	std::vector<at::Tensor> parameters;
	for (const auto& params : model->parameters(true)) {
		parameters.push_back(params);
	}
	std::cout << "Parameters size: " << parameters.size() << std::endl;
	std::cout << "Set up parameter DONE!\n";

	*/
	// Check parameter size
	std::vector<at::Tensor> parameters;
	for (const auto& params : model->parameters(true)) {
		parameters.push_back(params);
	}
	std::cout << "Parameters size: " << parameters.size() << std::endl;

	// Set up Optimizer
	/*
		Possible optimizer suppored by Libtorch:
			RMSprop
			SGD
			Adam
			Adagrad
			LBFGS
			LossClosureOptimizer
	*/

	// for Torch Script Model
	//torch::optim::SGD optimizer(parameters, torch::optim::SGDOptions(0.001).momentum(0.5));
	// for Torch NN Model

	torch::optim::SGD optimizer(std::move(model->parameters(true)), torch::optim::SGDOptions(0.01).momentum(0.5));


	// Loss function
	auto criterion = torch::nn::CrossEntropyLoss();


	std::cout << "Set up optimizer and loss function DONE!\n";

	// Train network
	int dataset_size = custom_dataset_construct.size().value();
	std::cout << "Data set size: " << dataset_size << std::endl;
	int n_epochs = 5; // number of epochs
	// Record for best loss
	float best_mse = std::numeric_limits<float>::max();

	std::cout << "Start training loop...\n";
	for (int epoch = 1; epoch <= n_epochs; epoch++) {
		//Track loss
		int batch_index = 0;
		float mse = 0; // mean square error
		float Acc = 0.0; // accuracy

		for (auto& batch : *data_loader) {
			auto data = batch.data;
			auto target = batch.target.squeeze();
			//std::cout << "Step 1\n";

			// Convert data to float32 format and target to Int64 format
			// Assuming you have labels as integers
			data = data.to(torch::kF32);
			target = target.to(torch::kInt64);
			//std::cout << "Step 2\n";

			// Clear the optimizer parameters
			optimizer.zero_grad();
			//std::cout << "Step 3\n";

			/*std::vector<torch::jit::IValue> input;
			input.push_back(data);*/

			//auto output = model.forward(input).toTensor();
			auto output = model->forward(data);
			//std::cout << "Step 4\n";

			// Compute the loss value to judge the prediction of our model
			// If the model has softmax layer at the output -> use nll_loss()
			// If the model doesnt have softmax layer at the output -> use criterion()
			//auto loss = torch::nll_loss(torch::log_softmax(output, 1), target);
			auto loss = criterion(output, target);  // the same with: auto loss = torch::nll_loss(torch::log_softmax(output, 1), target); 
			//std::cout << "Step 5\n";

			// Compute the gradients of the loss w.r.t. the parameters of our model
			loss.backward();
			//std::cout << "Step 6\n";

			// Update the parameters based on the calculated gradient.
			optimizer.step();
			//std::cout << "Step 7\n";

			// Record statistics
			auto acc = output.argmax(1).eq(target).sum();
			//std::cout << "accu" << std::endl;
			Acc += acc.template item<float>();
			//std::cout << "Step 8\n";

			mse += loss.template item<float>();

			std::printf("\rTrain Epoch: %d/%ld [%5ld/%5d] Loss: %.4f Acc/batch: %d/%d \n",
				epoch,
				n_epochs,
				(long)(batch_index * batch.data.size(0)),
				dataset_size,
				loss.template item<float>(),
				acc.template item<int>(),
				batch_size
			);

			batch_index++;
			std::cout << "Count batch loop: " << batch_index << std::endl;
		}

		mse /= (float)batch_index;
		std::cout << "Accuracy: " << Acc / dataset_size << " Mean squared error: " << mse << std::endl;

		// Save model every epoch
		// for Torch Script Model
		//model.save("resnet50_trained_model.pt");
		// for Torch NN Model
		torch::save(model, "vgg16_new.pt");

		std::cout << "Model SAVED" << std::endl;
		if (mse < best_mse)
		{
			best_mse = mse;
			std::cout << "Best Mean Square Error: " << best_mse << std::endl;
			std::cout << "Model TRAINED!!!\n";
		}
	}


	//-----------------VALIDATION---------------------------------
	float Acc = 0.0;
	int batch_index = 0;
	std::cout << "Start validation process...\n";
	for (auto& batch : *data_loader) {
		auto data = batch.data;
		auto target = batch.target.squeeze();
		data = data.to(torch::kF32);
		target = target.to(torch::kInt64);

		std::vector<torch::jit::IValue> input;
		input.push_back(data);

		//auto output = model.forward(input).toTensor();
		auto output = model->forward(data);
		auto acc = output.argmax(1).eq(target).sum();
		Acc += acc.template item<float>();

		batch_index++;
		std::printf("\rValidation -> Batch number: %d Acc/batch: %d/%d \n", batch_index, acc.template item<int>(), batch_size);
	}

	std::cout << "Validation finished -> Correct prediction: " << Acc << "/50000 Total accuracy: " << Acc / dataset_size << std::endl;
	// Save the model
	//torch::save(model, "best_model.pt");

	//std::cout << "--------TEST MODEL----------\n";

	//model.eval();
	////Read an image
//   cv::Mat image;
//   image = cv::imread(img_path, cv::IMREAD_COLOR);
//   if (!image.data) {
//       std::cout << "could not open or find the image" << std::endl;
//       return -1;
//   }

//   //prediction
//   std::string predict_obj;
//   predict_obj = predict(model, image);


//   //display image
//   std::string windowname = predict_obj; //name of the window
//   cv::namedWindow(windowname); // create a window
//   cv::imshow(windowname, image); // show our image inside the created window.
//   cv::waitKey(0); // wait for any keystroke in the window
//   cv::destroyWindow(windowname); //destroy the created window

	return 0;
}