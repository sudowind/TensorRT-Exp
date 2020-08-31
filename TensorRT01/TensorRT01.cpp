﻿#include "TensorRT01.h"

bool MyOnnxModel::build()
{
	auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
	if (!builder)
	{
		return false;
	}

	const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
	auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
	if (!network)
	{
		return false;
	}

	auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
	if (!config)
	{
		return false;
	}

	auto parser = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
	if (!parser)
	{
		return false;
	}

	auto constructed = constructNetwork(builder, network, config, parser);
	if (!constructed)
	{
		return false;
	}

	mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
		builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());
	if (!mEngine)
	{
		return false;
	}

	assert(network->getNbInputs() == 1);
	mInputDims = network->getInput(0)->getDimensions();
	assert(mInputDims.nbDims == 4);

	assert(network->getNbOutputs() == 1);
	mOutputDims = network->getOutput(0)->getDimensions();
	assert(mOutputDims.nbDims == 2);

	return true;
}

bool MyOnnxModel::infer()
{
	// Create RAII buffer manager object
	samplesCommon::BufferManager buffers(mEngine);

	auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
	if (!context)
	{
		return false;
	}

	// Read the input data into the managed buffers
	assert(mParams.inputTensorNames.size() == 1);
	if (!processInput(buffers))
	{
		return false;
	}

	// Memcpy from host input buffers to device input buffers
	buffers.copyInputToDevice();
	bool status;

	for (auto i : mInputDims.d) {
		cout << i << endl;
	}

	DWORD t1 = GetTickCount();
	for (int i = 0; i < 10000; ++i) {
		status = context->executeV2(buffers.getDeviceBindings().data());
	}
	DWORD t2 = GetTickCount();

	cout << "Using time: " << (t2 - t1) << endl;

	if (!status)
	{
		return false;
	}

	// Memcpy from device output buffers to host output buffers
	buffers.copyOutputToHost();

	// Verify results
	if (!verifyOutput(buffers))
	{
		return false;
	}

	return true;
}

bool MyOnnxModel::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder, SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config, SampleUniquePtr<nvonnxparser::IParser>& parser)
{
	auto parsed = parser->parseFromFile(
		locateFile(mParams.onnxFileName, mParams.dataDirs).c_str(), static_cast<int>(sample::gLogger.getReportableSeverity()));
	if (!parsed) {
		return false;
	}
	config->setMaxWorkspaceSize(16_MiB);
	if (mParams.fp16) {
		config->setFlag(BuilderFlag::kFP16);
	}
	if (mParams.int8) {
		config->setFlag(BuilderFlag::kINT8);
		samplesCommon::setAllTensorScales(network.get(), 127.0f, 127.0f);
	}

	samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);

	return true;
}

bool MyOnnxModel::processInput(const samplesCommon::BufferManager & buffers)
{
	const int inputH = mInputDims.d[2];
	const int inputW = mInputDims.d[3];

	// Read a random digit file
	srand(unsigned(time(nullptr)));
	std::vector<uint8_t> fileData(inputH * inputW);
	mNumber = rand() % 10;
	readPGMFile(locateFile(std::to_string(mNumber) + ".pgm", mParams.dataDirs), fileData.data(), inputH, inputW);

	// Print an ascii representation
	sample::gLogInfo << "Input:" << std::endl;
	for (int i = 0; i < inputH * inputW; i++)
	{
		sample::gLogInfo << (" .:-=+*#%@"[fileData[i] / 26]) << (((i + 1) % inputW) ? "" : "\n");
	}
	sample::gLogInfo << std::endl;

	float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
	for (int i = 0; i < inputH * inputW; i++)
	{
		hostDataBuffer[i] = 1.0 - float(fileData[i] / 255.0);
	}

	return true;
}

bool MyOnnxModel::verifyOutput(const samplesCommon::BufferManager & buffers)
{
	const int outputSize = mOutputDims.d[1];
	float* output = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[0]));
	float val{ 0.0f };
	int idx{ 0 };

	// Calculate Softmax
	float sum{ 0.0f };
	for (int i = 0; i < outputSize; i++)
	{
		output[i] = exp(output[i]);
		sum += output[i];
	}

	sample::gLogInfo << "Output:" << std::endl;
	for (int i = 0; i < outputSize; i++)
	{
		output[i] /= sum;
		val = std::max(val, output[i]);
		if (val == output[i])
		{
			idx = i;
		}

		sample::gLogInfo << " Prob " << i << "  " << std::fixed << std::setw(5) << std::setprecision(4) << output[i] << " "
			<< "Class " << i << ": " << std::string(int(std::floor(output[i] * 10 + 0.5f)), '*') << std::endl;
	}
	sample::gLogInfo << std::endl;

	return idx == mNumber && val > 0.9f;
}

void printHelpInfo()
{
	std::cout
		<< "Usage: ./sample_onnx_mnist [-h or --help] [-d or --datadir=<path to data directory>] [--useDLACore=<int>]"
		<< std::endl;
	std::cout << "--help          Display help information" << std::endl;
	std::cout << "--datadir       Specify path to a data directory, overriding the default. This option can be used "
		"multiple times to add multiple directories. If no data directories are given, the default is to use "
		"(data/samples/mnist/, data/mnist/)"
		<< std::endl;
	std::cout << "--useDLACore=N  Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, "
		"where n is the number of DLA engines on the platform."
		<< std::endl;
	std::cout << "--int8          Run in Int8 mode." << std::endl;
	std::cout << "--fp16          Run in FP16 mode." << std::endl;

}

samplesCommon::OnnxSampleParams initializeSampleParams(const samplesCommon::Args& args)
{
	samplesCommon::OnnxSampleParams params;
	if (args.dataDirs.empty()) //!< Use default directories if user hasn't provided directory paths
	{
		params.dataDirs.push_back("F:/TensorRT-7.1.3.4/data/mnist");
	}
	else //!< Use the data directory provided by the user
	{
		params.dataDirs = args.dataDirs;
	}
	params.onnxFileName = "mnist.onnx";
	params.inputTensorNames.push_back("Input3");
	params.outputTensorNames.push_back("Plus214_Output_0");
	params.dlaCore = args.useDLACore;
	params.int8 = args.runInInt8;
	params.fp16 = args.runInFp16;

	return params;

}