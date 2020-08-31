#include "utils.h"

bool OnnxModel::build()
{
	auto builder = OnnxUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
	if (!builder)
	{
		return false;
	}

	const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
	auto network = OnnxUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
	if (!network)
	{
		return false;
	}

	auto config = OnnxUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
	if (!config)
	{
		return false;
	}

	auto parser = OnnxUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
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

	//assert(network->getNbInputs() == 1);
	mInputDims = network->getInput(0)->getDimensions();
	mInputName = network->getInput(0)->getName();
	//assert(mInputDims.nbDims == 4);

	//assert(network->getNbOutputs() == 1);
	mOutputDims = network->getOutput(0)->getDimensions();
	mOutputName = network->getOutput(0)->getName();
	//assert(mOutputDims.nbDims == 2);

	return true;
}

bool OnnxModel::infer()
{
	return false;
}

bool OnnxModel::constructNetwork(OnnxUniquePtr<nvinfer1::IBuilder>& builder, OnnxUniquePtr<nvinfer1::INetworkDefinition>& network, OnnxUniquePtr<nvinfer1::IBuilderConfig>& config, OnnxUniquePtr<nvonnxparser::IParser>& parser)
{
	auto parsed = parser->parseFromFile(
		locateFile(mParams.onnxFileName, mParams.dataDirs).c_str(), static_cast<int>(sample::gLogger.getReportableSeverity()));
	if (!parsed)
	{
		return false;
	}

	config->setMaxWorkspaceSize(16_MiB);
	if (mParams.fp16)
	{
		config->setFlag(BuilderFlag::kFP16);
	}
	if (mParams.int8)
	{
		config->setFlag(BuilderFlag::kINT8);
		samplesCommon::setAllTensorScales(network.get(), 127.0f, 127.0f);
	}

	samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);

	return true;
}

bool OnnxModel::processInput(const samplesCommon::BufferManager & buffers)
{
	return false;
}

bool OnnxModel::verifyOutput(const samplesCommon::BufferManager & buffers)
{
	return false;
}

bool OnnxModel::dummyInfer()
{
	samplesCommon::BufferManager buffers(mEngine);

	int totalSize = 1;
	vector<int> shape;
	for (int i = 0; i < mInputDims.nbDims; i++) {
		totalSize *= mInputDims.d[i];
		shape.push_back(mInputDims.d[i]);
	}

	cout << totalSize << endl;

	xt::xarray<float> dummyInput = xt::ones<float>(shape);

	float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mInputName));

	memcpy(hostDataBuffer, dummyInput.data(), totalSize * sizeof(float));

	buffers.copyInputToDevice();

	auto context = OnnxUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());

	bool status = context->executeV2(buffers.getDeviceBindings().data());

	return status;
}

//samplesCommon::OnnxSampleParams initializeParams(string dataDir, string fileName, string inputName, string outputName)
samplesCommon::OnnxSampleParams initializeParams(string dataDir, string fileName)
{
	samplesCommon::OnnxSampleParams params;
	params.dataDirs.push_back(dataDir);
	params.onnxFileName = fileName;
	/*params.inputTensorNames.push_back(inputName);
	params.outputTensorNames.push_back(outputName);*/

	return params;
}