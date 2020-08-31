#pragma once
#include "utils.h"

class MyOnnxModel
{
	template <typename T>
	using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;
public:
	MyOnnxModel(const samplesCommon::OnnxSampleParams& params)
		: mParams(params), mEngine(nullptr) {}

	bool build();

	bool infer();

private:
	samplesCommon::OnnxSampleParams mParams;
	std::shared_ptr<nvinfer1::ICudaEngine> mEngine;

	int mNumber{ 0 };

	nvinfer1::Dims mInputDims;
	nvinfer1::Dims mOutputDims;

	bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
		SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
		SampleUniquePtr<nvonnxparser::IParser>& parser);

	bool processInput(const samplesCommon::BufferManager& buffers);

	bool verifyOutput(const samplesCommon::BufferManager& buffers);

};

void printHelpInfo();

samplesCommon::OnnxSampleParams initializeSampleParams(const samplesCommon::Args& args);