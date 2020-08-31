#pragma once
#define _CRT_SECURE_NO_WARNINGS

#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "parserOnnxConfig.h"

#include "NvInfer.h"
#include <cuda_runtime_api.h>

#include "xtensor\xarray.hpp"
#include "xtensor\xio.hpp"
#include "xtensor\xview.hpp"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <time.h>

class OnnxModel {
	template <typename T>
	using OnnxUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

public:
	OnnxModel(const samplesCommon::OnnxSampleParams& params)
		: mParams(params)
		, mEngine(nullptr)
	{
	}

	bool build();

	bool infer();

public:
	samplesCommon::OnnxSampleParams mParams;

	nvinfer1::Dims mInputDims;
	nvinfer1::Dims mOutputDims;

	int mNumber{ 0 };

	std::shared_ptr<nvinfer1::ICudaEngine> mEngine;

	bool constructNetwork(OnnxUniquePtr<nvinfer1::IBuilder>& builder,
		OnnxUniquePtr<nvinfer1::INetworkDefinition>& network, OnnxUniquePtr<nvinfer1::IBuilderConfig>& config,
		OnnxUniquePtr<nvonnxparser::IParser>& parser);

	bool processInput(const samplesCommon::BufferManager& buffers);

	bool verifyOutput(const samplesCommon::BufferManager& buffers);

	bool dummyInfer();

};

samplesCommon::OnnxSampleParams initializeParams(string dataDir, string fileName, string inputName, string outputName);