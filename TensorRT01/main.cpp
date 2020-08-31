#include "TensorRT01.h"

const std::string gSampleName = "TensorRT.sample_onnx_mnist";

int runMnist(int argc, char** argv) {
	samplesCommon::Args args;
	bool argsOK = samplesCommon::parseArgs(args, argc, argv);

	if (!argsOK)
	{
		sample::gLogError << "Invalid arguments" << std::endl;
		printHelpInfo();
		return EXIT_FAILURE;
	}
	if (args.help)
	{
		printHelpInfo();
		return EXIT_SUCCESS;
	}

	auto sampleTest = sample::gLogger.defineTest(gSampleName, argc, argv);

	sample::gLogger.reportTestStart(sampleTest);

	MyOnnxModel sample(initializeSampleParams(args));

	sample::gLogInfo << "Building and running a GPU inference engine for Onnx MNIST" << std::endl;

	if (!sample.build())
	{
		return sample::gLogger.reportFail(sampleTest);
	}
	if (!sample.infer())
	{
		return sample::gLogger.reportFail(sampleTest);
	}

	return sample::gLogger.reportPass(sampleTest);
}

int main(int argc, char** argv)
{
	//return runMnist(argc, argv);

	samplesCommon::OnnxSampleParams params = initializeParams("F:/TensorRT-7.1.3.4/data/mnist",
		"mnist.onnx", "Input3", "Plus214_Output_0");

	OnnxModel model(params);

	model.build();
	model.dummyInfer();
}