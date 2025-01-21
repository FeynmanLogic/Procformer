#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>

int main() {
    // Initialize ONNX Runtime environment
    onnxruntime::Env env(onnxruntime::LoggingLevel::WARNING, "ONNXModel");

    // Create session options and set execution providers (CPU in this case)
    onnxruntime::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1); // Set number of threads for multi-threading

    // Create ONNX model session
    onnxruntime::InferenceSession session(env, session_options);
    // Load the ONNX model
    onnxruntime::Status status = session.Load("model.onnx");
    if (!status.IsOK()) {
        std::cerr << "Error loading model: " << status.ErrorMessage() << std::endl;
        return 1;
    }

    // Prepare input data (input_ids for BERT)
    std::vector<int64_t> input_shape = {1, 128};  // Example shape, adjust based on model
    std::vector<int32_t> input_data(128, 0); // Example input data (tokenized input)

    // Create a tensor for input
    onnxruntime::Tensor input_tensor(onnxruntime::Tensor::FromDataType<int32_t>(), input_shape, input_data.data());

    // Create input feeds
    std::vector<std::string> input_names = {"input_ids"}; // Modify with your model input names
    std::vector<const onnxruntime::Tensor*> inputs = {&input_tensor};

    // Run inference
    std::vector<onnxruntime::Tensor> output_tensors;
    status = session.Run(onnxruntime::RunOptions(), input_names, inputs, {"output"}, output_tensors);
    if (!status.IsOK()) {
        std::cerr << "Error running inference: " << status.ErrorMessage() << std::endl;
        return 1;
    }

    // Process output (here assuming it's a single output tensor)
    auto output = output_tensors[0].Data<float>(); // Modify according to your output type

    // Print the output
    std::cout << "Output: " << output[0] << std::endl;  // Adjust for your model's output

    return 0;
}
