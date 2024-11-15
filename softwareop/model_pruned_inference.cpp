#include "cpu_provider_factory.h"

#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>

int main() {
    // Initialize ONNX Runtime environment
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNXRuntime");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);

    // Optional: Enable CPU execution
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CPU(session_options, 1));

    // Load the ONNX model
    const std::string model_path = "pruned_model.onnx"; // Path to your saved ONNX model
    Ort::Session session(env, model_path.c_str(), session_options);
    std::cout << "ONNX model loaded successfully.\n";

    // Prepare input data
    std::vector<int64_t> input_ids = {101, 2003, 2023, 1037, 2742, 102}; // Example input
    std::vector<int64_t> input_shape = {1, static_cast<int64_t>(input_ids.size())}; // Batch size = 1
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(
        memory_info, input_ids.data(), input_ids.size(), input_shape.data(), input_shape.size());

    // Input/output names
    const char* input_names[] = {"input_ids"}; // Must match ONNX input names
    const char* output_names[] = {"output"};   // Must match ONNX output names

    // Run inference
    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
    auto& output_tensor = output_tensors.front();

    // Process output
    float* output_data = output_tensor.GetTensorMutableData<float>();
    size_t output_size = output_tensor.GetTensorTypeAndShapeInfo().GetElementCount();
    std::cout << "Model output: ";
    for (size_t i = 0; i < output_size; ++i) {
        std::cout << output_data[i] << " ";
    }
    std::cout << "\n";

    // Get the predicted label
    auto predicted_label = std::distance(output_data, std::max_element(output_data, output_data + output_size));
    std::cout << "Predicted label: " << predicted_label << "\n";

    return 0;
}
