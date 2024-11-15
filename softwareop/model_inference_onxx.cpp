#include <torch/script.h> // For TorchScript
#include <torch/torch.h>  // For tensor operations
#include <iostream>
#include <vector>

int main() {
    // Load the TorchScript model
    const std::string model_path = "model.pt";
    std::shared_ptr<torch::jit::script::Module> model;

    try {
        model = torch::jit::load(model_path);
        std::cout << "Model loaded successfully.\n";
    } catch (const c10::Error &e) {
        std::cerr << "Error loading the model: " << e.what() << "\n";
        return -1;
    }

    // Prepare input data (example input_ids; replace with actual data)
    std::vector<int64_t> input_ids = {101, 2003, 2023, 1037, 2742, 102}; // Tokenized input
    torch::Tensor input_tensor = torch::tensor(input_ids, torch::kLong).unsqueeze(0); // Add batch dimension

    // Perform inference
    try {
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_tensor);

        auto output = model->forward(inputs).toTensor(); // Run the model
        std::cout << "Model output: " << output << "\n";

        // Get the predicted label
        auto predicted_label = output.argmax(1).item<int64_t>();
        std::cout << "Predicted label: " << predicted_label << "\n";
    } catch (const c10::Error &e) {
        std::cerr << "Error during inference: " << e.what() << "\n";
        return -1;
    }

    return 0;
}
