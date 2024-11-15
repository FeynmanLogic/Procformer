Respected sir, These is the Readme file to talk about how to generate and execute traces.
The primary issue is to generate memory profile/trace using the intel PIN tool, and then make them Champsim ready using the CVP converter.
The other is just how to execute the file and see what we are doing.

---

# Model Training, Pruning, and Performance Analysis with ChampSim Tracing

This project demonstrates training a Transformer-based model for sentiment analysis on the IMDB dataset, applying pruning to reduce model size, and analyzing memory efficiency before and after pruning with ChampSim traces.

---

## Table of Contents
1. [Requirements](#requirements)
2. [Code Explanation](#code-explanation)
3. [Execution Guide](#execution-guide)
4. [Analyzing Model Output](#analyzing-model-output)
5. [Generating ChampSim Traces](#generating-champsim-traces)
6. [Comparing Pre-Pruning and Post-Pruning Performance](#comparing-pre-pruning-and-post-pruning-performance)
7. [Troubleshooting and Tips](#troubleshooting-and-tips)

---

## 1. Requirements

Ensure you have the following installed:

- **Python** (>= 3.8)
- **PyTorch**
- **Hugging Face Transformers** (for tokenization and dataset handling)
- **ChampSim** (for cache simulation)
- **Intel Pin** or **perf** (for memory trace generation)

To install the Python dependencies, run:

```bash
pip install torch transformers datasets
```




## 2. Code Explanation

### `pre_pruning.py`

This script trains the model on the IMDB dataset and saves its state before pruning.

- **Model Setup**: Loads and tokenizes IMDB data, builds a Transformer model for binary sentiment classification.
- **Training**: Runs multiple epochs on the training set.
- **Saving State**: Saves the trained model state to `model_pre_pruning.pth`.

### `post_pruning.py`

This script loads the pre-trained model, applies pruning, and evaluates the pruned model.

- **Loading State**: Loads the model saved in `pre_pruning.py`.
- **Pruning**: Prunes neurons based on L2 norm.
- **Saving Post-Pruning State**: Saves the pruned model to `model_post_pruning.pth`.


---

## 3. Execution Guide

1. **Pre-Pruning Execution**

   Run `pre_pruning.py` to train and save the model before pruning:

   ```bash
   python3 pre_pruning.py
   ```

   This will:
   - Train the model on the IMDB dataset.
   - Save the model’s state to `model_pre_pruning.pth`.

2. **Post-Pruning Execution**

   Run `post_pruning.py` to load, prune, and evaluate the model:

   ```bash
   python3 post_pruning.py
   ```

   This will:
   - Load the pre-trained model.
   - Apply pruning and save the pruned model as `model_post_pruning.pth`.
   - Log predictions in `predictions_after.txt`.

---

## 4. Analyzing Model Output

Model predictions are logged before and after pruning in separate files for easy comparison:

- **Pre-Pruning Output**: `predictions_before.txt`
- **Post-Pruning Output**: `predictions_after.txt`

Each file includes:
- **Input Text**: The IMDB review text.
- **Predicted Label**: Model's sentiment prediction (Positive/Negative).

The logs make it easy to see if the pruning process affected the prediction accuracy.

---
# The problem is, we need to have C++ code. Here is how to get
# ONNX Runtime Inference Guide on Ubuntu

This guide explains how to set up and execute an ONNX Runtime-based inference script on an Ubuntu system, starting from installation to execution.

---

## Prerequisites

1. **Ubuntu 20.04 or later**
2. **g++** (GNU C++ compiler)
3. **CMake** (For building ONNX Runtime, if required)
4. Python (Optional, for exporting models to ONNX format)

---

## Step 1: Install Dependencies

Before setting up ONNX Runtime, install the required dependencies:

```bash
sudo apt update && sudo apt upgrade
sudo apt install -y build-essential wget cmake libgomp1
```

---

## Step 2: Download ONNX Runtime

1. Visit the [ONNX Runtime Releases](https://github.com/microsoft/onnxruntime/releases) page.
2. Download the precompiled ONNX Runtime package for Linux.
   
   Alternatively, download it via `wget`:

   ```bash
   wget https://github.com/microsoft/onnxruntime/releases/download/v1.20.0/onnxruntime-linux-x64-1.20.0.tgz
   ```

3. Extract the package:

   ```bash
tar -xvzf onnxruntime-linux-x64-1.20.0.tgz
   ```

4. Move the extracted directory to a suitable location:

   ```bash
   sudo mv onnxruntime-linux-x64-1.20.0 /usr/local/onnxruntime
   ```

---

## Step 3: Set Up Environment Variables

To ensure the system can locate the ONNX Runtime libraries during runtime:

1. Add ONNX Runtime's library path to your environment:

   ```bash
   echo 'export LD_LIBRARY_PATH=/usr/local/onnxruntime/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
   source ~/.bashrc
   ```

2. Verify the library path:

   ```bash
   echo $LD_LIBRARY_PATH
   ```

   You should see `/usr/local/onnxruntime/lib` included.

---

## Step 4: Write Your Inference Script

Create a C++ script for inference, for example, `onnx_inference.cpp`. Use the code provided earlier in this guide.

Save the file in your project directory:

```bash
nano onnx_inference.cpp
```

Paste the content, then save and exit.

---

## Step 5: Compile the Script

Compile the script using `g++` and link it to the ONNX Runtime library:

```bash
g++ onnx_inference.cpp -o onnx_inference \
    -I/usr/local/onnxruntime/include \
    -L/usr/local/onnxruntime/lib \
    -lonnxruntime \
    -std=c++17 -O3
```

---

## Step 6: Run the Inference Script

Run the compiled executable:

```bash
./onnx_inference
```

If you encounter errors about missing libraries, ensure the `LD_LIBRARY_PATH` is correctly set.

---

## Optional: Export an ONNX Model (Python)

If you don’t have an ONNX model, you can export one using PyTorch or TensorFlow. Here’s an example using PyTorch:

1. Install PyTorch and ONNX:

   ```bash
   pip install torch torchvision onnx
   ```

2. Export a model to ONNX:

   ```python
   import torch
   import torch.nn as nn

   # Example PyTorch Model
   class SimpleModel(nn.Module):
       def __init__(self):
           super(SimpleModel, self).__init__()
           self.fc = nn.Linear(10, 2)

       def forward(self, x):
           return self.fc(x)

   model = SimpleModel()
   model.eval()

   # Dummy input for the model
   dummy_input = torch.randn(1, 10)

   # Export the model
   torch.onnx.export(model, dummy_input, "simple_model.onnx", export_params=True, opset_version=11)
   print("Model exported to simple_model.onnx")
   ```

3. Move the exported `simple_model.onnx` to your project directory.

---

## Common Errors and Fixes

1. **Error: Library not found**
   - Ensure `LD_LIBRARY_PATH` includes `/usr/local/onnxruntime/lib`.

2. **Error: Header file not found**
   - Verify the `-I` path during compilation points to `/usr/local/onnxruntime/include`.

3. **Error: Input/Output names mismatch**
   - Inspect your ONNX model with [Netron](https://netron.app/) to verify the input and output names.

---

## Final Notes

You can now run your ONNX model inference script on Ubuntu. Modify the paths as needed for your specific setup. For advanced features like GPU inference, install the ONNX Runtime GPU version and follow similar steps.





# How to create traces; The most important part

We have included only 4 sample traces, taken from SPEC CPU 2006. These 
traces are short (10 million instructions), and do not necessarily cover the range of behaviors your 
replacement algorithm will likely see in the full competition trace list (not
included).  We STRONGLY recommend creating your own traces, covering
a wide variety of program types and behaviors.

The included Pin Tool champsim_tracer.cpp can be used to generate new traces.
We used Pin 3.2 (pin-3.2-81205-gcc-linux), and it may require 
installing libdwarf.so, libelf.so, or other libraries, if you do not already 
have them. Please refer to the Pin documentation (https://software.intel.com/sites/landingpage/pintool/docs/81205/Pin/html/)
for working with Pin 3.2.

Get this version of Pin:
```
wget http://software.intel.com/sites/landingpage/pintool/downloads/pin-3.2-81205-gcc-linux.tar.gz
```

**Note on compatibility**: If you are using newer linux kernels/Ubuntu versions (eg. 20.04LTS), you might run into issues (such as [[1](https://github.com/ChampSim/ChampSim/issues/102)],[[2](https://stackoverflow.com/questions/55698095/intel-pin-tools-32-bit-processsectionheaders-560-assertion-failed)],[[3](https://stackoverflow.com/questions/43589174/pin-tool-segmentation-fault-for-ubuntu-17-04)]) with the PIN3.2. ChampSim tracer works fine with newer PIN tool versions that can be downloaded from [here](https://software.intel.com/content/www/us/en/develop/articles/pin-a-binary-instrumentation-tool-downloads.html). PIN3.17 is [confirmed](https://github.com/ChampSim/ChampSim/issues/102) to work with Ubuntu 20.04.1 LTS.

Once downloaded, open tracer/make_tracer.sh and change PIN_ROOT to Pin's location.
Run ./make_tracer.sh to generate champsim_tracer.so.

**Use the Pin tool like this**
```
pin -t obj-intel64/champsim_tracer.so -- <your program here>
```

The tracer has three options you can set:
```
-o
Specify the output file for your trace.
The default is default_trace.champsim

-s <number>
Specify the number of instructions to skip in the program before tracing begins.
The default value is 0.

-t <number>
The number of instructions to trace, after -s instructions have been skipped.
The default value is 1,000,000.
```
For example, you could trace 200,000 instructions of the program ls, after
skipping the first 100,000 instructions, with this command:
```
pin -t obj/champsim_tracer.so -o traces/ls_trace.champsim -s 100000 -t 200000 -- ls
```
Traces created with the champsim_tracer.so are approximately 64 bytes per instruction,
but they generally compress down to less than a byte per instruction using xz compression.

### Convert the Pin Trace to ChampSim CvP Format
The cvp2champsim tracer comes as is with no guarantee that it covers every conversion case.

The tracer is used to convert the traces from the 2nd Championship Value 
Prediction (CVP) to a ChampSim-friendly format. 

CVP-1 Site: https://www.microarch.org/cvp1/
CVP-2 Site: https://www.microarch.org/cvp1/cvp2/rules.html

To use the tracer first compile it using g++:

g++ cvp2champsim.cc -o cvp_tracer

To convert a trace execute:

./cvp_tracer TRACE_NAME.gz

The ChampSim trace will be sent to standard output so to keep and compress the 
output trace run:

./cvp_tracer TRACE_NAME.gz | gzip > NEW_TRACE.champsim.gz

Adding the "-v" flag will print the dissassembly of the CVP trace to standard 
error output as well as the ChampSim format to standard output.
### Running ChampSim with Generated Traces

Use ChampSim to simulate cache performance on both traces.

```bash
# Pre-Pruning Trace
./run_champsim.sh bimodal-no-no-no-1core 1 10 traces/trace_before.champsimtrace

# Post-Pruning Trace
./run_champsim.sh bimodal-no-no-no-1core 1 10 traces/trace_after.champsimtrace
```

---

## 6. Comparing Pre-Pruning and Post-Pruning Performance

Compare the ChampSim outputs for `trace_before.champsimtrace` and `trace_after.champsimtrace`:

- **Key Metrics**:
  - **Cache Hit Rate**: Higher hit rates suggest better cache utilization.
  - **Memory Access Latency**: Lower latency indicates faster access times.
- **Execution Time**: Pruning may reduce memory footprint, leading to fewer cache misses and reduced latency.

### Example Analysis

Save outputs from each ChampSim run and analyze improvements in metrics like cache hit rate, miss rate, and latency. You can record results in a comparison table:

| Metric            | Pre-Pruning | Post-Pruning | Improvement |
|-------------------|-------------|--------------|-------------|
| L1 Cache Hit Rate | X%          | Y%           | +Z%         |
| Memory Latency    | X ms        | Y ms         | -Z ms       |

This allows you to quantitatively assess the effect of pruning on model performance.

---

## 7. Troubleshooting and Tips

- **Insufficient Disk Space**: Delete unnecessary files or use external storage for large traces.
- **Intel Pin Issues**: Ensure you have the correct version of Pin for your OS.
- **Perf Permissions**: If `perf` gives permission issues, run with `sudo` or adjust perf permissions:
  ```bash
  sudo sysctl -w kernel.perf_event_paranoid=-1
  ```
- **Memory Constraints**: Use smaller datasets or fewer epochs if running out of memory.

By following these steps, you’ll successfully execute the model, prune it, generate memory traces, and compare the memory performance with ChampSim. Let me know if you have any questions!