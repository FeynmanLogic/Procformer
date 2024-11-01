# 4-bit and 2-bit Quantized MAC VHDL Project

## Project Overview

This project implements a Multiply-Accumulate (MAC) operation for 4-bit data, with an option for quantized 2-bit MAC operations. Based on the `quantized` signal, the system dynamically selects between standard 4-bit and 2-bit quantized computations. The primary goal is to achieve flexible MAC operations and sigmoid activation with both full-precision and quantized precision, allowing efficient hardware synthesis while maintaining dynamic operational control through the quantized signal.

### Key Components
The dependencies for this project include:
- `mac_2bit`: 2-bit MAC operation module used in quantized mode.
- `rom_and`: Look-Up Table (LUT) for AND operations.
- `rom_xor`: LUT for XOR operations.
- `rom_sigmoid`: ROM-based sigmoid activation for non-quantized mode.
- `rom_quantized_sigmoid`: ROM-based sigmoid activation in quantized mode.

Please ensure these modules are compiled before compiling `lut.vhd`, as they are integral to the lookup table and MAC operations.

### Compilation Order
1. Compile dependencies:
   - `mac_2bit.vhd`
   - `rom_and.vhd`
   - `rom_xor.vhd`
   - `rom_sigmoid.vhd`
   - `rom_quantized_sigmoid.vhd`
2. Compile `lut.vhd` (main module).
3. Compile the test bench, `test_lut_tb.vhd`.

### Project Details
This implementation uses a LUT-based approach for basic operations:
- `rom_and` and `rom_xor` serve as LUTs for respective logical operations.
- `rom_sigmoid` and `rom_quantized_sigmoid` provide sigmoid activation values based on decimal notation, where, for example, `00001000` corresponds to a value calculated as \(2^{-3}\) or equivalent based on binary position.
# The goal of the project is this;


1. **All Components (LUTs and Calculations) Are Instantiated**: All the lookup tables (`rom_and`, `rom_xor`, `rom_sigmoid`, and `mac_2bit`) are instantiated outside the process block, so they are present in the synthesized design.

2. **Conditional Calculation Path Based on `quantized` Signal**: The process block only enables the calculations for either the 2-bit path (when `quantized = '1'`) or the 4-bit path (when `quantized = '0'`).
   - **When `quantized = '1'`**: The code inside this condition performs the calculations using `a_quantized` and `b_quantized` for the 2-bit calculation. The output `lut_newmul_out` is assigned to `sigmoid_bahar`, which corresponds to the quantized calculation result.
   - **When `quantized = '0'`**: The code performs the 4-bit calculations using the full 4-bit values of `a` and `b`. The signals `a_effective` and `b_effective` are populated with `a` and `b`, and the lookup tables perform the necessary steps to calculate `s_out1`, the full-precision result, which is then assigned to `lut_newmul_out`.

### Synthesis Behavior

During synthesis:
- **All Components are Physically Realized**: Every lookup table and instantiated component will be synthesized into hardware because they are outside the process block.
- **Calculation Path Control**: Although all components are instantiated, only one path's calculations are functionally active at a time, depending on the value of `quantized`. This ensures that only the necessary computations are conducted dynamically at runtime.
`quantized`.
### Testing with Test Bench
The test bench, `test_lut_tb.vhd`, evaluates the Mean Squared Error (MSE) between quantized and non-quantized outputs. Here’s the approach:
1. Converts binary representations from `lut_newmul_out` to decimal values.
2. Calculates MSE across all test cases.
3. Outputs a report to an MSE results file (`mse_results.txt`) included in the repository for analyzing quantization effects.

The test bench generates case-by-case MSE data, helping to analyze precision loss when using quantized MAC. You may check this output file for detailed insights.

### Synthesis and Process Control
The project uses a process block controlled by the `quantized` signal, dynamically switching between quantized and full-precision calculations. This ensures that:
- Only one of the calculation paths is synthesized based on the state of `quantized`.
- Each component is physically synthesized as specified.

Please verify synthesis compatibility and logic accuracy, as I am new to VHDL. Any improvements or error corrections you identify will be highly appreciated.

---

Thank you for your assistance in synthesizing this project and ensuring it adheres to the design specifications.