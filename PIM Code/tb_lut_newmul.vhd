library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;
use STD.TEXTIO.ALL;

entity tb_lut_newmul is
end tb_lut_newmul;

architecture Behavioral of tb_lut_newmul is

    component lut_newmul
        Port (
            lut_newmul_out : out std_logic_vector(7 downto 0);
            a              : in std_logic_vector(3 downto 0);
            b              : in std_logic_vector(3 downto 0);
            quantized      : in std_logic
        );
    end component;

    signal a, b       : std_logic_vector(3 downto 0) := "0000";
    signal quantized  : std_logic := '0';
    signal lut_newmul_out : std_logic_vector(7 downto 0);
    signal lut_newmul_out_nonquantized : std_logic_vector(7 downto 0);

    -- File variable for output
    file mse_file : text open write_mode is "mse_results.txt";

    -- Function to calculate squared difference between two real numbers
    function squared_error (a : real; b : real) return real is
        variable diff : real;
    begin
        diff := a - b;
        return diff * diff;
    end function;

    -- Converts an 8-bit purely fractional binary value to real decimal
    function to_decimal (bits : std_logic_vector(7 downto 0)) return real is
        variable fraction : real := 0.0;
    begin
        for i in 0 to 7 loop
            if bits(i) = '1' then
                fraction := fraction + (1.0 / (2.0 ** (i + 1)));
            end if;
        end loop;
        return fraction;
    end function;

begin
    uut: lut_newmul Port map (
        lut_newmul_out => lut_newmul_out,
        a              => a,
        b              => b,
        quantized      => quantized
    );

    -- Stimulus Process
    stim_proc: process
        variable mse_accumulator : real := 0.0;
        variable test_count : integer := 0;
        variable mse : real;
        variable line_out : line;
        variable nonquantized_val : real;
        variable quantized_val : real;
    begin
        for i in 0 to 15 loop
            a <= std_logic_vector(to_unsigned(i, 4));
            for j in 0 to 15 loop
                b <= std_logic_vector(to_unsigned(j, 4));
                
                -- Run in non-quantized mode
                quantized <= '0';
                wait for 10 ns;
                lut_newmul_out_nonquantized <= lut_newmul_out;

                -- Run in quantized mode
                quantized <= '1';
                wait for 10 ns;

                -- Convert outputs to decimal
                nonquantized_val := to_decimal(lut_newmul_out_nonquantized);
                quantized_val := to_decimal(lut_newmul_out);

                -- Calculate Mean Squared Error for this input pair
                mse := squared_error(quantized_val, nonquantized_val);
                
                -- Accumulate the squared error
                mse_accumulator := mse_accumulator + mse;
                test_count := test_count + 1;

                -- Write individual MSE to file
                write(line_out, string'("a: " & integer'image(i) & ", b: " & integer'image(j) &
                                     " | Non-Quantized: " & real'image(nonquantized_val) &
                                     " | Quantized: " & real'image(quantized_val) &
                                     " | MSE: " & real'image(mse)));
                writeline(mse_file, line_out);
            end loop;
        end loop;

        -- Calculate and write mean MSE to file
        write(line_out, string'("Overall Mean MSE: " & real'image(mse_accumulator / real(test_count))));
        writeline(mse_file, line_out);

        -- Close the file
        file_close(mse_file);

        wait;
    end process;

end Behavioral;
