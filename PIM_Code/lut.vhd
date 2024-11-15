library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

entity lut_newmul is
  Port (
    lut_newmul_out : out std_logic_vector(7 downto 0);
    a : in std_logic_vector(3 downto 0);
    b : in std_logic_vector(3 downto 0);
    quantized : in std_logic
  );
end lut_newmul;

architecture Behavioral of lut_newmul is

  -- Signals for quantized path (2-bit calculation)
  signal a_quantized, b_quantized : std_logic_vector(1 downto 0);
  signal mac_out_quant : std_logic_vector(3 downto 0);
  signal sigmoid_bahar : std_logic_vector(7 downto 0);

  -- Signals for full-precision path (4-bit calculation)
  signal a_effective, b_effective : std_logic_vector(3 downto 0);
  signal mac_out1, mac_out2 : std_logic_vector(3 downto 0);
  signal s_out1 : std_logic_vector(7 downto 0);

  -- Intermediate signals for AND and XOR results in full-precision mode
  signal and_out1, and_out2, and_out3, and_out4 : std_logic_vector(3 downto 0);
  signal and_out5, and_out6, and_out7, and_out9 : std_logic_vector(3 downto 0);
  signal and_out11, and_out12, and_out13, and_out14 : std_logic_vector(3 downto 0);
  signal xor_out1, xor_out2, xor_out3, xor_out4 : std_logic_vector(3 downto 0);
  signal xor_out5, xor_out6, xor_out7, xor_out9 : std_logic_vector(3 downto 0);
  signal xor_out11, xor_out12 : std_logic_vector(3 downto 0);
  signal a1, b1, a2, b2, a3, b3, a4, b4, a5, a6, a7, a8, a9, a10 : std_logic_vector(3 downto 0);
  signal b5, b6, b7, b8, b9, b10, b11, b12, b13 : std_logic_vector(3 downto 0);

begin

  -- Instantiate 2-bit MAC and sigmoid for quantized calculation
  mac_2bit_inst1: entity work.mac_2bit
    port map(macout => mac_out_quant, a => a_quantized, b => b_quantized);

  sig_quant_inst1: entity work.rom_quantized_sigmoid
    port map(sigmoid_out => sigmoid_bahar, address => mac_out_quant);

  -- Instantiate components for full-precision (4-bit) calculations
  rom_and_inst1 : entity work.rom_and
    port map(and_out => and_out1, a => a_effective, b => b1);

  rom_and_inst2 : entity work.rom_and
    port map(and_out => and_out2, a => a_effective, b => b2);

  rom_and_inst3 : entity work.rom_and
    port map(and_out => and_out3, a => a_effective, b => b3);

  rom_and_inst4 : entity work.rom_and
    port map(and_out => and_out4, a => a_effective, b => b4);

  rom_xor_inst1 : entity work.rom_xor
    port map(xor_out => xor_out1, a => a1, b => b5);

  rom_and_inst5 : entity work.rom_and
    port map(and_out => and_out5, a => a1, b => b5);

  rom_xor_inst2 : entity work.rom_xor
    port map(xor_out => xor_out2, a => a2, b => b6);

  rom_and_inst6 : entity work.rom_and
    port map(and_out => and_out6, a => a2, b => b6);

  rom_xor_inst3 : entity work.rom_xor
    port map(xor_out => xor_out3, a => a3, b => b7);

  rom_and_inst7 : entity work.rom_and
    port map(and_out => and_out7, a => a3, b => b7);

  rom_xor_inst4 : entity work.rom_xor
    port map(xor_out => xor_out4, a => a4, b => b8);

  rom_and_inst9 : entity work.rom_and
    port map(and_out => and_out9, a => a4, b => b8);

  rom_xor_inst5 : entity work.rom_xor
    port map(xor_out => xor_out5, a => a5, b => b9);

  rom_xor_inst6 : entity work.rom_xor
    port map(xor_out => xor_out6, a => a6, b => b10);

  rom_xor_inst7 : entity work.rom_xor
    port map(xor_out => xor_out7, a => a7, b => b11);

  rom_and_inst11 : entity work.rom_and
    port map(and_out => and_out11, a => a7, b => b11);

  rom_xor_inst9 : entity work.rom_xor
    port map(xor_out => xor_out9, a => a8, b => b12);

  rom_and_inst13 : entity work.rom_and
    port map(and_out => and_out13, a => a8, b => b12);

  rom_xor_inst11 : entity work.rom_xor
    port map(xor_out => xor_out11, a => a9, b => and_out13);

  rom_and_inst14 : entity work.rom_and
    port map(and_out => and_out14, a => a9, b => and_out13);

  rom_xor_inst12 : entity work.rom_xor
    port map(xor_out => xor_out12, a => a10, b => b13);

  rom_sigmoid_inst1 : entity work.rom_sigmoid
    port map(sigmoid_out => s_out1, a => mac_out1, b => mac_out2);

  -- Process to select calculation path based on quantized signal
process(quantized, a, b)
  begin
    if quantized = '1' then
      -- Quantized mode: Perform 2-bit calculations
      if a >= "0011" then
        a_quantized <= a(3 downto 2);
      else
        a_quantized <= a(1 downto 0);
      end if;

      if b >= "0011" then
        b_quantized <= b(3 downto 2);
      else
        b_quantized <= b(1 downto 0);
      end if;

      lut_newmul_out <= sigmoid_bahar;

    else
      -- Full-precision mode: Perform 4-bit calculations
      a_effective <= a;
      b_effective <= b;

      b1 <= b_effective(0) & b_effective(0) & b_effective(0) & b_effective(0);
      b2 <= b_effective(1) & b_effective(1) & b_effective(1) & b_effective(1);
      b3 <= b_effective(2) & b_effective(2) & b_effective(2) & b_effective(2);
      b4 <= b_effective(3) & b_effective(3) & b_effective(3) & b_effective(3);

      b5 <= '0' & and_out1(3) & and_out1(2) & and_out1(1);
      b6 <= '0' & and_out3(3) & and_out3(2) & and_out3(1);

      a1 <= and_out2(3) & and_out2(2) & and_out2(1) & and_out2(0);
      a2 <= and_out4(3) & and_out4(2) & and_out4(1) & and_out4(0);

      b7 <= and_out5(3) & and_out5(2) & and_out5(1) & and_out5(0);
      b8 <= and_out6(3) & and_out6(2) & and_out6(1) & and_out6(0);

      a3 <= '0' & xor_out1(3) & xor_out1(2) & xor_out1(1);
      a4 <= '0' & xor_out2(3) & xor_out2(2) & xor_out2(1);

      b9 <= xor_out3(3) & xor_out3(2) & xor_out3(1) & xor_out1(0);
      a5 <= and_out7(2) & and_out7(1) & and_out7(0) & and_out3(0);

      a6 <= '0' & and_out9(2) & and_out9(1) & and_out9(0);
      b10 <= xor_out4(3) & xor_out4(2) & xor_out4(1) & xor_out4(0);

      b11 <= xor_out6(2) & xor_out6(1) & xor_out6(0) & xor_out2(0);
      a7 <= '0' & xor_out5(3) & xor_out5(2) & xor_out5(1);

      b12 <= xor_out6(2) & xor_out6(1) & xor_out6(0) & xor_out2(0);
      a8 <= '0' & xor_out5(3) & xor_out5(2) & xor_out5(1);

      a9 <= xor_out7(3) & xor_out9(3) & xor_out9(2) & xor_out9(1);

      a10 <= xor_out7(3) & xor_out9(3) & xor_out9(2) & xor_out9(1);
      b13 <= xor_out6(2) & xor_out6(1) & xor_out6(0) & xor_out2(0);

      mac_out1 <= xor_out12(2) & xor_out12(1) & xor_out12(0) & xor_out9(0);
      mac_out2 <= xor_out7(0) & xor_out5(0) & xor_out1(0) & and_out1(0);

      lut_newmul_out <= s_out1;
    end if;
  end process;

end Behavioral;
