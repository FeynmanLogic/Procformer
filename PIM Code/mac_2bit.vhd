library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

entity mac_2bit is
  Port (
    macout : out std_logic_vector(3 downto 0); -- 8-bit output for quantized MAC result
    a         : in std_logic_vector(1 downto 0);  -- 2-bit input
    b         : in std_logic_vector(1 downto 0)   -- 2-bit input
  );
end mac_2bit;

architecture Behavioral of mac_2bit is
  signal and_out1 : std_logic_vector(1 downto 0);
  signal and_out2 : std_logic_vector(1 downto 0);
  signal xor_out1 : std_logic_vector(1 downto 0);
  signal acc_out  : std_logic_vector(3 downto 0);
  signal temp     : std_logic_vector(7 downto 0);
begin
  -- Adjust the bitwise operations
  and_out1(0) <= a(0) and b(0);
  and_out1(1) <= '0';
  and_out2(0) <= a(1) and b(1);
  and_out2(1) <= '0';

  xor_out1(0) <= and_out1(0) xor and_out2(0);
  xor_out1(1) <= and_out1(1) xor and_out2(1);

  -- Accumulation and Sigmoid
  acc_out <= "00" & xor_out1;
  
macout <=acc_out;
end Behavioral;
