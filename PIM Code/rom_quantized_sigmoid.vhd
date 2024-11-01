library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

entity rom_quantized_sigmoid is
  Port (
    sigmoid_out : out std_logic_vector(7 downto 0); -- 8-bit output after sigmoid
    address     : in std_logic_vector(3 downto 0)   -- 4-bit input address
  );
end rom_quantized_sigmoid;

architecture Behavioral of rom_quantized_sigmoid is
  type rom_sigmoid is array (0 to 15) of std_logic_vector(7 downto 0);
  constant memory : rom_sigmoid := (
    "00000000",  -- Sigmoid value for address 0
    "00010000",  -- Sigmoid value for address 1
    "00100000",  -- Sigmoid value for address 2
    "00110000",  -- Sigmoid value for address 3
    "01000000",  -- Sigmoid value for address 4
    "01010000",  -- Sigmoid value for address 5
    "01100000",  -- Sigmoid value for address 6
    "01110000",  -- Sigmoid value for address 7
    "10000000",  -- Sigmoid value for address 8
    "10010000",  -- Sigmoid value for address 9
    "10100000",  -- Sigmoid value for address 10
    "10110000",  -- Sigmoid value for address 11
    "11000000",  -- Sigmoid value for address 12
    "11010000",  -- Sigmoid value for address 13
    "11100000",  -- Sigmoid value for address 14
    "11110000"   -- Sigmoid value for address 15
  );
  
begin
  sigmoid_out <= memory(to_integer(unsigned(address)));
end Behavioral;
