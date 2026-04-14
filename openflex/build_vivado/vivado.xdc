
    create_clock -period 1.0 [get_ports clk] -name clk
    set_property HD.CLK_SRC BUFGCTRL_X0Y0 [get_ports clk]
    