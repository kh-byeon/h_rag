* AIMDC Capacitive Boosting Distance Macro
.OPTIONS NONLIN CONTINUE
.TEMP 25

.SUBCKT igzo_tft d g s
BIDS d s I={EXP(2.302585093 * ((-1.305308e+01) * (1.0 - (0.5 * (1.0 + TANH(1.153830e+01 * (V(g,s) - (-1.048026e-01)))))) + ((-9.437630e+00 + 4.929201e+00*V(g,s) + -2.344553e+00*V(g,s)*V(g,s) + 3.660973e-01*V(g,s)*V(g,s)*V(g,s))) * (0.5 * (1.0 + TANH(1.153830e+01 * (V(g,s) - (-1.048026e-01)))))))}
.ENDS igzo_tft

* ---- Left cell ----
XW1   BL    WL    SN     igzo_tft
Ccpl1 SL_bar SN     1e-13
XR1   MBL   RWL   N_mid1 igzo_tft
XS1   N_mid1 SN   ML     igzo_tft

* ---- Right cell (differential) ----
XW2   BL_bar WL   SN_bar igzo_tft
Ccpl2 SL     SN_bar 1e-13
XR2   MBL   RWL   N_mid2 igzo_tft
XS2   N_mid2 SN_bar ML   igzo_tft

* ---- Numerical DC path (ideal float; 1e15 Ohm ~ open) ----
R_leak1 SN     0 1e15
R_leak2 SN_bar 0 1e15

* ---- Biases ----
V_MBL  MBL   0 DC 0.1
V_ML   ML    0 DC 0.0
V_RWL  RWL   0 DC 3.0
V_WL   WL    0 DC -3.0
V_BL   BL    0 DC 0.0
V_BLb  BL_bar 0 DC 0.0

* ---- Stored weights (floating SN / SN_bar) ----
.IC V(SN)=1.0  V(SN_bar)=-1.0

* ---- Differential input PWL ----
V_SL    SL     0 PWL(0 0  9.9e-06 0  1e-05 1.0   2e-05 1.0)
V_SLb   SL_bar 0 PWL(0 0  9.9e-06 0  1e-05 -1.0  2e-05 -1.0)

.TRAN 0.1u 2e-05
.PRINT TRAN FORMAT=CSV file=sim_distance.csv V(SL) V(SN) V(SN_bar) I(V_MBL)
.END
