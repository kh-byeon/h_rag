* IGZO Native Behavioral Model (Composite Tanh)
Vd 2 0 DC 0.5
Vg 1 0 DC 0.0

B_igzo 2 0 I = { EXP(2.302585093 * ((-1.309362e+01) * (1.0 - (0.5 * (1.0 + TANH(6.721681e+00 * (V(1) - (-1.104277e+00)))))) + ((-6.768842e+00 + 1.227796e+00*V(1) + -6.822335e-01*V(1)*V(1) + 1.205168e-01*V(1)*V(1)*V(1))) * (0.5 * (1.0 + TANH(6.721681e+00 * (V(1) - (-1.104277e+00))))))) }

.DC Vg -3.0 3.0 0.1
.PRINT DC FORMAT=CSV file=sim_igzo.csv V(1) I(Vd)
.END
