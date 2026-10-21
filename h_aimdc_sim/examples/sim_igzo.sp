* Minimal Xyce deck: resistor divider (does not require compiled VA).
* Mount your real sim_igzo.sp here when using the full IGZO OSDI flow.
.title IGZO placeholder deck
V1 1 0 DC 1.0
R1 1 2 1k
R2 2 0 1k
.DC V1 1.0 1.0 0.1
.PRINT DC V(1) V(2)
.END
