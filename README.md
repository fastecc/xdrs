# XDRS

Authors: Chao Chen (Xidian University)

The provided code implements erasure-correcting encoding and decoding for Reed-Solomon codes based on the LCH-FFT algorithm, with AVX2 intrinsics used to accelerate finite-field multiplication and addition over GF(256).<br>
The algorithm supports switchable low and high code rates:<br>
Low rate: information-symbol length 0 < k ≤ 128<br>
High rate: information-symbol length 128 < k ≤ 256<br>

The src folder contains three source files:<br>
(1) benchmark.cpp (the main file)<br>
Initializes parameters, allocates memory, simulates erasures, generates test data, measures encoding/decoding times, performs correctness checks, and releases memory.<br>
(2) P_function.h<br>
Holds macro definitions and partial function declarations.<br>
(3) P_function.cpp<br>
Employs AVX2 intrinsics to accelerate the entire encoding/decoding pipeline.<br>
