# gVegascp

Trying to give better performance (as well as commenting all I can) to the code made in 2012 by Junichi Kanzaki (junichi.kanzaki@kek.jp) to integrate functions with the VEGAS algorithm (variation of Monte Carlo). The introductory publication is Monte Carlo integration on GPU (http://arxiv.org/abs/1010.2107).

The example function I used is norm(x)^2 in the [-1,-1]^6 box.

The code uses OpenMP to measure function times and can be compiled with "nvcc gVegasMain.cu -lgomp".
