################################################################################
#
# Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO USER:   
#
# This source code is subject to NVIDIA ownership rights under U.S. and 
# international Copyright laws.  
#
# NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
# CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
# IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
# REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
# MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.   
# IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
# OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
# OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
# OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE 
# OR PERFORMANCE OF THIS SOURCE CODE.  
#
# U.S. Government End Users.  This source code is a "commercial item" as 
# that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of 
# "commercial computer software" and "commercial computer software 
# documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) 
# and is provided to the U.S. Government only as a commercial end item.  
# Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
# 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
# source code with only those rights set forth herein.
#
################################################################################
#
# Build script for project
#
################################################################################
# Set the source directory
SRCDIR := ./source

# CUDA dependency files
CU_DEPS      := $(SRCDIR)/gVegasFunc.cu  $(SRCDIR)/gVegas.cu  $(SRCDIR)/gVegasCallFunc.cu  $(SRCDIR)/xorshift.cu
CCFILES	     :=  $(SRCDIR)/getrusage_sec.cpp
HFILES		 :=  $(SRCDIR)/const.h  $(SRCDIR)/getrusage_sec.h  $(SRCDIR)/gvegas.h  $(SRCDIR)/kernels.h  $(SRCDIR)/vegas.h $(SRCDIR)/vegasconst.h

# Options already here from gvegas
fastmath = 1
maxregisters = ${NREG}
ptxas = 1
NVCCFLAGS := -OPT:Olimit=0
#SMVERSIONFLAGS := -arch sm_13

# The variable, SMS, is recommended by DMC to ensure architectures are correct.
SMS ?= 35 60 70

################################################################################
# Custom flags for cuda compiler

# For Virutal Architecture with Kepler Support and GPU Features providing Basic and Kepler Support
# CUDACCFLAGS := --ptxas-options=-v --gpu-architecture=compute_32 --gpu-code=sm_32
CUDACCFLAGS := -w -lcublas -lcurand -lcudadevrt -lcudart -lgomp -O 0

################################################################################
# Rules and targets
.PHONY: all
all: reg_chi_1of4_mu2 reg_chi_2of4_mu2 reg_chi_3of4_mu2 reg_chi_4of4_mu2

reg_chi_1of4_mu2 :  $(SRCDIR)/gVegasMain1.cu $(CU_DEPS) $(CCFILES) $(HFILES)
	nvcc -o $@  $(SRCDIR)/gVegasMain1.cu $(CCFILES) $(CUDACCFLAGS)

reg_chi_2of4_mu2 :  $(SRCDIR)/gVegasMain2.cu $(CU_DEPS) $(CCFILES) $(HFILES)
	nvcc -o $@  $(SRCDIR)/gVegasMain2.cu $(CCFILES) $(CUDACCFLAGS)

reg_chi_3of4_mu2 :  $(SRCDIR)/gVegasMain3.cu $(CU_DEPS) $(CCFILES) $(HFILES)
	nvcc -o $@  $(SRCDIR)/gVegasMain3.cu $(CCFILES) $(CUDACCFLAGS)

reg_chi_4of4_mu2 :  $(SRCDIR)/gVegasMain4.cu $(CU_DEPS) $(CCFILES) $(HFILES)
	nvcc -o $@  $(SRCDIR)/gVegasMain4.cu $(CCFILES) $(CUDACCFLAGS)

.PHONY: clean
clean:
	rm reg_chi_1of4 reg_chi_2of4 reg_chi_3of4 reg_chi_4of4