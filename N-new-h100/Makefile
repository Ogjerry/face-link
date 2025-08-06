# ====================================================================
# Makefile for CUDA NeRF Project (Corrected & Final Version)
# ====================================================================

# --- Compiler and Flags ---
CXX := g++
NVCC := /usr/bin/nvcc
EXEC := nerf_app

# --- GPU Architecture ---
ARCH ?= 86

# --- Include Paths ---
INCLUDES := -I./include \
            -I/usr/local/cuda/include \
            -I/usr//local/include/opencv4

# --- Compiler Flags ---
OPTIMIZER_FLAGS := -O3 
DEBUGGER_FLAGS := -g -G

# CUDA Compiler flags (-Xcompiler passes flags to the host C++ compiler)
NVCCFLAGS := -std=c++17 $(OPTIMIZER_FLAGS) -gencode arch=compute_$(ARCH),code=sm_$(ARCH) \
             -diag-suppress=549 -diag-suppress=20096 \
             -Xcompiler "-Wall" -Xcompiler "-fopenmp"

# C++ Compiler flags
CXXFLAGS := -std=c++17 -Wall -Werror -Wpedantic

# --- Linker Flags and Libraries ---
LDFLAGS := -L/usr/local/lib -L/usr/local/cuda-12/lib64

# FIX: All libraries that start with '-l' belong here for the linker.
LDLIBS := -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc \
          -lcublas -lcublasLt -lcudart -lcusolver -lnccl -lgomp

# --- Source Files and Objects (Automatic Discovery) ---
SRC_DIRS := . src/nerf src/dataset
SRCS := $(foreach dir,$(SRC_DIRS),$(wildcard $(dir)/*.cu) $(wildcard $(dir)/*.cpp))
OBJS := $(SRCS:.cu=.o)
OBJS := $(OBJS:.cpp=.o)
DEPS := $(OBJS:.o=.d)

# --- Build Rules ---
all: $(EXEC)

# Rule to link the final executable
# This now correctly uses both LDFLAGS (for paths) and LDLIBS (for libraries)
$(EXEC): $(OBJS)
	$(NVCC) $^ -o $@ $(LDFLAGS) $(LDLIBS)
	@echo "✅ Linking complete. Executable created: $(EXEC)"

# Rule to compile .cu files
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@ -MMD -MP

# Rule to compile .cpp files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@ -MMD -MP

# --- Dependency Handling ---
-include $(DEPS)

# --- Clean Rule ---
clean:
	@echo "🧹 Cleaning up generated files..."
	rm -f $(EXEC) $(OBJS) $(DEPS)
	@echo "Cleanup complete."

.PHONY: all clean
