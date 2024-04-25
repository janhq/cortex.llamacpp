# Makefile for Cortex llamacpp engine - Build, Lint, Test, and Clean

CMAKE_EXTRA_FLAGS ?= ""

# Default target, does nothing
all:
	@echo "Specify a target to run"

# Build the Cortex engine
build:
ifeq ($(OS),Windows_NT)
	mkdir -p build
	cd build; \
	cmake .. $(CMAKE_EXTRA_FLAGS); \
	cmake --build . --config Release;
else ifeq ($(shell uname -s),Linux)
	mkdir build && cd build; \
	cmake .. $(CMAKE_EXTRA_FLAGS); \
	make -j$(nproc);
else
	mkdir build && cd build; \
	cmake .. $(CMAKE_EXTRA_FLAGS); \
	make -j$(sysctl -n hw.ncpu);
endif

code-sign: build
ifeq ($(OS),Windows_NT)
	@echo "Hello Windows";
else ifeq ($(shell uname -s),Linux)
	@echo "Hello Linux";
else
	@echo "Hello MacOS";
endif

package: build
ifeq ($(OS),Windows_NT)
	@echo "Hello Windows";
else ifeq ($(shell uname -s),Linux)
	@echo "Hello Linux";
else
	@echo "Hello MacOS";
endif