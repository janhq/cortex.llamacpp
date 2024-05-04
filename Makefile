# Makefile for Cortex llamacpp engine - Build, Lint, Test, and Clean

CMAKE_EXTRA_FLAGS ?= ""
RUN_TESTS ?= false
LLM_MODEL_URL ?= "https://delta.jan.ai/tinyllama-1.1b-chat-v0.3.Q2_K.gguf"
EMBEDDING_MODEL_URL ?= "https://catalog.jan.ai/dist/models/embeds/nomic-embed-text-v1.5.f16.gguf"

# Default target, does nothing
all:
	@echo "Specify a target to run"

# Build the Cortex engine
build-lib:
ifeq ($(OS),Windows_NT)
	@powershell -Command "cmake -S ./third-party -B ./build_deps/third-party;"
	@powershell -Command "cmake --build ./build_deps/third-party --config Release -j4;"
	@powershell -Command "mkdir -p build; cd build; cmake .. $(CMAKE_EXTRA_FLAGS); cmake --build . --config Release;"
else ifeq ($(shell uname -s),Linux)
	@cmake -S ./third-party -B ./build_deps/third-party;
	@make -C ./build_deps/third-party -j4;
	@rm -rf ./build_deps/third-party;
	@mkdir build && cd build; \
	cmake .. $(CMAKE_EXTRA_FLAGS); \
	make -j4;
else
	@cmake -S ./third-party -B ./build_deps/third-party
	@make -C ./build_deps/third-party -j4
	@rm -rf ./build_deps/third-party
	@mkdir build && cd build; \
	cmake .. $(CMAKE_EXTRA_FLAGS); \
	make -j4;
endif

build-example-server: build-lib
ifeq ($(OS),Windows_NT)
	@powershell -Command "mkdir -p .\examples\server\build; cd .\examples\server\build; cmake .. $(CMAKE_EXTRA_FLAGS); cmake --build . --config Release;"
else ifeq ($(shell uname -s),Linux)
	@mkdir -p examples/server/build && cd examples/server/build; \
	cmake .. $(CMAKE_EXTRA_FLAGS); \
	cmake --build . --config Release;
else
	@mkdir -p examples/server/build && cd examples/server/build; \
	cmake ..; \
	cmake --build . --config Release;
endif

package:
ifeq ($(OS),Windows_NT)
	@powershell -Command "mkdir -p cortex.llamacpp; cp build\Release\engine.dll cortex.llamacpp\; 7z a -ttar temp.tar cortex.llamacpp\*; 7z a -tgzip cortex.llamacpp.tar.gz temp.tar;"
else ifeq ($(shell uname -s),Linux)
	@mkdir -p cortex.llamacpp; \
	cp build/libengine.so cortex.llamacpp/; \
	tar -czvf cortex.llamacpp.tar.gz cortex.llamacpp;
else
	@mkdir -p cortex.llamacpp; \
	cp build/libengine.dylib cortex.llamacpp/; \
	tar -czvf cortex.llamacpp.tar.gz cortex.llamacpp;
endif

run-e2e-test:
ifeq ($(RUN_TESTS),false)
	@echo "Skipping tests"
	@exit 0
endif
ifeq ($(OS),Windows_NT)
	@powershell -Command "mkdir -p examples\server\build\Release\engines\cortex.llamacpp; cd examples\server\build\Release; cp ..\..\..\..\build\Release\engine.dll engines\cortex.llamacpp; ..\..\..\..\.github\scripts\e2e-test-server-windows.bat server.exe $(LLM_MODEL_URL) $(EMBEDDING_MODEL_URL);"
else ifeq ($(shell uname -s),Linux)
	@mkdir -p examples/server/build/engines/cortex.llamacpp; \
	cd examples/server/build/; \
	cp ../../../build/libengine.so engines/cortex.llamacpp/; \
	chmod +x ../../../.github/scripts/e2e-test-server-linux-and-mac.sh && ../../../.github/scripts/e2e-test-server-linux-and-mac.sh ./server $(LLM_MODEL_URL) $(EMBEDDING_MODEL_URL);
else
	@mkdir -p examples/server/build/engines/cortex.llamacpp; \
	cd examples/server/build/; \
	cp ../../../build/libengine.dylib engines/cortex.llamacpp/; \
	chmod +x ../../../.github/scripts/e2e-test-server-linux-and-mac.sh && ../../../.github/scripts/e2e-test-server-linux-and-mac.sh ./server $(LLM_MODEL_URL) $(EMBEDDING_MODEL_URL);
endif