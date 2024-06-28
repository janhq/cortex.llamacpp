# Makefile for Cortex llamacpp engine - Build, Lint, Test, and Clean

CMAKE_EXTRA_FLAGS ?= ""
RUN_TESTS ?= false
LLM_MODEL_URL ?= "https://delta.jan.ai/tinyllama-1.1b-chat-v0.3.Q2_K.gguf"
EMBEDDING_MODEL_URL ?= "https://catalog.jan.ai/dist/models/embeds/nomic-embed-text-v1.5.f16.gguf"
CODE_SIGN ?= false
AZURE_KEY_VAULT_URI ?= xxxx
AZURE_CLIENT_ID ?= xxxx
AZURE_TENANT_ID ?= xxxx
AZURE_CLIENT_SECRET ?= xxxx
AZURE_CERT_NAME ?= xxxx
DEVELOPER_ID ?= xxxx

# Default target, does nothing
all:
	@echo "Specify a target to run"

# Build the Cortex engine
build-lib:
ifeq ($(OS),Windows_NT)
	@powershell -Command "cmake -S ./third-party -B ./build_deps/third-party -DCMAKE_CXX_COMPILER_LAUNCHER=sccache -DCMAKE_C_COMPILER_LAUNCHER=sccache -DCMAKE_CUDA_COMPILER_LAUNCHER=sccache -DCMAKE_CXX_COMPILER=cl -DCMAKE_C_COMPILER=cl -DCMAKE_BUILD_TYPE=Release -GNinja;"
	@powershell -Command "cmake --build ./build_deps/third-party --config Release -j4;"
	@powershell -Command "mkdir -p build; cd build; cmake .. $(CMAKE_EXTRA_FLAGS); cmake --build . --config Release;"
else ifeq ($(shell uname -s),Linux)
	@cmake -S ./third-party -B ./build_deps/third-party;
	@make -C ./build_deps/third-party -j4;
	@rm -rf ./build_deps/third-party;
	@mkdir build && cd build; \
	cmake .. $(CMAKE_EXTRA_FLAGS); \
	cmake --build . --config Release --parallel 4;
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

pre-package:
ifeq ($(OS),Windows_NT)
	@powershell -Command "mkdir -p cortex.llamacpp; cp build\engine.dll cortex.llamacpp\;"
else ifeq ($(shell uname -s),Linux)
	@mkdir -p cortex.llamacpp; \
	cp build/libengine.so cortex.llamacpp/;
else
	@mkdir -p cortex.llamacpp; \
	cp build/libengine.dylib cortex.llamacpp/;
endif

codesign:
ifeq ($(CODE_SIGN),false)
	@echo "Skipping Code Sign"
	@exit 0
endif

ifeq ($(OS),Windows_NT)
	@powershell -Command "dotnet tool install --global AzureSignTool;"
	@powershell -Command 'azuresigntool.exe sign -kvu "$(AZURE_KEY_VAULT_URI)" -kvi "$(AZURE_CLIENT_ID)" -kvt "$(AZURE_TENANT_ID)" -kvs "$(AZURE_CLIENT_SECRET)" -kvc "$(AZURE_CERT_NAME)" -tr http://timestamp.globalsign.com/tsa/r6advanced1 -v ".\cortex.llamacpp\engine.dll";'
else ifeq ($(shell uname -s),Linux)
	@echo "Skipping Code Sign for linux"
	@exit 0
else
	find "cortex.llamacpp" -type f -exec codesign --force -s "$(DEVELOPER_ID)" --options=runtime {} \;
endif

package:
ifeq ($(OS),Windows_NT)
	@powershell -Command "7z a -ttar temp.tar cortex.llamacpp\*; 7z a -tgzip cortex.llamacpp.tar.gz temp.tar;"
else ifeq ($(shell uname -s),Linux)
	@tar -czvf cortex.llamacpp.tar.gz cortex.llamacpp;
else
	@tar -czvf cortex.llamacpp.tar.gz cortex.llamacpp;
endif

run-e2e-test:
ifeq ($(RUN_TESTS),false)
	@echo "Skipping tests"
	@exit 0
endif
ifeq ($(OS),Windows_NT)
	@powershell -Command "mkdir -p examples\server\build\engines\cortex.llamacpp; cd examples\server\build; cp ..\..\..\build\engine.dll engines\cortex.llamacpp; ..\..\..\.github\scripts\e2e-test-server-windows.bat server.exe $(LLM_MODEL_URL) $(EMBEDDING_MODEL_URL);"
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

run-e2e-submodule-test:
ifeq ($(RUN_TESTS),false)
	@echo "Skipping tests"
	@exit 0
endif
ifeq ($(OS),Windows_NT)
	@powershell -Command "python -m pip install --upgrade pip"
	@powershell -Command "python -m pip install requests;"
	@powershell -Command "mkdir -p examples\server\build\engines\cortex.llamacpp; cd examples\server\build; cp ..\..\..\build\engine.dll engines\cortex.llamacpp; python ..\..\..\.github\scripts\e2e-test-server.py server $(LLM_MODEL_URL) $(EMBEDDING_MODEL_URL);"
else ifeq ($(shell uname -s),Linux)
	python -m pip install --upgrade pip;
	python -m pip install requests;
	@mkdir -p examples/server/build/engines/cortex.llamacpp; \
	cd examples/server/build/; \
	cp ../../../build/libengine.so engines/cortex.llamacpp/; \
	python  ../../../.github/scripts/e2e-test-server.py server $(LLM_MODEL_URL) $(EMBEDDING_MODEL_URL);
else
	python -m pip install --upgrade pip;
	python -m pip install requests;
	@mkdir -p examples/server/build/engines/cortex.llamacpp; \
	cd examples/server/build/; \
	cp ../../../build/libengine.dylib engines/cortex.llamacpp/; \
	python  ../../../.github/scripts/e2e-test-server.py server $(LLM_MODEL_URL) $(EMBEDDING_MODEL_URL);
endif