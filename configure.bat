cmake -S ./third-party -B ./build_deps/third-party
cmake --build ./build_deps/third-party --config Release -j %NUMBER_OF_PROCESSORS%