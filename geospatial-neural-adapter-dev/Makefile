SHELL := /bin/bash
PYTHON ?= $(shell which python)
POETRY ?= $(shell which poetry)
EXECUTABLE := $(PYTHON) -m

.PHONY: clean install-dev build-cpp test run-local-jupyter help conda-env

## Clean up temporary files
clean:
	@echo "Cleaning up..."
	@find . -type f -name '*.py[co]' -delete
	@find . -type d -name '__pycache__' -delete
	@rm -rf build/ dist/ .eggs/
	@find . -name '*.egg-info' -exec rm -rf {} +
	@rm -f .coverage2gf
	@rm -rf .pytest_cache
	@rm -rf htmlcov
	@find . -type d -name '.ipynb_checkpoints' -exec rm -rf {} +

## Install development dependencies
install-dev:
	@echo "Installing development dependencies..."
	@echo "Activating conda environment 'geospatial-neural-adapter'..."
	@source $$(conda info --base)/etc/profile.d/conda.sh && conda activate geospatial-neural-adapter && \
	poetry lock && \
	poetry install && \
	poetry install --only-root

## Build C++ extensions
build-cpp:
	@echo "🔧 Building C++ extensions..."
	@echo "Activating conda environment 'geospatial-neural-adapter'..."
	@source $$(conda info --base)/etc/profile.d/conda.sh && conda activate geospatial-neural-adapter && \
	cd geospatial_neural_adapter/cpp_extensions && \
	rm -rf build CMakeCache.txt CMakeFiles cmake_install.cmake Makefile spatial_utils.so && \
	mkdir -p build && cd build && \
	cmake \
	  -DCMAKE_PREFIX_PATH=$$CONDA_PREFIX \
	  -DCMAKE_INCLUDE_PATH=$$CONDA_PREFIX/include \
	  -DCMAKE_LIBRARY_PATH=$$CONDA_PREFIX/lib \
	  -DCMAKE_BUILD_TYPE=Release .. && \
	make -j$$(nproc)
	@echo "📦 Installing package and copying C++ extensions..."
	@source $$(conda info --base)/etc/profile.d/conda.sh && conda activate geospatial-neural-adapter && poetry install && \
	PACKAGE_DIR=$$(python -c "import geospatial_neural_adapter; print(geospatial_neural_adapter.__file__)" | sed 's/__init__.py//') && \
	echo "Package directory: $$PACKAGE_DIR" && \
	if [ "$$PACKAGE_DIR" = "/home/u3369354/neural-low-rank-spatial-model/geospatial_neural_adapter/" ]; then \
		echo "✅ Development install detected - C++ extensions already in place"; \
	else \
		echo "Installing C++ extensions to: $$PACKAGE_DIR" && \
		cp geospatial_neural_adapter/cpp_extensions/spatial_utils.so $$PACKAGE_DIR/cpp_extensions/ && \
		echo "✅ C++ extensions installed successfully"; \
	fi && \
	echo "🔍 Verifying C++ extensions are accessible..." && \
	python -c "from geospatial_neural_adapter.cpp_extensions import spatial_utils; print('✅ C++ extensions imported successfully')"

## Run tests
test:
	@echo "Running tests..."
	@echo "Activating conda environment 'geospatial-neural-adapter'..."
	@source $$(conda info --base)/etc/profile.d/conda.sh && conda activate geospatial-neural-adapter && \
	$(EXECUTABLE) pytest --cov=geospatial_neural_adapter

## Run tests with coverage report
test-cov:
	@echo "Running tests with coverage report..."
	@echo "Activating conda environment 'geospatial-neural-adapter'..."
	@source $$(conda info --base)/etc/profile.d/conda.sh && conda activate geospatial-neural-adapter && \
	$(EXECUTABLE) pytest --cov=geospatial_neural_adapter --cov-report=html --cov-report=term-missing

## Run fast tests only (skip slow ones)
test-fast:
	@echo "Running fast tests..."
	@echo "Activating conda environment 'geospatial-neural-adapter'..."
	@source $$(conda info --base)/etc/profile.d/conda.sh && conda activate geospatial-neural-adapter && \
	$(EXECUTABLE) pytest -m "not slow" --cov=geospatial_neural_adapter

## Run GPU tests only
test-gpu:
	@echo "Running GPU tests..."
	@echo "Activating conda environment 'geospatial-neural-adapter'..."
	@source $$(conda info --base)/etc/profile.d/conda.sh && conda activate geospatial-neural-adapter && \
	$(EXECUTABLE) pytest -m "gpu" --cov=geospatial_neural_adapter

## Start Jupyter server locally
run-local-jupyter:
	@echo "Starting local Jupyter server..."
	@$(SHELL) envs/jupyter/start_jupyter_lab.sh --port 8501

## Set up Conda environment using the provided build script
conda-env:
	@echo "Setting up Conda environment using envs/conda/build_conda_env.sh ..."
	@bash envs/conda/build_conda_env.sh
	@echo "🔧 Building C++ extensions for new environment..."
	@make build-cpp

## Display help information
help:
	@echo "Available targets:"
	@echo "  clean                : Clean up temporary files"
	@echo "  install-dev          : Install development dependencies"
	@echo "  build-cpp            : Build C++ extensions"
	@echo "  test                 : Run tests"
	@echo "  test-cov             : Run tests with coverage report"
	@echo "  test-fast            : Run fast tests only"
	@echo "  test-gpu             : Run GPU tests only"
	@echo "  run-local-jupyter    : Start Jupyter server locally"
	@echo "  conda-env            : Set up Conda environment using build_conda_env.sh"
	@echo "  help                 : Display this help message"
