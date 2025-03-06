# Variables
PYTHON=python
SCRIPT=main.py
PIPELINE_SCRIPT=churn_model_pipeline.py
DOCKER_IMAGE_NAME=khaledbenahmed/mlops
DOCKER_TAG=latest
DOCKERFILE=Dockerfile
CONTAINER_NAME=mlops-container

# Default target
all: setup-dirs check-quality run-pipeline
	@echo "✅ All steps completed successfully!"

# Code Quality
lint:
	@echo "🔎 Running pylint..."
	-pylint $(SCRIPT) $(PIPELINE_SCRIPT) || true
flake:
	@echo "🔍 Running flake8..."
	-python -m flake8  $(SCRIPT) $(PIPELINE_SCRIPT) || true
type-check:
	@echo "🔎 Running mypy..."
	python -m mypy --ignore-missing-imports $(SCRIPT) $(PIPELINE_SCRIPT)

check-quality: lint flake type-check

# Code Formatting
format:
	@echo "🎨 Formatting code with black..."
	black $(SCRIPT) $(PIPELINE_SCRIPT)

# Data Pipeline
prepare-data:
	@echo "📦 Preparing data..."
	$(PYTHON) $(SCRIPT) --prepare
train-model:
	@echo "🤖 Training model..."
	$(PYTHON) $(SCRIPT) --train
grid-search-model:
	@echo "🔍 Running grid search for model..."
	$(PYTHON) $(SCRIPT) --grid-search
evaluate-model:
	@echo "📊 Evaluating model..."
	$(PYTHON) $(SCRIPT) --evaluate
predict:
	@echo "🔮 Making prediction..."
	$(PYTHON) $(SCRIPT) --predict

# Full Pipeline Execution
run-pipeline: prepare-data train-model evaluate-model predict
	@echo "✅ Complete pipeline execution finished!"

# Setup Environment
setup-dirs:
	@echo "📁 Creating required directories..."
	mkdir -p data models

# Docker Operations
build-image:
	@echo "🚀 Building Docker image..."
	docker build -t $(DOCKER_IMAGE_NAME):$(DOCKER_TAG) -f $(DOCKERFILE) .
        
run-container:
	@echo "🎮 Running Docker container..."
	docker run --name $(CONTAINER_NAME) -d $(DOCKER_IMAGE_NAME):$(DOCKER_TAG)
        
stop-container:
	@echo "🛑 Stopping Docker container..."
	docker stop $(CONTAINER_NAME) && docker rm $(CONTAINER_NAME)

push-image:
	@echo "📤 Pushing Docker image to registry..."
	docker push $(DOCKER_IMAGE_NAME):$(DOCKER_TAG)

pull-image:
	@echo "📥 Pulling Docker image from registry..."
	docker pull $(DOCKER_IMAGE_NAME):$(DOCKER_TAG)

clean:
	@echo "🧹 Cleaning up generated files..."
	rm -rf data models __pycache__

# Help
help:
	@echo "Churn Prediction Model Pipeline"
	@echo "------------------------------"
	@echo "Available commands:"
	@echo "  make all               - Setup directories, check code quality, and run the pipeline"
	@echo "  make prepare-data      - Prepare and preprocess the data"
	@echo "  make grid-search-model - Run grid search to find optimal hyperparameters"
	@echo "  make train-model       - Train the model with best parameters"
	@echo "  make evaluate-model    - Evaluate model performance"
	@echo "  make predict           - Make a prediction with the trained model"
	@echo "  make run-pipeline      - Run the complete pipeline"
	@echo "  make check-quality     - Run all code quality checks"
	@echo "  make format            - Format code with black"
	@echo "  make setup-dirs        - Create required directories"
	@echo "  make clean             - Clean up generated artifacts"
	@echo "  make build-image       - Build the Docker image"
	@echo "  make run-container     - Run the Docker container"
	@echo "  make stop-container    - Stop the Docker container"
	@echo "  make push-image        - Push the Docker image to the registry"
	@echo "  make pull-image        - Pull the Docker image from the registry"

.PHONY: all lint flake type-check check-quality format prepare-data train-model grid-search-model evaluate-model predict run-pipeline setup-dirs build-image run-container stop-container push-image pull-image clean help
