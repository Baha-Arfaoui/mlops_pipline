# MLOps Project

## Overview
This project focuses on implementing MLOps principles by automating and deploying a Machine Learning (ML) model using modern tools and best practices. The workflow includes modularization, automation, deployment, containerization, and monitoring.

## Project Structure
The project is structured as follows:
1. **Modularization**: Refactoring an ML model from a Jupyter Notebook into reusable and independent modules.
2. **Automation (CI/CD)**: Implementing automated workflows using a Makefile and tools such as Pylint, Flake8, MyPy, SonarQube, and Black to ensure code quality, security, and formatting.
3. **MLflow Integration**: Tracking experiments, managing model versions, and ensuring reproducibility using MLflow.
4. **Model Deployment**: Exposing the trained model as a REST API using FastAPI, with JSON-based documentation.
5. **Containerization**: Using Docker to containerize the API, publishing the image to DockerHub, and deploying the application.
6. **Monitoring**: Continuously tracking model performance using MLflow, Elasticsearch, and visualizing metrics with Kibana.

## Setup Instructions
### Prerequisites
- Python 3.x
- Docker & DockerHub account
- MLflow
- FastAPI
- Elasticsearch & Kibana
- Makefile (for CI/CD automation)

### Installation
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <project_directory>
   ```
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```
3. Run the CI/CD pipeline:
   ```bash
   make all
   ```
4. Train and track models with MLflow:
   ```bash
   python train.py
   ```
5. Deploy the model using FastAPI:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```
6. Build and deploy the Docker container:
   ```bash
   docker build -t <your_dockerhub_username>/mlops_project .
   docker push <your_dockerhub_username>/mlops_project
   docker run -p 8000:8000 <your_dockerhub_username>/mlops_project
   ```
7. Enable monitoring:
   - Start Elasticsearch and Kibana
   - Configure MLflow tracking for model monitoring

## Excellence Features
To enhance the project, consider implementing:
- **CI/CD with Jenkins or GitHub Actions**
- **Unit and functional tests**
- **WebSocket-based deployment**
- **Multi-container architecture with Docker Compose**
- **System monitoring (CPU, RAM, disk space, etc.)**

## Contributors
- [Your Name]
- [Other Contributors]

## License
This project is licensed under [Your Preferred License].

