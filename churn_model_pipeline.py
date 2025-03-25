"""
Module for training and evaluating a customer churn prediction model with MLflow and Elasticsearch integration and email notifications.
"""

import os
import pandas as pd
import numpy as np
import joblib
import mlflow
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)
from imblearn.combine import SMOTEENN
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import warnings
import time
import psutil
import docker
from elasticsearch import Elasticsearch
import smtplib
from email.mime.text import MIMEText

warnings.filterwarnings("ignore", category=UserWarning)

# Static configuration instead of loading from YAML
data80_path = "data/churn-bigml-20.csv"
data20_path = "data/churn-bigml-20.csv"
data_path = "data"
model_path = "models"
random_state = 42
test_size = 0.2
corr_threshold = 0.85
param_grid = {
    "n_estimators": [50, 200, 500],
    "max_depth": [10, 20, 30, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "bootstrap": [True, False],
}
model = None
X = None
sample_data = {
    "Account length": 25, # Shorter account length, higher churn risk
    "Area code": 408,
    "Number vmail messages": 0, # No voicemail messages, consistent with 'No' plan
    "Total day minutes": 280, # High day minutes usage
    "Total day calls": 100,
    "Total eve minutes": 250, # High evening minutes usage
    "Total eve calls": 80,
    "Total night minutes": 220,
    "Total night calls": 70,
    "Total intl minutes": 10,
    "Total intl calls": 5,
    "Customer service calls": 4, # Increased customer service calls - strong churn indicator
    "International plan_Yes": 1, # Keeping International plan as 'Yes' for consistency, could be either way for churn
    "Region_Northeast": 0,
    "Region_South": 1,
    "Region_West": 0,
}
# MLflow setup
mlflow.set_registry_uri("http://localhost:5000")
experiment_name = "Churn_Prediction_Experiment"
mlflow.set_tracking_uri("http://localhost:5000")

# Elasticsearch setup
def connect_elasticsearch():
    try:
        es = Elasticsearch(["http://localhost:9200"], http_auth=("elastic", "changeme"))
        if es.ping():
            print("✅ Connected to Elasticsearch")
            return es
        else:
            print("❌ Elasticsearch ping failed")
            return None
    except Exception as e:
        print(f"❌ Elasticsearch connection failed: {e}")
        return None

def log_to_elasticsearch(es, index, data):
    data["timestamp"] = int(time.time())
    if es:
        try:
            es.index(index=index, body=data)
            print(f"Logged to {index}: {data}")
        except Exception as e:
            print(f"❌ Failed to log to Elasticsearch: {e}")
    else:
        print("❌ No Elasticsearch connection for logging")

def monitor_system_resources(es):
    print("🔍 Monitoring system resources...")
    cpu_usage = psutil.cpu_percent(interval=1)
    mem_usage = psutil.virtual_memory().percent
    disk_usage = psutil.disk_usage("/").percent
    metrics = {
        "action": "system_monitoring",
        "cpu_usage": cpu_usage,
        "mem_usage": mem_usage,
        "disk_usage": disk_usage,
    }
    mlflow.log_metrics(
        {"cpu_usage": cpu_usage, "mem_usage": mem_usage, "disk_usage": disk_usage}
    )
    log_to_elasticsearch(es, "mlflow-metrics", metrics)
    return metrics

def monitor_docker_containers(es):
    print("🔍 Monitoring Docker containers...")
    try:
        client = docker.from_env()
        containers = {"elasticsearch": None, "kibana": None, "mlflow": None} # Added mlflow container monitoring
        for container in client.containers.list():
            if container.name in containers:
                stats = container.stats(stream=False)
                cpu_usage = stats["cpu_stats"]["cpu_usage"]["total_usage"] / stats["cpu_stats"]["system_cpu_usage"] * 100 if stats["cpu_stats"]["system_cpu_usage"] > 0 else 0
                mem_usage = stats["memory_stats"]["usage"] / stats["memory_stats"]["limit"] * 100 if stats["memory_stats"]["limit"] > 0 else 0
                metrics = {
                    "action": "docker_monitoring",
                    "container_name": container.name,
                    "cpu_usage": cpu_usage,
                    "mem_usage": mem_usage,
                }
                mlflow.log_metrics(
                    {
                        f"{container.name}_cpu_usage": cpu_usage,
                        f"{container.name}_mem_usage": mem_usage,
                    }
                )
                log_to_elasticsearch(es, "mlflow-metrics", metrics)
    except Exception as e:
        print(f"❌ Docker monitoring failed: {e}")

def monitor_data_drift(X_train, X_test, es):
    print("🔍 Monitoring data drift...")
    train_mean = np.mean(X_train, axis=0)
    test_mean = np.mean(X_test, axis=0)
    drift_score = np.mean(np.abs(train_mean - test_mean))
    metrics = {"action": "data_drift", "drift_score": drift_score}
    mlflow.log_metric("data_drift", drift_score)
    log_to_elasticsearch(es, "mlflow-metrics", metrics)
    return drift_score

# Email Notification Setup
def send_email_notification(subject, body, is_html=False):
    """Sends an email notification with support for HTML formatting."""
    sender_email = ""  # Replace with your sender email
    sender_password = ""  # Replace with your sender email password or app password
    receiver_email = ""  # Replace with your receiver email
    smtp_server = "smtp.gmail.com"  # Replace with your SMTP server, e.g., 'smtp.gmail.com' for Gmail
    smtp_port = 587  # Replace with your SMTP port, e.g., 587 for TLS

    msg = MIMEText(body, 'html' if is_html else 'plain') # Set subtype to 'html' if is_html is True
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = receiver_email

    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()  # Upgrade connection to secure TLS
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        print("📧 Email notification sent successfully!")
    except Exception as e:
        print(f"❌ Email notification failed: {e}")
        print(e)
    finally:
        if server:
            server.quit()


def load_raw_data() -> pd.DataFrame:
    """Load and concatenate raw datasets specified in the configuration."""
    data_80 = pd.read_csv(data80_path)
    data_20 = pd.read_csv(data20_path)
    data = pd.concat([data_80, data_20], axis=0, ignore_index=True)
    print(f"Loaded data shape: {data.shape}")
    return data

def handle_outliers(data: pd.DataFrame) -> pd.DataFrame:
    """Replace outliers with the median in numeric columns."""
    numeric_data = data.select_dtypes(include=["float64", "int64"])
    q1 = numeric_data.quantile(0.25)
    q3 = numeric_data.quantile(0.75)
    iqr = q3 - q1

    for column in numeric_data.columns:
        outliers = (numeric_data[column] < (q1[column] - 1.5 * iqr[column])) | (
            numeric_data[column] > (q3[column] + 1.5 * iqr[column])
        )
        numeric_data.loc[outliers, column] = numeric_data[column].median()

    non_numeric_data = data.drop(columns=numeric_data.columns)
    combined_data = pd.concat([numeric_data, non_numeric_data], axis=1)
    return combined_data

def map_states_to_regions(data: pd.DataFrame) -> pd.DataFrame:
    """Map U.S. state abbreviations to regions."""
    region_mapping = {
        "CT": "Northeast",
        "ME": "Northeast",
        "MA": "Northeast",
        "NH": "Northeast",
        "RI": "Northeast",
        "VT": "Northeast",
        "NJ": "Northeast",
        "NY": "Northeast",
        "PA": "Northeast",
        "IL": "Midwest",
        "IN": "Midwest",
        "IA": "Midwest",
        "KS": "Midwest",
        "MI": "Midwest",
        "MN": "Midwest",
        "MO": "Midwest",
        "NE": "Midwest",
        "ND": "Midwest",
        "OH": "Midwest",
        "SD": "Midwest",
        "WI": "Midwest",
        "AL": "South",
        "AR": "South",
        "DE": "South",
        "FL": "South",
        "GA": "South",
        "KY": "South",
        "LA": "South",
        "MD": "South",
        "MS": "South",
        "NC": "South",
        "OK": "South",
        "SC": "South",
        "TN": "South",
        "TX": "South",
        "VA": "South",
        "WV": "South",
        "DC": "South",
        "AK": "West",
        "AZ": "West",
        "CA": "West",
        "CO": "West",
        "HI": "West",
        "ID": "West",
        "MT": "West",
        "NV": "West",
        "NM": "West",
        "OR": "West",
        "UT": "West",
        "WA": "West",
        "WY": "West",
    }
    data["Region"] = data["State"].map(region_mapping)
    data.drop(columns="State", inplace=True)
    return data

def drop_highly_correlated_features(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Drop features with correlation above the threshold."""
    corr_matrix = df.corr().abs()
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    to_drop = [
        col
        for col in upper_triangle.columns
        if any(upper_triangle[col] > threshold)
    ]

    if to_drop:
        print("Dropped features due to high correlation:", to_drop)
    else:
        print("No features dropped based on the correlation threshold.")

    return df.drop(columns=to_drop)

def apply_oversampling(X: pd.DataFrame, y: pd.Series) -> tuple:
    """Apply SMOTEENN to handle class imbalance."""
    smote_enn = SMOTEENN(random_state=random_state)
    X_resampled, y_resampled = smote_enn.fit_resample(X, y)

    print(f"Original dataset shape: {X.shape, y.shape}")
    print(f"Resampled dataset shape: {X_resampled.shape, y_resampled.shape}")

    joblib.dump(X_resampled, f"{data_path}/X_train_res.joblib")
    joblib.dump(y_resampled, f"{data_path}/y_train_res.joblib")
    return X_resampled, y_resampled

def encoding(data: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical variables using one-hot encoding."""
    return pd.get_dummies(data, drop_first=True).astype(int)

def prepare_data() -> None:
    """Prepare and preprocess the data for training."""
    es = connect_elasticsearch() # Connect to Elasticsearch

    data = load_raw_data()
    data = handle_outliers(data)
    data = map_states_to_regions(data)
    data = encoding(data)

    features_df = drop_highly_correlated_features(
        data.drop(columns=["Churn"]), corr_threshold
    )
    X = features_df
    y = data["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    monitor_data_drift(X_train, X_test, es) # Monitor data drift after split
    X_train_res, y_train_res = apply_oversampling(X_train, y_train)
    joblib.dump(X_test, f"{data_path}/X_test.joblib")
    joblib.dump(y_test, f"{data_path}/y_test.joblib")

    # Save feature names for model signature
    joblib.dump(list(X.columns), f"{data_path}/feature_names.joblib")

    print("✅ Data prepared and saved.")
    send_email_notification(subject="Data Preparation Completed", body="Data preparation step finished successfully.")


def perform_grid_search() -> None:
    """Perform grid search to find the best hyperparameters."""
    es = connect_elasticsearch() # Connect to Elasticsearch
    monitor_system_resources(es) # Monitor system resources before grid search
    monitor_docker_containers(es) # Monitor docker containers before grid search

    X_train = joblib.load(f"{data_path}/X_train_res.joblib")
    y_train = joblib.load(f"{data_path}/y_train_res.joblib")

    # Create MLflow experiment if it doesn't exist
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)

    # Initialize base model
    base_model = ExtraTreesClassifier(random_state=random_state)

    # Setup grid search
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        verbose=1,
        scoring="f1",
    )

    # Perform grid search with MLflow tracking
    with mlflow.start_run(run_name="grid_search") as run:
        mlflow.log_params({"grid_search": str(param_grid)})
        log_to_elasticsearch(es, "mlflow-metrics", {"action": "grid_search_params", "params": str(param_grid)}) # Log grid search params to ES

        grid_search.fit(X_train, y_train)

        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best F1 score: {grid_search.best_score_:.4f}")

        # Log best parameters
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("best_cv_f1", grid_search.best_score_)
        log_to_elasticsearch(es, "mlflow-metrics", {"action": "grid_search_best_params", "best_params": grid_search.best_params_, "best_cv_f1": grid_search.best_score_}) # Log best params and score to ES

        # Save the best model
        model = grid_search.best_estimator_
        joblib.dump(model, f"{model_path}/model.joblib")
        joblib.dump(
            grid_search.best_params_, f"{model_path}/best_params.joblib"
        )

        # Log feature importances
        feature_names = joblib.load(f"{data_path}/feature_names.joblib")
        feature_importance = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        # Create feature importance plot
        plt.figure(figsize=(10, 6))
        plt.barh(
            feature_importance["feature"][:10],
            feature_importance["importance"][:10],
        )
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.title("Top 10 Feature Importances")
        plt.tight_layout()
        importance_path = f"{model_path}/feature_importance.png"
        plt.savefig(importance_path)

        # Log the feature importance plot
        mlflow.log_artifact(importance_path)

        print("✅ Grid search completed and best model saved.")
        send_email_notification(subject="Grid Search Completed", body=f"Grid search finished. Best parameters: {grid_search.best_params_}, Best F1 score: {grid_search.best_score_:.4f}")


    monitor_system_resources(es) # Monitor system resources after grid search
    monitor_docker_containers(es) # Monitor docker containers after grid search


def train_model() -> None:
    """Train the model with the best parameters from grid search or default if no grid search was performed."""
    es = connect_elasticsearch() # Connect to Elasticsearch
    monitor_system_resources(es) # Monitor system resources before training
    monitor_docker_containers(es) # Monitor docker containers before training

    print("here")
    X = joblib.load(f"{data_path}/X_train_res.joblib")
    y_train = joblib.load(f"{data_path}/y_train_res.joblib")

    # Create MLflow experiment if it doesn't exist
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)

    # Check if there is an active run and end it to avoid errors
    if mlflow.active_run():
        mlflow.end_run()

    # Initialize and train model
    with mlflow.start_run(run_name="model_training") as run:
        # Log training parameters
        mlflow.log_param("random_state", random_state)
        log_to_elasticsearch(es, "mlflow-metrics", {"action": "training_params", "random_state": random_state}) # Log training params to ES
        # Create model with best parameters
        model = ExtraTreesClassifier(n_estimators=50, n_jobs=4, random_state=random_state)
        model.fit(X, y_train)

        # Save the model
        joblib.dump(model, f"{model_path}/model.joblib")

        # Log model to MLflow
        X_test = joblib.load(f"{data_path}/X_test.joblib")
        signature = infer_signature(X_test, model.predict(X_test))

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            registered_model_name="churn_prediction_model",
        )
        log_to_elasticsearch(es, "mlflow-metrics", {"action": "model_registered", "model_name": "churn_prediction_model"}) # Log model registration to ES

        print("✅ Model trained, saved locally, and registered with MLflow.")
        send_email_notification(subject="Model Training Completed", body="Model training finished and model registered in MLflow.")


    monitor_system_resources(es) # Monitor system resources after training
    monitor_docker_containers(es) # Monitor docker containers after training


def load_model() -> object:
    """Load the trained model from the specified path."""
    model = joblib.load(f"{model_path}/model.joblib")
    print("✅ Model loaded successfully.")
    return model

def retrain_model():
    """Retrain the model without using MLflow."""
    print("🔄 Retraining model ...")

    # Load training data
    X_train = joblib.load(f"{data_path}/X_train_res.joblib")
    y_train = joblib.load(f"{data_path}/y_train_res.joblib")

    # Train a new model
    model = ExtraTreesClassifier(n_estimators=50, n_jobs=4, random_state=random_state)
    model.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(model, f"{model_path}/model_retrained.joblib")
    print("✅ Model retrained and saved successfully!")
    send_email_notification(subject="Model Retraining Completed", body="Model retraining finished successfully.")
    return model

def evaluate_model() -> None:
    """Evaluate the model and log metrics to MLflow."""
    es = connect_elasticsearch() # Connect to Elasticsearch
    monitor_system_resources(es) # Monitor system resources before evaluation
    monitor_docker_containers(es) # Monitor docker containers before evaluation

    model = joblib.load(f"{model_path}/model.joblib")
    X_test = joblib.load(f"{data_path}/X_test.joblib")
    y_test = joblib.load(f"{data_path}/y_test.joblib")

    # Create MLflow experiment if it doesn't exist
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)

    # Check if there is an active run and end it to avoid errors
    if mlflow.active_run():
        mlflow.end_run()

    with mlflow.start_run(run_name="model_evaluation") as run:
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        metrics = {
            "test_accuracy": accuracy_score(y_test, y_pred),
            "test_precision": precision_score(y_test, y_pred),
            "test_recall": recall_score(y_test, y_pred),
            "test_f1": f1_score(y_test, y_pred),
            "test_roc_auc": roc_auc_score(y_test, y_pred),
        }

        # Log metrics to MLflow
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        log_to_elasticsearch(es, "mlflow-metrics", {"action": "evaluation_metrics", "metrics_raw": metrics}) # Log metrics to ES in raw format as well for potential Kibana dashboarding

        # Format metrics for styled email body
        metrics_html = "".join([f"<tr><td><b>{metric_name}</b></td><td>{metric_value:.4f}</td></tr>" for metric_name, metric_value in metrics.items()])


        print("📊 Evaluation metrics:", metrics)

        # Create and save confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        cm_display = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=["Not Churn", "Churn"]
        )
        plt.figure(figsize=(8, 6))
        cm_display.plot(cmap="Blues")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        cm_path = f"{model_path}/confusion_matrix.png"
        plt.savefig(cm_path)

        # Log confusion matrix to MLflow
        mlflow.log_artifact(cm_path)

        # Create and save ROC curve
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f'ROC (AUC = {metrics["test_roc_auc"]:.4f})')
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        roc_path = f"{model_path}/roc_curve.png"
        plt.savefig(roc_path)

        # Log ROC curve to MLflow
        mlflow.log_artifact(roc_path)

        # Create HTML body for email
        evaluation_email_body = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Evaluation Completed</title>
        </head>
        <body>
            <h1>Model Evaluation Completed</h1>
            <p>The churn prediction model evaluation has finished. Here are the key metrics:</p>
            <table border="1">
                <thead>
                    <tr><th>Metric</th><th>Value</th></tr>
                </thead>
                <tbody>
                    {metrics_html}
                </tbody>
            </table>
            <p>Evaluation artifacts (confusion matrix, ROC curve) have been saved and logged to MLflow.</p>
        </body>
        </html>
        """


        print("📊 Evaluation artifacts saved and logged to MLflow.")
        send_email_notification(
            subject="Model Evaluation Completed", body=evaluation_email_body, is_html=True
        )


    monitor_system_resources(es) # Monitor system resources after evaluation
    monitor_docker_containers(es) # Monitor docker containers after evaluation


def predict() -> None:
    """Make a prediction using the trained model."""
    es = connect_elasticsearch() # Connect to Elasticsearch
    monitor_system_resources(es) # Monitor system resources before prediction
    monitor_docker_containers(es) # Monitor docker containers before prediction

    model = joblib.load(f"{model_path}/model.joblib")
    sample_data_df = pd.DataFrame([sample_data])

    # Create MLflow experiment if it doesn't exist
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)

    # Check if there is an active run and end it to avoid errors
    if mlflow.active_run():
        mlflow.end_run()

    with mlflow.start_run(run_name="prediction_demo") as run:
        prediction = model.predict(sample_data_df)
        prediction_prob = model.predict_proba(sample_data_df)[:, 1]

        # Log inputs and outputs
        mlflow.log_param("sample_input", str(sample_data))
        mlflow.log_metric("prediction", int(prediction[0]))
        mlflow.log_metric("prediction_probability", float(prediction_prob[0]))
        log_to_elasticsearch(es, "mlflow-metrics", {"action": "prediction", "sample_input": str(sample_data), "prediction": int(prediction[0]), "prediction_probability": float(prediction_prob[0])}) # Log prediction info to ES


        result = "Churn" if prediction[0] else "Not Churn"

        # Format sample data for HTML email
        sample_data_html = "".join([f"<tr><td><b>{key}</b></td><td>{value}</td></tr>" for key, value in sample_data.items()])

        # Create HTML body for prediction email
        prediction_email_body = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Prediction Completed</title>
        </head>
        <body>
            <h1>Prediction Completed</h1>
            <p>Here is the churn prediction for the provided sample data:</p>
            <table border="1">
                <thead>
                    <tr><th>Feature</th><th>Value</th></tr>
                </thead>
                <tbody>
                    {sample_data_html}
                </tbody>
            </table>
            <h2>Prediction Result: {result}</h2>
            <p><b>Probability of Churn:</b> {prediction_prob[0]:.2f}</p>
        </body>
        </html>
        """


        print(f"🔮 Prediction: {result} (Probability: {prediction_prob[0]:.2f})")
        send_email_notification(
            subject="Prediction Completed", body=prediction_email_body, is_html=True
        )
