"""
Main entry point for the Churn Prediction model pipeline.
"""
import argparse
import os
# Import all functions directly from the module
import churn_model_pipeline as cmp

def main():
    """Parse command-line arguments and execute the pipeline."""
    parser = argparse.ArgumentParser(description="Customer Churn Prediction Pipeline")
    parser.add_argument("--prepare", action="store_true", help="Prepare data")
    parser.add_argument(
        "--grid-search", action="store_true", help="Perform grid search"
    )
    parser.add_argument("--train", action="store_true", help="Train model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate model")
    parser.add_argument("--predict", action="store_true", help="Make prediction")
    parser.add_argument("--load", action="store_true", help="Load model")
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Call functions directly from the module instead of using a class instance
    if args.prepare:
        cmp.prepare_data()
    if args.grid_search:
        cmp.perform_grid_search()
    if args.train:
        cmp.train_model()
    if args.evaluate:
        cmp.evaluate_model()
    if args.predict:
        cmp.predict()
    if args.load:
        cmp.load_model()
    
    # If no arguments were provided, display help
    if not any(vars(args).values()):
        parser.print_help()

if __name__ == "__main__":
    main()
