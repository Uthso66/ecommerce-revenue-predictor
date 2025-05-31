# run.py
from src.data.preprocess_data import main as preprocess_main
from src.models.train_model import main as train_main
from src.models.evaluate_test import main as eval_main

if __name__ == "__main__":
    print("🚀 Running full pipeline...")
    preprocess_main()
    train_main()
    eval_main()
    print("✅ Pipeline completed.")
