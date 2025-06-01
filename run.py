import time
from src.data.preprocess_data import main as preprocess_main
from src.models.train_model import main as train_main
from src.models.evaluate_test import main as eval_main

if __name__ == "__main__":
    print("🚀 Launching pipeline...")
    
    stages = {
        "🛠 PREPROCESSING": preprocess_main,
        "🤖 TRAINING": train_main,
        "🧪 EVALUATION": eval_main
    }
    
    for name, stage in stages.items():
        start = time.time()
        print(f"\n{name} STARTED")
        stage()
        print(f"⏱ {name} COMPLETED in {time.time()-start:.1f}s")
    
    print("\n🎉 PIPELINE SUCCESS! ✅")