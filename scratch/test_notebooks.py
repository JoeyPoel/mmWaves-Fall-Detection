import json
import traceback
from pathlib import Path

notebooks = [
    "notebooks/00_train_val_test_split.ipynb",
    "notebooks/01_dataset_explorer.ipynb",
    "notebooks/02_data_preparation.ipynb",
    "notebooks/03_train_and_evaluate_models.ipynb",
    "notebooks/04_hyperparameter_tuning_and_ablation.ipynb"
]

print("=== Starting Notebook Execution Verification ===")

for nb_path in notebooks:
    print(f"\n--- Testing: {nb_path} ---", flush=True)
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    global_env = {}
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        code = "".join(cell["source"])
        clean_lines = [line for line in code.split("\n") if not line.strip().startswith("%")]
        clean_code = "\n".join(clean_lines)
        
        try:
            exec(clean_code, global_env)
            print(f"  [PASS] Cell {i}", flush=True)
        except Exception as e:
            print(f"  [FAIL] Cell {i} in {nb_path}", flush=True)
            print(f"Error: {e}")
            traceback.print_exc()
            raise e

print("\n=== ALL 5 NOTEBOOKS PASSED VERIFICATION WITH 0 ERRORS! ===", flush=True)
