"""CLI: python scripts/train_model.py [path_to_dataset]"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as C
from src.data_loader import DatasetError, find_default_dataset, load_dataset
from src.train import save_artifacts, train_and_evaluate


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else find_default_dataset()
    if not path or not path.exists():
        print("No dataset found. Place the UCI/Kaggle file in ./data/ (see README) or pass a path.")
        sys.exit(1)
    try:
        df, rep = load_dataset(path, require_target=True)
    except DatasetError as e:
        print(f"Dataset error: {e}"); sys.exit(2)
    print(f"Loaded {rep['rows_out']:,} rows from {path.name}; default rate {rep['default_rate']:.3f}")
    res = train_and_evaluate(df, progress=print, save=False)
    save_artifacts(res, report_extra=rep)
    m = res["models"][res["final_model"]]
    print(f"\nFinal model: {res['final_model']}")
    print(f"Test ROC-AUC {m['test']['roc_auc']:.3f} | PR-AUC {m['test']['pr_auc']:.3f} | F1 {m['test']['f1']:.3f}")
    tk = m["topk_500"]; print(f"Test Top-{tk['k_used']}: precision {tk['precision_at_k']:.3f}, recall {tk['recall_at_k']:.3f}, lift {tk['lift_at_k']:.2f}")
    o = res["oof_topk"]; print(f"Population Top-{o['k_used']} (out-of-fold, n={res['n_total']:,}): precision {o['precision_at_k']:.3f}, recall {o['recall_at_k']:.3f}, lift {o['lift_at_k']:.2f}")
    print(f"Ablation ROC-AUC without/with demographics: {res['ablation']['test_roc_auc_without_demographics']:.3f} / {res['ablation']['test_roc_auc_with_demographics']:.3f}")
    ml = res["mlflow"]; print("MLflow:", "logged " + str(len(ml['runs'])) + " runs" if ml["enabled"] else ml["warning"])
    print("Artifacts written to ./artifacts/")


if __name__ == "__main__":
    main()
