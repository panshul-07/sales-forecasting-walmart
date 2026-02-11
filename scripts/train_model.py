import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.services.data import load_walmart_data
from backend.app.services.features import make_features
from backend.app.services.modeling import train_pipeline


def main():
    root = ROOT
    raw = load_walmart_data(root)
    feat = make_features(raw)
    artifacts = train_pipeline(feat, root / "artifacts")
    print("Training complete")
    print("Metrics:")
    for k, v in artifacts.metrics.items():
        print(f"- {k}: {v:.6f}")
    print("Key diagnostics:")
    print(f"- jarque_bera_p: {artifacts.diagnostics['normality']['jarque_bera_p']:.6f}")
    print(f"- residual_kurtosis: {artifacts.diagnostics['normality']['residual_kurtosis']:.6f}")
    print(f"- breusch_pagan_p: {artifacts.diagnostics['heteroskedasticity']['breusch_pagan_p']:.6f}")


if __name__ == "__main__":
    main()
