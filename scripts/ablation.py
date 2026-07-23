"""Run a small, reproducible Domain-SFT hyperparameter ablation."""

from __future__ import annotations

import os
import subprocess
import sys

from .config import ARTIFACTS_DIR, ROOT, load_json, save_json


ABLATION_CONFIGS = [
    {"name": "ablation_3ep_lr1e-4", "epochs": 3, "learning_rate": 1e-4},
    {"name": "ablation_5ep_lr1e-4", "epochs": 5, "learning_rate": 1e-4},
    {"name": "ablation_5ep_lr5e-5", "epochs": 5, "learning_rate": 5e-5},
]


def select_best_validation(results):
    """Select the run with the lowest validation loss observed."""
    return min(results, key=lambda name: min(results[name]["validation"]))


def render_markdown(results):
    best = select_best_validation(results)
    lines = [
        "# Domain SFT Ablation Results",
        "",
        "This report compares training epochs and learning rate while keeping the data split and seed fixed.",
        "",
        "| Run | Best validation loss | Epoch |",
        "| --- | ---: | ---: |",
    ]
    for name, metrics in results.items():
        values = metrics["validation"]
        best_value = min(values)
        best_epoch = values.index(best_value) + 1
        lines.append(f"| `{name}` | {best_value:.4f} | {best_epoch} |")
    lines.extend(["", f"Best observed run: `{best}`.", ""])
    return "\n".join(lines)


def main():
    results = {}
    base_env = os.environ.copy()
    for config in ABLATION_CONFIGS:
        env = base_env.copy()
        env["RUN_NAME"] = config["name"]
        env["DOMAIN_EPOCHS"] = str(config["epochs"])
        env["DOMAIN_LR"] = str(config["learning_rate"])
        print(f"running={config['name']}")
        subprocess.run(
            [sys.executable, "-m", "scripts.train_domain"],
            cwd=ROOT,
            env=env,
            check=True,
        )
        results[config["name"]] = load_json(
            ARTIFACTS_DIR / "metrics" / f"{config['name']}.json"
        )

    save_json(results, ROOT / "reports" / "ablation_results.json")
    (ROOT / "reports" / "ablation_results.md").write_text(
        render_markdown(results), encoding="utf-8"
    )
    print(f"best_run={select_best_validation(results)}")


if __name__ == "__main__":
    main()
