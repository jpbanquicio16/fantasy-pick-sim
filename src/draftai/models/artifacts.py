"""Versioned model artifacts: LightGBM native text files, ridge pipelines via joblib,
and a JSON manifest (inputs, versions, blend weights, conformal corrections)."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import date
from pathlib import Path

import joblib
import lightgbm as lgb
import sklearn

from draftai.config import REPO_ROOT, TrainConfig
from draftai.models.components import BlendModel, ProjectionModel, target_specs
from draftai.models.quantiles import QuantileModel
from draftai.models.train import Models


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "nogit"


def config_hash(cfg: TrainConfig) -> str:
    return hashlib.sha256(cfg.model_dump_json().encode()).hexdigest()[:8]


def artifact_name(kind: str, snapshot: date, cfg: TrainConfig) -> str:
    return f"{kind}_{snapshot.isoformat()}_{_git_commit()}_{config_hash(cfg)}"


def save_models(models: Models, directory: Path, cfg: TrainConfig, snapshot: date, extra: dict | None = None) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    pm = models.projection
    for name, m in pm.models.items():
        booster = m.lgbm.booster_ if hasattr(m.lgbm, "booster_") else m.lgbm
        booster.save_model(str(directory / f"proj_{name}.txt"))
        joblib.dump(m.ridge, directory / f"proj_{name}_ridge.joblib")
    for label, qm in (("fp_q", models.fp_q), ("share_q", models.share_q)):
        for q, m in qm.models.items():
            booster = m.booster_ if hasattr(m, "booster_") else m
            booster.save_model(str(directory / f"{label}_{q}.txt"))
    manifest = {
        "snapshot_date": snapshot.isoformat(),
        "git_commit": _git_commit(),
        "config_hash": config_hash(cfg),
        "train_config": json.loads(cfg.model_dump_json()),
        "versions": {"lightgbm": lgb.__version__, "sklearn": sklearn.__version__},
        "blends": pm.blends,
        "minutes_budget": models.minutes_budget,
        "fp_q": {"target": models.fp_q.target, "features": models.fp_q.features, "quantiles": list(models.fp_q.quantiles),
                 "adj": [models.fp_q.adj_lo, models.fp_q.adj_hi], "min_gp": models.fp_q.min_gp, "anchor": models.fp_q.anchor,
                 "clip": list(models.fp_q.clip)},
        "share_q": {"target": models.share_q.target, "features": models.share_q.features,
                    "quantiles": list(models.share_q.quantiles), "adj": [models.share_q.adj_lo, models.share_q.adj_hi],
                    "min_gp": models.share_q.min_gp, "anchor": models.share_q.anchor, "clip": list(models.share_q.clip)},
        **(extra or {}),
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return directory


def load_models(directory: Path) -> Models:
    man = json.loads((directory / "manifest.json").read_text())
    cfg = TrainConfig(**man["train_config"])
    pm = ProjectionModel(cfg.lgbm, cfg.ridge_alpha, cfg.recency_half_life_seasons, cfg.seed, blends=man["blends"])
    for spec in target_specs():
        m = BlendModel(spec, cfg.lgbm, cfg.ridge_alpha, cfg.recency_half_life_seasons, cfg.seed,
                       blend=man["blends"].get(spec.name, 0.5))
        m.lgbm = lgb.Booster(model_file=str(directory / f"proj_{spec.name}.txt"))
        m.ridge = joblib.load(directory / f"proj_{spec.name}_ridge.joblib")
        pm.models[spec.name] = m
    qms = {}
    for label in ("fp_q", "share_q"):
        info = man[label]
        qm = QuantileModel(info["target"], info["features"], cfg.lgbm, cfg.recency_half_life_seasons, cfg.seed,
                           tuple(info["quantiles"]), min_gp=info["min_gp"], clip=tuple(info["clip"]),
                           anchor=info.get("anchor"))
        qm.adj_lo, qm.adj_hi = info["adj"]
        for q in qm.quantiles:
            qm.models[q] = lgb.Booster(model_file=str(directory / f"{label}_{q}.txt"))
        qms[label] = qm
    # Models saved before the team minutes budget existed project without it.
    return Models(projection=pm, fp_q=qms["fp_q"], share_q=qms["share_q"],
                  minutes_budget=float(man.get("minutes_budget", 0.0)))
