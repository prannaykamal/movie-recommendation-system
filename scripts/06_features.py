"""Phase 8a: reranker features for each stage's candidates.

Training stages (s0, s1) keep users with a liked movie among their candidates
(no others can teach the reranker anything). Evaluation stages (s2, s3) keep
users with a liked movie in the target block (only they are scored).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from hybridrec.cli import log, parser, timed
from hybridrec.config import get_config
from hybridrec.data.dataset import Dataset
from hybridrec.eval.protocol import stage_data
from hybridrec.rerank.features import ItemTable, stage_features
from hybridrec.retrieval.registry import text_vectors


def main() -> None:
    args = parser(__doc__, stage=True).parse_args()
    cfg = get_config(args.dataset)
    ds = Dataset.load(cfg)
    it = ItemTable.build(ds, text_vectors(ds))
    for stage in cfg.stages if args.stage == "all" else (args.stage,):
        sdir = cfg.stage_dir(stage)
        cands = pd.read_parquet(sdir / "candidates.parquet")
        if stage in ("s0", "s1"):
            keep = cands.groupby("user_id")["label"].transform("max") > 0
        else:
            keep = cands["user_id"].isin(set(stage_data(ds, stage).truth))
        cands = cands[keep]
        tt_path = sdir / "twotower_items.npy"
        tt = np.load(tt_path) if tt_path.exists() else None
        with timed(f"[{stage}] features for {cands.user_id.nunique()} users, {len(cands)} rows"):
            feats, users, seq = stage_features(ds, stage, cands, it, tt)
        feats.to_parquet(sdir / "features.parquet", index=False)
        np.savez(sdir / "history.npz", users=users, seq=seq)
        log.info("[%s] positive rate %.4f", stage, feats["label"].mean())


if __name__ == "__main__":
    main()
