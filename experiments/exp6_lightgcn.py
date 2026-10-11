"""Suggestion 6: LightGCN (He et al. 2020) on the user-item graph, BPR loss, CPU.
usage: python experiments/exp6_lightgcn.py <val|random|temporal> <all|pos> [max_epochs] [eval_every]
On this CPU one epoch took ~100 s, so a full run takes many hours."""
import sys
import time

import numpy as np
import torch
from exp_common import *

torch.manual_seed(0)
torch.set_num_threads(max(1, torch.get_num_threads()))


def norm_adj(X):
    """Symmetric-normalised bipartite adjacency D^-1/2 A D^-1/2 as a torch sparse tensor."""
    X = X.tocoo()
    nu, ni = X.shape
    du = np.asarray(X.sum(axis=1)).ravel()
    di = np.asarray(X.sum(axis=0)).ravel()
    w = 1.0 / np.sqrt(np.maximum(du[X.row], 1) * np.maximum(di[X.col], 1))
    rows = np.concatenate([X.row, X.col + nu])
    cols = np.concatenate([X.col + nu, X.row])
    vals = np.concatenate([w, w]).astype(np.float32)
    return torch.sparse_coo_tensor(np.vstack([rows, cols]), vals, (nu + ni, nu + ni)).coalesce()


class LightGCN(torch.nn.Module):
    def __init__(self, n_users, n_items, A, dim=64, layers=3):
        super().__init__()
        self.nu, self.A, self.layers = n_users, A, layers
        self.emb = torch.nn.Embedding(n_users + n_items, dim)
        torch.nn.init.normal_(self.emb.weight, std=0.1)

    def propagate(self):
        e = self.emb.weight
        out = [e]
        for _ in range(self.layers):
            e = torch.sparse.mm(self.A, e)
            out.append(e)
        e = torch.stack(out).mean(0)
        return e[:self.nu], e[self.nu:]


def positives(R, mode):
    X = R.copy()
    if mode == "pos":
        X.data = (X.data >= 4).astype(np.float32)
        X.eliminate_zeros()
    else:
        X.data[:] = 1
    return X.tocsr()


def train_lightgcn(R, mode, eval_fn=None, max_epochs=300, batch=8192, lr=5e-3, reg=1e-4, dim=64, layers=3,
                   patience=4, every=10):
    X = positives(R, mode)
    nu, ni = X.shape
    model = LightGCN(nu, ni, norm_adj(X), dim, layers)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    coo = X.tocoo()
    pu, pi = coo.row.astype(np.int64), coo.col.astype(np.int64)
    rng = np.random.default_rng(0)
    best, best_state, bad = -1, None, 0
    for ep in range(1, max_epochs + 1):
        t = time.time()
        perm = rng.permutation(len(pu))
        tot = 0.0
        for s in range(0, len(perm), batch):
            idx = perm[s:s + batch]
            u, i = torch.from_numpy(pu[idx]), torch.from_numpy(pi[idx])
            j = torch.from_numpy(rng.integers(0, ni, len(idx)))  # uniform negatives
            eu, ei = model.propagate()
            x = (eu[u] * (ei[i] - ei[j])).sum(1)
            e0 = model.emb.weight
            l2 = (e0[u].pow(2).sum() + e0[nu + i].pow(2).sum() + e0[nu + j].pow(2).sum()) / len(idx)
            loss = -torch.nn.functional.logsigmoid(x).mean() + reg * l2
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
        if eval_fn and ep % every == 0:
            with torch.no_grad():
                U, V = (t_.numpy() for t_ in model.propagate())
            score = eval_fn(U, V)
            print(f"epoch {ep} loss {tot:.2f} val ndcg@10 {score:.4f} ({time.time() - t:.1f}s/epoch)", flush=True)
            if score > best:
                best, best_state, bad = score, {k: v.clone() for k, v in model.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    with torch.no_grad():
        U, V = (t_.numpy() for t_ in model.propagate())
    return U, V, best


if __name__ == "__main__":
    which, mode = sys.argv[1], sys.argv[2]
    max_epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    every = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    tr, te = split(which)
    R, _ = matrices(tr)
    _, uidx_s = sample_users(1500)
    rel, uidx = rel_and_users(te, uidx_s if which == "val" else None)
    fn = lambda U, V: evaluate(lambda b: U[b] @ V.T, R, rel, uidx)["ndcg@10"]
    U, V, best = train_lightgcn(R, mode, fn, max_epochs, every=every)
    np.savez(RESULTS / f"lightgcn_{which}_{mode}.npz", U=U, V=V)
    print("best", best)
