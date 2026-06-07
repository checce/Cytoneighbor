import os, io, math, json, time, gc, zipfile, pickle, hashlib

from typing import Optional, Dict, Any, List, Tuple, Union

from collections import Counter, defaultdict, deque



import numpy as np

import pandas as pd

import scipy.sparse as sp

import anndata as ad



import torch

import torch.nn as nn

import torch.optim as optim



from sklearn.metrics import roc_auc_score



import faiss

import umap

from tqdm.auto import tqdm



_ENV_KEYS = ['batch']





# ---------------- Utils ----------------

def _log(msg): print(msg, flush=True)

EPS = 1e-8



def _as_numpy_dense(X):

    if sp.issparse(X): return X.toarray()

    return np.asarray(X)



def _feature_hash(feats: List[str]) -> str:

    s = '\n'.join(map(str, feats))

    return hashlib.sha1(s.encode('utf-8')).hexdigest()



def _set_seed(seed: Optional[int], use_cuda: bool, deterministic: bool=False):

    if seed is None: return

    seed = int(seed)

    np.random.seed(seed); torch.manual_seed(seed)

    if use_cuda and torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    if deterministic:

        torch.backends.cudnn.deterministic = True

        torch.backends.cudnn.benchmark = False



class Timer:

    def __init__(self, msg): self.msg=msg; self.t0=None

    def __enter__(self): _log(self.msg); self.t0=time.time()

    def __exit__(self, et, ev, tb): _log(f"[done in {time.time()-self.t0:.2f}s]")



# ---------- AMP ----------

def _autocast_cuda(enabled: bool, amp_dtype: str = "auto"):

    if not enabled:

        try:

            from torch.amp import autocast as amp_autocast

            return amp_autocast('cuda', enabled=False)

        except Exception:

            return torch.cuda.amp.autocast(enabled=False)

    use_bf16 = False

    if amp_dtype == "bf16": use_bf16 = True

    elif amp_dtype == "auto":

        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    try:

        from torch.amp import autocast as amp_autocast

        return amp_autocast('cuda', dtype=(torch.bfloat16 if use_bf16 else torch.float16))

    except Exception:

        return torch.cuda.amp.autocast(dtype=(torch.bfloat16 if use_bf16 else torch.float16))



def _grad_scaler(enabled: bool, amp_dtype: str = "auto"):

    use_bf16 = (amp_dtype == "bf16") or (amp_dtype == "auto" and torch.cuda.is_available() and torch.cuda.is_bf16_supported())

    want_enabled = bool(enabled and (not use_bf16))

    try:

        from torch.amp import GradScaler as AmpGradScaler

        return AmpGradScaler('cuda', enabled=want_enabled)

    except Exception:

        return torch.cuda.amp.GradScaler(enabled=want_enabled)



def _sigmoid(x): return torch.sigmoid(x)



# ---------- Stabilization Utilities ----------

def _ensure_finite_np(X: np.ndarray, clip_abs: Optional[float]=None) -> np.ndarray:

    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32, copy=False)

    if clip_abs is not None:

        c = float(clip_abs); X = np.clip(X, -c, c, out=X)

    return X



def _ensure_finite_t(x: torch.Tensor, clip_abs: Optional[float]=None) -> torch.Tensor:

    x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)

    if clip_abs is not None: x = torch.clamp(x, -float(clip_abs), float(clip_abs))

    return x



def _to_numpy(x: torch.Tensor) -> np.ndarray:

    return x.to(torch.float32).detach().cpu().numpy()



# ---------- Value Preprocessing ----------

def _value_preproc_auto_decide(sample: np.ndarray) -> str:

    try: m = float(np.nanmax(sample))

    except: m = 0.0

    return "log1p" if m > 50.0 else "none"



def _apply_value_preproc_np(X: np.ndarray, mode: str) -> np.ndarray:

    if mode == "none": return X.astype(np.float32, copy=False)

    if mode == "log1p": return np.log1p(np.maximum(X, 0.0)).astype(np.float32, copy=False)

    if mode == "sqrt":  return np.sqrt(np.maximum(X, 0.0)).astype(np.float32, copy=False)

    return X.astype(np.float32, copy=False)



# ---------------- Storage & I/O ----------------

def _init_store(adata: ad.AnnData):

    if 'cytoneighbor' not in adata.uns:

        adata.uns['cytoneighbor'] = {

            'feature_names': None,

            'label_mapping': None,

            'model': None,

            'preproc': {"mode": "none"},

            'ann': {'P_ann': None, 'd_ann': None},

            'history': {},

            'env_key': None,

            '_train_X': None, '_train_y': None,

            # --- Synthesis-related slots (statistics / config / last centers) ---

            'synth': {

                'enabled': True,

                'last_info': None,     # Statistics from the previous synthesis round

                'centers_R': None,     # (donor, type) centers in ANN projection space

                'radius_intra': None,  # (donor, type) intra-class radius threshold

                'nearest_other': None, # (donor, type) distance to nearest other-class center

            }

        }



def load_data(adata: ad.AnnData, data=None, labels=None, file_path: Optional[str]=None,

              data_key='data', labels_key='labels'):

    """Minimal loader: store numpy/sparse matrix + labels directly into adata.uns['cytoneighbor']."""

    _init_store(adata)

    if file_path:

        arr = np.load(file_path, allow_pickle=True)

        data, labels = arr[data_key], arr[labels_key]

    if data is None or labels is None:

        raise ValueError("Must provide either (data, labels) or file_path.")

    X = _as_numpy_dense(data).astype(np.float32); y = np.asarray(labels)

    if adata.is_view: adata._init_as_actual(adata.copy())

    if adata.var_names is None or len(adata.var_names)==0:

        adata.var_names = [f"g{i}" for i in range(X.shape[1])]

    z = adata.uns['cytoneighbor']

    z['_train_X'] = X; z['_train_y'] = y

    z['feature_names'] = list(adata.var_names)   # v7: record full feature set without HVG filtering

    _log(f"Data loaded: {X.shape[0]} cells, {X.shape[1]} genes.")



def save_model(adata: ad.AnnData, filepath: str):

    z = adata.uns['cytoneighbor']; model = z.get('model', None)

    if model is None: raise ValueError("No model to save.")

    feats = z.get('feature_names', None)

    if feats is None: raise ValueError("Missing feature_names.")

    ann=z.get('ann',{})

    if ann.get('P_ann', None) is None: raise ValueError("Missing P_ann.")

    state_dict = {k:v.detach().cpu() for k,v in model.state_dict().items()}

    cfg = {

        "feature_names": feats,

        "label_mapping": z.get("label_mapping", None),

        "preproc": z.get("preproc", {"mode": "none"}),

        "env_key": z.get("env_key", None),

        "arch": "SymRelEnc",

        "d_model": int(model.d_model),

        "depth": int(model.depth),

        "hidden": int(model.hidden),

        "dropout": float(model.dropout),

        "diff_mode": getattr(model, "diff_mode", "abs"),

        "d_ann": int(ann.get("d_ann",512)),

    }

    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

    with zipfile.ZipFile(filepath, 'w', zipfile.ZIP_DEFLATED) as zf:

        bio = io.BytesIO(); torch.save(state_dict, bio)

        zf.writestr("model_state.pt", bio.getvalue())

        zf.writestr("config.json", json.dumps(cfg).encode('utf-8'))

        bb = io.BytesIO(); np.save(bb, np.asarray(ann['P_ann'], np.float32))

        zf.writestr("P_ann.npy", bb.getvalue())

    _log(f"Model saved to {filepath}")



def load_model(adata: ad.AnnData, filepath: str) -> ad.AnnData:

    _init_store(adata)

    if not os.path.exists(filepath): raise FileNotFoundError(filepath)

    with zipfile.ZipFile(filepath,'r') as zf:

        cfg = json.loads(zf.read("config.json").decode('utf-8'))

        feats = cfg["feature_names"]

        d_model=int(cfg.get("d_model",1024)); depth=int(cfg.get("depth",6))

        hidden=int(cfg.get("hidden",2048)); dropout=float(cfg.get("dropout",0.1))

        d_ann=int(cfg.get("d_ann",512))

        diff_mode=str(cfg.get("diff_mode","abs"))

        P_ann = np.load(io.BytesIO(zf.read("P_ann.npy")), allow_pickle=False)

        model = SymRelEnc(n_genes=len(feats), d_model=d_model, depth=depth, hidden=hidden, dropout=dropout, diff_mode=diff_mode)

        st = torch.load(io.BytesIO(zf.read("model_state.pt")), map_location='cpu')

        miss = model.load_state_dict(st, strict=True)

        if getattr(miss, "missing_keys", []) or getattr(miss, "unexpected_keys", []):

            _log(f"[load_model] missing={getattr(miss,'missing_keys',[])} unexpected={getattr(miss,'unexpected_keys',[])}")

    z = adata.uns['cytoneighbor']

    z.update({

        "feature_names": feats,

        "label_mapping": cfg.get("label_mapping", None),

        "ann": {"P_ann": P_ann.astype(np.float32), "d_ann": int(d_ann)},

        "model": model, "env_key": cfg.get("env_key", None),

        "preproc": cfg.get("preproc", {"mode":"none"})

    })

    _log(f"Model loaded. SymRelEnc d_model={d_model}, depth={depth}, d_ann={d_ann}, diff_mode={diff_mode}.")

    return adata



# ---------------- Environment / Split / ANN ----------------

# --- _ENV_KEYS definition moved to external user parameters; read-only here ---

# This variable is defined in the user-configurable section of the main script



def _resolve_env(adata: ad.AnnData) -> Tuple[Optional[np.ndarray], Optional[str]]:

    for k in _ENV_KEYS:

        if k in adata.obs.columns:

            _log(f"[Env] Automatically enabling environment key: [{k}]")

            return adata.obs[k].astype(str).fillna("NA").to_numpy(), k

    _log(f"[Env] No environment keys detected (e.g., {_ENV_KEYS}); falling back to ungrouped split.")

    return None, None



def _group_split(y: np.ndarray, groups: Optional[np.ndarray], val_ratio: float, random_state: int):

    idx = np.arange(len(y))

    if groups is None or len(np.unique(groups)) < 2:

        try: tr, va = np.split(np.random.RandomState(random_state).permutation(idx),

                               [int(round((1.0-val_ratio)*len(idx)))])

        except: tr, va = np.split(idx, [int(round((1.0-val_ratio)*len(idx)))])

        _log(f"[Split] Standard split: train={len(tr)} val={len(va)}")

        return np.sort(tr), np.sort(va)

    

    donors = pd.Index(pd.Series(groups).astype('category').cat.categories)

    d2idx = {d: np.where(groups==d)[0] for d in donors}

    total = Counter(y); target = {c: val_ratio*v for c,v in total.items()}

    def score(cur): return sum((cur.get(c,0)-target.get(c,0))**2 for c in total.keys())

    cur = defaultdict(float); vald=set(); trd=set()

    for d in sorted(donors, key=lambda d: len(d2idx[d]), reverse=True):

        cand = Counter(y[d2idx[d]]); cur2 = cur.copy()

        for c,v in cand.items(): cur2[c]+=v

        if score(cur2) <= score(cur): vald.add(d); cur=cur2

        else: trd.add(d)

    tr = np.concatenate([d2idx[d] for d in trd]) if trd else np.array([],int)

    va = np.concatenate([d2idx[d] for d in vald]) if vald else np.array([],int)

    if len(tr)==0 or len(va)==0:

        _log("[Split] Group split too extreme; falling back to standard split.")

        try: tr, va = np.split(np.random.RandomState(random_state).permutation(idx),

                               [int(round((1.0-val_ratio)*len(idx)))])

        except: tr, va = np.split(idx, [int(round((1.0-val_ratio)*len(idx)))])

    _log(f"[Split-Group] Group-aware split: train={len(tr)} val={len(va)}")

    return np.sort(tr), np.sort(va)



def _faiss_hnsw_build(X: np.ndarray, m: int=32, efc: int=200) -> faiss.Index:

    d = X.shape[1]

    idx = faiss.IndexHNSWFlat(d, m); idx.hnsw.efConstruction = efc

    with Timer(f"[FAISS] HNSW add ({X.shape[0]}x{d}) ..."):

        if X.shape[0] > 0: idx.add(X.astype(np.float32))

    idx.hnsw.efSearch = 64

    return idx



def _faiss_search(idx: faiss.Index, Q: np.ndarray, k: int) -> np.ndarray:

    k = int(max(1, k))

    if Q.shape[0] == 0: return np.zeros((0, k), dtype=np.int64)

    _, I = idx.search(Q.astype(np.float32), k)

    return I



def _faiss_search_DI(idx: faiss.Index, Q: np.ndarray, k: int):

    if Q.shape[0] == 0:

        return np.full((0, k), np.float32(np.inf)), np.zeros((0, k), dtype=np.int64)

    D, I = idx.search(Q.astype(np.float32), k)

    return D, I



def _ann_rep(Xz: np.ndarray, P_ann: np.ndarray) -> np.ndarray:

    return (Xz @ P_ann.T).astype(np.float32)



# ------------------- Rule / Prototype / Targeted Sampling -------------------

def _build_label_donor_index(y: np.ndarray, donors: Optional[np.ndarray]):

    idx_by_label = {int(c): np.where(y==c)[0] for c in np.unique(y)}

    if donors is None:

        idx_by_label_donor = None

        donors_unique = None

    else:

        donors_unique = pd.Index(pd.Series(donors).astype('category').cat.categories).to_numpy()

        idx_by_label_donor = {}

        for c in np.unique(y):

            mask_c = (y==c)

            for d in donors_unique:

                ids = np.where(mask_c & (donors==d))[0]

                if ids.size>0: idx_by_label_donor[(int(c), str(d))] = ids

    return idx_by_label, idx_by_label_donor, donors_unique



def _sample_pos_simple(rng, anc:int, y:np.ndarray, donors:Optional[np.ndarray],

                       idx_by_label, idx_by_label_donor) -> Optional[int]:

    """Cross-donor same-type; fall back to global same-type if unavailable; None if both fail."""

    cls = int(y[anc])

    pool = None

    if donors is not None and idx_by_label_donor is not None:

        d_anc = str(donors[anc])

        # First choice: cross-donor

        cross = []

        for (c, d), ids in idx_by_label_donor.items():

            if c==cls and d!=d_anc: cross.append(ids)

        if cross:

            pool = np.concatenate(cross)

    if pool is None or (pool is not None and pool.size==0):

        # Fallback: global same-type (excluding self)

        pool = idx_by_label.get(cls, np.array([], dtype=np.int64))

        pool = pool[pool != anc]

    if pool.size==0: return None

    return int(rng.choice(pool))



def _sample_neg_simple(rng, anc:int, y:np.ndarray, donors:Optional[np.ndarray],

                       idx_by_label, idx_by_label_donor) -> Optional[int]:

    """Same-donor other-type; fall back to global other-type if unavailable."""

    cls = int(y[anc])

    pool = None

    if donors is not None and idx_by_label_donor is not None:

        d_anc = str(donors[anc])

        same_d = []

        # Collect all other types within this donor

        for (c, d), ids in idx_by_label_donor.items():

            if d==d_anc and c!=cls:

                same_d.append(ids)

        if same_d:

            pool = np.concatenate(same_d)

    if pool is None or pool.size==0:

        # Fallback: global other-type

        all_neg = np.where(y!=cls)[0]

        pool = all_neg

    if pool.size==0: return None

    return int(rng.choice(pool))



def _kmeans_np(X: np.ndarray, K: int, iters: int=8, seed: int=0) -> np.ndarray:

    """Minimal KMeans (CPU, few iters) for prototype construction; X: [n, d]."""

    n, d = X.shape

    K = max(1, min(K, n))

    rng = np.random.RandomState(seed)

    # k-means++ style initialization (simplified)

    centers = np.empty((K, d), np.float32)

    i0 = rng.randint(0, n); centers[0] = X[i0]

    closest = np.sum((X - centers[0])**2, axis=1)

    for k in range(1, K):

        probs = closest / (closest.sum() + 1e-8)

        i = rng.choice(n, p=probs)

        centers[k] = X[i]

        dist2 = np.sum((X - centers[k])**2, axis=1)

        closest = np.minimum(closest, dist2)

    # Iterate

    for _ in range(iters):

        # assign

        d2 = ((X[:, None, :] - centers[None, :, :])**2).sum(axis=2)  # [n,K]

        lab = np.argmin(d2, axis=1)

        # update

        for k in range(K):

            ids = np.where(lab==k)[0]

            if ids.size>0:

                centers[k] = X[ids].mean(axis=0)

            else:

                centers[k] = X[rng.randint(0, n)]

    return centers.astype(np.float32)



def _build_prototypes_R(R: np.ndarray, y: np.ndarray, per_class: int=2, seed: int=0):

    """Build multiple prototypes in ANN projection space R; returns (P: [M,d], proto_cls: [M])."""

    rng = np.random.RandomState(seed)

    protos = []

    proto_cls = []

    for c in np.unique(y):

        ids = np.where(y==c)[0]

        if ids.size == 0: continue

        if ids.size <= per_class:

            C = R[ids].mean(axis=0, dtype=np.float32, keepdims=True)

            protos.append(C); proto_cls += [int(c)]

            continue

        K = min(per_class, ids.size)

        C = _kmeans_np(R[ids], K=K, iters=6, seed=int(seed+13*c))

        protos.append(C); proto_cls += [int(c)]*C.shape[0]

    P = np.vstack(protos) if protos else np.zeros((0, R.shape[1]), np.float32)

    return P.astype(np.float32), np.asarray(proto_cls, np.int64)



# ---------------- Legacy FAISS-based pair construction (retained for compatibility) ----------------

def _classwise_knn_radius(R: np.ndarray, y: np.ndarray, *, k_intra: int = 16, quantile_hi: float = 0.995) -> Dict[int, float]:

    thr = {}

    classes = np.unique(y)

    for c in classes:

        ids_c = np.where(y == c)[0]

        if ids_c.size < 2:

            thr[int(c)] = float("inf"); continue

        Rc = R[ids_c]; idx = _faiss_hnsw_build(Rc)

        Di, Ii = idx.search(Rc.astype(np.float32), min(k_intra + 1, max(2, Rc.shape[0])))

        radii = []

        for i in range(Rc.shape[0]):

            di = Di[i]; ii = Ii[i]

            mask = (ii != i); di = di[mask]

            if di.size == 0: continue

            kpos = min(k_intra, di.size) - 1; kpos = max(0, kpos)

            radii.append(float(np.sort(di)[kpos]))

        thr[int(c)] = float(np.quantile(np.asarray(radii, np.float32), float(quantile_hi))) if len(radii) else float("inf")

    return thr



# ---------------- Encoder ----------------

class GEGLU(nn.Module):

    def __init__(self, d_in, d_out):

        super().__init__()

        self.proj = nn.Linear(d_in, d_out*2)

        self.act = nn.GELU()

    def forward(self, x):

        a,b = self.proj(x).chunk(2, dim=-1)

        return self.act(a) * b



class GatedResidualMLPBlock(nn.Module):

    def __init__(self, d, hidden, dropout=0.1):

        super().__init__()

        self.norm = nn.LayerNorm(d)

        self.ffn1 = GEGLU(d, hidden)

        self.drop = nn.Dropout(dropout)

        self.ffn2 = nn.Linear(hidden, d)

    def forward(self, x):

        h = self.norm(x)

        h = self.ffn2(self.drop(self.ffn1(h)))

        return x + self.drop(h)



class SymRelEnc(nn.Module):

    """

    Two difference modes are supported:

      - diff_mode='abs'   : element-wise absolute difference |x1 - x2| (default; G-dim vector)

      - diff_mode='symdiv': symmetric division raw=((x1+1)^2+(x2+1)^2)/((x1+1)*(x2+1)), local=log1p(max(raw-2,0))

    """

    def __init__(self, n_genes: int, d_model: int=1024, depth: int=6, hidden: int=2048, dropout: float=0.1,

                 diff_mode: str = "abs"):

        super().__init__()

        self.G = int(n_genes)

        in_dim = self.G

        self.d_model = int(d_model); self.depth=int(depth); self.hidden=int(hidden); self.dropout=float(dropout)

        self.diff_mode = str(diff_mode)



        self.proj = nn.Linear(in_dim, self.d_model, bias=True)

        self.inp  = nn.Sequential(nn.GELU(), nn.Linear(self.d_model, self.d_model), nn.GELU(), nn.Dropout(dropout))

        self.blocks = nn.ModuleList([GatedResidualMLPBlock(self.d_model, self.hidden, dropout) for _ in range(self.depth)])

        self.head = nn.Sequential(

            nn.LayerNorm(self.d_model),

            nn.Linear(self.d_model, self.hidden//2), nn.GELU(), nn.Dropout(dropout),

            nn.Linear(self.hidden//2, 1)

        )

        self.register_parameter("bias", nn.Parameter(torch.tensor(0.0), requires_grad=False))



    @torch.amp.autocast("cuda", enabled=False)

    def _abs_local_feats(self, xa: torch.Tensor, xb: torch.Tensor) -> torch.Tensor:

        xa = xa.float(); xb = xb.float()

        local = torch.abs(xa - xb)

        return _ensure_finite_t(local, clip_abs=1e6)



    @torch.amp.autocast("cuda", enabled=False)

    def _sym_local_feats(self, xa: torch.Tensor, xb: torch.Tensor) -> torch.Tensor:

        xa = xa.float(); xb = xb.float()

        eps = 1e-6

        a = xa + 1.0; b = xb + 1.0

        raw = (a.square() + b.square()) / (a.mul(b).clamp_min(eps))

        local = torch.log1p(torch.clamp(raw - 2.0, min=0.0))

        return _ensure_finite_t(local, clip_abs=1e4)



    def forward_pair_logits(self, xa: torch.Tensor, xb: torch.Tensor, *, amp_dtype: str = "auto") -> torch.Tensor:

        if self.diff_mode == "abs":

            local = self._abs_local_feats(xa, xb)

        elif self.diff_mode == "symdiv":

            local = self._sym_local_feats(xa, xb)

        else:

            raise ValueError("diff_mode must be 'abs' or 'symdiv'")

        x = _ensure_finite_t(local, clip_abs=1e6)

        with _autocast_cuda(enabled=xa.is_cuda, amp_dtype=amp_dtype):

            z = self.proj(x)

            z = self.inp(z)

            for blk in self.blocks: z = blk(z)

            logit = self.head(z).squeeze(-1) + self.bias

        return _ensure_finite_t(logit, clip_abs=1e6)



    def forward_scores(self, xa: torch.Tensor, xb: torch.Tensor, *, amp_dtype: str = "auto") -> torch.Tensor:

        if xb.dim()==2: xb = xb.unsqueeze(1)

        B,K,G = xb.shape

        xa_rep = xa.unsqueeze(1).expand(B,K,G)

        logits = self.forward_pair_logits(xa_rep.reshape(-1,G), xb.reshape(-1,G), amp_dtype=amp_dtype)

        return logits.reshape(B,K)



# -------------------- Gene-CutMix Data Synthesis --------------------

def _contiguous_slices(G: int, parts: List[float]) -> List[slice]:

    """Split [0, G) into contiguous segments by proportion; return slice list."""

    parts = np.asarray(parts, np.float32); parts = parts / max(1e-8, parts.sum())

    cuts = np.cumsum(np.round(parts * G)).astype(int)

    cuts[-1] = G

    s = 0; out=[]

    for e in cuts:

        out.append(slice(s, e)); s = e

    # Some segments may have length 0 due to extreme rounding; harmless (callers will skip).

    return out



def _gene_cutmix_two_parents(par1: np.ndarray, par2: np.ndarray, *,

                             ratio_range=(0.35, 0.65), rng: np.random.RandomState) -> np.ndarray:

    """Two-parent contiguous segment CutMix: first r%, last (1-r)%; preserves gene order."""

    G = par1.shape[0]

    lo, hi = float(ratio_range[0]), float(ratio_range[1])

    r = float(rng.uniform(lo, hi))

    k = int(round(r * G))

    if k <= 0: return par2.copy()

    if k >= G: return par1.copy()

    out = np.empty_like(par1, dtype=np.float32)

    out[:k] = par1[:k]

    out[k:] = par2[k:]

    return out



def _gene_cutmix_three_parents(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, *,

                               parts=(0.3, 0.4, 0.3)) -> np.ndarray:

    """Three-parent contiguous segment CutMix: front/middle/back; preserves gene order."""

    G = p1.shape[0]

    slices = _contiguous_slices(G, list(parts))

    out = np.empty_like(p1, dtype=np.float32)

    if len(slices) >= 1 and slices[0].stop - slices[0].start > 0:

        out[slices[0]] = p1[slices[0]]

    if len(slices) >= 2 and slices[1].stop - slices[1].start > 0:

        out[slices[1]] = p2[slices[1]]

    if len(slices) >= 3 and slices[2].stop - slices[2].start > 0:

        out[slices[2]] = p3[slices[2]]

    # Extra segments (if len(parts) > 3): rotate p1/p2/p3

    for i in range(3, len(slices)):

        src = (p1, p2, p3)[i % 3]

        if slices[i].stop - slices[i].start > 0:

            out[slices[i]] = src[slices[i]]

    return out



def _compute_centers_R_by_donor_type(R: np.ndarray, y: np.ndarray, donors: np.ndarray

                                     ) -> Tuple[Dict[Tuple[int,str], np.ndarray],

                                                Dict[Tuple[int,str], float],

                                                Dict[Tuple[int,str], float]]:

    """

    Returns:

      centers[(cls, donor)] = class center (R-space)

      radius_intra[(cls, donor)] = intra-class radius (P95 Euclidean distance to center; +inf when fewer than 3 samples)

      nearest_other[(cls, donor)] = distance to nearest other-class center within same donor (+inf when fewer than 2 classes)

    """

    centers: Dict[Tuple[int,str], np.ndarray] = {}

    radius_intra: Dict[Tuple[int,str], float] = {}

    nearest_other: Dict[Tuple[int,str], float] = {}



    classes = np.unique(y)

    donors_unique = pd.Index(pd.Series(donors).astype('category').cat.categories).to_numpy()

    # First, compute all (donor, type) centers

    for d in donors_unique:

        mask_d = (donors == d)

        y_d = y[mask_d]

        R_d = R[mask_d]

        for c in np.unique(y_d):

            ids = np.where(y_d == c)[0]

            if ids.size == 0: continue

            key = (int(c), str(d))

            centers[key] = R_d[ids].mean(axis=0).astype(np.float32, copy=False)

            if ids.size >= 3:

                dist = np.sqrt(((R_d[ids] - centers[key])**2).sum(axis=1))

                radius_intra[key] = float(np.quantile(dist, 0.95))

            else:

                radius_intra[key] = float("inf")



    # Then compute nearest other-class center distance within same donor

    for d in donors_unique:

        keys_d = [(int(c), str(d)) for c in classes if (int(c), str(d)) in centers]

        for key in keys_d:

            c, _ = key

            others = [k for k in keys_d if k[0] != c]

            if not others:

                nearest_other[key] = float("inf")

                continue

            dist = [np.linalg.norm(centers[key] - centers[k2]) for k2 in others]

            nearest_other[key] = float(min(dist)) if len(dist) else float("inf")



    return centers, radius_intra, nearest_other



def _filter_synth_by_ann(S: np.ndarray, cls: int, donor: str, *,

                         P_ann: np.ndarray,

                         centers_R: Dict[Tuple[int,str], np.ndarray],

                         radius_intra: Dict[Tuple[int,str], float],

                         nearest_other: Dict[Tuple[int,str], float],

                         alpha: float = 1.0, gamma: float = 1.0) -> np.ndarray:

    """

    Dual-threshold filtering:

      dist_to_center <= min(alpha * radius_intra, gamma * nearest_other)

      If one threshold is missing (=inf), use the other; if both are inf, pass all.

    Return boolean mask of retained samples.

    """

    key = (int(cls), str(donor))

    if key not in centers_R:

        # No center available (this donor x type combination does not exist); discard all

        return np.zeros(S.shape[0], dtype=bool)

    C = centers_R[key]

    R = _ann_rep(S.astype(np.float32), P_ann)

    dist = np.sqrt(((R - C[None, :])**2).sum(axis=1))

    thr_a = float(alpha) * float(radius_intra.get(key, float("inf")))

    thr_g = float(gamma) * float(nearest_other.get(key, float("inf")))

    thr = min(thr_a, thr_g)

    if not np.isfinite(thr):

        # Both are inf: no filtering

        return np.ones(S.shape[0], dtype=bool)

    return dist <= thr



# ========== Synthesis Toolkit (Same-Donor Same-Type Gene-CutMix + Radius/Margin Thresholds) ==========



def _class_centers_R(R: np.ndarray, y: np.ndarray) -> Dict[int, np.ndarray]:

    centers = {}

    for c in np.unique(y):

        ids = np.where(y==c)[0]

        if ids.size:

            centers[int(c)] = R[ids].mean(axis=0).astype(np.float32)

    return centers



def _compute_center_margins_R(centers: Dict[int, np.ndarray]) -> Dict[int, float]:

    keys = sorted(centers.keys())

    if len(keys) < 2:

        return {int(k): float("inf") for k in keys}

    mat = np.stack([centers[k] for k in keys], axis=0).astype(np.float32)

    margins = {}

    for i, ci in enumerate(keys):

        d = np.linalg.norm(mat[i][None,:] - mat, axis=1)

        d[i] = np.inf

        margins[int(ci)] = float(np.min(d))

    return margins



def _safe_parts(parts: Optional[List[slice]], G: int) -> List[slice]:

    """Convert arbitrary input parts into a valid slice list (at least 2 segments, covering 0..G)."""

    if not isinstance(G, (int, np.integer)) or G <= 1:

        return [slice(0, max(1, int(G)))]

    if (parts is None) or (not isinstance(parts, (list, tuple))) or (len(parts) == 0):

        k = max(1, min(G-1, G // 2))

        return [slice(0, k), slice(k, G)]

    # Normalize, merge, clip boundaries

    norm = []

    last_end = 0

    for s in parts:

        if not isinstance(s, slice):

            continue

        a = 0 if s.start is None else int(s.start)

        b = G if s.stop  is None else int(s.stop)

        a = max(0, min(G, a))

        b = max(0, min(G, b))

        if b <= a:

            continue

        # If there is a gap from the previous segment, fill it first

        if a > last_end:

            norm.append(slice(last_end, a))

        norm.append(slice(a, b))

        last_end = b

        if last_end >= G:

            break

    if last_end < G:

        norm.append(slice(last_end, G))

    if len(norm) == 0:  # Extreme degenerate case

        k = max(1, min(G-1, G // 2))

        norm = [slice(0, k), slice(k, G)]

    return norm



def _pick_cutmix_parts(G: int,

                       rng: np.random.RandomState,

                       ratio_range: Tuple[float,float]=(0.35,0.65),

                       three_parent_p: float=0.25,

                       three_parts: Tuple[float,float,float]=(0.3,0.4,0.3)) -> List[slice]:

    """Randomly generate 2-segment or 3-segment gene slices (preserving gene order)."""

    if rng.rand() < float(three_parent_p):

        a,b,c = three_parts

        a = float(np.clip(a, 0.05, 0.9))

        b = float(np.clip(b, 0.05, 0.9))

        c = float(np.clip(c, 0.05, 0.9))

        # Normalize to sum to 1

        s = a + b + c

        if s <= 0:  # Safety guard

            a,b,c = 0.33, 0.34, 0.33

        else:

            a, b, c = a/s, b/s, c/s

        s1 = max(1, min(G-2, int(round(G*a))))

        s2 = max(1, min(G-1-s1, int(round(G*b))))

        parts = [slice(0, s1), slice(s1, s1+s2), slice(s1+s2, G)]

    else:

        lo, hi = ratio_range

        lo = float(np.clip(lo, 0.1, 0.9))

        hi = float(np.clip(hi, lo, 0.9))

        k = int(round(G * rng.uniform(lo, hi)))

        k = max(1, min(G-1, k))

        parts = [slice(0, k), slice(k, G)]

    return _safe_parts(parts, G)





def _adjust_parts_for_k(parts: List[slice], k_parents: int, G: int) -> List[slice]:

    """When parent count is 2 but parts has 3 segments, merge trailing segments until count == k_parents."""

    cur = _safe_parts(parts, G)

    if k_parents <= 1:

        return [slice(0, G)]

    while len(cur) > k_parents:

        p1, p2 = cur[-2], cur[-1]

        cur = cur[:-2] + [slice(p1.start, p2.stop)]

    return cur



def _gene_cutmix_synthesize(X: np.ndarray,

                            parents: List[int],

                            parts: Optional[List[slice]]) -> np.ndarray:

    """Copy gene segments from multiple parents by partition, strictly preserving order."""

    G = X.shape[1]

    parts = _safe_parts(parts, G)

    out = np.zeros(G, np.float32)

    take = min(len(parents), len(parts))

    if take <= 0:

            # Extreme fallback: return a copy of the first parent directly

        return X[parents[0]].astype(np.float32, copy=True)

    for i in range(take):

        out[parts[i]] = X[parents[i], parts[i]]

    # Fill uncovered boundaries with the last parent (_safe_parts should cover the full interval)

    covered = np.zeros(G, dtype=bool)

    for i in range(take): covered[parts[i]] = True

    if not covered.all():

        last = parents[take-1]

        out[~covered] = X[last, ~covered]

    return out.astype(np.float32, copy=False)



def build_synth_pool(

    X_real: np.ndarray, y_real: np.ndarray, donors_real: np.ndarray,

    P_ann: np.ndarray,

    mult_per_dt: float = 1.2,

    cap_per_dt: int = 600,

    overgen_factor: float = 1.5,

    ratio_range: Tuple[float,float]=(0.35,0.65),

    p_three_parent: float = 0.25,

    parts_three: Tuple[float,float,float]=(0.3,0.4,0.3),

    alpha: float = 1.0,   # Radius threshold scaling

    gamma: float = 1.0,   # Nearest other-class center margin scaling

    random_state: int = 0

) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str,int]]:

    rng = np.random.RandomState(random_state)

    Xr = _ensure_finite_np(X_real, 1e6)

    R = _ann_rep(Xr, P_ann)

    G = Xr.shape[1]; y = y_real; donors = donors_real



    # donor/type -> index

    dts = defaultdict(list)

    for i in range(len(y)):

        dts[(int(y[i]), str(donors[i]))].append(i)

    for k in list(dts.keys()):

        dts[k] = np.asarray(dts[k], np.int64)



    centers = _class_centers_R(R, y)

    margins = _compute_center_margins_R(centers)

    rad = _classwise_knn_radius(R, y, k_intra=8, quantile_hi=0.95)



    synth = []; synth_y = []; synth_d = []; is_synth = []



    for (c, d), idx in dts.items():

        n_real = idx.size

        if n_real < 2:

            continue  # Requires at least two parents

        goal = int(min(int(round(mult_per_dt * n_real)), int(cap_per_dt)))

        over = int(round(goal * max(1.0, float(overgen_factor))))

        if over <= 0: continue



        produced = 0

        trials = min(over, max(8, n_real * 16))  # Moderate number of attempts

        for _ in range(trials):

            try:

                parts = _pick_cutmix_parts(G, rng, ratio_range, p_three_parent, parts_three)

                k_parents = min(3, n_real)

                if k_parents < 2:

                    break

                parts_use = _adjust_parts_for_k(parts, k_parents, G)

                pa = rng.choice(idx, size=k_parents, replace=False)

                xnew = _gene_cutmix_synthesize(Xr, list(pa), parts_use)



                # R-space threshold: class radius + nearest other-class margin

                rnew = (xnew[None,:] @ P_ann.T).astype(np.float32)

                rc = centers[c][None,:]

                dist_c = float(np.linalg.norm(rnew - rc))

                thr_rad = float(alpha) * float(rad.get(int(c), float('inf')))

                thr_margin = float(gamma) * float(margins.get(int(c), float('inf')))



                ok = True

                if not np.isfinite(thr_rad): thr_rad = float('inf')

                if dist_c > thr_rad: ok = False

                # Nearest other-class center distance

                mind_other = float('inf')

                for cc, ctr in centers.items():

                    if cc == c: continue

                    mind_other = min(mind_other, float(np.linalg.norm(rnew - ctr[None,:])))

                if mind_other < thr_margin: ok = False

                if not ok: continue



                synth.append(xnew); synth_y.append(int(c)); synth_d.append(str(d)); is_synth.append(1)

                produced += 1

                if produced >= goal: break

            except (ValueError, TypeError, RuntimeError):

                # Boundary issue: skip this iteration, continue trying

                continue



        # If still insufficient after filtering, relax radius by 10% and retry

        if produced < goal:

            need = goal - produced

            fill_try = max(need*3, 8)

            fill_cnt = 0

            for _ in range(fill_try):

                try:

                    parts = _pick_cutmix_parts(G, rng, ratio_range, p_three_parent, parts_three)

                    k_parents = min(3, n_real)

                    if k_parents < 2: break

                    parts_use = _adjust_parts_for_k(parts, k_parents, G)

                    pa = rng.choice(idx, size=k_parents, replace=False)

                    xnew = _gene_cutmix_synthesize(Xr, list(pa), parts_use)

                    rnew = (xnew[None,:] @ P_ann.T).astype(np.float32)

                    rc = centers[c][None,:]

                    dist_c = float(np.linalg.norm(rnew - rc))

                    thr_rad = 1.10 * float(alpha) * float(rad.get(int(c), float('inf')))

                    mind_other = float('inf')

                    for cc, ctr in centers.items():

                        if cc == c: continue

                        mind_other = min(mind_other, float(np.linalg.norm(rnew - ctr[None,:])))

                    thr_margin = float(gamma) * float(margins.get(int(c), float('inf')))

                    if (dist_c <= thr_rad) and (mind_other >= thr_margin):

                        synth.append(xnew); synth_y.append(int(c)); synth_d.append(str(d)); is_synth.append(1)

                        fill_cnt += 1

                        if fill_cnt >= need: break

                except (ValueError, TypeError, RuntimeError):

                    continue



    if synth:

        Xs = np.stack(synth, axis=0).astype(np.float32)

        ys = np.asarray(synth_y, np.int64)

        ds = np.asarray(synth_d, object)

        flags = np.asarray(is_synth, np.int8)

        X_aug = np.concatenate([Xr, Xs], axis=0)

        y_aug = np.concatenate([y_real, ys], axis=0)

        d_aug = np.concatenate([donors_real, ds], axis=0)

        is_s_aug = np.concatenate([np.zeros(Xr.shape[0], np.int8), flags], axis=0)

    else:

        X_aug, y_aug, d_aug = Xr, y_real, donors_real

        is_s_aug = np.zeros(Xr.shape[0], np.int8)



    info = {"n_real": int(Xr.shape[0]),

            "n_synth": int((is_s_aug==1).sum()),

            "n_aug": int(X_aug.shape[0]),

            "mult_per_dt": float(mult_per_dt),

            "cap_per_dt": int(cap_per_dt),

            "alpha": float(alpha),

            "gamma": float(gamma)}

    _log(f"[Synth] Synthesis complete: real={info['n_real']} synth={info['n_synth']} aug={info['n_aug']} "

         f"(mult_per_dt={mult_per_dt}, cap_per_dt={cap_per_dt}, alpha={alpha}, gamma={gamma})")

    return X_aug, y_aug, d_aug, is_s_aug, info





# ---------------- Evaluation ----------------

@torch.inference_mode()

def _eval_pairs(model, X_t, a, b, y, device, batch=2048, amp_dtype: str = "auto"):

    model.eval(); all_y=[]; all_s=[]; tot=0.0; steps=0

    Bv = min(batch, max(64, len(a)//8))

    bce = nn.BCEWithLogitsLoss()

    for s in range(0, len(a), Bv):

        e=min(len(a), s+Bv)

        ai = torch.from_numpy(a[s:e]).to(device)

        bi = torch.from_numpy(b[s:e]).to(device)

        lab = torch.from_numpy(y[s:e]).float().to(device)

        xa = X_t.index_select(0, ai)

        xb = X_t.index_select(0, bi).unsqueeze(1)

        logits = model.forward_scores(xa, xb, amp_dtype=amp_dtype)[:,0]

        logits = _ensure_finite_t(logits, clip_abs=1e6)

        loss = bce(logits.to(torch.float32), lab.to(torch.float32))

        tot += float(loss.detach().cpu()); steps+=1

        all_y.append(_to_numpy(lab)); all_s.append(_to_numpy(_sigmoid(logits)))

    all_y = np.concatenate(all_y) if all_y else np.array([])

    all_s = np.concatenate(all_s) if all_s else np.array([])

    try: auc = roc_auc_score(all_y, all_s) if all_y.size else float("nan")

    except: auc = float("nan")

    return tot/max(1,steps), auc, (all_y, all_s)

















# ---------------- Main Training API (Integrated Dynamic Synthesis) ----------------

# def fit(adata: ad.AnnData,

#         preproc: str = "auto",

#         validation_split: Union[float,str,None]='auto',

#         monitor: str = "dyn_val_auc",

#         random_state: int = 0,

#         use_cuda: bool = True,

#         use_amp: bool = True,

#         amp_dtype: str = "bf16",

#         torch_matmul_precision: str = "high",

#         preload_to_gpu: bool = True,

#         d_model: int = 1024,

#         depth: int = 6,

#         hidden: int = 2048,

#         dropout: float = 0.1,

#         diff_mode: str = "abs",

#         ann_ratio: float = 0.08,

#         ann_dim_min: int = 256,

#         ann_dim_max: int = 1024,

#         sampler_mode: str = "mixed",

#         simple_frac: float = 0.50,

#         proto_frac: float  = 0.20,

#         global_frac: float = 0.30,

#         train_pairs_per_epoch: int = 6000,

#         proto_per_class: int = 2,

#         proto_update_every: int = 3,

#         proto_pos_pool: int = 512,

#         global_probe_frac: float = 0.30,

#         global_probe_candidates: int = 4096,

#         global_probe_topk: int = 32,

#         probe_interval: int = 1,

#         neg_cache_pick_p: float = 0.90,

#         # Dynamic synthesis

#         synth_enable: bool = True,

#         synth_refresh_every: int = 1,

#         synth_mult_per_dt: float = 1.2,

#         synth_cap_per_dt: int = 600,

#         synth_overgen_factor: float = 1.5,

#         synth_ratio_range: Tuple[float,float]=(0.35,0.65),

#         synth_p_three_parent: float = 0.25,

#         synth_parts_three: Tuple[float,float,float]=(0.3,0.4,0.3),

#         synth_alpha: float = 1.0,

#         synth_gamma: float = 1.0,

#         anchor_real_only: bool = True,

#         pos_semihard_q: Tuple[float,float]=(0.6, 0.9),

#         rneg_k: int = 64,

#         epochs: int = 100,

#         lr: float = 1e-3,

#         weight_decay: float = 1e-4,

#         lr_sched: str = "cosine",

#         warmup_ratio: float = 0.05,

#         plateau_factor: float = 0.5,

#         plateau_patience: int = 8,

#         lambda_bce: float = 0.4,

#         label_smoothing: float = 0.0,

#         lambda_rank: float = 1.0,

#         rank_margin: float = 0.5,

#         self_pair_frac: float = 0.50,

#         lambda_self_rank: float = 0.2,

#         self_margin: float = 0.5,

#         margin_warmup_epochs: int = 6,

#         noise_mode: str = "feature_jitter",

#         noise_sigma0: float = 0.03,

#         noise_gamma: float = 1.5,

#         noise_shared_rho: float = 0.0,

#         noise_on_train: bool = True,

#         val_refresh_every: int = 1,

#         val_window: int = 3,

#         val_eps: float = 1e-3,

#         val_pairs_max: int = 60_000,

#         val_global_probe_frac: float = 0.05,

#         val_global_candidates: int = 512,

#         val_global_topk: int = 8,

#         gpu_mem_budget_mb: float = 2000.0,

#         batch_size_override: Optional[int] = None,

#         save_best_model_path: Optional[str] = None,

#         save_ckpt_dir: Optional[str] = None) -> ad.AnnData:



#     _init_store(adata)

#     z = adata.uns['cytoneighbor']

#     X_all = z.get('_train_X', None); y_raw = z.get('_train_y', None)

#     if X_all is None or y_raw is None: raise ValueError("Please call load_data() first.")



#     feats = list(adata.var_names); z['feature_names'] = feats

#     _log(f"Features: {len(feats)} genes (v7: no HVG/Scaler)")



#     try: torch.set_float32_matmul_precision(str(torch_matmul_precision))

#     except Exception: pass



#     cats = pd.Index(pd.Series(y_raw).astype('category').cat.categories)

#     lab2id = {c:i for i,c in enumerate(cats)}

#     y_all = np.array([lab2id[v] for v in y_raw], dtype=np.int64)

#     z['label_mapping'] = {'classes': list(map(str, cats))}



#     donors_all, donor_key = _resolve_env(adata); z['env_key'] = donor_key



#     X_all = _as_numpy_dense(X_all)

#     preproc_mode = preproc if preproc != "auto" else _value_preproc_auto_decide(X_all[:min(10000, X_all.shape[0])])

#     X_all = _apply_value_preproc_np(X_all, preproc_mode)

#     z['preproc'] = {"mode": str(preproc_mode)}

#     X_all = _ensure_finite_np(X_all, 1e6)



#     # Split

#     if validation_split == 'auto':

#         val_ratio = 0.2; no_val = False

#     elif validation_split in (None, 0, 0.0, 'none'):

#         no_val = True; val_ratio = 0.0

#         if monitor == 'dyn_val_auc':

#             _log('[Warn] monitor="dyn_val_auc" but no validation set, falling back to "shadow_auc".'); monitor='shadow_auc'

#     else:

#         no_val = False; val_ratio = float(validation_split)



#     if not no_val:

#         tr_idx, va_idx = _group_split(y_all, donors_all, float(val_ratio), int(random_state))

#         if tr_idx.size==0 or va_idx.size==0: raise RuntimeError("Split failed: training or validation set is empty.")

#     else:

#         tr_idx = np.arange(X_all.shape[0], dtype=np.int64); va_idx = np.array([], np.int64)

#         _log("[Split] Using full dataset for training (no fixed validation set)")



#     Xtr_real = X_all[tr_idx]; ytr_real = y_all[tr_idx]

#     Xva = X_all[va_idx] if not no_val else None; yva = y_all[va_idx] if not no_val else None

#     donors_tr_real = (adata.obs.loc[adata.obs.index[tr_idx], donor_key].astype(str).fillna("NA").to_numpy()

#                       if (donor_key is not None) else None)

#     donors_va = (adata.obs.loc[adata.obs.index[va_idx], donor_key].astype(str).fillna("NA").to_numpy()

#                  if ((not no_val) and (donor_key is not None)) else None)



#     device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

#     _log(f"[Device] using {device.type.upper()}")

#     _set_seed(random_state, use_cuda=(device.type=='cuda'), deterministic=False)

#     amp_ok = bool(use_amp and device.type=='cuda')

#     scaler = _grad_scaler(amp_ok, amp_dtype=amp_dtype)



#     d_ann_cand = int(round(max(1, ann_ratio * Xtr_real.shape[1])))

#     d_ann = int(min(ann_dim_max, max(ann_dim_min, d_ann_cand)))

#     rng = np.random.RandomState(random_state)

#     P_ann = np.linalg.qr(rng.normal(size=(d_ann, Xtr_real.shape[1])).astype(np.float32).T)[0].T.astype(np.float32)

#     z['ann'] = {"P_ann": P_ann.astype(np.float32), "d_ann": int(d_ann)}

#     _log(f"[ANN] d_ann={d_ann} (ratio={ann_ratio}, min={ann_dim_min}, max={ann_dim_max})")



#     model = SymRelEnc(n_genes=Xtr_real.shape[1], d_model=d_model, depth=depth, hidden=hidden, dropout=dropout, diff_mode=diff_mode).to(device)

#     opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)



#     if lr_sched == "cosine":

#         scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs)); warmup_epochs = max(1, int(warmup_ratio * epochs))

#     elif lr_sched == "plateau":

#         scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=plateau_factor, patience=plateau_patience, verbose=False); warmup_epochs = 0

#     else:

#         scheduler = None; warmup_epochs = 0



#     if (not no_val) and Xva is not None and Xva.shape[0] > 1:

#         Xva = _ensure_finite_np(Xva, 1e6)

#         Xva_t = torch.from_numpy(Xva).to(device, non_blocking=True) if (device.type=='cuda' and preload_to_gpu) else torch.from_numpy(Xva).to(device)

#     else:

#         Xva_t = None



#     if batch_size_override is not None:

#         B_target = int(max(8, batch_size_override))

#     else:

#         B_target = int(math.sqrt(max(1, (float(gpu_mem_budget_mb)*1024*1024)/(Xtr_real.shape[1]*4.0))))

#         B_target = max(128, min(4096, B_target))



#     hist={'train_loss': [], 'val_loss': [], 'val_auc': [], 'monitor': str(monitor),

#           'loss_bce': [], 'loss_rank': [], 'loss_self_rank': [],

#           'triplet_active_ratio': [],

#           'val_refresh_every': int(val_refresh_every), 'val_window': int(val_window), 'val_eps': float(val_eps),

#           'diff_mode': str(diff_mode), 'sampler_mode': str(sampler_mode),

#           'synth': {'enable': bool(synth_enable)}}

#     best_auc = -np.inf; best_state_cpu = None



#     # Validation pairs (adaptive cap)

#     def _refresh_val_pairs(epoch:int):

#         if no_val or Xva is None or Xva.shape[0] < 2:

#             return None

#         rng_e = np.random.RandomState(random_state + 101*epoch + 7)

#         idx_by_label_v, idx_by_label_donor_v, _ = _build_label_donor_index(yva, donors_va)

#         n_val = int(Xva.shape[0]); C = int(len(idx_by_label_v))

#         N_max = int(min(int(val_pairs_max), 16 * max(1,n_val) * max(1,C)))

#         a_list=[]; b_list=[]; l_list=[]

#         while len(l_list) < N_max:

#             classes, counts = np.unique(yva, return_counts=True)

#             p = counts / counts.sum()

#             c = int(rng_e.choice(classes, p=p))

#             ids = idx_by_label_v.get(int(c), np.array([], np.int64))

#             if ids.size<=1: continue

#             ai = int(rng_e.choice(ids))

#             pi = _sample_pos_simple(rng_e, ai, yva, donors_va, idx_by_label_v, idx_by_label_donor_v)

#             if pi is not None:

#                 a_list.append(ai); b_list.append(pi); l_list.append(1)

#             ni = _sample_neg_simple(rng_e, ai, yva, donors_va, idx_by_label_v, idx_by_label_donor_v)

#             if ni is not None:

#                 a_list.append(ai); b_list.append(ni); l_list.append(0)

#         a_va = np.asarray(a_list, np.int64); b_va = np.asarray(b_list, np.int64); l_va = np.asarray(l_list, np.int64)

#         _log(f"[ValPairs] epoch={epoch} built={len(l_va)} (pos={(l_va==1).sum()}, neg={(l_va==0).sum()})")

#         return a_va, b_va, l_va



#     a_va = b_va = l_va = None

#     if (not no_val) and Xva_t is not None:

#         cur_val = _refresh_val_pairs(epoch=1)

#         if cur_val is None: raise RuntimeError("Validation set too small; cannot construct validation pairs.")

#         a_va, b_va, l_va = cur_val



#     _log("==================== Training: SymRelEnc (+ Dynamic Gene-CutMix) ====================")

#     bce_logits = nn.BCEWithLogitsLoss(reduction='none')

#     pos_smooth = float(1.0 - label_smoothing); neg_smooth = float(label_smoothing)

#     softplus = nn.Softplus()

#     hard_cache: Dict[int, Tuple[np.ndarray,int]] = {}



#     def _decay_cache():

#         drop=[]

#         for k,(ids,ttl) in list(hard_cache.items()):

#             ttl -= 1

#             if ttl <= 0: drop.append(k)

#             else: hard_cache[k]=(ids,ttl)

#         for k in drop: hard_cache.pop(k, None)



#     # ====== Training Loop ======

#     for ep in range(1, epochs+1):

#         t0=time.time(); model.train()

#         if lr_sched=="cosine" and ep <= warmup_epochs:

#             for g in opt.param_groups: g['lr'] = lr * ep / max(1,warmup_epochs)



#         # Refresh synthesis pool

#         if synth_enable and (donors_tr_real is not None) and ((ep==1) or (ep % max(1,int(synth_refresh_every)) == 0)):

#             Xtr_aug, ytr_aug, donors_tr_aug, is_synth, info = build_synth_pool(

#                 Xtr_real, ytr_real, donors_tr_real, P_ann,

#                 mult_per_dt=synth_mult_per_dt, cap_per_dt=synth_cap_per_dt,

#                 overgen_factor=synth_overgen_factor, ratio_range=synth_ratio_range,

#                 p_three_parent=synth_p_three_parent, parts_three=synth_parts_three,

#                 alpha=synth_alpha, gamma=synth_gamma, random_state=int(random_state+991*ep)

#             )

#             z['synth'] = info; hard_cache.clear()

#         else:

#             Xtr_aug, ytr_aug = _ensure_finite_np(Xtr_real, 1e6), ytr_real

#             donors_tr_aug = donors_tr_real

#             is_synth = np.zeros(Xtr_aug.shape[0], np.int8)



#         Xtr_t = torch.from_numpy(Xtr_aug).to(device, non_blocking=True) if (device.type=='cuda' and preload_to_gpu) else torch.from_numpy(Xtr_aug).to(device)

#         Rtr = _ann_rep(Xtr_aug, P_ann)



#         # Donor index

#         idx_by_label, idx_by_label_donor, _ = _build_label_donor_index(ytr_aug, donors_tr_aug)



#         # Prototypes

#         def _refresh_prototypes(epoch:int):

#             if sampler_mode == "simple": return (None, None)

#             if (epoch==1) or (epoch % max(1,int(proto_update_every)) == 0):

#                 Pp, Pc = _build_prototypes_R(Rtr, ytr_aug, per_class=int(proto_per_class), seed=int(random_state+epoch*7))

#                 _log(f"[Proto] built {len(Pc)} prototypes (per_class={proto_per_class})")

#                 return (Pp, Pc)

#             return (proto_P, proto_cls)

#         proto_P, proto_cls = _refresh_prototypes(ep)



#         # Positives: semi-hard (R-space 0.6-0.9 quantile; fallback when candidates are few)

#         def _pos_semihard(ai: int, rng_e: np.random.RandomState) -> Optional[int]:

#             cls = int(ytr_aug[ai])

#             # Candidates: cross-donor same-type -> fallback global same-type

#             cand = None

#             if donors_tr_aug is not None and idx_by_label_donor is not None:

#                 d_anc = str(donors_tr_aug[ai]); packs=[]

#                 for (c,d), ids in idx_by_label_donor.items():

#                     if c==cls and d!=d_anc and ids.size>0:

#                         packs.append(ids)

#                 if packs: cand = np.concatenate(packs)

#             if cand is None or cand.size==0:

#                 cand = idx_by_label.get(cls, np.array([], np.int64))

#             cand = cand[cand!=ai]

#             if cand.size==0: return None



#             if cand.size < 4:

#                 return int(rng_e.choice(cand))  # Too few candidates, choose randomly



#             r_ai = Rtr[ai:ai+1]; dist = ((Rtr[cand]-r_ai)**2).sum(axis=1)

#             lo_q, hi_q = pos_semihard_q

#             lo = float(np.quantile(dist, lo_q)); hi = float(np.quantile(dist, hi_q))

#             mask = (dist >= lo) & (dist <= hi)

#             pool = cand[mask] if mask.any() else cand

#             return int(rng_e.choice(pool))



#         # Negatives: semi-hard (same-donor other-type nearest few), fallback when few candidates

#         def _neg_semihard(ai: int, rng_e: np.random.RandomState) -> Optional[int]:

#             cls = int(ytr_aug[ai]); r_ai = Rtr[ai:ai+1]

#             # Same-donor other-type

#             if donors_tr_aug is not None:

#                 ids = np.where((donors_tr_aug == donors_tr_aug[ai]) & (ytr_aug != cls))[0]

#             else:

#                 ids = np.where(ytr_aug != cls)[0]

#             if ids.size == 0:

#                 return None

#             take = min(int(rneg_k), ids.size)

#             if take <= 1:

#                 return int(rng_e.choice(ids))

#             dist = ((Rtr[ids]-r_ai)**2).sum(axis=1)

#             part = np.argpartition(dist, kth=take-1)[:take]

#             cand = ids[part]

#             return int(rng_e.choice(cand)) if cand.size>0 else None



#         # Anchor: real samples only

#         if anchor_real_only:

#             real_idx = np.where(is_synth==0)[0]

#             if real_idx.size == 0: real_idx = np.arange(ytr_aug.size)

#         else:

#             real_idx = np.arange(ytr_aug.size)



#         # Training pair count adaptive

#         max_pairs = int(16 * max(1, real_idx.size))

#         N_pairs_target = int(min(int(train_pairs_per_epoch), max_pairs))



#         def _refresh_train_pairs(epoch:int):

#             rng_e = np.random.RandomState(random_state + 17*epoch)

#             classes, counts = np.unique(ytr_aug[real_idx], return_counts=True)

#             p = counts / counts.sum() if counts.sum() > 0 else np.ones_like(counts)/max(1,len(counts))

#             by_cls_real = {c: real_idx[ytr_aug[real_idx]==c] for c in classes}

#             anc=[]

#             while len(anc) < N_pairs_target and len(classes)>0:

#                 c = rng_e.choice(classes, p=p)

#                 ids = by_cls_real[c]

#                 if ids.size <= 1:

#                     # Avoid infinite loop: exclude this class

#                     mask = classes != c

#                     classes = classes[mask]; p = p[mask];

#                     if p.sum() > 0: p = p/p.sum()

#                     continue

#                 anc.append(int(rng_e.choice(ids)))

#             if len(anc) == 0:

#                 raise RuntimeError("Training pair sampling failed: check if training set is too small.")

#             anc = np.asarray(anc, np.int64)



#             n_simple = int(round(simple_frac * N_pairs_target)) if sampler_mode!="simple" else N_pairs_target

#             n_proto  = int(round(proto_frac  * N_pairs_target)) if sampler_mode!="simple" else 0

#             n_rest   = N_pairs_target - n_simple - n_proto

#             if n_rest < 0: n_rest = 0



#             order = rng_e.permutation(N_pairs_target)

#             seg_simple = order[:n_simple]; seg_proto  = order[n_simple:n_simple+n_proto]; seg_rest   = order[n_simple+n_proto:]



#             a_pos=[]; p_pos=[]

#             for i in seg_simple:

#                 ai = anc[i]; pi = _pos_semihard(ai, rng_e)

#                 if pi is None: continue

#                 a_pos.append(int(ai)); p_pos.append(int(pi))

#             if sampler_mode!="simple" and proto_P is not None and proto_cls is not None and len(proto_cls)>0:

#                 for i in seg_proto:

#                     ai = anc[i]; cls = int(ytr_aug[ai])

#                     cand = None

#                     if donors_tr_aug is not None and idx_by_label_donor is not None:

#                         d_anc = str(donors_tr_aug[ai]); packs=[]

#                         for (c,d), ids in idx_by_label_donor.items():

#                             if c==cls and d!=d_anc: packs.append(ids)

#                         if packs: cand = np.concatenate(packs)

#                     if cand is None or cand.size==0:

#                         cand = idx_by_label.get(cls, np.array([], np.int64))

#                         cand = cand[cand!=ai]

#                     if cand.size==0: continue

#                     take = min(int(proto_pos_pool), cand.size)

#                     cand = rng_e.choice(cand, size=take, replace=False)

#                     r_ai = Rtr[ai:ai+1]

#                     dist = ((Rtr[cand]-r_ai)**2).sum(axis=1)

#                     j = int(np.argmin(dist))

#                     a_pos.append(int(ai)); p_pos.append(int(cand[j]))

#             for i in seg_rest:

#                 ai = anc[i]; pi = _pos_semihard(ai, rng_e)

#                 if pi is None: continue

#                 a_pos.append(int(ai)); p_pos.append(int(pi))



#             a_pos = np.asarray(a_pos, np.int64); p_pos = np.asarray(p_pos, np.int64)

#             _log(f"[TrainPairs] epoch={epoch} built pos={len(a_pos)} (simple={n_simple}, proto={n_proto}, rest={n_rest})")

#             return a_pos, p_pos, anc



#         def _update_cache_global(epoch:int, anchors: np.ndarray):

#             if sampler_mode == "simple": return

#             if (epoch % max(1,int(probe_interval))) != 0: return

#             if anchors.size == 0: return

#             rng_local = np.random.RandomState(random_state+31*epoch)

#             sel = rng_local.choice(anchors, size=max(1, int(round(global_probe_frac*anchors.size))), replace=False)

#             _log(f"[Probe] epoch={epoch} anchors={len(sel)} cand={global_probe_candidates} topk={global_probe_topk}")

#             Bc = max(1, min(128, len(sel))); Cc = max(64, min(global_probe_candidates, 4096))

#             for s in tqdm(range(0, len(sel), Bc), leave=False, desc="[Probe]"):

#                 e = min(len(sel), s+Bc)

#                 anc_ids = sel[s:e]

#                 xa = Xtr_t.index_select(0, torch.from_numpy(anc_ids).to(device))

#                 for i, anc in enumerate(anc_ids):

#                     neg_pool = np.where(ytr_aug != ytr_aug[anc])[0]

#                     if neg_pool.size == 0: continue

#                     cand = rng_local.choice(neg_pool, size=min(Cc, neg_pool.size), replace=False)

#                     logits_all = []

#                     chunk = 512

#                     for c0 in range(0, cand.size, chunk):

#                         c1 = min(cand.size, c0+chunk)

#                         xb = Xtr_t.index_select(0, torch.from_numpy(cand[c0:c1]).to(device)).unsqueeze(0)

#                         lg = model.forward_scores(xa[i:i+1], xb, amp_dtype=amp_dtype).squeeze(0).detach().cpu().to(torch.float32).numpy()

#                         logits_all.append(lg); del xb

#                     lg = np.concatenate(logits_all, axis=0)

#                     if lg.size == 0: continue

#                     topk = min(int(global_probe_topk), lg.size)

#                     part = np.argpartition(-lg, kth=topk-1)[:topk]

#                     hard_ids = cand[part]

#                     hard_cache[int(anc)] = (hard_ids.astype(np.int64), int(4))

#                 del xa



#         # ===== Training Step =====

#         _decay_cache()

#         a_tr, p_tr, anc_all = _refresh_train_pairs(ep)

#         _update_cache_global(ep, anchors=np.unique(anc_all))



#         ya_all = ytr_aug[a_tr]; classes = np.unique(ya_all)

#         C_each = min(10, len(classes)); M_per = max(2, B_target // max(1, C_each))

#         steps_per_epoch = max(1, int(math.ceil(len(a_tr)/max(1,C_each*M_per))))

#         rng_batch = np.random.RandomState(random_state+42+ep)

#         _log(f"[BatchPlan] ep={ep} steps={steps_per_epoch} | C_each={C_each} | M_per={M_per} | B≈{C_each*M_per}")



#         def _plan_batches(a_tr, y_arr, C_each, M_per, rng):

#             ya = y_arr[a_tr]; cls = np.unique(ya)

#             by_cls = {c: np.where(ya==c)[0] for c in cls}

#             while True:

#                 if len(cls) == 0:

#                     # Defense: no class to select

#                     yield np.array([], dtype=np.int64)

#                 choose = rng.choice(cls, size=min(C_each, len(cls)), replace=(len(cls)<C_each))

#                 sel=[]

#                 for c in choose:

#                     pool = by_cls[c]

#                     if len(pool)==0: continue

#                     take = rng.choice(pool, size=M_per, replace=(len(pool)<M_per))

#                     sel.append(take)

#                 if not sel: continue

#                 sel = np.concatenate(sel)

#                 if len(np.unique(ya[sel])) < 2: continue

#                 yield sel



#         batch_iter = _plan_batches(a_tr, ytr_aug, C_each, M_per, rng_batch)



#         # Margin curriculum

#         if ep <= int(margin_warmup_epochs): warm = float(ep) / max(1.0, float(margin_warmup_epochs))

#         else: warm = 1.0

#         m_rank_now  = float(rank_margin) * warm

#         m_self_now  = float(self_margin) * warm



#         # Noise

#         if (noise_on_train and (noise_mode=='feature_jitter') and (noise_sigma0>0)):

#             frac = min(1.0, ep / max(1, epochs))

#             sigma_now = float(noise_sigma0) * (math.cos(0.5*math.pi*frac) ** float(noise_gamma))

#         else:

#             sigma_now = 0.0



#         agg = dict(bce=0.0, rank=0.0, self_rank=0.0, tri_active=0.0, n=0)

#         for _ in tqdm(range(steps_per_epoch), leave=False, desc=f"[SymRel] Epoch {ep}/{epochs}"):

#             sel = next(batch_iter)

#             if sel.size == 0:

#                 continue

#             a_batch = a_tr[sel]; p_batch = p_tr[sel]; B = a_batch.shape[0]



#             # Negatives: cached hard negatives first -> semi-hard / rule-based fallback

#             n_batch = np.zeros(B, dtype=np.int64)

#             for i in range(B):

#                 anc = int(a_batch[i])

#                 use_cache = (np.random.rand() < float(neg_cache_pick_p)) and (anc in hard_cache) and (hard_cache[anc][0].size>0)

#                 if use_cache:

#                     ids, ttl = hard_cache[anc]

#                     n_batch[i] = int(ids[np.random.randint(0, ids.size)])

#                     continue

#                 ni = _neg_semihard(anc, np.random)

#                 if ni is None:

#                     ni = _sample_neg_simple(np.random, anc, ytr_aug, donors_tr_aug, idx_by_label, idx_by_label_donor)

#                     if ni is None:

#                         pool = np.where(ytr_aug!=ytr_aug[anc])[0]

#                         ni = int(np.random.choice(pool)) if pool.size>0 else int(a_batch[i])

#                 n_batch[i] = int(ni)



#             xa = Xtr_t.index_select(0, torch.from_numpy(a_batch).to(device))

#             xp = Xtr_t.index_select(0, torch.from_numpy(p_batch).to(device))

#             xn = Xtr_t.index_select(0, torch.from_numpy(n_batch).to(device))

#             xa = _ensure_finite_t(xa, 1e6); xp = _ensure_finite_t(xp, 1e6); xn = _ensure_finite_t(xn, 1e6)



#             if sigma_now > 0.0:

#                 eps_a = torch.randn_like(xa) * sigma_now

#                 eps_p = torch.randn_like(xp) * sigma_now

#                 eps_n = torch.randn_like(xn) * sigma_now

#                 if noise_shared_rho > 0:

#                     rho = float(noise_shared_rho)

#                     shared = torch.randn((xa.size(0), 1), device=xa.device)

#                     eps_a = (math.sqrt(rho)*shared + math.sqrt(1-rho)*eps_a)

#                     eps_p = (math.sqrt(rho)*shared + math.sqrt(1-rho)*eps_p)

#                     eps_n = (math.sqrt(rho)*shared + math.sqrt(1-rho)*eps_n)

#                 xa = _ensure_finite_t(xa + eps_a, 1e6)

#                 xp = _ensure_finite_t(xp + eps_p, 1e6)

#                 xn = _ensure_finite_t(xn + eps_n, 1e6)



#             log_ap = model.forward_pair_logits(xa, xp, amp_dtype=amp_dtype)

#             log_an = model.forward_pair_logits(xa, xn, amp_dtype=amp_dtype)

#             log_pn = model.forward_pair_logits(xp, xn, amp_dtype=amp_dtype)



#             L_rank = softplus(m_rank_now - (log_ap - log_an)).mean() + softplus(m_rank_now - (log_ap - log_pn)).mean()



#             if lambda_bce > 0.0:

#                 labels_ap = torch.full_like(log_ap, fill_value=pos_smooth)

#                 labels_an = torch.full_like(log_an, fill_value=neg_smooth)

#                 labels_pn = torch.full_like(log_pn, fill_value=neg_smooth)

#                 pos_cnt = B; neg_cnt = 2*B

#                 pos_weight = torch.tensor(float(neg_cnt)/max(1.0,float(pos_cnt)), device=device)

#                 def _bce(logit, lab):

#                     loss = bce_logits(logit, lab)

#                     w = torch.where(lab>0.5, pos_weight, torch.tensor(1.0, device=device))

#                     return (loss * w).mean()

#                 L_bce = _bce(log_ap, labels_ap) + _bce(log_an, labels_an) + _bce(log_pn, labels_pn)

#             else:

#                 L_bce = torch.tensor(0.0, device=device)



#             n_self = int(max(1, round(self_pair_frac * B)))

#             idx_all = torch.randperm(B, device=device)[:n_self]

#             xs = xa.index_select(0, idx_all)

#             log_self = model.forward_pair_logits(xs, xs, amp_dtype=amp_dtype)

#             L_self_rank = 0.5*( softplus(m_self_now - (log_self - log_ap.index_select(0, idx_all))).mean()

#                                +softplus(m_self_now - (log_self - log_an.index_select(0, idx_all))).mean() )



#             loss = lambda_rank*L_rank + lambda_bce*L_bce + lambda_self_rank*L_self_rank

#             if not torch.isfinite(loss):

#                 _log("[Warn] non-finite loss encountered; skipping this step."); continue



#             opt.zero_grad(set_to_none=True)

#             scaler.scale(loss).backward()

#             nn.utils.clip_grad_norm_(model.parameters(), 5.0)

#             scaler.step(opt); scaler.update()



#             s_ap = _sigmoid(log_ap); s_an = _sigmoid(log_an); s_pn = _sigmoid(log_pn)

#             tri_raw = (1.0 - s_ap) + (1.0 - s_pn) - (1.0 - s_an)

#             tri_active = (tri_raw > 0).float().mean()



#             agg['bce'] += float(L_bce.detach().cpu())

#             agg['rank'] += float(L_rank.detach().cpu())

#             agg['self_rank'] += float(L_self_rank.detach().cpu())

#             agg['tri_active'] += float(tri_active.detach().cpu())

#             agg['n'] += 1



#         tr_loss = (lambda_rank*agg['rank'] + lambda_bce*agg['bce'] + lambda_self_rank*agg['self_rank'])/max(1,agg['n'])



#         # ===== Validation / Best Model Save =====

#         if (monitor == 'dyn_val_auc') and (not no_val) and Xva_t is not None:

#             if (ep==1) or (ep % max(1,int(val_refresh_every)) == 0):

#                 a_va, b_va, l_va = _refresh_val_pairs(epoch=ep)

#             cur_loss, cur_auc, _ = _eval_pairs(model, Xva_t, a_va, b_va, l_va, device, amp_dtype=amp_dtype)

#             if (cur_auc > best_auc + 2*float(val_eps)) or (best_state_cpu is None):

#                 old = best_auc; best_auc = float(cur_auc)

#                 best_state_cpu = {k:v.detach().cpu() for k,v in model.state_dict().items()}

#                 z['model'] = model

#                 _log(f"[Best] AUC {old:.4f} -> {best_auc:.4f}")

#                 out_path = save_best_model_path or "cynb_best.zip"

#                 save_model(adata, out_path)

#             mon_auc = cur_auc; val_loss = cur_loss

#         elif (monitor == 'val_auc') and (not no_val) and Xva_t is not None:

#             val_loss, cur_auc, _ = _eval_pairs(model, Xva_t, a_va, b_va, l_va, device, amp_dtype=amp_dtype)

#             if (cur_auc > best_auc + 1e-6) or (not np.isfinite(best_auc)):

#                 old = best_auc; best_auc = float(cur_auc)

#                 best_state_cpu = {k:v.detach().cpu() for k,v in model.state_dict().items()}

#                 z['model'] = model

#                 _log(f"[Best] AUC {old:.4f} -> {best_auc:.4f}")

#                 out_path = save_best_model_path or "cynb_best.zip"

#                 save_model(adata, out_path)

#             mon_auc = cur_auc

#         else:

#             mon_auc = float("nan"); val_loss = float("nan")



#         _log(f"  Train {tr_loss:.4f} | ValLoss {val_loss:.4f} | MonAUC {best_auc:.4f} | "

#              f"time {time.time()-t0:.2f}s")



#         hist['train_loss'].append(tr_loss); hist['val_loss'].append(val_loss); hist['val_auc'].append(best_auc)

#         hist['loss_bce'].append(agg['bce']/max(1,agg['n'])); hist['loss_rank'].append(agg['rank']/max(1,agg['n']))

#         hist['loss_self_rank'].append(agg['self_rank']/max(1,agg['n']))

#         hist['triplet_active_ratio'].append(agg['tri_active']/max(1,agg['n']))



#         if lr_sched == "cosine":

#             if ep > warmup_epochs: scheduler.step()

#         elif lr_sched == "plateau":

#             scheduler.step(best_auc)



#         if device.type=='cuda':

#             torch.cuda.empty_cache()



#     # Finalize: restore/save best model

#     if best_state_cpu is not None:

#         model.load_state_dict(best_state_cpu, strict=True); z['model']=model

#         _log(f"Loaded best model (AUC={best_auc:.4f}).")

#         out_path = save_best_model_path or "cynb_best.zip"

#         save_model(adata, out_path)

#     else:

#         z['model']=model; _log("Warning: using last state.")

#     z['history']=hist

#     if save_ckpt_dir: save_model(adata, os.path.join(save_ckpt_dir, f"last_epoch.zip"))

#     z.pop('_train_X', None); z.pop('_train_y', None)

#     if device.type=='cuda':

#         gc.collect(); torch.cuda.empty_cache()

#     return adata





# ---------------- Main Training API (Integrated Dynamic Synthesis + Early Stopping) ----------------

def fit(adata: ad.AnnData,

        preproc: str = "auto",

        validation_split: Union[float,str,None]='auto',

        monitor: str = "dyn_val_auc",

        random_state: int = 0,

        use_cuda: bool = True,

        use_amp: bool = True,

        amp_dtype: str = "bf16",

        torch_matmul_precision: str = "high",

        preload_to_gpu: bool = True,

        d_model: int = 1024,

        depth: int = 6,

        hidden: int = 2048,

        dropout: float = 0.1,

        diff_mode: str = "abs",

        ann_ratio: float = 0.08,

        ann_dim_min: int = 256,

        ann_dim_max: int = 1024,

        sampler_mode: str = "mixed",

        simple_frac: float = 0.50,

        proto_frac: float  = 0.20,

        global_frac: float = 0.30,

        train_pairs_per_epoch: int = 6000,

        proto_per_class: int = 2,

        proto_update_every: int = 3,

        proto_pos_pool: int = 512,

        global_probe_frac: float = 0.30,

        global_probe_candidates: int = 4096,

        global_probe_topk: int = 32,

        probe_interval: int = 1,

        neg_cache_pick_p: float = 0.90,

        # Dynamic synthesis

        synth_enable: bool = True,

        synth_refresh_every: int = 1,

        synth_mult_per_dt: float = 1.2,

        synth_cap_per_dt: int = 600,

        synth_overgen_factor: float = 1.5,

        synth_ratio_range: Tuple[float,float]=(0.35,0.65),

        synth_p_three_parent: float = 0.25,

        synth_parts_three: Tuple[float,float,float]=(0.3,0.4,0.3),

        synth_alpha: float = 1.0,

        synth_gamma: float = 1.0,

        anchor_real_only: bool = True,

        pos_semihard_q: Tuple[float,float]=(0.6, 0.9),

        rneg_k: int = 64,

        epochs: int = 100,

        lr: float = 1e-3,

        weight_decay: float = 1e-4,

        lr_sched: str = "cosine",

        warmup_ratio: float = 0.05,

        plateau_factor: float = 0.5,

        plateau_patience: int = 8,

        lambda_bce: float = 0.4,

        label_smoothing: float = 0.0,

        lambda_rank: float = 1.0,

        rank_margin: float = 0.5,

        self_pair_frac: float = 0.50,

        lambda_self_rank: float = 0.2,

        self_margin: float = 0.5,

        margin_warmup_epochs: int = 6,

        noise_mode: str = "feature_jitter",

        noise_sigma0: float = 0.03,

        noise_gamma: float = 1.5,

        noise_shared_rho: float = 0.0,

        noise_on_train: bool = True,

        val_refresh_every: int = 1,

        val_window: int = 3,

        val_eps: float = 1e-3,

        val_pairs_max: int = 60_000,

        val_global_probe_frac: float = 0.05,

        val_global_candidates: int = 512,

        val_global_topk: int = 8,

        gpu_mem_budget_mb: float = 2000.0,

        batch_size_override: Optional[int] = None,

        save_best_model_path: Optional[str] = None,

        save_ckpt_dir: Optional[str] = None,

        # --- Early stopping parameters ---

        early_stopping_patience: Optional[int] = 10,  # Stop after N epochs without improvement; None disables

        early_stopping_monitor: str = "dyn_val_auc", # Monitoring metric: dyn_val_auc, val_auc, or val_loss

        early_stopping_min_delta: float = 1e-4      # Minimum change to qualify as an improvement

        ) -> ad.AnnData:



    _init_store(adata)

    z = adata.uns['cytoneighbor']

    X_all = z.get('_train_X', None); y_raw = z.get('_train_y', None)

    if X_all is None or y_raw is None: raise ValueError("Please call load_data() first.")



    feats = list(adata.var_names); z['feature_names'] = feats

    _log(f"Features: {len(feats)} genes (v7: no HVG/Scaler)")



    try: torch.set_float32_matmul_precision(str(torch_matmul_precision))

    except Exception: pass



    cats = pd.Index(pd.Series(y_raw).astype('category').cat.categories)

    lab2id = {c:i for i,c in enumerate(cats)}

    y_all = np.array([lab2id[v] for v in y_raw], dtype=np.int64)

    z['label_mapping'] = {'classes': list(map(str, cats))}



    donors_all, donor_key = _resolve_env(adata); z['env_key'] = donor_key



    X_all = _as_numpy_dense(X_all)

    preproc_mode = preproc if preproc != "auto" else _value_preproc_auto_decide(X_all[:min(10000, X_all.shape[0])])

    X_all = _apply_value_preproc_np(X_all, preproc_mode)

    z['preproc'] = {"mode": str(preproc_mode)}

    X_all = _ensure_finite_np(X_all, 1e6)



    # Split

    if validation_split == 'auto':

        val_ratio = 0.2; no_val = False

    elif validation_split in (None, 0, 0.0, 'none'):

        no_val = True; val_ratio = 0.0

        if monitor == 'dyn_val_auc':

            _log('[Warn] monitor="dyn_val_auc" but no validation set, falling back to "shadow_auc".'); monitor='shadow_auc'

    else:

        no_val = False; val_ratio = float(validation_split)



    if not no_val:

        tr_idx, va_idx = _group_split(y_all, donors_all, float(val_ratio), int(random_state))

        if tr_idx.size==0 or va_idx.size==0: raise RuntimeError("Split failed: training or validation set is empty.")

    else:

        tr_idx = np.arange(X_all.shape[0], dtype=np.int64); va_idx = np.array([], np.int64)

        _log("[Split] Using full dataset for training (no fixed validation set)")



    Xtr_real = X_all[tr_idx]; ytr_real = y_all[tr_idx]

    Xva = X_all[va_idx] if not no_val else None; yva = y_all[va_idx] if not no_val else None

    donors_tr_real = (adata.obs.loc[adata.obs.index[tr_idx], donor_key].astype(str).fillna("NA").to_numpy()

                      if (donor_key is not None) else None)

    donors_va = (adata.obs.loc[adata.obs.index[va_idx], donor_key].astype(str).fillna("NA").to_numpy()

                 if ((not no_val) and (donor_key is not None)) else None)



    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    _log(f"[Device] using {device.type.upper()}")

    _set_seed(random_state, use_cuda=(device.type=='cuda'), deterministic=False)

    amp_ok = bool(use_amp and device.type=='cuda')

    scaler = _grad_scaler(amp_ok, amp_dtype=amp_dtype)



    d_ann_cand = int(round(max(1, ann_ratio * Xtr_real.shape[1])))

    d_ann = int(min(ann_dim_max, max(ann_dim_min, d_ann_cand)))

    rng = np.random.RandomState(random_state)

    P_ann = np.linalg.qr(rng.normal(size=(d_ann, Xtr_real.shape[1])).astype(np.float32).T)[0].T.astype(np.float32)

    z['ann'] = {"P_ann": P_ann.astype(np.float32), "d_ann": int(d_ann)}

    _log(f"[ANN] d_ann={d_ann} (ratio={ann_ratio}, min={ann_dim_min}, max={ann_dim_max})")



    model = SymRelEnc(n_genes=Xtr_real.shape[1], d_model=d_model, depth=depth, hidden=hidden, dropout=dropout, diff_mode=diff_mode).to(device)

    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)



    if lr_sched == "cosine":

        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs)); warmup_epochs = max(1, int(warmup_ratio * epochs))

    elif lr_sched == "plateau":

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=plateau_factor, patience=plateau_patience, verbose=False); warmup_epochs = 0

    else:

        scheduler = None; warmup_epochs = 0



    if (not no_val) and Xva is not None and Xva.shape[0] > 1:

        Xva = _ensure_finite_np(Xva, 1e6)

        Xva_t = torch.from_numpy(Xva).to(device, non_blocking=True) if (device.type=='cuda' and preload_to_gpu) else torch.from_numpy(Xva).to(device)

    else:

        Xva_t = None



    if batch_size_override is not None:

        B_target = int(max(8, batch_size_override))

    else:

        B_target = int(math.sqrt(max(1, (float(gpu_mem_budget_mb)*1024*1024)/(Xtr_real.shape[1]*4.0))))

        B_target = max(128, min(4096, B_target))



    hist={'train_loss': [], 'val_loss': [], 'val_auc': [], 'monitor': str(monitor),

          'loss_bce': [], 'loss_rank': [], 'loss_self_rank': [],

          'triplet_active_ratio': [],

          'val_refresh_every': int(val_refresh_every), 'val_window': int(val_window), 'val_eps': float(val_eps),

          'diff_mode': str(diff_mode), 'sampler_mode': str(sampler_mode),

          'synth': {'enable': bool(synth_enable)}}

    best_auc = -np.inf; best_state_cpu = None



    # --- Early stopping state initialization ---

    patience_counter = 0

    # If monitoring loss, initialize best value to infinity

    best_metric_for_es = -np.inf if "auc" in early_stopping_monitor else np.inf

    # If no validation set, disable early stopping

    if no_val and early_stopping_patience is not None:

        _log("[Warn] No validation set; early stopping automatically disabled.")

        early_stopping_patience = None

    # --- End of early stopping initialization ---





    # Validation pairs (adaptive cap)

    def _refresh_val_pairs(epoch:int):

        if no_val or Xva is None or Xva.shape[0] < 2:

            return None

        rng_e = np.random.RandomState(random_state + 101*epoch + 7)

        idx_by_label_v, idx_by_label_donor_v, _ = _build_label_donor_index(yva, donors_va)

        n_val = int(Xva.shape[0]); C = int(len(idx_by_label_v))

        N_max = int(min(int(val_pairs_max), 16 * max(1,n_val) * max(1,C)))

        a_list=[]; b_list=[]; l_list=[]

        while len(l_list) < N_max:

            classes, counts = np.unique(yva, return_counts=True)

            p = counts / counts.sum()

            c = int(rng_e.choice(classes, p=p))

            ids = idx_by_label_v.get(int(c), np.array([], np.int64))

            if ids.size<=1: continue

            ai = int(rng_e.choice(ids))

            pi = _sample_pos_simple(rng_e, ai, yva, donors_va, idx_by_label_v, idx_by_label_donor_v)

            if pi is not None:

                a_list.append(ai); b_list.append(pi); l_list.append(1)

            ni = _sample_neg_simple(rng_e, ai, yva, donors_va, idx_by_label_v, idx_by_label_donor_v)

            if ni is not None:

                a_list.append(ai); b_list.append(ni); l_list.append(0)

        a_va = np.asarray(a_list, np.int64); b_va = np.asarray(b_list, np.int64); l_va = np.asarray(l_list, np.int64)

        _log(f"[ValPairs] epoch={epoch} built={len(l_va)} (pos={(l_va==1).sum()}, neg={(l_va==0).sum()})")

        return a_va, b_va, l_va



    a_va = b_va = l_va = None

    if (not no_val) and Xva_t is not None:

        cur_val = _refresh_val_pairs(epoch=1)

        if cur_val is None: raise RuntimeError("Validation set too small; cannot construct validation pairs.")

        a_va, b_va, l_va = cur_val



    _log("==================== Training: SymRelEnc (+ Dynamic Gene-CutMix) ====================")

    bce_logits = nn.BCEWithLogitsLoss(reduction='none')

    pos_smooth = float(1.0 - label_smoothing); neg_smooth = float(label_smoothing)

    softplus = nn.Softplus()

    hard_cache: Dict[int, Tuple[np.ndarray,int]] = {}



    def _decay_cache():

        drop=[]

        for k,(ids,ttl) in list(hard_cache.items()):

            ttl -= 1

            if ttl <= 0: drop.append(k)

            else: hard_cache[k]=(ids,ttl)

        for k in drop: hard_cache.pop(k, None)



    # ====== Training Loop ======

    for ep in range(1, epochs+1):

        t0=time.time(); model.train()

        if lr_sched=="cosine" and ep <= warmup_epochs:

            for g in opt.param_groups: g['lr'] = lr * ep / max(1,warmup_epochs)



        # Refresh synthesis pool

        if synth_enable and (donors_tr_real is not None) and ((ep==1) or (ep % max(1,int(synth_refresh_every)) == 0)):

            Xtr_aug, ytr_aug, donors_tr_aug, is_synth, info = build_synth_pool(

                Xtr_real, ytr_real, donors_tr_real, P_ann,

                mult_per_dt=synth_mult_per_dt, cap_per_dt=synth_cap_per_dt,

                overgen_factor=synth_overgen_factor, ratio_range=synth_ratio_range,

                p_three_parent=synth_p_three_parent, parts_three=synth_parts_three,

                alpha=synth_alpha, gamma=synth_gamma, random_state=int(random_state+991*ep)

            )

            z['synth'] = info; hard_cache.clear()

        else:

            Xtr_aug, ytr_aug = _ensure_finite_np(Xtr_real, 1e6), ytr_real

            donors_tr_aug = donors_tr_real

            is_synth = np.zeros(Xtr_aug.shape[0], np.int8)



        Xtr_t = torch.from_numpy(Xtr_aug).to(device, non_blocking=True) if (device.type=='cuda' and preload_to_gpu) else torch.from_numpy(Xtr_aug).to(device)

        Rtr = _ann_rep(Xtr_aug, P_ann)



        # Donor index

        idx_by_label, idx_by_label_donor, _ = _build_label_donor_index(ytr_aug, donors_tr_aug)



        # Prototypes

        def _refresh_prototypes(epoch:int):

            if sampler_mode == "simple": return (None, None)

            if (epoch==1) or (epoch % max(1,int(proto_update_every)) == 0):

                Pp, Pc = _build_prototypes_R(Rtr, ytr_aug, per_class=int(proto_per_class), seed=int(random_state+epoch*7))

                _log(f"[Proto] built {len(Pc)} prototypes (per_class={proto_per_class})")

                return (Pp, Pc)

            return (proto_P, proto_cls)

        proto_P, proto_cls = _refresh_prototypes(ep)



        # Positives: semi-hard (R-space 0.6-0.9 quantile; fallback when candidates are few)

        def _pos_semihard(ai: int, rng_e: np.random.RandomState) -> Optional[int]:

            cls = int(ytr_aug[ai])

            # Candidates: cross-donor same-type -> fallback global same-type

            cand = None

            if donors_tr_aug is not None and idx_by_label_donor is not None:

                d_anc = str(donors_tr_aug[ai]); packs=[]

                for (c,d), ids in idx_by_label_donor.items():

                    if c==cls and d!=d_anc and ids.size>0:

                        packs.append(ids)

                if packs: cand = np.concatenate(packs)

            if cand is None or cand.size==0:

                cand = idx_by_label.get(cls, np.array([], np.int64))

            cand = cand[cand!=ai]

            if cand.size==0: return None



            if cand.size < 4:

                return int(rng_e.choice(cand))  # Too few candidates, choose randomly



            r_ai = Rtr[ai:ai+1]; dist = ((Rtr[cand]-r_ai)**2).sum(axis=1)

            lo_q, hi_q = pos_semihard_q

            lo = float(np.quantile(dist, lo_q)); hi = float(np.quantile(dist, hi_q))

            mask = (dist >= lo) & (dist <= hi)

            pool = cand[mask] if mask.any() else cand

            return int(rng_e.choice(pool))



        # Negatives: semi-hard (same-donor other-type nearest few), fallback when few candidates

        def _neg_semihard(ai: int, rng_e: np.random.RandomState) -> Optional[int]:

            cls = int(ytr_aug[ai]); r_ai = Rtr[ai:ai+1]

            # Same-donor other-type

            if donors_tr_aug is not None:

                ids = np.where((donors_tr_aug == donors_tr_aug[ai]) & (ytr_aug != cls))[0]

            else:

                ids = np.where(ytr_aug != cls)[0]

            if ids.size == 0:

                return None

            take = min(int(rneg_k), ids.size)

            if take <= 1:

                return int(rng_e.choice(ids))

            dist = ((Rtr[ids]-r_ai)**2).sum(axis=1)

            part = np.argpartition(dist, kth=take-1)[:take]

            cand = ids[part]

            return int(rng_e.choice(cand)) if cand.size>0 else None



        # Anchor: real samples only

        if anchor_real_only:

            real_idx = np.where(is_synth==0)[0]

            if real_idx.size == 0: real_idx = np.arange(ytr_aug.size)

        else:

            real_idx = np.arange(ytr_aug.size)



        # Adapt number of training pairs

        max_pairs = int(16 * max(1, real_idx.size))

        N_pairs_target = int(min(int(train_pairs_per_epoch), max_pairs))



        def _refresh_train_pairs(epoch:int):

            rng_e = np.random.RandomState(random_state + 17*epoch)

            classes, counts = np.unique(ytr_aug[real_idx], return_counts=True)

            p = counts / counts.sum() if counts.sum() > 0 else np.ones_like(counts)/max(1,len(counts))

            by_cls_real = {c: real_idx[ytr_aug[real_idx]==c] for c in classes}

            anc=[]

            while len(anc) < N_pairs_target and len(classes)>0:

                c = rng_e.choice(classes, p=p)

                ids = by_cls_real[c]

                if ids.size <= 1:

                    # Avoid infinite loop: exclude this class

                    mask = classes != c

                    classes = classes[mask]; p = p[mask];

                    if p.sum() > 0: p = p/p.sum()

                    continue

                anc.append(int(rng_e.choice(ids)))

            if len(anc) == 0:

                raise RuntimeError("Training pair sampling failed: check if the training set is too small.")

            anc = np.asarray(anc, np.int64)



            n_simple = int(round(simple_frac * N_pairs_target)) if sampler_mode!="simple" else N_pairs_target

            n_proto  = int(round(proto_frac  * N_pairs_target)) if sampler_mode!="simple" else 0

            n_rest   = N_pairs_target - n_simple - n_proto

            if n_rest < 0: n_rest = 0



            order = rng_e.permutation(N_pairs_target)

            seg_simple = order[:n_simple]; seg_proto  = order[n_simple:n_simple+n_proto]; seg_rest   = order[n_simple+n_proto:]



            a_pos=[]; p_pos=[]

            for i in seg_simple:

                ai = anc[i]; pi = _pos_semihard(ai, rng_e)

                if pi is None: continue

                a_pos.append(int(ai)); p_pos.append(int(pi))

            if sampler_mode!="simple" and proto_P is not None and proto_cls is not None and len(proto_cls)>0:

                for i in seg_proto:

                    ai = anc[i]; cls = int(ytr_aug[ai])

                    cand = None

                    if donors_tr_aug is not None and idx_by_label_donor is not None:

                        d_anc = str(donors_tr_aug[ai]); packs=[]

                        for (c,d), ids in idx_by_label_donor.items():

                            if c==cls and d!=d_anc: packs.append(ids)

                        if packs: cand = np.concatenate(packs)

                    if cand is None or cand.size==0:

                        cand = idx_by_label.get(cls, np.array([], np.int64))

                        cand = cand[cand!=ai]

                    if cand.size==0: continue

                    take = min(int(proto_pos_pool), cand.size)

                    cand = rng_e.choice(cand, size=take, replace=False)

                    r_ai = Rtr[ai:ai+1]

                    dist = ((Rtr[cand]-r_ai)**2).sum(axis=1)

                    j = int(np.argmin(dist))

                    a_pos.append(int(ai)); p_pos.append(int(cand[j]))

            for i in seg_rest:

                ai = anc[i]; pi = _pos_semihard(ai, rng_e)

                if pi is None: continue

                a_pos.append(int(ai)); p_pos.append(int(pi))



            a_pos = np.asarray(a_pos, np.int64); p_pos = np.asarray(p_pos, np.int64)

            _log(f"[TrainPairs] epoch={ep} built pos={len(a_pos)} (simple={n_simple}, proto={n_proto}, rest={n_rest})")

            return a_pos, p_pos, anc



        def _update_cache_global(epoch:int, anchors: np.ndarray):

            if sampler_mode == "simple": return

            if (epoch % max(1,int(probe_interval))) != 0: return

            if anchors.size == 0: return

            rng_local = np.random.RandomState(random_state+31*epoch)

            sel = rng_local.choice(anchors, size=max(1, int(round(global_probe_frac*anchors.size))), replace=False)

            _log(f"[Probe] epoch={epoch} anchors={len(sel)} cand={global_probe_candidates} topk={global_probe_topk}")

            Bc = max(1, min(128, len(sel))); Cc = max(64, min(global_probe_candidates, 4096))

            for s in tqdm(range(0, len(sel), Bc), leave=False, desc="[Probe]"):

                e = min(len(sel), s+Bc)

                anc_ids = sel[s:e]

                xa = Xtr_t.index_select(0, torch.from_numpy(anc_ids).to(device))

                for i, anc in enumerate(anc_ids):

                    neg_pool = np.where(ytr_aug != ytr_aug[anc])[0]

                    if neg_pool.size == 0: continue

                    cand = rng_local.choice(neg_pool, size=min(Cc, neg_pool.size), replace=False)

                    logits_all = []

                    chunk = 512

                    for c0 in range(0, cand.size, chunk):

                        c1 = min(cand.size, c0+chunk)

                        xb = Xtr_t.index_select(0, torch.from_numpy(cand[c0:c1]).to(device)).unsqueeze(0)

                        lg = model.forward_scores(xa[i:i+1], xb, amp_dtype=amp_dtype).squeeze(0).detach().cpu().to(torch.float32).numpy()

                        logits_all.append(lg); del xb

                    lg = np.concatenate(logits_all, axis=0)

                    if lg.size == 0: continue

                    topk = min(int(global_probe_topk), lg.size)

                    part = np.argpartition(-lg, kth=topk-1)[:topk]

                    hard_ids = cand[part]

                    hard_cache[int(anc)] = (hard_ids.astype(np.int64), int(4))

                del xa



        # ===== Training Step =====

        _decay_cache()

        a_tr, p_tr, anc_all = _refresh_train_pairs(ep)

        _update_cache_global(ep, anchors=np.unique(anc_all))



        ya_all = ytr_aug[a_tr]; classes = np.unique(ya_all)

        C_each = min(10, len(classes)); M_per = max(2, B_target // max(1, C_each))

        steps_per_epoch = max(1, int(math.ceil(len(a_tr)/max(1,C_each*M_per))))

        rng_batch = np.random.RandomState(random_state+42+ep)

        _log(f"[BatchPlan] ep={ep} steps={steps_per_epoch} | C_each={C_each} | M_per={M_per} | B≈{C_each*M_per}")



        def _plan_batches(a_tr, y_arr, C_each, M_per, rng):

            ya = y_arr[a_tr]; cls = np.unique(ya)

            by_cls = {c: np.where(ya==c)[0] for c in cls}

            while True:

                if len(cls) == 0:

                    # Defense: no class to select

                    yield np.array([], dtype=np.int64)

                choose = rng.choice(cls, size=min(C_each, len(cls)), replace=(len(cls)<C_each))

                sel=[]

                for c in choose:

                    pool = by_cls[c]

                    if len(pool)==0: continue

                    take = rng.choice(pool, size=M_per, replace=(len(pool)<M_per))

                    sel.append(take)

                if not sel: continue

                sel = np.concatenate(sel)

                if len(np.unique(ya[sel])) < 2: continue

                yield sel



        batch_iter = _plan_batches(a_tr, ytr_aug, C_each, M_per, rng_batch)



        # Margin curriculum

        if ep <= int(margin_warmup_epochs): warm = float(ep) / max(1.0, float(margin_warmup_epochs))

        else: warm = 1.0

        m_rank_now  = float(rank_margin) * warm

        m_self_now  = float(self_margin) * warm



        # Noise

        if (noise_on_train and (noise_mode=='feature_jitter') and (noise_sigma0>0)):

            frac = min(1.0, ep / max(1, epochs))

            sigma_now = float(noise_sigma0) * (math.cos(0.5*math.pi*frac) ** float(noise_gamma))

        else:

            sigma_now = 0.0



        agg = dict(bce=0.0, rank=0.0, self_rank=0.0, tri_active=0.0, n=0)

        for _ in tqdm(range(steps_per_epoch), leave=False, desc=f"[SymRel] Epoch {ep}/{epochs}"):

            sel = next(batch_iter)

            if sel.size == 0:

                continue

            a_batch = a_tr[sel]; p_batch = p_tr[sel]; B = a_batch.shape[0]



            # Negatives: cached hard negatives first -> semi-hard / rule-based fallback

            n_batch = np.zeros(B, dtype=np.int64)

            for i in range(B):

                anc = int(a_batch[i])

                use_cache = (np.random.rand() < float(neg_cache_pick_p)) and (anc in hard_cache) and (hard_cache[anc][0].size>0)

                if use_cache:

                    ids, ttl = hard_cache[anc]

                    n_batch[i] = int(ids[np.random.randint(0, ids.size)])

                    continue

                ni = _neg_semihard(anc, np.random)

                if ni is None:

                    ni = _sample_neg_simple(np.random, anc, ytr_aug, donors_tr_aug, idx_by_label, idx_by_label_donor)

                    if ni is None:

                        pool = np.where(ytr_aug!=ytr_aug[anc])[0]

                        ni = int(np.random.choice(pool)) if pool.size>0 else int(a_batch[i])

                n_batch[i] = int(ni)



            xa = Xtr_t.index_select(0, torch.from_numpy(a_batch).to(device))

            xp = Xtr_t.index_select(0, torch.from_numpy(p_batch).to(device))

            xn = Xtr_t.index_select(0, torch.from_numpy(n_batch).to(device))

            xa = _ensure_finite_t(xa, 1e6); xp = _ensure_finite_t(xp, 1e6); xn = _ensure_finite_t(xn, 1e6)



            if sigma_now > 0.0:

                eps_a = torch.randn_like(xa) * sigma_now

                eps_p = torch.randn_like(xp) * sigma_now

                eps_n = torch.randn_like(xn) * sigma_now

                if noise_shared_rho > 0:

                    rho = float(noise_shared_rho)

                    shared = torch.randn((xa.size(0), 1), device=xa.device)

                    eps_a = (math.sqrt(rho)*shared + math.sqrt(1-rho)*eps_a)

                    eps_p = (math.sqrt(rho)*shared + math.sqrt(1-rho)*eps_p)

                    eps_n = (math.sqrt(rho)*shared + math.sqrt(1-rho)*eps_n)

                xa = _ensure_finite_t(xa + eps_a, 1e6)

                xp = _ensure_finite_t(xp + eps_p, 1e6)

                xn = _ensure_finite_t(xn + eps_n, 1e6)



            log_ap = model.forward_pair_logits(xa, xp, amp_dtype=amp_dtype)

            log_an = model.forward_pair_logits(xa, xn, amp_dtype=amp_dtype)

            log_pn = model.forward_pair_logits(xp, xn, amp_dtype=amp_dtype)



            L_rank = softplus(m_rank_now - (log_ap - log_an)).mean() + softplus(m_rank_now - (log_ap - log_pn)).mean()



            if lambda_bce > 0.0:

                labels_ap = torch.full_like(log_ap, fill_value=pos_smooth)

                labels_an = torch.full_like(log_an, fill_value=neg_smooth)

                labels_pn = torch.full_like(log_pn, fill_value=neg_smooth)

                pos_cnt = B; neg_cnt = 2*B

                pos_weight = torch.tensor(float(neg_cnt)/max(1.0,float(pos_cnt)), device=device)

                def _bce(logit, lab):

                    loss = bce_logits(logit, lab)

                    w = torch.where(lab>0.5, pos_weight, torch.tensor(1.0, device=device))

                    return (loss * w).mean()

                L_bce = _bce(log_ap, labels_ap) + _bce(log_an, labels_an) + _bce(log_pn, labels_pn)

            else:

                L_bce = torch.tensor(0.0, device=device)



            n_self = int(max(1, round(self_pair_frac * B)))

            idx_all = torch.randperm(B, device=device)[:n_self]

            xs = xa.index_select(0, idx_all)

            log_self = model.forward_pair_logits(xs, xs, amp_dtype=amp_dtype)

            L_self_rank = 0.5*( softplus(m_self_now - (log_self - log_ap.index_select(0, idx_all))).mean()

                               +softplus(m_self_now - (log_self - log_an.index_select(0, idx_all))).mean() )



            loss = lambda_rank*L_rank + lambda_bce*L_bce + lambda_self_rank*L_self_rank

            if not torch.isfinite(loss):

                _log("[Warn] non-finite loss encountered; skipping this step."); continue



            opt.zero_grad(set_to_none=True)

            scaler.scale(loss).backward()

            nn.utils.clip_grad_norm_(model.parameters(), 5.0)

            scaler.step(opt); scaler.update()



            s_ap = _sigmoid(log_ap); s_an = _sigmoid(log_an); s_pn = _sigmoid(log_pn)

            tri_raw = (1.0 - s_ap) + (1.0 - s_pn) - (1.0 - s_an)

            tri_active = (tri_raw > 0).float().mean()



            agg['bce'] += float(L_bce.detach().cpu())

            agg['rank'] += float(L_rank.detach().cpu())

            agg['self_rank'] += float(L_self_rank.detach().cpu())

            agg['tri_active'] += float(tri_active.detach().cpu())

            agg['n'] += 1



        tr_loss = (lambda_rank*agg['rank'] + lambda_bce*agg['bce'] + lambda_self_rank*agg['self_rank'])/max(1,agg['n'])



        # ===== Validation / Best Model Save / Early Stopping =====

        # Initialize monitoring metric for this round

        current_mon_metric = -np.inf if "auc" in early_stopping_monitor else np.inf



        if (monitor == 'dyn_val_auc') and (not no_val) and Xva_t is not None:

            if (ep==1) or (ep % max(1,int(val_refresh_every)) == 0):

                a_va, b_va, l_va = _refresh_val_pairs(epoch=ep)

            cur_loss, cur_auc, _ = _eval_pairs(model, Xva_t, a_va, b_va, l_va, device, amp_dtype=amp_dtype)

            if (cur_auc > best_auc + 2*float(val_eps)) or (best_state_cpu is None):

                old = best_auc; best_auc = float(cur_auc)

                best_state_cpu = {k:v.detach().cpu() for k,v in model.state_dict().items()}

                z['model'] = model

                _log(f"[Best] AUC {old:.4f} -> {best_auc:.4f}")

                out_path = save_best_model_path or "cynb_best.zip"

                save_model(adata, out_path)

            

            # Update monitoring metric for this round

            if early_stopping_monitor == 'dyn_val_auc': current_mon_metric = cur_auc

            elif early_stopping_monitor == 'val_loss': current_mon_metric = cur_loss

            

            mon_auc = cur_auc; val_loss = cur_loss

        elif (monitor == 'val_auc') and (not no_val) and Xva_t is not None:

            val_loss, cur_auc, _ = _eval_pairs(model, Xva_t, a_va, b_va, l_va, device, amp_dtype=amp_dtype)

            if (cur_auc > best_auc + 1e-6) or (not np.isfinite(best_auc)):

                old = best_auc; best_auc = float(cur_auc)

                best_state_cpu = {k:v.detach().cpu() for k,v in model.state_dict().items()}

                z['model'] = model

                _log(f"[Best] AUC {old:.4f} -> {best_auc:.4f}")

                out_path = save_best_model_path or "cynb_best.zip"

                save_model(adata, out_path)

            

            # Update monitoring metric for this round

            if early_stopping_monitor == 'val_auc': current_mon_metric = cur_auc

            elif early_stopping_monitor == 'val_loss': current_mon_metric = val_loss

            

            mon_auc = cur_auc

        else:

            mon_auc = float("nan"); val_loss = float("nan")



        _log(f"  Train {tr_loss:.4f} | ValLoss {val_loss:.4f} | MonAUC {best_auc:.4f} | "

             f"time {time.time()-t0:.2f}s")



        hist['train_loss'].append(tr_loss); hist['val_loss'].append(val_loss); hist['val_auc'].append(best_auc)

        hist['loss_bce'].append(agg['bce']/max(1,agg['n'])); hist['loss_rank'].append(agg['rank']/max(1,agg['n']))

        hist['loss_self_rank'].append(agg['self_rank']/max(1,agg['n']))

        hist['triplet_active_ratio'].append(agg['tri_active']/max(1,agg['n']))

        

        # --- Early stopping check ---

        if early_stopping_patience is not None:

            improved = False

            # Determine if the metric improved (higher AUC = better, lower Loss = better)

            if "auc" in early_stopping_monitor:

                if current_mon_metric > best_metric_for_es + early_stopping_min_delta:

                    improved = True

            else: # Monitoring loss

                if current_mon_metric < best_metric_for_es - early_stopping_min_delta:

                    improved = True

            

            if improved:

                best_metric_for_es = current_mon_metric

                patience_counter = 0

                _log(f"  [EarlyStopping] Metric improved to {best_metric_for_es:.4f}. Patience reset to 0.")

            else:

                patience_counter += 1

                _log(f"  [EarlyStopping] Metric did not improve. Patience is {patience_counter}/{early_stopping_patience}.")



            if patience_counter >= early_stopping_patience:

                _log(f"--- Early stopping triggered after {ep} epochs. ---")

                break # Early exit from training loop

        # --- End of early stopping ---



        if lr_sched == "cosine":

            if ep > warmup_epochs: scheduler.step()

        elif lr_sched == "plateau":

            scheduler.step(best_auc)



        if device.type=='cuda':

            torch.cuda.empty_cache()



    # Finalize: restore/save best model

    if best_state_cpu is not None:

        model.load_state_dict(best_state_cpu, strict=True); z['model']=model

        _log(f"Loaded best model (AUC={best_auc:.4f}).")

        out_path = save_best_model_path or "cynb_best.zip"

        save_model(adata, out_path)

    else:

        z['model']=model; _log("Warning: using last state.")

    z['history']=hist

    if save_ckpt_dir: save_model(adata, os.path.join(save_ckpt_dir, f"last_epoch.zip"))

    z.pop('_train_X', None); z.pop('_train_y', None)

    if device.type=='cuda':

        gc.collect(); torch.cuda.empty_cache()

    return adata











# ---------------- Transform (Retrieval + Re-ranking) ----------------

def _weights_from_dist(dist: np.ndarray, mode: str="softmax", temperature: float=1.0) -> np.ndarray:

    if mode == "softmax":

        tau = (np.median(dist, axis=1, keepdims=True) + 1e-6) * float(temperature)

        z = np.exp(-dist / tau); z = z / (z.sum(axis=1, keepdims=True) + 1e-8); return z.astype(np.float32)

    elif mode == "inv":

        z = 1.0 / (dist + 1e-6); z = z / (z.sum(axis=1, keepdims=True) + 1e-8); return z.astype(np.float32)

    elif mode == "rank":

        ranks = np.argsort(np.argsort(dist, axis=1), axis=1).astype(np.float32) + 1.0

        z = 1.0 / ranks; z = z / (z.sum(axis=1, keepdims=True) + 1e-8); return z.astype(np.float32)

    else:

        return _weights_from_dist(dist, mode="softmax", temperature=temperature)



def _write_neighbors(adata_out: ad.AnnData, idx: np.ndarray, dist: np.ndarray,

                     neighbors_key="neighbors", random_state=0):

    n = adata_out.n_obs; assert idx.shape == dist.shape

    if (idx < 0).any():

        raise ValueError("KNN index contains negative values.")

    for i in range(n):

        row = idx[i]; drow = dist[i]

        mask = (row != i)

        if not mask.all():

            row2 = row[mask]; d2 = drow[mask]

            if row2.size == 0: raise RuntimeError("KNN row is empty.")

            take = min(len(row2), len(row))

            pad_ids = row2[-1] * np.ones(len(row)-take, dtype=np.int64)

            pad_ds  = np.full(len(row)-take, float(d2[-1] if d2.size>0 else 1.0), dtype=np.float32)

            idx[i]  = np.concatenate([row2[:take], pad_ids])

            dist[i] = np.concatenate([d2[:take],  pad_ds])

    dist = np.asarray(dist, np.float32)

    dist[~np.isfinite(dist)] = np.float32(1.0)

    dist = np.maximum(dist, 0.0).astype(np.float32)



    rows = np.repeat(np.arange(n), idx.shape[1])

    cols = idx.ravel()

    vals = dist.ravel().astype(np.float32)

    dkey = "distances" if neighbors_key == "neighbors" else f"{neighbors_key}_distances"

    ckey = "connectivities" if neighbors_key == "neighbors" else f"{neighbors_key}_connectivities"

    adata_out.obsp[dkey] = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))



    conn, _, _ = umap.umap_.fuzzy_simplicial_set(

        X=np.zeros((n,1)), n_neighbors=idx.shape[1],

        random_state=np.random.RandomState(random_state),

        metric="precomputed", knn_indices=idx, knn_dists=dist

    )

    conn = conn.tocsr().copy(); conn = conn.maximum(conn.T)

    conn.setdiag(0); conn.eliminate_zeros()

    adata_out.obsp[ckey] = conn

    adata_out.uns[neighbors_key] = {

        "connectivities_key": ckey, "distances_key": dkey,

        "params": {"n_neighbors": int(idx.shape[1]), "method": "umap", "metric": "precomputed",

                   "random_state": int(random_state), "knn": True}

    }



@torch.inference_mode()

def transform(adata: ad.AnnData,

              n_neighbors: int=15,

              k_candidate: int=500,

              neighbors_key="neighbors",

              use_cuda: bool=True,

              batch_q: int=256,

              random_state: int=0,

              copy: bool=False,

              rerank_chunk: int=128,

              use_amp: bool = True,

              amp_dtype: str = "auto",

              preload_to_gpu: bool = False,

              keep_model_on_device: Optional[bool] = None,

              pin_memory: bool = True,

              efsearch_override: Optional[int] = None,

              dist_from: str = "logit_softplus",     # "logit_softplus" | "sigmoid_1minus"

              dist_temperature: float = 1.0,

              dist_row_normalize: bool = True):

    ad_out = adata.copy() if copy else adata

    z = ad_out.uns.get('cytoneighbor', {})

    model: SymRelEnc = z.get('model', None)

    feats: List[str] = z.get('feature_names', None)

    ann = z.get('ann', {})

    if model is None or feats is None:

        raise ValueError("Please load/train a model first.")

    if ann.get("P_ann", None) is None:

        raise RuntimeError("Missing ANN projection matrix.")

    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")



    def _align_features(adata_like: ad.AnnData, feats: List[str], verbose=True) -> np.ndarray:

        var = adata_like.var_names

        idx = var.get_indexer(feats); present = idx >= 0

        X = adata_like.X; n = adata_like.n_obs

        out = np.zeros((n, len(feats)), dtype=np.float32)

        if np.any(present):

            if sp.issparse(X): out[:, present] = X[:, idx[present]].toarray().astype(np.float32)

            else: out[:, present] = np.asarray(X)[:, idx[present]].astype(np.float32)

        miss_cnt = int((~present).sum())

        if miss_cnt and verbose: _log(f"[Align] Missing {miss_cnt} features -> filled with 0.")

        return out



    X = _align_features(ad_out, feats, verbose=True)

    preproc_mode = z.get('preproc', {"mode":"none"}).get("mode", "none")

    X = _apply_value_preproc_np(X, preproc_mode)

    X = _ensure_finite_np(X, clip_abs=1e6)

    N, G = X.shape

    if N <= 1: raise RuntimeError("transform: dataset has only one cell; cannot build graph.")



    P_ann = ann["P_ann"].astype(np.float32)

    R = _ann_rep(X, P_ann)



    k_candidate = int(min(max(1, k_candidate), max(1, N-1)))

    index = _faiss_hnsw_build(R)

    try:

        if efsearch_override is not None:

            index.hnsw.efSearch = int(efsearch_override)

        else:

            index.hnsw.efSearch = int(max(128, min(4*int(k_candidate), 1024)))

    except Exception:

        pass



    I = _faiss_search(index, R, k_candidate + 1)



    # Distance mapping

    def _logits_to_dist_np(z: np.ndarray) -> np.ndarray:

        if dist_from == "logit_softplus":

            az = np.abs(z, dtype=np.float32)

            d = np.maximum(-z, 0, dtype=np.float32) + np.log1p(np.exp(-az, dtype=np.float32))

            return d.astype(np.float32)

        elif dist_from == "sigmoid_1minus":

            s = 1.0 / (1.0 + np.exp(-z, dtype=np.float32))

            return (1.0 - s).astype(np.float32)

        else:

            raise ValueError("dist_from must be 'logit_softplus' or 'sigmoid_1minus'.")



    try:

        try:

            any_param = next(model.parameters())

            orig_device = any_param.device

        except StopIteration:

            orig_device = torch.device('cpu')

        model.to(device).eval()



        if preload_to_gpu and device.type == 'cuda':

            X_host = torch.from_numpy(X).pin_memory() if pin_memory else torch.from_numpy(X)

            X_store = X_host.to(device, non_blocking=True); X_is_gpu = True

        else:

            X_store = torch.from_numpy(X)

            if pin_memory and device.type == 'cuda':

                try: X_store = X_store.pin_memory()

                except Exception: pass

            X_is_gpu = False



        acc_idx, acc_dst = [[] for _ in range(N)], [[] for _ in range(N)]

        B = max(1, int(batch_q))

        rerank_chunk = max(1, int(rerank_chunk))



        with Timer("--- Reranking with SymRelEnc (streamed to GPU) ---"):

            for s in tqdm(range(0, N, B), leave=False):

                e = min(N, s + B)

                cand = I[s:e].copy()

                for i in range(e - s):

                    gi = s + i

                    row = cand[i]

                    row = row[(row >= 0) & (row != gi)]

                    if row.size == 0:

                        qI = _faiss_search(index, R[gi:gi+1], k_candidate + 1)

                        qI = qI[0] if qI.shape[0] > 0 else np.array([], dtype=np.int64)

                        row = qI[(qI >= 0) & (qI != gi)]

                    if row.size == 0:

                        row = np.array([(gi + 1) % N], dtype=np.int64)

                    take = min(row.size, cand.shape[1])

                    cand[i, :take] = row[:take]

                    if take < cand.shape[1]:

                        cand[i, take:] = row[take - 1]



                xa = X_store[s:e] if X_is_gpu else X_store[s:e].to(device, non_blocking=True)



                K = cand.shape[1]

                logits_rows = []

                for c0 in range(0, K, rerank_chunk):

                    c1 = min(K, c0 + rerank_chunk)

                    cand_flat = cand[:, c0:c1].reshape(-1)

                    idx_t = torch.from_numpy(cand_flat)

                    xb_cpu = X_store.index_select(0, idx_t)

                    xb = xb_cpu if X_is_gpu else xb_cpu.to(device, non_blocking=True)

                    xb = xb.reshape(e - s, c1 - c0, G)

                    logits = model.forward_scores(xa, xb, amp_dtype=amp_dtype)

                    logits = _ensure_finite_t(logits, clip_abs=1e6)

                    logits_rows.append(logits.detach().cpu().to(torch.float32))

                    del xb, xb_cpu, logits



                logits_np = torch.cat(logits_rows, dim=1).numpy()

                dist = _logits_to_dist_np(logits_np)



                for i in range(e - s):

                    gi = s + i

                    ids = cand[i].astype(np.int64, copy=False)

                    drow = dist[i].astype(np.float32, copy=False)

                    order_all = np.argsort(drow)

                    ids_sorted = ids[order_all]; d_sorted = drow[order_all]

                    _, uniq_idx = np.unique(ids_sorted, return_index=True)

                    keep = np.sort(uniq_idx)

                    ids_unique = ids_sorted[keep]; d_unique = d_sorted[keep]

                    if dist_row_normalize:

                        tau = (np.median(d_unique) + 1e-6) * float(dist_temperature)

                        d_unique = (d_unique / tau).astype(np.float32)

                    take = min(len(ids_unique), int(n_neighbors))

                    ids_take = ids_unique[:take]; d_take = d_unique[:take]

                    if take < n_neighbors:

                        need = n_neighbors - take

                        qI = _faiss_search(index, R[gi:gi+1], n_neighbors + 1)

                        qI = [j for j in (qI[0] if qI.shape[0] > 0 else [])

                              if (j >= 0 and j != gi and j not in ids_take)][:need]

                        ids_take = np.concatenate([ids_take, np.array(qI, dtype=np.int64)])

                        d_take = np.concatenate([d_take, np.full(len(qI), float(1.0), dtype=np.float32)])

                    acc_idx[gi] = ids_take.tolist()

                    acc_dst[gi] = d_take.astype(np.float32).tolist()



                del xa, logits_rows, logits_np, dist, cand



        Kmin = min((len(v) for v in acc_idx if len(v) > 0), default=0)

        if Kmin <= 0: raise RuntimeError("transform: candidate list is empty.")

        Kmin = min(Kmin, int(n_neighbors))



        knn_idx = np.zeros((N, Kmin), np.int64)

        knn_dst = np.zeros((N, Kmin), np.float32)

        for i in range(N):

            row_idx = [j for j in acc_idx[i][:Kmin] if j != i]

            row_dst = [d for (j, d) in zip(acc_idx[i][:Kmin], acc_dst[i][:Kmin]) if j != i]

            if len(row_idx) < Kmin:

                need = Kmin - len(row_idx)

                qI = _faiss_search(index, R[i:i+1], Kmin + 1)

                qI = [j for j in (qI[0] if qI.shape[0] > 0 else [])

                      if (j >= 0 and j != i and j not in row_idx)][:need]

                row_idx += qI; row_dst += [float(1.0)] * len(qI)

            knn_idx[i] = np.array(row_idx[:Kmin], np.int64)

            knn_dst[i] = np.array(row_dst[:Kmin], np.float32)



        _write_neighbors(ad_out, knn_idx, knn_dst, neighbors_key=neighbors_key, random_state=random_state)

        _log(f"--- transform finished --- neighbors_key='{neighbors_key}' ready. "

             f"(k_candidate={k_candidate}, efSearch={getattr(index.hnsw, 'efSearch', 0)})")

        return ad_out if copy else None



    finally:

        try:

            if keep_model_on_device is None:

                try: any_param = next(model.parameters()); orig_device = any_param.device

                except: orig_device = torch.device('cpu')

                if orig_device.type == 'cpu':

                    model.to('cpu')

            elif not keep_model_on_device:

                model.to('cpu')

        except Exception:

            pass

        try:

            for name in list(locals().keys()):

                obj = locals().get(name, None)

                if isinstance(obj, torch.Tensor):

                    del obj

            gc.collect()

            if torch.cuda.is_available():

                torch.cuda.synchronize()

                torch.cuda.empty_cache()

        except Exception:

            pass



# ---------------- Annotate (Voting) ----------------

@torch.inference_mode()

def annotate(reference: ad.AnnData, reference_obs_key: str,

             query: ad.AnnData, *,

             k_ref: int = 64,

             k_search: Optional[int] = None,

             efsearch_override: Optional[int] = None,

             score_mode: str = "logit",          # "logit" | "sim" | "power"

             temperature: float = 1.0,

             power_alpha: float = 1.5,

             clip_logits_quantile: Optional[float] = None,

             allow_self: bool = False,

             enable_unknown: bool = True,

             unknown_label: str = "Unknown",

             min_top1: float = 0.60,

             min_margin: float = 0.15,

             max_norm_entropy: float = 0.80,

             min_top1_sim: float = 0.55,

             unknown_topk: int = 3,

             pred_key: str = "cytoneighbor_prediction",

             conf_key: str = "cytoneighbor_confidence",

             proba_key: str = "cytoneighbor_probs",

             unknown_detail_key: str = "cytoneighbor_unknown_detail",

             pred_pretty_key: str = "cytoneighbor_prediction_pretty",

             write_diagnostics: bool = True,

             top1_prob_key: str = "cytoneighbor_top1_prob",

             top2_prob_key: str = "cytoneighbor_top2_prob",

             margin_key: str = "cytoneighbor_margin",

             entropy_key: str = "cytoneighbor_norm_entropy",

             top1_sim_key: str = "cytoneighbor_top1_sim",

             is_unknown_key: str = "cytoneighbor_is_unknown",

             batch_q: int = 256,

             rerank_chunk: int = 128,

             use_cuda: bool = True,

             use_amp: bool = True,

             amp_dtype: str = "auto",

             store_neighbors: bool = False,

             neighbors_store_key: str = "cytoneighbor_knn"):

    if reference_obs_key not in reference.obs.columns:

        raise KeyError(f"reference.obs does not contain column '{reference_obs_key}'")



    def _pick_carrier():

        for A in (query, reference):

            zA = A.uns.get('cytoneighbor', {})

            if zA.get('model', None) is not None and zA.get('feature_names', None) is not None and zA.get('ann', {}).get('P_ann', None) is not None:

                return A

        return None

    carrier = _pick_carrier()

    if carrier is None:

        raise RuntimeError("annotate: no loaded model found. Please call load_model() or fit() on reference or query first.")



    z = carrier.uns['cytoneighbor']

    model: SymRelEnc = z['model']

    feats: List[str] = z['feature_names']

    ann = z['ann']; P_ann = ann['P_ann'].astype(np.float32)



    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    model.to(device).eval()



    def _align_features(adata_like: ad.AnnData, feats: List[str], verbose=True) -> np.ndarray:

        var = adata_like.var_names

        idx = var.get_indexer(feats); present = idx >= 0

        X = adata_like.X; n = adata_like.n_obs

        out = np.zeros((n, len(feats)), dtype=np.float32)

        if np.any(present):

            if sp.issparse(X): out[:, present] = X[:, idx[present]].toarray().astype(np.float32)

            else: out[:, present] = np.asarray(X)[:, idx[present]].astype(np.float32)

        miss_cnt = int((~present).sum())

        if miss_cnt and verbose: _log(f"[Align] Missing {miss_cnt} features -> filled with 0.")

        return out



    X_ref = _align_features(reference, feats, verbose=False)

    X_qry = _align_features(query, feats, verbose=False)

    preproc_mode = z.get('preproc', {"mode":"none"}).get("mode", "none")

    X_ref = _apply_value_preproc_np(X_ref, preproc_mode)

    X_qry = _apply_value_preproc_np(X_qry, preproc_mode)

    X_ref = _ensure_finite_np(X_ref, clip_abs=1e6)

    X_qry = _ensure_finite_np(X_qry, clip_abs=1e6)



    nR, nQ = X_ref.shape[0], X_qry.shape[0]

    if nR < 2 or nQ < 1:

        raise RuntimeError("annotate: reference or query set is insufficient.")



    ref_lab_raw = reference.obs[reference_obs_key].astype(str).fillna("NA").to_numpy()

    classes = pd.Index(pd.Series(ref_lab_raw).astype('category').cat.categories)

    cls2id = {c:i for i,c in enumerate(classes)}

    y_ref = np.array([cls2id[v] for v in ref_lab_raw], dtype=np.int64)

    C = len(classes)

    _log(f"[Annotate] reference cells={nR}, classes={C}, query cells={nQ}")



    # Retrieval

    R_ref = _ann_rep(X_ref, P_ann)

    index = _faiss_hnsw_build(R_ref)

    if k_search is None:

        k_search = int(min(nR, max(4*int(k_ref), 128)))

    ef_s = int(efsearch_override) if efsearch_override is not None else int(max(128, min(4*int(k_search), 2048)))

    try: index.hnsw.efSearch = ef_s

    except Exception: pass



    def _to_logits_from(sim: np.ndarray) -> np.ndarray:

        if score_mode == "sim":

            return sim / max(1e-6, float(temperature))

        elif score_mode == "power":

            a = max(1e-6, float(power_alpha))

            return (np.clip(sim, 0.0, 1.0) ** a) / max(1e-6, float(temperature))

        else:

            raise ValueError("score_mode must be 'logit' | 'sim' | 'power'.")



    def _stabilize_logits(z: np.ndarray) -> np.ndarray:

        if clip_logits_quantile is None: return z

        q = float(clip_logits_quantile)

        hi = np.quantile(z, q, axis=1, keepdims=True)

        lo = np.quantile(z, 1.0-q, axis=1, keepdims=True)

        return np.clip(z, lo, hi)



    probs = np.zeros((nQ, C), dtype=np.float32)

    pred_ids = np.zeros(nQ, dtype=np.int64)

    conf = np.zeros(nQ, dtype=np.float32)

    unknown_detail = np.array([""]*nQ, dtype=object)

    pred_pretty = np.array([""]*nQ, dtype=object)

    is_unknown = np.zeros(nQ, dtype=bool)

    top1_prob = np.zeros(nQ, dtype=np.float32)

    top2_prob = np.zeros(nQ, dtype=np.float32)

    margin = np.zeros(nQ, dtype=np.float32)

    norm_entropy = np.zeros(nQ, dtype=np.float32)

    top1_sim = np.zeros(nQ, dtype=np.float32)



    if store_neighbors:

        knn_idx_all = np.zeros((nQ, int(k_ref)), dtype=np.int64)

        knn_sim_all = np.zeros((nQ, int(k_ref)), dtype=np.float32)

        knn_logit_all = np.zeros((nQ, int(k_ref)), dtype=np.float32)



    B = max(1, int(batch_q))

    rerank_chunk = max(1, int(rerank_chunk))

    same_set = (reference is query)



    for s in tqdm(range(0, nQ, B), leave=False, desc="[Annotate]"):

        e = min(nQ, s+B)

        R_q = _ann_rep(X_qry[s:e], P_ann)

        K = int(min(nR, max(int(k_search), int(k_ref))))

        I = _faiss_search(index, R_q, K)

        I_clean = np.empty_like(I)

        for i in range(e - s):

            gi = s + i

            row = I[i]; row = row[(row >= 0) & (row < nR)]

            if same_set and (not allow_self): row = row[row != gi]

            if row.size == 0: row = np.array([ (gi + 1) % nR ], dtype=np.int64)

            take = min(row.size, K)

            I_clean[i, :take] = row[:take]

            if take < K: I_clean[i, take:] = row[take - 1]



        xa = torch.from_numpy(np.ascontiguousarray(X_qry[s:e])).to(device, non_blocking=True)

        logits_rows = []

        for c0 in range(0, K, rerank_chunk):

            c1 = min(K, c0+rerank_chunk)

            cand_flat = I_clean[:, c0:c1].reshape(-1).astype(np.int64, copy=False)

            xb_cpu = np.ascontiguousarray(X_ref[cand_flat])

            xb = torch.from_numpy(xb_cpu).to(device, non_blocking=True).reshape(e-s, c1-c0, -1)

            logits = model.forward_scores(xa, xb, amp_dtype=amp_dtype)

            logits = _ensure_finite_t(logits, clip_abs=1e6)

            logits_rows.append(logits.detach().cpu().to(torch.float32))

            del xb, xb_cpu, logits



        logits_all = torch.cat(logits_rows, dim=1).numpy()   # [B, K]

        sim_all = 1.0 / (1.0 + np.exp(-logits_all))



        k_take = min(int(k_ref), K)

        part_idx = np.argpartition(-logits_all, kth=k_take-1, axis=1)[:, :k_take]

        rr = np.arange(e-s)[:, None]

        logits_top = logits_all[rr, part_idx]

        sim_top    = sim_all[rr, part_idx]

        labs       = y_ref[I_clean[rr, part_idx]]

        order_local = np.argsort(-logits_top, axis=1)

        logits_top = logits_top[rr, order_local]

        sim_top    = sim_top[rr, order_local]

        labs       = labs[rr, order_local]



        if score_mode == "logit":

            zlog = logits_top / max(1e-6, float(temperature))

        else:

            zlog = _to_logits_from(sim_top)

        zlog = _stabilize_logits(zlog)

        zlog = zlog - zlog.max(axis=1, keepdims=True)

        w = np.exp(zlog); w = w / (w.sum(axis=1, keepdims=True) + 1e-8)



        for i in range(e-s):

            wi = w[i]; li = labs[i]

            acc = np.zeros(C, dtype=np.float32)

            np.add.at(acc, li, wi)

            sm = acc.sum(); acc = (acc / sm) if sm > 0 else (np.ones(C, np.float32)/C)

            probs[s+i] = acc

            p_sorted_idx = np.argsort(-acc)

            p1 = float(acc[p_sorted_idx[0]]); p2 = float(acc[p_sorted_idx[1]]) if C >= 2 else 0.0

            top1_prob[s+i] = p1; top2_prob[s+i] = p2; margin[s+i] = p1 - p2

            ent = -np.sum(acc * (np.log(acc + 1e-12))); ent_max = math.log(C) if C > 1 else 1.0

            norm_entropy[s+i] = float(ent / max(ent_max, 1e-12))

            top1_sim[s+i] = float(np.max(sim_top[i]))

            pid = int(p_sorted_idx[0]); pred_ids[s+i] = pid; conf[s+i] = p1

            unk = False

            if enable_unknown:

                if (p1 < float(min_top1)) or (margin[s+i] < float(min_margin)) \
                   or (norm_entropy[s+i] > float(max_norm_entropy)) \
                   or (top1_sim[s+i] < float(min_top1_sim)):

                    unk = True

            is_unknown[s+i] = unk

            if unk:

                topk = min(int(unknown_topk), C)

                idx_top = p_sorted_idx[:topk]

                parts = [f"{classes[j]}:{acc[j]:.2f}" for j in idx_top]

                txt = f"{unknown_label} (" + ", ".join(parts) + ")"

                unknown_detail[s+i] = txt; pred_pretty[s+i] = txt

            else:

                unknown_detail[s+i] = ""; pred_pretty[s+i] = str(classes[pid])



            if store_neighbors:

                sel = I_clean[i, part_idx[i][order_local[i]]]

                knn_idx_all[s+i, :k_take] = sel

                knn_sim_all[s+i, :k_take] = sim_top[i]

                knn_logit_all[s+i, :k_take] = logits_top[i]



    pred_labels = np.array([classes[i] for i in pred_ids], dtype=object)

    if enable_unknown:

        pred_labels[is_unknown] = unknown_label

        cat_order = list(classes) + [unknown_label]

    else:

        cat_order = list(classes)



    query.obs[pred_key] = pd.Categorical(pred_labels, categories=cat_order)

    query.obs[conf_key] = conf

    query.obs[pred_pretty_key] = pred_pretty

    query.obs[unknown_detail_key] = unknown_detail

    query.obsm[proba_key] = probs



    if write_diagnostics:

        query.obs[top1_prob_key] = top1_prob

        query.obs[top2_prob_key] = top2_prob

        query.obs[margin_key] = margin

        query.obs[entropy_key] = norm_entropy

        query.obs[top1_sim_key] = top1_sim

        query.obs[is_unknown_key] = is_unknown



    qz = query.uns.setdefault('cytoneighbor', {})

    qz['classes_for_probs'] = list(map(str, classes))

    _log(f"[Annotate] Written to obs['{pred_key}'], obs['{pred_pretty_key}'], obs['{conf_key}'], obsm['{proba_key}']; C={C}; Unknown={int(is_unknown.sum())}; efSearch={getattr(index.hnsw,'efSearch',0)}")

    return query

