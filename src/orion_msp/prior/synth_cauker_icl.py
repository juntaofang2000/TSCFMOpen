from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Callable

warnings.filterwarnings(
    "ignore",
    message=r".*cupy\.random\.(multivariate_normal|RandomState\.multivariate_normal).*experimental.*",
    category=FutureWarning,
)

try:
    import cupy as cp
except Exception:
    cp = None
import networkx as nx
import numpy as np
import torch
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    DotProduct,
    ExpSineSquared,
    RBF,
    RationalQuadratic,
    WhiteKernel,
)
from torch.utils.data import IterableDataset, get_worker_info


_CUPY_WORKER_PATH_LOGGED = False


def build_kernel_bank(time_length: int) -> List:
    return [
        ExpSineSquared(periodicity=24 / time_length),
        ExpSineSquared(periodicity=48 / time_length),
        ExpSineSquared(periodicity=96 / time_length),
        ExpSineSquared(periodicity=24 * 7 / time_length),
        ExpSineSquared(periodicity=48 * 7 / time_length),
        ExpSineSquared(periodicity=96 * 7 / time_length),
        ExpSineSquared(periodicity=7 / time_length),
        ExpSineSquared(periodicity=14 / time_length),
        ExpSineSquared(periodicity=30 / time_length),
        ExpSineSquared(periodicity=60 / time_length),
        ExpSineSquared(periodicity=365 / time_length),
        ExpSineSquared(periodicity=365 * 2 / time_length),
        ExpSineSquared(periodicity=4 / time_length),
        ExpSineSquared(periodicity=26 / time_length),
        ExpSineSquared(periodicity=52 / time_length),
        ExpSineSquared(periodicity=4 / time_length),
        ExpSineSquared(periodicity=6 / time_length),
        ExpSineSquared(periodicity=12 / time_length),
        ExpSineSquared(periodicity=4 / time_length),
        ExpSineSquared(periodicity=(4 * 10) / time_length),
        ExpSineSquared(periodicity=10 / time_length),
        DotProduct(sigma_0=0.0),
        DotProduct(sigma_0=1.0),
        DotProduct(sigma_0=10.0),
        RBF(length_scale=0.1),
        RBF(length_scale=1.0),
        RBF(length_scale=10.0),
        RationalQuadratic(alpha=0.1),
        RationalQuadratic(alpha=1.0),
        RationalQuadratic(alpha=10.0),
        WhiteKernel(noise_level=0.1),
        WhiteKernel(noise_level=1.0),
        ConstantKernel(),
    ]


def sample_from_gp_prior_efficient_gpu(
    kernel,
    X: np.ndarray,
    random_seed: Optional[int] = None,
    method: str = "eigh",
    mean_vec: Optional[np.ndarray] = None,
    num_samples: int = 1,
    cov_cpu: Optional[np.ndarray] = None,
):
    if X.ndim == 1:
        X = X[:, None]
    cov_cpu = kernel(X) if cov_cpu is None else cov_cpu
    n = X.shape[0]
    mean_vec = np.zeros(n, dtype=np.float64) if mean_vec is None else mean_vec
    num_samples = max(1, int(num_samples))

    worker = get_worker_info()
    allow_cupy_in_worker = str(os.getenv("ORION_CUPY_IN_WORKER", "1")).lower() in {"1", "true", "yes", "on"}
    if cp is None or (worker is not None and not allow_cupy_in_worker):
        rng = np.random.default_rng(None if random_seed is None else int(random_seed))
        draws = rng.multivariate_normal(mean=mean_vec, cov=cov_cpu, method=method, size=num_samples)
        return np.asarray(draws, dtype=np.float64)

    try:
        global _CUPY_WORKER_PATH_LOGGED
        if worker is not None and allow_cupy_in_worker:
            device_idx = int(os.getenv("ORION_CUPY_DEVICE", os.getenv("LOCAL_RANK", "0")))
            cp.cuda.Device(device_idx).use()
            rank = int(os.getenv("RANK", "0"))
            if (not _CUPY_WORKER_PATH_LOGGED) and rank == 0 and int(getattr(worker, "id", -1)) == 0:
                print(f"[rank0] CuPy worker path enabled (worker_id={worker.id}, device={device_idx})", flush=True)
                _CUPY_WORKER_PATH_LOGGED = True
        cov_gpu = cp.asarray(cov_cpu)
        mean_gpu = cp.asarray(mean_vec)
        if random_seed is not None:
            cp.random.seed(int(random_seed))
        ts_gpu = cp.random.multivariate_normal(mean=mean_gpu, cov=cov_gpu, method=method, size=num_samples)
        return cp.asnumpy(ts_gpu)
    except Exception:
        rng = np.random.default_rng(None if random_seed is None else int(random_seed))
        draws = rng.multivariate_normal(mean=mean_vec, cov=cov_cpu, method=method, size=num_samples)
        return np.asarray(draws, dtype=np.float64)


@dataclass
class MeanSpec:
    kind: str
    params: Dict[str, object]


@dataclass
class RootSpec:
    node_id: int
    kernel: object
    mean_spec: MeanSpec


@dataclass
class NodeSpec:
    node_id: int
    parents: List[int]
    lags: List[int]
    w_orig: np.ndarray
    w_lag: np.ndarray
    bias: float
    act: str
    act_params: Dict[str, float]
    noise_std: float


@dataclass
class Program:
    level: int
    seed: int
    time_length: int
    num_nodes: int
    max_parents: int
    max_lag: int
    dag: nx.DiGraph
    topological_order: List[int]
    root_specs: Dict[int, RootSpec]
    node_specs: Dict[int, NodeSpec]
    chosen_nodes: List[int]
    base_noise_std: float


def _difficulty_cfg(level: int, max_parents: int, max_lag: int) -> Dict[str, object]:
    level = int(level)
    if level <= 0:
        return {
            "kernel_terms": (1, 2),
            "nonlinear_prob": 0.15,
            "act_weights": {
                "linear": 0.85,
                "relu": 0.08,
                "sin": 0.03,
                "mod": 0.02,
                "leakyrelu": 0.02,
                "sigmoid": 0.00,
            },
            "max_parents": min(max_parents, 2),
            "max_lag": min(max_lag, 2),
            "root_noise_std": 0.03,
            "node_noise_std": 0.03,
        }
    if level == 1:
        return {
            "kernel_terms": (2, 4),
            "nonlinear_prob": 0.45,
            "act_weights": {
                "linear": 0.55,
                "relu": 0.15,
                "sin": 0.10,
                "mod": 0.08,
                "leakyrelu": 0.07,
                "sigmoid": 0.05,
            },
            "max_parents": min(max_parents, 3),
            "max_lag": min(max_lag, 4),
            "root_noise_std": 0.05,
            "node_noise_std": 0.05,
        }
    if level == 2:
        return {
            "kernel_terms": (3, 6),
            "nonlinear_prob": 0.70,
            "act_weights": {
                "linear": 0.30,
                "relu": 0.20,
                "sin": 0.18,
                "mod": 0.12,
                "leakyrelu": 0.10,
                "sigmoid": 0.10,
            },
            "max_parents": min(max_parents, 5),
            "max_lag": min(max_lag, 8),
            "root_noise_std": 0.08,
            "node_noise_std": 0.08,
        }
    return {
        "kernel_terms": (3, 7),
        "nonlinear_prob": 0.85,
        "act_weights": {
            "linear": 0.18,
            "relu": 0.20,
            "sin": 0.20,
            "mod": 0.15,
            "leakyrelu": 0.12,
            "sigmoid": 0.15,
        },
        "max_parents": min(max_parents, 6),
        "max_lag": min(max_lag, 12),
        "root_noise_std": 0.10,
        "node_noise_std": 0.12,
    }


def _generate_random_dag_rng(num_nodes: int, max_parents: int, rng: np.random.Generator) -> nx.DiGraph:
    G = nx.DiGraph()
    nodes = list(range(num_nodes))
    rng.shuffle(nodes)
    G.add_nodes_from(nodes)
    for i in range(num_nodes):
        possible_parents = nodes[:i]
        max_here = min(len(possible_parents), max_parents)
        num_par = int(rng.integers(0, max_here + 1)) if max_here > 0 else 0
        if num_par > 0:
            parents = rng.choice(possible_parents, size=num_par, replace=False)
            for p in parents.tolist():
                G.add_edge(int(p), int(nodes[i]))
    return G


def _sample_mean_spec(rng: np.random.Generator, time_length: int) -> MeanSpec:
    kind = str(rng.choice(["zero", "linear", "exponential", "anomaly"], p=[0.15, 0.45, 0.20, 0.20]))
    if kind == "zero":
        return MeanSpec(kind=kind, params={})
    if kind == "linear":
        return MeanSpec(kind=kind, params={"a": float(rng.uniform(-1.0, 1.0)), "b": float(rng.uniform(-1.0, 1.0))})
    if kind == "exponential":
        return MeanSpec(kind=kind, params={"a": float(rng.uniform(0.5, 1.5)), "b": float(rng.uniform(0.5, 1.5))})
    num_anomalies = int(rng.integers(1, 6))
    return MeanSpec(
        kind=kind,
        params={
            "positions": rng.integers(0, time_length, size=num_anomalies).astype(int).tolist(),
            "amplitudes": rng.uniform(-5.0, 5.0, size=num_anomalies).astype(float).tolist(),
        },
    )


def _mean_from_spec(X: np.ndarray, spec: MeanSpec) -> np.ndarray:
    if spec.kind == "zero":
        return np.zeros_like(X)
    if spec.kind == "linear":
        return spec.params["a"] * X + spec.params["b"]
    if spec.kind == "exponential":
        return spec.params["a"] * np.exp(spec.params["b"] * X)
    m = np.zeros_like(X)
    pos = spec.params.get("positions", [])
    amp = spec.params.get("amplitudes", [])
    for p, a in zip(pos, amp):
        if 0 <= int(p) < len(X):
            m[int(p)] += float(a)
    return m


def _sample_activation_spec(rng: np.random.Generator, cfg: Dict[str, object]) -> tuple[str, Dict[str, float]]:
    weights = cfg["act_weights"]
    acts = list(weights.keys())
    probs = np.array([weights[a] for a in acts], dtype=np.float64)
    probs = probs / probs.sum()
    act = str(rng.choice(acts, p=probs))
    if act == "linear":
        return act, {"a": float(rng.uniform(0.5, 2.0)), "b": float(rng.uniform(-1.0, 1.0))}
    if act == "mod":
        return act, {"c": float(rng.uniform(1.0, 5.0))}
    if act == "leakyrelu":
        return act, {"alpha": float(rng.uniform(0.01, 0.30))}
    return act, {}


def _apply_activation(z: np.ndarray, act: str, params: Dict[str, float]) -> np.ndarray:
    if act == "linear":
        return float(params.get("a", 1.0)) * z + float(params.get("b", 0.0))
    if act == "relu":
        return np.maximum(0.0, z)
    if act == "sigmoid":
        z_clipped = np.clip(z, -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-z_clipped))
    if act == "sin":
        return np.sin(z)
    if act == "mod":
        c = max(1e-6, float(params.get("c", 2.0)))
        return np.mod(z, c)
    alpha = float(params.get("alpha", 0.1))
    return np.where(z > 0, z, alpha * z)


def sample_program(level: int, time_length: int, num_nodes: int, max_parents: int, max_lag: int, seed: int) -> Program:
    rng = np.random.default_rng(seed)
    cfg = _difficulty_cfg(level=level, max_parents=max_parents, max_lag=max_lag)
    dag = _generate_random_dag_rng(num_nodes=num_nodes, max_parents=int(cfg["max_parents"]), rng=rng)
    topological_order = list(nx.topological_sort(dag))

    root_nodes = [n for n in dag.nodes if dag.in_degree(n) == 0]
    kernel_bank = build_kernel_bank(time_length)

    root_specs: Dict[int, RootSpec] = {}
    kmin, kmax = cfg["kernel_terms"]
    for r in root_nodes:
        n_terms = int(rng.integers(kmin, kmax + 1))
        terms = rng.choice(kernel_bank, size=n_terms, replace=True)
        kernel = terms[0]
        for t in terms[1:]:
            if rng.uniform() < 0.5:
                kernel = kernel + t
            else:
                kernel = kernel * t
        mean_spec = _sample_mean_spec(rng, time_length)
        root_specs[int(r)] = RootSpec(node_id=int(r), kernel=kernel, mean_spec=mean_spec)

    node_specs: Dict[int, NodeSpec] = {}
    for node in topological_order:
        if node in root_specs:
            continue
        parents = list(dag.predecessors(node))
        p = len(parents)
        lags = rng.integers(1, int(cfg["max_lag"]) + 1, size=p).astype(int).tolist() if p > 0 else []
        w_orig = rng.normal(loc=0.0, scale=1.0, size=p).astype(np.float32) if p > 0 else np.zeros((0,), dtype=np.float32)
        w_lag = rng.normal(loc=0.0, scale=1.0, size=p).astype(np.float32) if p > 0 else np.zeros((0,), dtype=np.float32)
        bias = float(rng.normal(loc=0.0, scale=1.0))

        if rng.uniform() > float(cfg["nonlinear_prob"]):
            act, act_params = "linear", {"a": 1.0, "b": 0.0}
        else:
            act, act_params = _sample_activation_spec(rng, cfg)

        node_specs[int(node)] = NodeSpec(
            node_id=int(node),
            parents=[int(pp) for pp in parents],
            lags=lags,
            w_orig=w_orig,
            w_lag=w_lag,
            bias=bias,
            act=act,
            act_params=act_params,
            noise_std=float(cfg["node_noise_std"]),
        )

    chosen_nodes = rng.choice(topological_order, size=min(num_nodes, len(topological_order)), replace=False)

    return Program(
        level=int(level),
        seed=int(seed),
        time_length=int(time_length),
        num_nodes=int(num_nodes),
        max_parents=int(cfg["max_parents"]),
        max_lag=int(cfg["max_lag"]),
        dag=dag,
        topological_order=[int(x) for x in topological_order],
        root_specs=root_specs,
        node_specs=node_specs,
        chosen_nodes=[int(x) for x in chosen_nodes.tolist()],
        base_noise_std=float(cfg["root_noise_std"]),
    )


@lru_cache(maxsize=4096)
def _sample_program_cached(
    level: int,
    time_length: int,
    num_nodes: int,
    max_parents: int,
    max_lag: int,
    seed: int,
) -> Program:
    return sample_program(
        level=int(level),
        time_length=int(time_length),
        num_nodes=int(num_nodes),
        max_parents=int(max_parents),
        max_lag=int(max_lag),
        seed=int(seed),
    )


def _apply_level3_augment(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n, c, t = x.shape
    for i in range(n):
        if rng.uniform() < 0.35:
            seg_len = int(rng.integers(max(4, t // 32), max(8, t // 12)))
            start = int(rng.integers(0, max(1, t - seg_len)))
            x[i, :, start : start + seg_len] = 0.0
        if rng.uniform() < 0.35:
            m = int(rng.integers(1, 6))
            idx_t = rng.integers(0, t, size=m)
            idx_c = rng.integers(0, c, size=m)
            x[i, idx_c, idx_t] += rng.normal(0.0, 4.0, size=m)
    return x


def render_program(program: Program, n: int, num_features: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    T = int(program.time_length)
    C = int(num_features)
    if C > len(program.chosen_nodes):
        raise ValueError(f"num_features={C} exceeds chosen_nodes={len(program.chosen_nodes)} in Program")

    X_axis = np.linspace(0.0, 1.0, T)
    node_data: Dict[int, np.ndarray] = {}

    for root_id, rs in program.root_specs.items():
        mean_vec = _mean_from_spec(X_axis, rs.mean_spec)
        gp_seed = int(rng.integers(0, 2**31 - 1))
        cov_cpu = rs.kernel(X_axis[:, None])
        samples = sample_from_gp_prior_efficient_gpu(
            kernel=rs.kernel,
            X=X_axis,
            mean_vec=mean_vec,
            random_seed=gp_seed,
            num_samples=n,
            cov_cpu=cov_cpu,
        )
        samples = samples + rng.normal(0.0, program.base_noise_std, size=samples.shape)
        samples = np.asarray(samples, dtype=np.float32)
        node_data[int(root_id)] = samples

    for node in program.topological_order:
        if node in node_data:
            continue
        spec = program.node_specs[node]
        z = np.full((n, T), fill_value=spec.bias, dtype=np.float32)
        for j, parent_id in enumerate(spec.parents):
            parent = node_data[int(parent_id)]
            lag = int(spec.lags[j])
            shifted = np.roll(parent, lag, axis=1)
            shifted[:, :lag] = 0.0
            z += float(spec.w_orig[j]) * parent + float(spec.w_lag[j]) * shifted
        out = _apply_activation(z, spec.act, spec.act_params)
        out = out + rng.normal(0.0, spec.noise_std, size=out.shape)
        node_data[int(node)] = out.astype(np.float32)

    chosen = program.chosen_nodes[:C]
    x = np.stack([node_data[int(nid)] for nid in chosen], axis=1).astype(np.float32)

    if program.level >= 3:
        x = _apply_level3_augment(x, rng)

    return x.astype(np.float32)


def generate_episode(
    task_id: int,
    seed: int,
    K: int,
    n_ctx_per_class: int,
    n_qry_per_class: int,
    time_length: int = 512,
    num_features: int = 6,
    level: int = 0,
    num_nodes: int = 18,
    max_parents: int = 5,
    max_lag: int = 5,
    program_pool_size: int = 0,
) -> dict:
    if K <= 0:
        raise ValueError("K must be positive")
    if n_ctx_per_class < 1:
        raise ValueError("n_ctx_per_class must be >= 1 to guarantee min-one-per-class")
    if num_features > num_nodes:
        raise ValueError(f"num_features ({num_features}) must be <= num_nodes ({num_nodes})")

    rng = np.random.default_rng(seed + task_id * 1009)

    xs_ctx: List[np.ndarray] = []
    ys_ctx: List[np.ndarray] = []
    xs_qry: List[np.ndarray] = []
    ys_qry: List[np.ndarray] = []

    for k in range(K):
        if int(program_pool_size) > 0:
            pool_slot = int((task_id * K + k) % int(program_pool_size))
            class_seed = int(seed + pool_slot * 10_007)
        else:
            class_seed = int(seed + task_id * 100_003 + k * 10_007)

        program = _sample_program_cached(
            level=int(level),
            time_length=int(time_length),
            num_nodes=int(num_nodes),
            max_parents=int(max_parents),
            max_lag=int(max_lag),
            seed=int(class_seed),
        )
        if len(program.chosen_nodes) != int(num_features):
            prng = np.random.default_rng(class_seed + 31337)
            chosen_nodes = prng.choice(
                np.array(program.topological_order, dtype=np.int64),
                size=int(num_features),
                replace=False,
            ).astype(int).tolist()
            old_nodes = program.chosen_nodes
            program.chosen_nodes = chosen_nodes
            xk = render_program(
                program=program,
                n=n_ctx_per_class + n_qry_per_class,
                num_features=num_features,
                seed=class_seed + 777,
            )
            program.chosen_nodes = old_nodes
        else:
            xk = render_program(
                program=program,
                n=n_ctx_per_class + n_qry_per_class,
                num_features=num_features,
                seed=class_seed + 777,
            )

        perm = rng.permutation(xk.shape[0])
        xk = xk[perm]

        x_ctx_k = xk[:n_ctx_per_class]
        x_qry_k = xk[n_ctx_per_class : n_ctx_per_class + n_qry_per_class]

        y_ctx_k = np.full((x_ctx_k.shape[0],), k, dtype=np.int64)
        y_qry_k = np.full((x_qry_k.shape[0],), k, dtype=np.int64)

        xs_ctx.append(x_ctx_k)
        ys_ctx.append(y_ctx_k)
        xs_qry.append(x_qry_k)
        ys_qry.append(y_qry_k)

    x_ctx = np.concatenate(xs_ctx, axis=0).astype(np.float32)
    y_ctx = np.concatenate(ys_ctx, axis=0).astype(np.int64)
    x_qry = np.concatenate(xs_qry, axis=0).astype(np.float32)
    y_qry = np.concatenate(ys_qry, axis=0).astype(np.int64)

    p_ctx = rng.permutation(x_ctx.shape[0])
    p_qry = rng.permutation(x_qry.shape[0])
    x_ctx, y_ctx = x_ctx[p_ctx], y_ctx[p_ctx]
    x_qry, y_qry = x_qry[p_qry], y_qry[p_qry]

    ctx_classes = set(np.unique(y_ctx).tolist())
    qry_classes = set(np.unique(y_qry).tolist())
    if not qry_classes.issubset(ctx_classes):
        raise RuntimeError("Invalid episode: query contains unseen classes")

    class_perm = rng.permutation(K).astype(np.int64)
    y_ctx = class_perm[y_ctx]
    y_qry = class_perm[y_qry]

    return {
        "x_ctx": x_ctx,
        "y_ctx": y_ctx,
        "x_qry": x_qry,
        "y_qry": y_qry,
        "n_ctx": int(x_ctx.shape[0]),
        "n_qry": int(x_qry.shape[0]),
        "K": int(K),
        "C": int(num_features),
        "T": int(time_length),
        "task_id": int(task_id),
        "seed": int(seed),
        "difficulty_level": int(level),
        "class_perm": class_perm,
    }


class SynthCaukerICLDataset(IterableDataset):
    def __init__(
        self,
        *,
        num_tasks_per_epoch: int,
        base_seed: int,
        K: int,
        n_ctx_per_class: int,
        n_qry_per_class: int,
        time_length: int = 512,
        num_features: int = 6,
        num_nodes: int = 18,
        max_parents: int = 5,
        max_lag: int = 5,
        level: int = 0,
        level_sampler: Optional[Callable[[int, np.random.Generator], int]] = None,
    ):
        super().__init__()
        self.num_tasks_per_epoch = int(num_tasks_per_epoch)
        self.base_seed = int(base_seed)
        self.K = int(K)
        self.n_ctx_per_class = int(n_ctx_per_class)
        self.n_qry_per_class = int(n_qry_per_class)
        self.time_length = int(time_length)
        self.num_features = int(num_features)
        self.num_nodes = int(num_nodes)
        self.max_parents = int(max_parents)
        self.max_lag = int(max_lag)
        self.level = int(level)
        self.level_sampler = level_sampler

    def _pick_level(self, task_id: int, rng: np.random.Generator) -> int:
        if self.level_sampler is not None:
            return int(self.level_sampler(task_id, rng))
        return self.level

    def __iter__(self):
        worker = get_worker_info()
        if worker is None:
            start = 0
            stride = 1
            worker_seed = self.base_seed
        else:
            start = worker.id
            stride = worker.num_workers
            worker_seed = self.base_seed + worker.id * 1_000_003

        rng = np.random.default_rng(worker_seed)
        produced = 0
        task_id = start
        while produced < self.num_tasks_per_epoch:
            lvl = self._pick_level(task_id, rng)
            ep_seed = int(self.base_seed + task_id * 97 + rng.integers(0, 10_000))
            episode = generate_episode(
                task_id=task_id,
                seed=ep_seed,
                K=self.K,
                n_ctx_per_class=self.n_ctx_per_class,
                n_qry_per_class=self.n_qry_per_class,
                time_length=self.time_length,
                num_features=self.num_features,
                level=lvl,
                num_nodes=self.num_nodes,
                max_parents=self.max_parents,
                max_lag=self.max_lag,
            )
            yield episode
            produced += 1
            task_id += stride


def collate_episode_batch(batch: List[dict]) -> dict:
    x_ctx = torch.stack([torch.from_numpy(b["x_ctx"]) for b in batch], dim=0).float()
    y_ctx = torch.stack([torch.from_numpy(b["y_ctx"]) for b in batch], dim=0).long()
    x_qry = torch.stack([torch.from_numpy(b["x_qry"]) for b in batch], dim=0).float()
    y_qry = torch.stack([torch.from_numpy(b["y_qry"]) for b in batch], dim=0).long()

    x_all = torch.cat([x_ctx, x_qry], dim=1)
    y_all = torch.cat([y_ctx, y_qry], dim=1)

    return {
        "x_ctx": x_ctx,
        "y_ctx": y_ctx,
        "x_qry": x_qry,
        "y_qry": y_qry,
        "x_all": x_all,
        "y_all": y_all,
        "n_ctx": torch.tensor([b["n_ctx"] for b in batch], dtype=torch.long),
        "n_qry": torch.tensor([b["n_qry"] for b in batch], dtype=torch.long),
        "K": torch.tensor([b["K"] for b in batch], dtype=torch.long),
        "C": torch.tensor([b["C"] for b in batch], dtype=torch.long),
        "T": torch.tensor([b["T"] for b in batch], dtype=torch.long),
        "task_id": torch.tensor([b["task_id"] for b in batch], dtype=torch.long),
        "seed": torch.tensor([b["seed"] for b in batch], dtype=torch.long),
        "difficulty_level": torch.tensor([b["difficulty_level"] for b in batch], dtype=torch.long),
        "class_perm": torch.stack([torch.from_numpy(b["class_perm"]) for b in batch], dim=0).long(),
    }


def generate_episodes_to_npz(
    out_dir: str | Path,
    num_episodes: int,
    *,
    base_seed: int,
    K: int,
    n_ctx_per_class: int,
    n_qry_per_class: int,
    time_length: int = 512,
    num_features: int = 6,
    level: int = 0,
    num_nodes: int = 18,
    max_parents: int = 5,
    max_lag: int = 5,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for task_id in range(int(num_episodes)):
        ep = generate_episode(
            task_id=task_id,
            seed=int(base_seed + task_id * 131),
            K=K,
            n_ctx_per_class=n_ctx_per_class,
            n_qry_per_class=n_qry_per_class,
            time_length=time_length,
            num_features=num_features,
            level=level,
            num_nodes=num_nodes,
            max_parents=max_parents,
            max_lag=max_lag,
        )
        np.savez_compressed(out / f"episode_{task_id:06d}.npz", **ep)


def sanity_check_episode() -> None:
    ep = generate_episode(
        task_id=0,
        seed=42,
        K=5,
        n_ctx_per_class=2,
        n_qry_per_class=5,
        num_features=6,
        time_length=512,
        level=1,
        num_nodes=18,
        max_parents=5,
        max_lag=5,
    )
    print("[Sanity] x_ctx:", ep["x_ctx"].shape, ep["x_ctx"].dtype)
    print("[Sanity] y_ctx:", ep["y_ctx"].shape, ep["y_ctx"].dtype)
    print("[Sanity] x_qry:", ep["x_qry"].shape, ep["x_qry"].dtype)
    print("[Sanity] y_qry:", ep["y_qry"].shape, ep["y_qry"].dtype)

    unique_ctx, cnt_ctx = np.unique(ep["y_ctx"], return_counts=True)
    unique_qry, cnt_qry = np.unique(ep["y_qry"], return_counts=True)
    print("[Sanity] ctx class counts:", dict(zip(unique_ctx.tolist(), cnt_ctx.tolist())))
    print("[Sanity] qry class counts:", dict(zip(unique_qry.tolist(), cnt_qry.tolist())))

    if not set(unique_qry.tolist()).issubset(set(unique_ctx.tolist())):
        raise RuntimeError("Sanity failed: query contains class not in context")
    print("[Sanity] PASS: query classes are subset of context classes.")


if __name__ == "__main__":
    sanity_check_episode()
