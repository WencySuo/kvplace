"""Pure arithmetic: model dims, attention shapes, roofline predictions, capacity table.

No torch, no CUDA. Everything downstream (bench, plot, audit) imports from here so the
numbers in the report are derived in exactly one place.

Conventions
  R      query rows in one grouped attention call = B * q  (B requests sharing the prefix,
         q verifier positions each; q=1 is ordinary decode, q=16 is a DFlash block)
  S, L   shared-prefix length in tokens (one layer's K/V is [L, h_kv, d])
  G      GQA group size h_q / h_kv
  bytes  bf16 everywhere (ELEM = 2)
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path

ELEM = 2  # bf16 bytes
CONFIG_DIR = Path(__file__).parent / "model_configs"


@dataclass(frozen=True)
class ModelDims:
    name: str
    layers: int
    h_q: int
    h_kv: int
    d: int
    hidden: int
    intermediate: int
    vocab: int
    tie_embeddings: bool
    revision: str | None = None
    block_size: int | None = None  # drafter only

    @property
    def G(self) -> int:
        assert self.h_q % self.h_kv == 0
        return self.h_q // self.h_kv

    @property
    def kv_bytes_per_token(self) -> int:
        """All layers, K and V, bf16. Storage uses h_kv, never h_q."""
        return 2 * self.layers * self.h_kv * self.d * ELEM

    @property
    def params(self) -> int:
        """Config-derived parameter count (dense attention + SwiGLU MLP + embeddings)."""
        attn = self.hidden * (self.h_q * self.d) * 2 + self.hidden * (self.h_kv * self.d) * 2
        mlp = 3 * self.hidden * self.intermediate
        embed = self.vocab * self.hidden * (1 if self.tie_embeddings else 2)
        return self.layers * (attn + mlp) + embed

    @property
    def weight_bytes(self) -> int:
        return self.params * ELEM


def load_dims(key: str) -> ModelDims:
    """Read an HF config.json committed under model_configs/ (revision from REVISIONS.json)."""
    cfg = json.loads((CONFIG_DIR / f"{key}.json").read_text())
    revs = json.loads((CONFIG_DIR / "REVISIONS.json").read_text())
    return ModelDims(
        name=key,
        layers=cfg["num_hidden_layers"],
        h_q=cfg["num_attention_heads"],
        h_kv=cfg["num_key_value_heads"],
        d=cfg["head_dim"],
        hidden=cfg["hidden_size"],
        intermediate=cfg["intermediate_size"],
        vocab=cfg["vocab_size"],
        tie_embeddings=bool(cfg.get("tie_word_embeddings", False)),
        revision=revs.get(key, {}).get("sha"),
        block_size=cfg.get("block_size"),
    )


# ----------------------------------------------------------------------------- shapes

@dataclass(frozen=True)
class AttnShapes:
    model: str
    R: int
    L: int
    h_q: int
    h_kv: int
    d: int

    @property
    def q(self) -> tuple[int, int, int]:
        return (self.R, self.h_q, self.d)

    @property
    def kv(self) -> tuple[int, int, int]:
        return (self.L, self.h_kv, self.d)

    @property
    def G(self) -> int:
        return self.h_q // self.h_kv

    @property
    def kv_bytes(self) -> int:
        """One layer, K and V."""
        return 2 * self.L * self.h_kv * self.d * ELEM

    @property
    def flops(self) -> int:
        """QK^T and PV, one layer, no causal mask: 4 * R * L * h_q * d."""
        return 4 * self.R * self.L * self.h_q * self.d

    @property
    def intensity(self) -> float:
        """FLOP per K/V byte. Equals G * R for bf16 GQA; asserted at construction."""
        return self.flops / self.kv_bytes

    def __post_init__(self):
        assert self.h_q % self.h_kv == 0
        assert math.isclose(self.flops / self.kv_bytes, self.G * self.R), "intensity != G*R"


def make_shapes(dims: ModelDims, R: int, L: int) -> AttnShapes:
    return AttnShapes(model=dims.name, R=R, L=L, h_q=dims.h_q, h_kv=dims.h_kv, d=dims.d)


# ----------------------------------------------------------------------------- roofline

@dataclass(frozen=True)
class Machine:
    """Measured or assumed machine constants. Units: bytes/s and FLOP/s."""
    F_roof: float          # attention compute plateau (hbm arm at largest R)
    B_hbm: float           # HBM bandwidth
    beta_dma: float        # copy-engine host->GPU (feeds host_staged, host_copy)
    beta_sm: float         # SM-initiated host reads (feeds host_direct*)
    tile_rows: int = 128   # packed query rows per prefill tile (verify per backend)
    l2_caches_host: bool = False  # tier-0 shared-footprint test decides this
    launch_us: float = 5.0        # per-kernel launch + event overhead
    source: str = "assumed"


# Assumed GH200 numbers used only for --dry-run predictions; replaced by measurements.
GH200_ASSUMED = Machine(
    F_roof=400e12, B_hbm=4.0e12, beta_dma=400e9, beta_sm=300e9,
    tile_rows=128, l2_caches_host=False, launch_us=5.0, source="assumed",
)

# PCIe presets: host memory is NOT cached in GPU L2 over PCIe; SM reads and DMA share the link.
H100_PCIE5_ASSUMED = Machine(
    F_roof=400e12, B_hbm=3.35e12, beta_dma=50e9, beta_sm=40e9,
    tile_rows=128, l2_caches_host=False, launch_us=5.0, source="assumed h100 pcie5 (p5.4xlarge)",
)
L40S_PCIE4_ASSUMED = Machine(
    F_roof=180e12, B_hbm=0.86e12, beta_dma=25e9, beta_sm=20e9,
    tile_rows=128, l2_caches_host=False, launch_us=5.0, source="assumed l40s pcie4 (g6e)",
)
PRESETS = {"gh200": GH200_ASSUMED, "h100_pcie5": H100_PCIE5_ASSUMED, "l40s_pcie4": L40S_PCIE4_ASSUMED}

ARMS = ("hbm", "host_direct", "host_direct_chunked", "host_staged", "host_copy")


def q_tiles(sh: AttnShapes, tile_rows: int) -> int:
    """Query tiles per KV head for a FlashAttention-style kernel with GQA packing."""
    return max(1, math.ceil(sh.R * sh.G / tile_rows))


def link_bytes(sh: AttnShapes, arm: str, m: Machine) -> int:
    """Bytes expected to cross C2C per layer per step, by arm."""
    if arm == "hbm":
        return 0
    if arm == "host_direct":
        return sh.kv_bytes * (1 if m.l2_caches_host else q_tiles(sh, m.tile_rows))
    return sh.kv_bytes  # chunked, staged, copy: once per byte by construction


def crossover_R(dims: ModelDims, m: Machine, beta: float | None = None) -> float:
    """R* = F / (beta * G): rows at which a once-per-byte host read is hidden by compute."""
    beta = m.beta_dma if beta is None else beta
    return m.F_roof / (beta * dims.G)


def t_pred_s(
    sh: AttnShapes, arm: str, m: Machine,
    chunk_tokens: int | None = None,
    t_hbm_chunk_s: float | None = None,
    t_merge_s: float = 0.0,
) -> float:
    """Predicted seconds per layer per step for one arm.

    hbm / host_direct: max(bytes/bandwidth, flops/F_roof), single kernel.
    host_direct_chunked / host_staged: per-chunk pipeline with pipeline fill and merge.
      Per-chunk compute uses t_hbm_chunk_s (measured hbm arm at L=chunk_tokens) when given,
      because small per-chunk kernels do not reach F_roof.
    host_copy: serial memcpy + hbm kernel.
    """
    compute = sh.flops / m.F_roof
    if arm == "hbm":
        return max(sh.kv_bytes / m.B_hbm, compute)
    if arm == "host_direct":
        return max(link_bytes(sh, arm, m) / m.beta_sm, compute)
    if arm == "host_copy":
        return sh.kv_bytes / m.beta_dma + max(sh.kv_bytes / m.B_hbm, compute)
    if arm in ("host_direct_chunked", "host_staged"):
        assert chunk_tokens, "chunked arms need chunk_tokens"
        n_chunks = math.ceil(sh.L / chunk_tokens)
        chunk_bytes = sh.kv_bytes / n_chunks
        beta = m.beta_dma if arm == "host_staged" else m.beta_sm
        xfer = chunk_bytes / beta
        if t_hbm_chunk_s is None:
            # fall back to scaled roofline; optimistic for small per-chunk kernels
            t_hbm_chunk_s = max(chunk_bytes / m.B_hbm, compute / n_chunks)
        per_chunk = t_hbm_chunk_s + m.launch_us * 1e-6
        if arm == "host_direct_chunked":
            # no DMA: the kernel itself pulls the chunk; cannot overlap with its own compute
            # unless L2 caches host lines, in which case re-reads inside the chunk are free.
            per_chunk = max(xfer, per_chunk) if m.l2_caches_host else xfer * q_tiles(sh, m.tile_rows) + per_chunk
            return n_chunks * per_chunk + t_merge_s
        return xfer + n_chunks * max(xfer, per_chunk) + t_merge_s
    raise ValueError(arm)


# ----------------------------------------------------------------------------- capacity (Stage A)

@dataclass(frozen=True)
class CapacityRow:
    target: str
    drafter: str
    S: int
    P: int
    q: int
    M: int
    hbm_bytes: int
    weights_bytes: int
    workspace_bytes: int
    free_hbm_bytes: int
    shared_bytes: int          # M distinct prefixes, one physical copy each
    private_bytes_per_req: int  # target KV for P + q tokens
    drafter_bytes_per_req: int  # drafter KV over S + P + q tokens, no dedup (config-derived)
    per_req_bytes: int
    fits_resident: bool
    B_max_resident: int
    B_max_offloaded: int        # optimistic zero-overhead bound, not a throughput bound
    extra_requests: int
    link_s_per_step_per_prefix: float  # shared KV all layers / beta, if read from host each step
    note: str


def capacity_rows(
    target: ModelDims, drafter: ModelDims, hbm_bytes: int, workspace_bytes: int,
    S_list, P_list, M_list, q: int, beta: float,
) -> list[CapacityRow]:
    weights = target.weight_bytes + drafter.weight_bytes
    free = hbm_bytes - weights - workspace_bytes
    rows = []
    for S in S_list:
        for P in P_list:
            for M in M_list:
                shared = M * S * target.kv_bytes_per_token
                priv = (P + q) * target.kv_bytes_per_token
                dr = (S + P + q) * drafter.kv_bytes_per_token
                per_req = priv + dr
                fits = shared <= free
                b_res = max(0, (free - shared) // per_req) if fits else 0
                b_off = max(0, free // per_req)
                note = "offload necessary: shared set exceeds free HBM" if not fits else (
                    "offload optional" if shared < 0.1 * free else "shared set >=10% of free HBM")
                rows.append(CapacityRow(
                    target=target.name, drafter=drafter.name, S=S, P=P, q=q, M=M,
                    hbm_bytes=hbm_bytes, weights_bytes=weights, workspace_bytes=workspace_bytes,
                    free_hbm_bytes=free, shared_bytes=shared, private_bytes_per_req=priv,
                    drafter_bytes_per_req=dr, per_req_bytes=per_req, fits_resident=fits,
                    B_max_resident=b_res, B_max_offloaded=b_off, extra_requests=b_off - b_res,
                    link_s_per_step_per_prefix=S * target.kv_bytes_per_token / beta,
                    note=note,
                ))
    return rows


def row_dict(obj) -> dict:
    return asdict(obj)
