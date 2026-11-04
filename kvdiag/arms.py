"""Attention arms: same semantics, different K/V placement and schedule.

STATUS: UNVERIFIED on GH200 as of 2026-11-03. Written against the FlashInfer 0.2/0.3 API from
memory; run tests/test_arms.py on any CUDA GPU first (host arms work over PCIe via UVA; only
the numbers need the GH200).

Every arm: `run(Q, K, V) -> O` with Q [R, h_q, d], K/V [L, h_kv, d], bf16, no causal mask,
every row attends to all L prefix tokens. Private-suffix attention is not part of the arm.

hbm                 BatchPrefillWithRaggedKVCacheWrapper, ONE request of R rows (planner may
                    split KV for small R; record backend). K/V in HBM.
host_direct         identical call; K/V are host_tensor views (SM reads over C2C).
host_direct_chunked per host chunk: same kernel on K[j0:j1] (a view, no copy) with return_lse,
                    then flashinfer.merge_states. KV-stationary if L2 caches host lines.
host_staged         double-buffered cudaMemcpyAsync (copy engine, via cpu_view) per chunk on a
                    copy stream, kernel per chunk on the compute stream, merge at the end.
host_copy           one memcpy of the whole K/V into HBM buffers, then the hbm kernel. Serial.
"""
from __future__ import annotations

import math

import torch

from kvdiag.shapes import AttnShapes

BF16 = torch.bfloat16
_WORKSPACE = None


def _workspace():
    global _WORKSPACE
    if _WORKSPACE is None:
        _WORKSPACE = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    return _WORKSPACE


def make_q(sh: AttnShapes, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(sh.q, generator=g, dtype=BF16, device="cuda")


def make_kv(sh: AttnShapes, placement: str, seed: int):
    """Identical contents for a given seed regardless of placement."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    K = torch.randn(sh.kv, generator=g, dtype=BF16, device="cuda")
    V = torch.randn(sh.kv, generator=g, dtype=BF16, device="cuda")
    if placement == "hbm":
        return K, V
    assert placement == "host"
    from kvdiag import hostmem
    out = []
    for src in (K, V):
        h = hostmem.host_tensor(sh.kv, BF16)
        h.cpu_view.copy_(src.cpu())
        out.append(h)
    del K, V
    torch.cuda.synchronize()
    return tuple(out)


def pointer_attrs(t: torch.Tensor) -> dict:
    try:
        from kvdiag import hostmem
        return hostmem.pointer_attrs(t.data_ptr())
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)[:100]}


class _Prefill:
    """One ragged-prefill plan for (R rows) x (kv_len tokens)."""

    def __init__(self, sh: AttnShapes, kv_len: int):
        import flashinfer
        self.w = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(_workspace(), "NHD")
        qo = torch.tensor([0, sh.R], dtype=torch.int32, device="cuda")
        kv = torch.tensor([0, kv_len], dtype=torch.int32, device="cuda")
        self.w.plan(qo, kv, sh.h_q, sh.h_kv, sh.d, causal=False, q_data_type=BF16, kv_data_type=BF16)
        self.backend = getattr(self.w, "_backend", None) or type(self.w).__name__

    def __call__(self, Q, K, V, return_lse=False):
        return self.w.run(Q, K, V, return_lse=return_lse)


class HBM:
    placement = "hbm"

    def __init__(self, sh: AttnShapes, chunk_tokens=None):
        self.sh = sh
        self.p = _Prefill(sh, sh.L)
        self.backend = self.p.backend

    def run(self, Q, K, V):
        return self.p(Q, K, V)


class HostDirect(HBM):
    placement = "host_pinned"


class _Chunked:
    def __init__(self, sh: AttnShapes, chunk_tokens: int):
        self.sh, self.ct = sh, chunk_tokens
        self.n_chunks = math.ceil(sh.L / chunk_tokens)
        self.bounds = [(j * chunk_tokens, min(sh.L, (j + 1) * chunk_tokens)) for j in range(self.n_chunks)]
        self.plans = {n: _Prefill(sh, n) for n in {b - a for a, b in self.bounds}}
        self.backend = next(iter(self.plans.values())).backend

    @staticmethod
    def merge(outs, lses):
        import flashinfer
        if len(outs) == 1:
            return outs[0]
        O, _ = flashinfer.merge_states(torch.stack(outs, dim=1), torch.stack(lses, dim=1))
        return O


class HostDirectChunked(_Chunked):
    placement = "host_pinned"

    def run(self, Q, K, V):
        outs, lses = [], []
        for a, b in self.bounds:
            O, lse = self.plans[b - a](Q, K[a:b], V[a:b], return_lse=True)
            outs.append(O); lses.append(lse)
        return self.merge(outs, lses)


class HostStaged(_Chunked):
    placement = "host_pinned"

    def __init__(self, sh, chunk_tokens):
        super().__init__(sh, chunk_tokens)
        shape = (chunk_tokens, sh.h_kv, sh.d)
        self.buf = [(torch.empty(shape, dtype=BF16, device="cuda"), torch.empty(shape, dtype=BF16, device="cuda"))
                    for _ in range(2)]
        self.copy_stream = torch.cuda.Stream()
        self.ev_copied = [torch.cuda.Event() for _ in range(2)]
        self.ev_consumed = [torch.cuda.Event() for _ in range(2)]

    def run(self, Q, K, V):
        Kc, Vc = K.cpu_view, V.cpu_view  # pinned CPU views -> copy engine
        cur = torch.cuda.current_stream()
        outs, lses = [], []
        for j, (a, b) in enumerate(self.bounds):
            i, n = j % 2, b - a
            Kb, Vb = self.buf[i]
            with torch.cuda.stream(self.copy_stream):
                if j >= 2:
                    self.copy_stream.wait_event(self.ev_consumed[i])
                Kb[:n].copy_(Kc[a:b], non_blocking=True)
                Vb[:n].copy_(Vc[a:b], non_blocking=True)
                self.ev_copied[i].record(self.copy_stream)
            cur.wait_event(self.ev_copied[i])
            O, lse = self.plans[n](Q, Kb[:n], Vb[:n], return_lse=True)
            self.ev_consumed[i].record(cur)
            outs.append(O); lses.append(lse)
        return self.merge(outs, lses)


class HostCopy(HBM):
    placement = "host_pinned"

    def __init__(self, sh, chunk_tokens=None):
        super().__init__(sh)
        self.Kb = torch.empty(sh.kv, dtype=BF16, device="cuda")
        self.Vb = torch.empty(sh.kv, dtype=BF16, device="cuda")

    def run(self, Q, K, V):
        self.Kb.copy_(K.cpu_view, non_blocking=True)
        self.Vb.copy_(V.cpu_view, non_blocking=True)
        return self.p(Q, self.Kb, self.Vb)


ARMS = {
    "hbm": HBM, "host_direct": HostDirect, "host_direct_chunked": HostDirectChunked,
    "host_staged": HostStaged, "host_copy": HostCopy,
}


def make_arm(name: str, sh: AttnShapes, chunk_tokens: int | None = None):
    return ARMS[name](sh, chunk_tokens)
