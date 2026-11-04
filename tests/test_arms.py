"""Correctness gate. The numpy reference and the LSE merge run anywhere; arm tests need CUDA
(any GPU: host arms work over PCIe via UVA; only the timings need a GH200)."""
import math

import numpy as np
import pytest

from kvdiag.shapes import load_dims, make_shapes

REL_L2_TOL = 1e-2


def ref_attention(Q, K, V):
    """fp32 dense GQA attention. Q [R,h_q,d], K/V [L,h_kv,d] -> O [R,h_q,d], lse [R,h_q] (natural log)."""
    R, h_q, d = Q.shape
    L, h_kv, _ = K.shape
    G = h_q // h_kv
    Kx = np.repeat(K, G, axis=1)  # head expansion is fine in the reference, never in a timed arm
    Vx = np.repeat(V, G, axis=1)
    s = np.einsum("rhd,lhd->hrl", Q, Kx) / math.sqrt(d)
    m = s.max(-1, keepdims=True)
    p = np.exp(s - m)
    lse = (m[..., 0] + np.log(p.sum(-1))).T  # [R, h_q]
    O = np.einsum("hrl,lhd->rhd", p / p.sum(-1, keepdims=True), Vx)
    return O.astype(np.float32), lse.astype(np.float32)


def merge_lse(outs, lses):
    """Combine independently normalized partial outputs via log-sum-exp statistics (fp32)."""
    lse = np.logaddexp.reduce(np.stack(lses), axis=0)
    O = sum(np.exp(l - lse)[..., None] * o for o, l in zip(outs, lses))
    return O.astype(np.float32), lse


def rel_l2(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def _rand(shape, seed):
    return np.random.default_rng(seed).standard_normal(shape, dtype=np.float32)


@pytest.mark.parametrize("R,L", [(1, 64), (16, 512), (32, 512), (3, 1)])
def test_reference_matches_naive_softmax(R, L):
    Q, K, V = _rand((R, 8, 16), 0), _rand((L, 2, 16), 1), _rand((L, 2, 16), 2)
    O, lse = ref_attention(Q, K, V)
    # independent check for one head
    h, g = 5, 5 // 4
    s = Q[:, h] @ K[:, g].T / 4.0
    p = np.exp(s - s.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
    assert rel_l2(O[:, h], p @ V[:, g]) < 1e-5
    assert np.isfinite(O).all() and np.isfinite(lse).all()


@pytest.mark.parametrize("n_chunks", [2, 3, 8])
def test_lse_merge_equals_full(n_chunks):
    Q, K, V = _rand((32, 8, 16), 0), _rand((512, 2, 16), 1), _rand((512, 2, 16), 2)
    full, _ = ref_attention(Q, K, V)
    bounds = np.linspace(0, 512, n_chunks + 1).astype(int)
    parts = [ref_attention(Q, K[a:b], V[a:b]) for a, b in zip(bounds[:-1], bounds[1:])]
    O, _ = merge_lse([p[0] for p in parts], [p[1] for p in parts])
    assert rel_l2(O, full) < 1e-5
    # and that naive addition of normalized partials is wrong (guards the merge formula)
    assert rel_l2(sum(p[0] for p in parts), full) > 1e-2


# ----------------------------------------------------------------------------- GPU arms

torch = pytest.importorskip("torch")
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
SMALL = dict(R=32, L=512)   # B=2 requests x q=16, S=512


@pytest.fixture(scope="module")
def small():
    from kvdiag import arms as A
    sh = make_shapes(load_dims("qwen3-8b"), **SMALL)
    Q = A.make_q(sh, seed=7)
    K, V = A.make_kv(sh, "hbm", seed=0)
    Kh, Vh = A.make_kv(sh, "host", seed=0)
    assert torch.equal(K.cpu(), Kh.cpu_view) and torch.equal(V.cpu(), Vh.cpu_view)
    ref, _ = ref_attention(Q.float().cpu().numpy(), K.float().cpu().numpy(), V.float().cpu().numpy())
    return sh, Q, (K, V), (Kh, Vh), ref


@cuda
@pytest.mark.parametrize("arm", ["hbm", "host_direct", "host_direct_chunked", "host_staged", "host_copy"])
def test_arm_matches_reference(small, arm):
    from kvdiag import arms as A
    sh, Q, hbm, host, ref = small
    K, V = hbm if arm == "hbm" else host
    obj = A.make_arm(arm, sh, chunk_tokens=128)   # 4 chunks so the merge runs
    mem0 = torch.cuda.memory_allocated()
    O = obj.run(Q, K, V)
    torch.cuda.synchronize()
    out = O.float().cpu().numpy()
    assert np.isfinite(out).all()
    assert rel_l2(out, ref) < REL_L2_TOL, (arm, rel_l2(out, ref), float(np.abs(out - ref).max()))
    if arm in ("host_direct", "host_direct_chunked"):
        assert torch.cuda.memory_allocated() - mem0 < sh.kv_bytes // 4, "host arm silently copied K/V to HBM"


@cuda
def test_host_pointer_is_mapped_host_memory(small):
    from kvdiag import arms as A
    _, _, hbm, host, _ = small
    a = A.pointer_attrs(host[0])
    assert a["type"] == "host", a
    assert A.pointer_attrs(hbm[0])["type"] == "device"


@cuda
def test_q1_decode_anchor_runs(small):
    from kvdiag import arms as A
    sh = make_shapes(load_dims("qwen3-8b"), R=1, L=512)
    Q = A.make_q(sh, 1); K, V = A.make_kv(sh, "hbm", 0)
    O = A.make_arm("hbm", sh).run(Q, K, V)
    ref, _ = ref_attention(Q.float().cpu().numpy(), K.float().cpu().numpy(), V.float().cpu().numpy())
    assert rel_l2(O.float().cpu().numpy(), ref) < REL_L2_TOL
