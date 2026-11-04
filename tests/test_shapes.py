import math
import pytest

from kvdiag.shapes import (load_dims, make_shapes, t_pred_s, link_bytes, q_tiles, crossover_R,
                           capacity_rows, GH200_ASSUMED, ARMS)


@pytest.fixture(scope="module")
def dims():
    return load_dims("qwen3-8b"), load_dims("qwen3-8b-dflash-b16")


def test_dims_from_config(dims):
    t, d = dims
    assert (t.layers, t.h_q, t.h_kv, t.d, t.G) == (36, 32, 8, 128, 4)
    assert t.kv_bytes_per_token == 144 * 1024
    assert 8.0e9 < t.params < 8.4e9          # Qwen3-8B is 8.2B
    assert (d.layers, d.block_size) == (5, 16)
    assert d.kv_bytes_per_token == 20 * 1024
    assert t.revision and d.revision


@pytest.mark.parametrize("R", [1, 7, 16, 128, 4096])
@pytest.mark.parametrize("L", [512, 8192, 32768])
def test_intensity_is_G_times_R(dims, R, L):
    sh = make_shapes(dims[0], R, L)
    assert math.isclose(sh.intensity, sh.G * R)
    assert sh.kv_bytes == 2 * L * 8 * 128 * 2


def test_L32k_kv_is_128MiB(dims):
    assert make_shapes(dims[0], 1, 32768).kv_bytes == 128 * 2**20


def test_predictions_ordered(dims):
    m = GH200_ASSUMED
    for R in (1, 16, 128, 1024, 4096):
        sh = make_shapes(dims[0], R, 32768)
        t = {a: t_pred_s(sh, a, m, chunk_tokens=4096) for a in ARMS}
        assert all(v > 0 for v in t.values())
        assert t["hbm"] <= min(t[a] for a in ARMS if a != "hbm")
        assert t["host_staged"] <= t["host_copy"] * 1.05   # overlap never worse than serial


def test_host_direct_rereads_per_tile(dims):
    m = GH200_ASSUMED
    sh = make_shapes(dims[0], 4096, 32768)
    assert q_tiles(sh, m.tile_rows) == 4096 * 4 // 128
    assert link_bytes(sh, "host_direct", m) == sh.kv_bytes * q_tiles(sh, m.tile_rows)
    assert link_bytes(sh, "host_staged", m) == sh.kv_bytes


def test_crossover(dims):
    m = GH200_ASSUMED
    assert math.isclose(crossover_R(dims[0], m), 400e12 / (400e9 * 4))   # 250
    # at the predicted crossover, staged time is within 2x of hbm time (fill + launch overhead)
    sh = make_shapes(dims[0], 256, 32768)
    assert t_pred_s(sh, "host_staged", m, chunk_tokens=4096) < 2 * t_pred_s(sh, "hbm", m)


def test_capacity_rows(dims):
    t, d = dims
    rows = capacity_rows(t, d, 96 * 2**30, 2 * 2**30, [8192, 32768], [256, 4096], [1, 4, 16, 64], 16, 400e9)
    assert len(rows) == 16
    r1 = next(r for r in rows if (r.S, r.P, r.M) == (32768, 4096, 1))
    assert math.isclose(r1.shared_bytes / 2**30, 4.5, rel_tol=0.01)
    assert r1.fits_resident and r1.extra_requests >= 0
    r64 = next(r for r in rows if (r.S, r.P, r.M) == (32768, 4096, 64))
    assert not r64.fits_resident and r64.B_max_resident == 0
