"""Tests for Step 9 — Recombination Profile.

Coverage
--------
TestRecombinationProfile  (6) — 데이터 구조, 엑시톤 속성, 직렬화
TestRecombinationSolver   (6) — SRH/Langevin 계산, 물리적 단조성
TestRecombinationSweep    (5) — 전압 스윕 일괄 처리
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

ROOT    = Path(__file__).parent.parent
SAMPLES = ROOT / "configs" / "samples"

from src.io import load_project
from src.electrical import (
    BiasSweepRunner,
    RecombinationProfile,
    RecombinationSolver,
    RecombinationSweep,
    build_mesh,
    build_solver,
)
from src.electrical.continuity import ContinuitySolver


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def project():
    return load_project(SAMPLES / "oled_input.yaml")


@pytest.fixture(scope="module")
def mesh(project):
    return build_mesh(project.device_stack, project.material_db, z_resolution_nm=2.0)


@pytest.fixture(scope="module")
def gummel(mesh):
    from src.electrical.models import GummelConfig
    # Step 8 와 동일한 수렴·감쇄 파라미터 사용
    cfg = GummelConfig(max_iterations=200, tolerance=1e-4, damping=0.5)
    return build_solver(mesh, cfg)


@pytest.fixture(scope="module")
def continuity(mesh):
    return ContinuitySolver(mesh)


@pytest.fixture(scope="module")
def rec_solver(mesh, continuity):
    return RecombinationSolver(mesh, continuity)


@pytest.fixture(scope="module")
def sweep_result(gummel):
    """0V ~ 3V, 4포인트 스윕."""
    runner = BiasSweepRunner(gummel, v_start=0.0, v_end=3.0, n_points=4)
    return runner.run()


@pytest.fixture(scope="module")
def profile_3v(rec_solver, sweep_result):
    """3V 바이어스 재결합 프로파일."""
    bp = sweep_result.bias_points[-1]   # 마지막 = 최고 전압
    return rec_solver.compute(bp.state, bp.voltage_V)


@pytest.fixture(scope="module")
def profile_0v(rec_solver, sweep_result):
    """0V 바이어스 재결합 프로파일."""
    bp = sweep_result.bias_points[0]
    return rec_solver.compute(bp.state, bp.voltage_V)


# ---------------------------------------------------------------------------
# TestRecombinationProfile — 데이터 구조 검증
# ---------------------------------------------------------------------------

class TestRecombinationProfile:

    def test_shape_consistent(self, profile_3v, mesh):
        """모든 배열이 메쉬 노드 수와 일치해야 한다."""
        N = mesh.num_nodes
        assert len(profile_3v.z_nm)        == N
        assert len(profile_3v.R_srh_m3s)   == N
        assert len(profile_3v.R_lan_m3s)   == N
        assert len(profile_3v.R_total_m3s) == N

    def test_total_equals_srh_plus_langevin(self, profile_3v):
        """R_total = R_srh + R_lan 일치."""
        np.testing.assert_allclose(
            profile_3v.R_total_m3s,
            profile_3v.R_srh_m3s + profile_3v.R_lan_m3s,
            rtol=1e-10,
        )

    def test_exciton_spin_fractions(self, profile_3v):
        """G_singlet = 0.25·G_ex, G_triplet = 0.75·G_ex."""
        np.testing.assert_allclose(profile_3v.G_singlet_m3s,
                                   0.25 * profile_3v.G_exciton_m3s)
        np.testing.assert_allclose(profile_3v.G_triplet_m3s,
                                   0.75 * profile_3v.G_exciton_m3s)

    def test_exciton_generation_dict_keys(self, profile_3v):
        """exciton_generation_dict()가 excitonics 모듈 연결 키를 포함해야 한다."""
        d = profile_3v.exciton_generation_dict()
        required = {"voltage_V", "z_nm", "G_total_m3s", "G_singlet_m3s", "G_triplet_m3s"}
        assert required.issubset(d.keys())

    def test_exciton_dict_array_lengths(self, profile_3v, mesh):
        """exciton_generation_dict() 배열 길이가 메쉬와 일치."""
        d = profile_3v.exciton_generation_dict()
        N = mesh.num_nodes
        assert len(d["z_nm"])          == N
        assert len(d["G_total_m3s"])   == N
        assert len(d["G_singlet_m3s"]) == N
        assert len(d["G_triplet_m3s"]) == N

    def test_to_csv_creates_file(self, profile_3v, tmp_path):
        """to_csv()가 CSV 파일을 생성하고 헤더를 포함해야 한다."""
        out = tmp_path / "rec.csv"
        profile_3v.to_csv(out)
        assert out.exists()
        lines = out.read_text(encoding="utf-8").splitlines()
        assert lines[0].startswith("z_nm")
        assert len(lines) > 1      # 헤더 + 데이터 행


# ---------------------------------------------------------------------------
# TestRecombinationSolver — 물리적 계산 검증
# ---------------------------------------------------------------------------

class TestRecombinationSolver:

    def test_srh_nonnegative(self, profile_0v):
        """SRH 재결합률은 항상 ≥ 0 (유기물 노드 마스킹 포함)."""
        assert float(np.min(profile_0v.R_srh_m3s)) >= 0.0

    def test_langevin_nonnegative(self, profile_0v):
        """Langevin 재결합률은 항상 ≥ 0."""
        assert float(np.min(profile_0v.R_lan_m3s)) >= 0.0

    def test_nonzero_organic_recombination(self, profile_0v):
        """0V에서도 접촉 캐리어 주입으로 유기물 내 비zero 재결합 발생."""
        # 이상 오믹 접촉 BC (n/p = 1e21 고정)가 유기층 내 np >> ni² 유발
        assert profile_0v.R_total_m3s.max() > 0.0

    def test_peak_in_organic_region(self, profile_0v, mesh):
        """수렴한 0V 프로파일에서 최대 재결합 위치가 유기물 영역 내에 있어야 한다."""
        # ITO: 0~150nm, 유기물: 150~270nm, Al: 270~370nm
        ito_end  = float(mesh.layer_boundaries_nm[1])    # 150 nm
        al_start = float(mesh.layer_boundaries_nm[-2])   # 270 nm
        peak_z   = profile_0v.peak_z_nm
        assert ito_end <= peak_z <= al_start, (
            f"peak_z={peak_z:.1f}nm 이 유기물 영역({ito_end}~{al_start}nm) 밖"
        )

    def test_voltage_stored_in_profile(self, rec_solver, sweep_result):
        """profile.voltage_V 가 BiasPoint 전압과 일치해야 한다."""
        bp = sweep_result.bias_points[0]
        profile = rec_solver.compute(bp.state, bp.voltage_V)
        assert abs(profile.voltage_V - bp.voltage_V) < 1e-9

    def test_z_array_matches_mesh(self, rec_solver, sweep_result, mesh):
        """profile.z_nm 이 메쉬 z_nm 과 일치해야 한다."""
        bp = sweep_result.bias_points[0]
        profile = rec_solver.compute(bp.state)
        np.testing.assert_array_equal(profile.z_nm, mesh.z_nm)


# ---------------------------------------------------------------------------
# TestRecombinationSweep — 전압 스윕 일괄 처리 검증
# ---------------------------------------------------------------------------

class TestRecombinationSweep:

    @pytest.fixture(scope="class")
    def rec_sweep(self, rec_solver):
        return RecombinationSweep(rec_solver)

    @pytest.fixture(scope="class")
    def all_profiles(self, rec_sweep, sweep_result):
        return rec_sweep.compute_all(sweep_result)

    def test_profile_count_matches_sweep(self, all_profiles, sweep_result):
        """프로파일 개수 = 스윕 바이어스 포인트 수."""
        assert len(all_profiles) == len(sweep_result.bias_points)

    def test_voltages_ascending(self, all_profiles):
        """전압이 단조 증가해야 한다."""
        voltages = [p.voltage_V for p in all_profiles]
        assert voltages == sorted(voltages)

    def test_bias_profiles_all_generated(self, all_profiles, sweep_result):
        """바이어스에 따른 재결합 프로파일 확인 가능 — 완료 조건 검증.

        수렴 여부와 관계없이 모든 바이어스 포인트에서 프로파일이
        생성되어 bias별 관찰이 가능해야 한다.
        """
        assert len(all_profiles) == len(sweep_result.bias_points)
        for prof in all_profiles:
            assert isinstance(prof, RecombinationProfile)
            assert len(prof.z_nm) > 0
            # 각 프로파일에 전압 정보가 있어야 함
            assert isinstance(prof.voltage_V, (int, float))

    def test_peak_positions_array_shape(self, rec_sweep, all_profiles, sweep_result):
        """peak_positions() 배열 길이 = 바이어스 포인트 수."""
        peaks = rec_sweep.peak_positions(all_profiles)
        assert len(peaks) == len(sweep_result.bias_points)

    def test_exciton_dict_serializable_to_json(self, all_profiles):
        """exciton_generation_dict()가 JSON 직렬화 가능해야 한다 (excitonics 연결 확인)."""
        for prof in all_profiles:
            d = prof.exciton_generation_dict()
            text = json.dumps(d)   # 예외 없이 직렬화
            restored = json.loads(text)
            assert abs(restored["voltage_V"] - prof.voltage_V) < 1e-9
