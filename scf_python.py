#!/usr/bin/env python3
"""Python translation of the Szabo/Ostlund HeH+ STO-nG SCF toy code.

This script mirrors the structure of the C implementation in this repository:
- integral setup (`calc_integrals.c`)
- formatting into matrices (`format_integrals.c`)
- SCF loop (`perform_scf.c`)

It intentionally favors readability over performance so the algorithmic steps are easy
for learners to inspect.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import List

MAX_ELEC = 2
PI = math.pi


@dataclass
class Settings:
    sto_ng: int = 3
    bond_len: float = 1.4632
    zeta_a: float = 2.0925
    zeta_b: float = 1.24
    z_a: float = 2.0
    z_b: float = 1.0
    conv_crit: float = 1e-4
    max_iter: int = 25


@dataclass
class Integrals:
    s12: float = 0.0
    t11: float = 0.0
    t12: float = 0.0
    t22: float = 0.0
    v11_nuc_a: float = 0.0
    v12_nuc_a: float = 0.0
    v22_nuc_a: float = 0.0
    v11_nuc_b: float = 0.0
    v12_nuc_b: float = 0.0
    v22_nuc_b: float = 0.0
    v1111: float = 0.0
    v2111: float = 0.0
    v2121: float = 0.0
    v2211: float = 0.0
    v2221: float = 0.0
    v2222: float = 0.0


@dataclass
class ScfState:
    s_mat: List[List[float]] = field(default_factory=lambda: [[0.0, 0.0], [0.0, 0.0]])
    x_mat: List[List[float]] = field(default_factory=lambda: [[0.0, 0.0], [0.0, 0.0]])
    xt_mat: List[List[float]] = field(default_factory=lambda: [[0.0, 0.0], [0.0, 0.0]])
    h_mat: List[List[float]] = field(default_factory=lambda: [[0.0, 0.0], [0.0, 0.0]])
    f_mat: List[List[float]] = field(default_factory=lambda: [[0.0, 0.0], [0.0, 0.0]])
    g_mat: List[List[float]] = field(default_factory=lambda: [[0.0, 0.0], [0.0, 0.0]])
    c_mat: List[List[float]] = field(default_factory=lambda: [[0.0, 0.0], [0.0, 0.0]])
    fprime_mat: List[List[float]] = field(default_factory=lambda: [[0.0, 0.0], [0.0, 0.0]])
    cprime_mat: List[List[float]] = field(default_factory=lambda: [[0.0, 0.0], [0.0, 0.0]])
    dens_mat: List[List[float]] = field(default_factory=lambda: [[0.0, 0.0], [0.0, 0.0]])
    olddens_mat: List[List[float]] = field(default_factory=lambda: [[0.0, 0.0], [0.0, 0.0]])
    energy_mat: List[List[float]] = field(default_factory=lambda: [[0.0, 0.0], [0.0, 0.0]])
    mulliken_mat: List[List[float]] = field(default_factory=lambda: [[0.0, 0.0], [0.0, 0.0]])
    tt_mat: List[List[List[List[float]]]] = field(
        default_factory=lambda: [[[[0.0 for _ in range(MAX_ELEC)] for _ in range(MAX_ELEC)] for _ in range(MAX_ELEC)] for _ in range(MAX_ELEC)]
    )


COEF_STO_NG_BASIS = [
    [1.000000, 0.000000, 0.000000],
    [0.678914, 0.430129, 0.000000],
    [0.444635, 0.535328, 0.154329],
]

EXP_STO_NG_BASIS = [
    [0.270950, 0.000000, 0.000000],
    [0.151623, 0.851819, 0.000000],
    [0.109818, 0.405771, 2.227660],
]


def overlap(exp_a: float, exp_b: float, r2: float) -> float:
    return (PI / (exp_a + exp_b)) ** 1.5 * math.exp(-exp_a * exp_b * r2 / (exp_a + exp_b))


def kinetic_energy(exp_a: float, exp_b: float, r2: float) -> float:
    pref = exp_a * exp_b / (exp_a + exp_b)
    bracket = 3.0 - (2.0 * exp_a * exp_b * r2 / (exp_a + exp_b))
    return pref * bracket * (PI / (exp_a + exp_b)) ** 1.5 * math.exp(-exp_a * exp_b * r2 / (exp_a + exp_b))


def f0(arg: float) -> float:
    # Small-argument expansion used in original C code
    if arg >= 1e-6:
        return 0.5 * math.sqrt(PI / arg) * math.erf(math.sqrt(arg))
    return 1.0 - arg / 3.0


def one_elec_nuc_attr(exp_a: float, exp_b: float, rab2: float, rcp2: float, zc: float) -> float:
    return 2.0 * (PI / (exp_a + exp_b)) * f0((exp_a + exp_b) * rcp2) * math.exp(-exp_a * exp_b * rab2 / (exp_a + exp_b)) * (-zc)


def two_elec_repulsion(exp_a: float, exp_b: float, exp_c: float, exp_d: float, rab2: float, rcd2: float, rpq2: float) -> float:
    denom = (exp_a + exp_b) * (exp_c + exp_d) * math.sqrt(exp_a + exp_b + exp_c + exp_d)
    boys_arg = (exp_a + exp_b) * (exp_c + exp_d) * rpq2 / (exp_a + exp_b + exp_c + exp_d)
    return 2.0 * (PI ** 2.5) / denom * f0(boys_arg) * math.exp(
        -exp_a * exp_b * rab2 / (exp_a + exp_b) - exp_c * exp_d * rcd2 / (exp_c + exp_d)
    )


def prepare_basis(settings: Settings) -> tuple[List[float], List[float], List[float], List[float]]:
    row = settings.sto_ng - 1
    scaled_exp_a, scaled_exp_b = [], []
    renorm_a, renorm_b = [], []
    for i in range(settings.sto_ng):
        ea = (settings.zeta_a ** 2) * EXP_STO_NG_BASIS[row][i]
        eb = (settings.zeta_b ** 2) * EXP_STO_NG_BASIS[row][i]
        scaled_exp_a.append(ea)
        scaled_exp_b.append(eb)
        renorm_a.append((2.0 * ea / PI) ** 0.75 * COEF_STO_NG_BASIS[row][i])
        renorm_b.append((2.0 * eb / PI) ** 0.75 * COEF_STO_NG_BASIS[row][i])
    return scaled_exp_a, scaled_exp_b, renorm_a, renorm_b


def calc_integrals(settings: Settings) -> Integrals:
    ints = Integrals()
    bond_len_sq = settings.bond_len * settings.bond_len
    scaled_a, scaled_b, renorm_a, renorm_b = prepare_basis(settings)

    for i in range(settings.sto_ng):
        for j in range(settings.sto_ng):
            r_ap = (scaled_b[j] * settings.bond_len) / (scaled_a[i] + scaled_b[j])
            r_ap_sq = r_ap * r_ap
            r_bp_sq = (settings.bond_len - r_ap) ** 2

            ints.s12 += renorm_a[i] * renorm_b[j] * overlap(scaled_a[i], scaled_b[j], bond_len_sq)
            ints.t11 += renorm_a[i] * renorm_a[j] * kinetic_energy(scaled_a[i], scaled_a[j], 0.0)
            ints.t12 += renorm_a[i] * renorm_b[j] * kinetic_energy(scaled_a[i], scaled_b[j], bond_len_sq)
            ints.t22 += renorm_b[i] * renorm_b[j] * kinetic_energy(scaled_b[i], scaled_b[j], 0.0)

            ints.v11_nuc_a += renorm_a[i] * renorm_a[j] * one_elec_nuc_attr(scaled_a[i], scaled_a[j], 0.0, 0.0, settings.z_a)
            ints.v12_nuc_a += renorm_a[i] * renorm_b[j] * one_elec_nuc_attr(scaled_a[i], scaled_b[j], bond_len_sq, r_ap_sq, settings.z_a)
            ints.v22_nuc_a += renorm_b[i] * renorm_b[j] * one_elec_nuc_attr(scaled_b[i], scaled_b[j], 0.0, bond_len_sq, settings.z_a)
            ints.v11_nuc_b += renorm_a[i] * renorm_a[j] * one_elec_nuc_attr(scaled_a[i], scaled_a[j], 0.0, bond_len_sq, settings.z_b)
            ints.v12_nuc_b += renorm_a[i] * renorm_b[j] * one_elec_nuc_attr(scaled_a[i], scaled_b[j], bond_len_sq, r_bp_sq, settings.z_b)
            ints.v22_nuc_b += renorm_b[i] * renorm_b[j] * one_elec_nuc_attr(scaled_b[i], scaled_b[j], 0.0, 0.0, settings.z_b)

    for i in range(settings.sto_ng):
        for j in range(settings.sto_ng):
            for k in range(settings.sto_ng):
                for l in range(settings.sto_ng):
                    r_ap = scaled_b[i] * settings.bond_len / (scaled_b[i] + scaled_a[j])
                    r_bp = settings.bond_len - r_ap
                    r_aq = scaled_b[k] * settings.bond_len / (scaled_b[k] + scaled_a[l])
                    r_bq = settings.bond_len - r_aq
                    r_pq = r_ap - r_aq

                    ints.v1111 += renorm_a[i] * renorm_a[j] * renorm_a[k] * renorm_a[l] * two_elec_repulsion(scaled_a[i], scaled_a[j], scaled_a[k], scaled_a[l], 0.0, 0.0, 0.0)
                    ints.v2111 += renorm_b[i] * renorm_a[j] * renorm_a[k] * renorm_a[l] * two_elec_repulsion(scaled_b[i], scaled_a[j], scaled_a[k], scaled_a[l], bond_len_sq, 0.0, r_ap**2)
                    ints.v2121 += renorm_b[i] * renorm_a[j] * renorm_b[k] * renorm_a[l] * two_elec_repulsion(scaled_b[i], scaled_a[j], scaled_b[k], scaled_a[l], bond_len_sq, bond_len_sq, r_pq**2)
                    ints.v2211 += renorm_b[i] * renorm_b[j] * renorm_a[k] * renorm_a[l] * two_elec_repulsion(scaled_b[i], scaled_b[j], scaled_a[k], scaled_a[l], 0.0, 0.0, bond_len_sq)
                    ints.v2221 += renorm_b[i] * renorm_b[j] * renorm_b[k] * renorm_a[l] * two_elec_repulsion(scaled_b[i], scaled_b[j], scaled_b[k], scaled_a[l], 0.0, bond_len_sq, r_bq**2)
                    ints.v2222 += renorm_b[i] * renorm_b[j] * renorm_b[k] * renorm_b[l] * two_elec_repulsion(scaled_b[i], scaled_b[j], scaled_b[k], scaled_b[l], 0.0, 0.0, 0.0)

    return ints


def matmul_2x2(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    return [
        [a[0][0] * b[0][0] + a[0][1] * b[1][0], a[0][0] * b[0][1] + a[0][1] * b[1][1]],
        [a[1][0] * b[0][0] + a[1][1] * b[1][0], a[1][0] * b[0][1] + a[1][1] * b[1][1]],
    ]


def diag_2x2(fprime: List[List[float]]) -> tuple[List[List[float]], List[List[float]]]:
    if abs(fprime[0][0] - fprime[1][1]) < 1e-20:
        theta = PI / 4
    else:
        theta = 0.5 * math.atan(2.0 * fprime[0][1] / (fprime[0][0] - fprime[1][1]))

    cprime = [
        [math.cos(theta), math.sin(theta)],
        [math.sin(theta), -math.cos(theta)],
    ]

    energy = [[0.0, 0.0], [0.0, 0.0]]
    energy[0][0] = fprime[0][0] * math.cos(theta) ** 2 + fprime[1][1] * math.sin(theta) ** 2 + fprime[0][1] * math.sin(2 * theta)
    energy[1][1] = fprime[1][1] * math.cos(theta) ** 2 + fprime[0][0] * math.sin(theta) ** 2 - fprime[0][1] * math.sin(2 * theta)

    if energy[1][1] < energy[0][0]:
        energy[0][0], energy[1][1] = energy[1][1], energy[0][0]
        cprime[0][0], cprime[0][1] = cprime[0][1], cprime[0][0]
        cprime[1][0], cprime[1][1] = cprime[1][1], cprime[1][0]

    return cprime, energy


def setup_matrices(ints: Integrals, state: ScfState) -> None:
    state.h_mat = [
        [ints.t11 + ints.v11_nuc_a + ints.v11_nuc_b, ints.t12 + ints.v12_nuc_a + ints.v12_nuc_b],
        [ints.t12 + ints.v12_nuc_a + ints.v12_nuc_b, ints.t22 + ints.v22_nuc_a + ints.v22_nuc_b],
    ]

    state.s_mat = [[1.0, ints.s12], [ints.s12, 1.0]]

    state.x_mat = [
        [1.0 / math.sqrt(2.0 * (1.0 + ints.s12)), 1.0 / math.sqrt(2.0 * (1.0 - ints.s12))],
        [1.0 / math.sqrt(2.0 * (1.0 + ints.s12)), -1.0 / math.sqrt(2.0 * (1.0 - ints.s12))],
    ]
    state.xt_mat = [[state.x_mat[0][0], state.x_mat[1][0]], [state.x_mat[0][1], state.x_mat[1][1]]]

    tt = state.tt_mat
    tt[0][0][0][0] = ints.v1111
    tt[1][0][0][0] = tt[0][1][0][0] = tt[0][0][1][0] = tt[0][0][0][1] = ints.v2111
    tt[1][0][1][0] = tt[0][1][1][0] = tt[1][0][0][1] = tt[0][1][0][1] = ints.v2121
    tt[1][1][0][0] = tt[0][0][1][1] = ints.v2211
    tt[1][1][1][0] = tt[1][1][0][1] = tt[1][0][1][1] = tt[0][1][1][1] = ints.v2221
    tt[1][1][1][1] = ints.v2222


def form_g(state: ScfState) -> None:
    for i in range(MAX_ELEC):
        for j in range(MAX_ELEC):
            total = 0.0
            for k in range(MAX_ELEC):
                for l in range(MAX_ELEC):
                    total += state.dens_mat[k][l] * (state.tt_mat[i][j][k][l] - 0.5 * state.tt_mat[i][l][k][j])
            state.g_mat[i][j] = total


def run_scf(settings: Settings, state: ScfState) -> tuple[float, float, int, bool]:
    converged = False
    elec_energy = 0.0

    for iteration in range(1, settings.max_iter + 1):
        form_g(state)

        for i in range(MAX_ELEC):
            for j in range(MAX_ELEC):
                state.f_mat[i][j] = state.h_mat[i][j] + state.g_mat[i][j]

        elec_energy = 0.0
        for i in range(MAX_ELEC):
            for j in range(MAX_ELEC):
                elec_energy += 0.5 * state.dens_mat[i][j] * (state.h_mat[i][j] + state.f_mat[i][j])

        state.fprime_mat = matmul_2x2(state.xt_mat, matmul_2x2(state.f_mat, state.x_mat))
        state.cprime_mat, state.energy_mat = diag_2x2(state.fprime_mat)
        state.c_mat = matmul_2x2(state.x_mat, state.cprime_mat)

        for i in range(MAX_ELEC):
            for j in range(MAX_ELEC):
                state.olddens_mat[i][j] = state.dens_mat[i][j]
                state.dens_mat[i][j] = 2.0 * state.c_mat[i][0] * state.c_mat[j][0]

        delta_sq = 0.0
        for i in range(MAX_ELEC):
            for j in range(MAX_ELEC):
                delta_sq += (state.dens_mat[i][j] - state.olddens_mat[i][j]) ** 2
        delta = math.sqrt(delta_sq / 4.0)

        if delta < settings.conv_crit:
            converged = True
            return elec_energy, elec_energy + (settings.z_a * settings.z_b) / settings.bond_len, iteration, converged

    return elec_energy, elec_energy + (settings.z_a * settings.z_b) / settings.bond_len, settings.max_iter, converged


def print_matrix(name: str, mat: List[List[float]]) -> None:
    print(f"{name}:")
    for row in mat:
        print("  " + " ".join(f"{v: .6f}" for v in row))


def main() -> None:
    settings = Settings()
    state = ScfState()

    ints = calc_integrals(settings)
    setup_matrices(ints, state)
    elec_energy, total_energy, n_iter, converged = run_scf(settings, state)

    state.mulliken_mat = matmul_2x2(state.dens_mat, state.s_mat)

    print("Python translation of SCF_Szabo (HeH+, STO-nG, 2x2 RHF)")
    print(f"Converged: {converged} in {n_iter} iterations")
    print(f"Electronic Energy: {elec_energy:.6f}")
    print(f"Total Energy:      {total_energy:.6f}")
    print_matrix("P (density)", state.dens_mat)
    print_matrix("F (Fock)", state.f_mat)
    print_matrix("C (coefficients)", state.c_mat)
    print_matrix("Mulliken", state.mulliken_mat)


if __name__ == "__main__":
    main()
