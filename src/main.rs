const MAX_ELEC: usize = 2;
const PI: f64 = std::f64::consts::PI;

#[derive(Clone, Copy)]
struct Settings {
    sto_ng: usize,
    bond_len: f64,
    zeta_a: f64,
    zeta_b: f64,
    z_a: f64,
    z_b: f64,
    conv_crit: f64,
    max_iter: usize,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            sto_ng: 3,
            bond_len: 1.4632,
            zeta_a: 2.0925,
            zeta_b: 1.24,
            z_a: 2.0,
            z_b: 1.0,
            conv_crit: 1e-4,
            max_iter: 25,
        }
    }
}

#[derive(Default)]
struct Integrals {
    s12: f64,
    t11: f64,
    t12: f64,
    t22: f64,
    v11_nuc_a: f64,
    v12_nuc_a: f64,
    v22_nuc_a: f64,
    v11_nuc_b: f64,
    v12_nuc_b: f64,
    v22_nuc_b: f64,
    v1111: f64,
    v2111: f64,
    v2121: f64,
    v2211: f64,
    v2221: f64,
    v2222: f64,
}

#[derive(Clone)]
struct ScfState {
    s_mat: [[f64; MAX_ELEC]; MAX_ELEC],
    x_mat: [[f64; MAX_ELEC]; MAX_ELEC],
    xt_mat: [[f64; MAX_ELEC]; MAX_ELEC],
    h_mat: [[f64; MAX_ELEC]; MAX_ELEC],
    f_mat: [[f64; MAX_ELEC]; MAX_ELEC],
    g_mat: [[f64; MAX_ELEC]; MAX_ELEC],
    c_mat: [[f64; MAX_ELEC]; MAX_ELEC],
    fprime_mat: [[f64; MAX_ELEC]; MAX_ELEC],
    cprime_mat: [[f64; MAX_ELEC]; MAX_ELEC],
    dens_mat: [[f64; MAX_ELEC]; MAX_ELEC],
    olddens_mat: [[f64; MAX_ELEC]; MAX_ELEC],
    energy_mat: [[f64; MAX_ELEC]; MAX_ELEC],
    mulliken_mat: [[f64; MAX_ELEC]; MAX_ELEC],
    tt_mat: [[[[f64; MAX_ELEC]; MAX_ELEC]; MAX_ELEC]; MAX_ELEC],
}

impl Default for ScfState {
    fn default() -> Self {
        Self {
            s_mat: [[0.0; MAX_ELEC]; MAX_ELEC],
            x_mat: [[0.0; MAX_ELEC]; MAX_ELEC],
            xt_mat: [[0.0; MAX_ELEC]; MAX_ELEC],
            h_mat: [[0.0; MAX_ELEC]; MAX_ELEC],
            f_mat: [[0.0; MAX_ELEC]; MAX_ELEC],
            g_mat: [[0.0; MAX_ELEC]; MAX_ELEC],
            c_mat: [[0.0; MAX_ELEC]; MAX_ELEC],
            fprime_mat: [[0.0; MAX_ELEC]; MAX_ELEC],
            cprime_mat: [[0.0; MAX_ELEC]; MAX_ELEC],
            dens_mat: [[0.0; MAX_ELEC]; MAX_ELEC],
            olddens_mat: [[0.0; MAX_ELEC]; MAX_ELEC],
            energy_mat: [[0.0; MAX_ELEC]; MAX_ELEC],
            mulliken_mat: [[0.0; MAX_ELEC]; MAX_ELEC],
            tt_mat: [[[[0.0; MAX_ELEC]; MAX_ELEC]; MAX_ELEC]; MAX_ELEC],
        }
    }
}

const COEF: [[f64; 3]; 3] = [
    [1.0, 0.0, 0.0],
    [0.678914, 0.430129, 0.0],
    [0.444635, 0.535328, 0.154329],
];
const EXP: [[f64; 3]; 3] = [
    [0.270950, 0.0, 0.0],
    [0.151623, 0.851819, 0.0],
    [0.109818, 0.405771, 2.227660],
];

fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    sign * y
}

fn f0(arg: f64) -> f64 {
    if arg >= 1e-6 {
        0.5 * (PI / arg).sqrt() * erf(arg.sqrt())
    } else {
        1.0 - arg / 3.0
    }
}
fn overlap(a: f64, b: f64, r2: f64) -> f64 {
    (PI / (a + b)).powf(1.5) * (-(a * b * r2) / (a + b)).exp()
}
fn kinetic(a: f64, b: f64, r2: f64) -> f64 {
    let p = a * b / (a + b);
    p * (3.0 - 2.0 * a * b * r2 / (a + b)) * overlap(a, b, r2)
}
fn nuc_attr(a: f64, b: f64, rab2: f64, rcp2: f64, zc: f64) -> f64 {
    2.0 * PI / (a + b) * f0((a + b) * rcp2) * (-(a * b * rab2) / (a + b)).exp() * -zc
}
fn two_elec(a: f64, b: f64, c: f64, d: f64, rab2: f64, rcd2: f64, rpq2: f64) -> f64 {
    let denom = (a + b) * (c + d) * (a + b + c + d).sqrt();
    let boys_arg = (a + b) * (c + d) * rpq2 / (a + b + c + d);
    2.0 * PI.powf(2.5) / denom
        * f0(boys_arg)
        * (-(a * b * rab2) / (a + b) - (c * d * rcd2) / (c + d)).exp()
}

fn prepare_basis(s: Settings) -> ([f64; 3], [f64; 3], [f64; 3], [f64; 3]) {
    let row = s.sto_ng - 1;
    let mut ea = [0.0; 3];
    let mut eb = [0.0; 3];
    let mut na = [0.0; 3];
    let mut nb = [0.0; 3];
    for i in 0..s.sto_ng {
        ea[i] = s.zeta_a.powi(2) * EXP[row][i];
        eb[i] = s.zeta_b.powi(2) * EXP[row][i];
        na[i] = (2.0 * ea[i] / PI).powf(0.75) * COEF[row][i];
        nb[i] = (2.0 * eb[i] / PI).powf(0.75) * COEF[row][i];
    }
    (ea, eb, na, nb)
}

fn calc_integrals(s: Settings) -> Integrals {
    let mut ints = Integrals::default();
    let r2 = s.bond_len * s.bond_len;
    let (ea, eb, na, nb) = prepare_basis(s);
    for i in 0..s.sto_ng {
        for j in 0..s.sto_ng {
            let r_ap = eb[j] * s.bond_len / (ea[i] + eb[j]);
            let r_ap2 = r_ap * r_ap;
            let r_bp2 = (s.bond_len - r_ap).powi(2);
            ints.s12 += na[i] * nb[j] * overlap(ea[i], eb[j], r2);
            ints.t11 += na[i] * na[j] * kinetic(ea[i], ea[j], 0.0);
            ints.t12 += na[i] * nb[j] * kinetic(ea[i], eb[j], r2);
            ints.t22 += nb[i] * nb[j] * kinetic(eb[i], eb[j], 0.0);
            ints.v11_nuc_a += na[i] * na[j] * nuc_attr(ea[i], ea[j], 0.0, 0.0, s.z_a);
            ints.v12_nuc_a += na[i] * nb[j] * nuc_attr(ea[i], eb[j], r2, r_ap2, s.z_a);
            ints.v22_nuc_a += nb[i] * nb[j] * nuc_attr(eb[i], eb[j], 0.0, r2, s.z_a);
            ints.v11_nuc_b += na[i] * na[j] * nuc_attr(ea[i], ea[j], 0.0, r2, s.z_b);
            ints.v12_nuc_b += na[i] * nb[j] * nuc_attr(ea[i], eb[j], r2, r_bp2, s.z_b);
            ints.v22_nuc_b += nb[i] * nb[j] * nuc_attr(eb[i], eb[j], 0.0, 0.0, s.z_b);
        }
    }
    for i in 0..s.sto_ng {
        for j in 0..s.sto_ng {
            for k in 0..s.sto_ng {
                for l in 0..s.sto_ng {
                    let r_ap = eb[i] * s.bond_len / (eb[i] + ea[j]);
                    let r_aq = eb[k] * s.bond_len / (eb[k] + ea[l]);
                    let r_bq = s.bond_len - r_aq;
                    let r_pq = r_ap - r_aq;
                    ints.v1111 += na[i]
                        * na[j]
                        * na[k]
                        * na[l]
                        * two_elec(ea[i], ea[j], ea[k], ea[l], 0.0, 0.0, 0.0);
                    ints.v2111 += nb[i]
                        * na[j]
                        * na[k]
                        * na[l]
                        * two_elec(eb[i], ea[j], ea[k], ea[l], r2, 0.0, r_ap.powi(2));
                    ints.v2121 += nb[i]
                        * na[j]
                        * nb[k]
                        * na[l]
                        * two_elec(eb[i], ea[j], eb[k], ea[l], r2, r2, r_pq.powi(2));
                    ints.v2211 += nb[i]
                        * nb[j]
                        * na[k]
                        * na[l]
                        * two_elec(eb[i], eb[j], ea[k], ea[l], 0.0, 0.0, r2);
                    ints.v2221 += nb[i]
                        * nb[j]
                        * nb[k]
                        * na[l]
                        * two_elec(eb[i], eb[j], eb[k], ea[l], 0.0, r2, r_bq.powi(2));
                    ints.v2222 += nb[i]
                        * nb[j]
                        * nb[k]
                        * nb[l]
                        * two_elec(eb[i], eb[j], eb[k], eb[l], 0.0, 0.0, 0.0);
                }
            }
        }
    }
    ints
}

fn matmul(a: [[f64; 2]; 2], b: [[f64; 2]; 2]) -> [[f64; 2]; 2] {
    [
        [
            a[0][0] * b[0][0] + a[0][1] * b[1][0],
            a[0][0] * b[0][1] + a[0][1] * b[1][1],
        ],
        [
            a[1][0] * b[0][0] + a[1][1] * b[1][0],
            a[1][0] * b[0][1] + a[1][1] * b[1][1],
        ],
    ]
}

fn diag(f: [[f64; 2]; 2]) -> ([[f64; 2]; 2], [[f64; 2]; 2]) {
    let theta = if (f[0][0] - f[1][1]).abs() < 1e-20 {
        PI / 4.0
    } else {
        0.5 * (2.0 * f[0][1] / (f[0][0] - f[1][1])).atan()
    };
    let mut c = [[theta.cos(), theta.sin()], [theta.sin(), -theta.cos()]];
    let mut e = [[0.0; 2]; 2];
    e[0][0] = f[0][0] * theta.cos().powi(2)
        + f[1][1] * theta.sin().powi(2)
        + f[0][1] * (2.0 * theta).sin();
    e[1][1] = f[1][1] * theta.cos().powi(2) + f[0][0] * theta.sin().powi(2)
        - f[0][1] * (2.0 * theta).sin();
    if e[1][1] < e[0][0] {
        e.swap(0, 1);
        c[0].swap(0, 1);
        c[1].swap(0, 1);
    }
    (c, e)
}

fn setup_matrices(ints: &Integrals, st: &mut ScfState) {
    st.h_mat = [
        [
            ints.t11 + ints.v11_nuc_a + ints.v11_nuc_b,
            ints.t12 + ints.v12_nuc_a + ints.v12_nuc_b,
        ],
        [
            ints.t12 + ints.v12_nuc_a + ints.v12_nuc_b,
            ints.t22 + ints.v22_nuc_a + ints.v22_nuc_b,
        ],
    ];
    st.s_mat = [[1.0, ints.s12], [ints.s12, 1.0]];
    st.x_mat = [
        [
            1.0 / (2.0 * (1.0 + ints.s12)).sqrt(),
            1.0 / (2.0 * (1.0 - ints.s12)).sqrt(),
        ],
        [
            1.0 / (2.0 * (1.0 + ints.s12)).sqrt(),
            -1.0 / (2.0 * (1.0 - ints.s12)).sqrt(),
        ],
    ];
    st.xt_mat = [
        [st.x_mat[0][0], st.x_mat[1][0]],
        [st.x_mat[0][1], st.x_mat[1][1]],
    ];
    st.tt_mat[0][0][0][0] = ints.v1111;
    st.tt_mat[1][0][0][0] = ints.v2111;
    st.tt_mat[0][1][0][0] = ints.v2111;
    st.tt_mat[0][0][1][0] = ints.v2111;
    st.tt_mat[0][0][0][1] = ints.v2111;
    st.tt_mat[1][0][1][0] = ints.v2121;
    st.tt_mat[0][1][1][0] = ints.v2121;
    st.tt_mat[1][0][0][1] = ints.v2121;
    st.tt_mat[0][1][0][1] = ints.v2121;
    st.tt_mat[1][1][0][0] = ints.v2211;
    st.tt_mat[0][0][1][1] = ints.v2211;
    st.tt_mat[1][1][1][0] = ints.v2221;
    st.tt_mat[1][1][0][1] = ints.v2221;
    st.tt_mat[1][0][1][1] = ints.v2221;
    st.tt_mat[0][1][1][1] = ints.v2221;
    st.tt_mat[1][1][1][1] = ints.v2222;
}

fn form_g(st: &mut ScfState) {
    for i in 0..2 {
        for j in 0..2 {
            let mut total = 0.0;
            for k in 0..2 {
                for l in 0..2 {
                    total +=
                        st.dens_mat[k][l] * (st.tt_mat[i][j][k][l] - 0.5 * st.tt_mat[i][l][k][j]);
                }
            }
            st.g_mat[i][j] = total;
        }
    }
}

fn run_scf(s: Settings, st: &mut ScfState) -> (f64, f64, usize, bool) {
    let mut elec = 0.0;
    for iter in 1..=s.max_iter {
        form_g(st);
        for i in 0..2 {
            for j in 0..2 {
                st.f_mat[i][j] = st.h_mat[i][j] + st.g_mat[i][j];
                elec += 0.5 * st.dens_mat[i][j] * (st.h_mat[i][j] + st.f_mat[i][j]);
            }
        }
        st.fprime_mat = matmul(st.xt_mat, matmul(st.f_mat, st.x_mat));
        (st.cprime_mat, st.energy_mat) = diag(st.fprime_mat);
        st.c_mat = matmul(st.x_mat, st.cprime_mat);
        let mut delta_sq = 0.0;
        for i in 0..2 {
            for j in 0..2 {
                st.olddens_mat[i][j] = st.dens_mat[i][j];
                st.dens_mat[i][j] = 2.0 * st.c_mat[i][0] * st.c_mat[j][0];
                delta_sq += (st.dens_mat[i][j] - st.olddens_mat[i][j]).powi(2);
            }
        }
        if (delta_sq / 4.0).sqrt() < s.conv_crit {
            return (elec, elec + s.z_a * s.z_b / s.bond_len, iter, true);
        }
        elec = 0.0;
    }
    (elec, elec + s.z_a * s.z_b / s.bond_len, s.max_iter, false)
}

fn print_matrix(name: &str, mat: [[f64; 2]; 2]) {
    println!("{name}:");
    for row in mat {
        println!("  {:>9.6} {:>9.6}", row[0], row[1]);
    }
}

fn main() {
    let settings = Settings::default();
    let mut state = ScfState::default();
    let ints = calc_integrals(settings);
    setup_matrices(&ints, &mut state);
    let (elec, total, n_iter, converged) = run_scf(settings, &mut state);
    state.mulliken_mat = matmul(state.dens_mat, state.s_mat);
    println!("Rust SCF_Szabo (HeH+, STO-nG, 2x2 RHF)");
    println!("Converged: {converged} in {n_iter} iterations");
    println!("Electronic Energy: {elec:.6}");
    println!("Total Energy:      {total:.6}");
    print_matrix("P (density)", state.dens_mat);
    print_matrix("F (Fock)", state.f_mat);
    print_matrix("C (coefficients)", state.c_mat);
    print_matrix("Mulliken", state.mulliken_mat);
}
