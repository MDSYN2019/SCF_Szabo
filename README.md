# SCF_Szabo

This repository contains a Rust implementation of the SCF sample calculation from
Appendix B of *Modern Quantum Chemistry - An Introduction to Electronic Structure*
by A. Szabo and N. Ostlund. The program intentionally favors readability over
computational efficiency so each step of the HeH+ restricted Hartree-Fock (RHF)
procedure is easy to follow.

The calculation is intentionally narrow in scope:

- Molecule: HeH+
- Basis: STO-nG, with the default set to STO-3G
- Matrix size: 2x2, matching the educational example in the book

The original C translation is still present for reference, but the maintained
entry point is now Rust.

## Requirements

Install Rust and Cargo, for example with [rustup](https://rustup.rs/):

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Python 3 is optional and is only needed if you want to use the Python frontend.

## Run the Rust implementation

```bash
cargo run
```

For an optimized build:

```bash
cargo run --release
```

The included makefile provides shortcuts:

```bash
make build
make run
```

## Python frontend

`scf_python.py` is now a lightweight Python frontend that invokes the Rust
implementation through Cargo. It is useful when you want a Python-friendly entry
point without duplicating the SCF algorithm in two languages.

Run it with:

```bash
python3 scf_python.py
```

Run the optimized Rust binary through the frontend with:

```bash
python3 scf_python.py --release
```

## Legacy C version

The historical C source files remain in the repository for comparison with the
book and with the Rust port. You can still build them with:

```bash
make legacy-c
```

## Program flow

The Rust code follows the same three-phase flow as the original educational
program:

1. Evaluate one- and two-electron integrals.
2. Build overlap, core Hamiltonian, and orthogonalization matrices.
3. Iterate the RHF SCF cycle until density convergence.
