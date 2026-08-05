.PHONY: build run python clean legacy-c

build:
	cargo build

run:
	cargo run

python:
	python3 scf_python.py

clean:
	cargo clean

legacy-c:
	gcc -o scf_basic_version_1_0 main.c calc_integrals.c format_integrals.c perform_scf.c -I. -lm
