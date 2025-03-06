# VQCount

This package implements the VQCount algorithm for approximatively solving #P-hard problems. The VQCount algorithm leverages the equivalence between almost-uniform sampling and approximate counting, as proposed by Jerrum, Valiant and Vazirani, using the Quantum Alternating Operator Ansatz (QAOA) as the solution generator. For a detailed explanation of VQCount, please refer to our paper.

## Installation

Dependencies are listed in the `requirements.txt` file. This package requires building the model counter [Ganak](https://github.com/meelgroup/ganak). If needed, you can modify the build path in `vqcount/utils/ganak_wrapper.py`.

## Usage

Refer to the optimization example in `example_optimization.py` or the counting example in `example_counting.py`.