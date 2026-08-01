Design Principles

1. Mathematical models belong in models.py.
2. Simulation workflows belong in simulation.py.
3. IRFs are generated independently from decay models.
4. Convolution is an independent operation.
5. Noise sampling is independent from signal generation.
6. Evaluation functions are organized by evaluated quantity, not by algorithm.
7. Public APIs use explicit names:
   monoexponential_decay()
   simulate_monoexponential_decay()
   fit_monoexponential_decay()
8. Convenience wrappers may compose lower-level functions.