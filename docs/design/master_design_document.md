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
9. Preprocessing is analysis-dependent. The toolkit should provide reusable transformations, not impose one universal pipeline. There is no single universally correct TCSPC preprocessing pipeline.
10. Imported experimental histograms preserve explicit time units, raw-count versus processed-intensity semantics, and source provenance. Internal time coordinates are in nanoseconds.
11. A sampled measured IRF retains its imported values and provenance separately from the decay histogram. Reconvolution normalizes a compatible IRF for forward modelling without changing the imported trace.
12. Experimental estimation does not imply known ground truth. Truth-based error metrics require an explicitly supplied trusted reference; synthetic generating parameters and experimental reference values have different meanings.
