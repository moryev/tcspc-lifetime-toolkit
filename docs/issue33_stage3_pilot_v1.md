# Issue #33 Stage 3 pilot report

Execution lock: `fa830e76cc756c2d618e0a31ffcddcd871c20bd1cd92719905297326188ed6eb`.

Independent agreement does not prove global optimality. Audit incompleteness does not by itself establish production-fitter failure.

Primary gates: `null`.

| Condition / arm | Accepted / scheduled | Median absolute relative errors | Signed biases |
|---|---:|---|---|
| C00 / matched | 1/1 | [0.004869567151408044, 0.002276654850798948] | [0.004869567151408044, 0.002276654850798948] |
| C26 / matched | 1/1 | [0.0004811363153482684, 0.005191361981084774] | [-0.0004811363153482684, -0.005191361981084774] |
| C26 / gaussian_0.3 | 1/1 | [0.4236203544911993, 0.08006640800285944] | [0.4236203544911993, 0.08006640800285944] |

Component errors use accepted, available components with explicit denominators in the JSON. No warning-based exclusions. Secondary conditions are diagnostic.

Audit classes and all scheduled task outcomes are retained in the companion JSON.
