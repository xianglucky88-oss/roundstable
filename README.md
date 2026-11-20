# roundstable-aggregate

Numerical state and transform components.

Python 3.11 or newer. The primary interfaces are importable component libraries and a local JSON HTTP service.

## Install

```sh
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements-dev.txt
./.venv/bin/python -m pip install --no-deps -e .
```

## Run

```sh
./.venv/bin/python -m roundstable demo
./.venv/bin/python -m roundstable serve --port 8080
./.venv/bin/python -m pytest -q
```

The server binds only to `127.0.0.1`. `GET /health` reports process readiness; `GET /capabilities` lists operations.
Each component has `POST /v1/<operation>` accepting its JSON request object. Responses contain `result` or an explicit error. The service accepts bodies up to 1 MB and at most four concurrent operation handlers. It does not expose a shell, arbitrary file paths or remote URLs.

## Components

| Operation | Behavior |
|---|---|
| `accumulator` | Exact rational accumulation of binary64 inputs with mergeable state and explicit nonfinite-value policy. |
| `intervals` | Outward-rounded interval add/subtract/multiply/divide and bounded expression evaluation. |
| `polynomials` | Exact Sturm root counting and independently verifiable rational real-root intervals. |
| `sketch` | Frequent-Directions matrix summaries, state restoration and approximate merges. |
| `fourier` | Native radix-2 FFT, bounded small direct DFT fallback and inverse transforms. |
| `periodicity` | Regular-sampling periodicity tests over supported FFT windows: explicit time grid, observation masks, trend handling, white-noise null with Fisher and joint Monte Carlo evidence. |

## Request examples

Run `python -m roundstable.examples` to print valid JSON requests. The output is generated locally and contains no credentials.

```sh
./.venv/bin/python -m roundstable.examples > requests.json
curl -s http://127.0.0.1:8080/capabilities
```

Use one operation request from that output as the JSON body for its `/v1/<operation>` endpoint. The `demo` command executes all five operations and prints their computed results.

## Supported scope

Exact accumulation favors correctness over high-throughput vector performance and final zero is positive. Interval outputs may be intentionally conservative; division by an interval containing zero is rejected. Root results count distinct real roots, do not report multiplicity, and use bounded degree/work. Matrix summaries are approximate and not bitwise invariant under merge-tree changes. FFT lengths are at most 4096; non-power-of-two fallback is limited to 64.

Periodicity analysis only tests regularly sampled series. Missing samples are never zero-filled: the mask splits the series into fully observed, mutually disjoint power-of-two windows (at most 4096 each), and gaps or sub-minimum tails are returned as `excluded_segments`. Constant series, irregular timestamps, unsupported noise nulls and zero residual variance are reported as `untestable` with a reason instead of a p-value. The null is Gaussian white noise with a single variance pooled across every effective window; each window also carries an exact scale-invariant Fisher g p-value, while a joint Monte Carlo max statistic controls the family-wise error across all windows and eligible frequencies in one shared population. Frequencies at or above Nyquist cannot be separated from aliases and are flagged. A budget that cannot buy the minimum number of replicates yields `incomplete`/`inconclusive_incomplete`, which is distinct from `no_period_found`; every candidate points back to its effective window, observed sample range and statistic convention.

Runtime state uses explicit temporary directories or caller-provided library objects. Service examples are per-request computations; use the library interfaces for persistent state and long-lived readers.

## Dependencies

Pinned Python dependencies are listed in `requirements.txt`; the development test runner is pinned in `requirements-dev.txt`. System libraries or a native compiler, when required above, are detected at runtime. No accounts, network connection strings, tokens or private-key files are required.
