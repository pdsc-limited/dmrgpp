# Causal-Cholesky timestep comparison through t = 3

This technical comparison checks the sensitivity of the non-self-consistent
`U=10`, `Lb=16` star-SIAM trajectory to halving the endpoint and propagation
timestep from `0.05` to `0.025`.

It is **not yet a Wolf et al. reproduction claim**. In particular, the cosine
ramp duration `t1=0.25` is provisional: it is borrowed from the independent
GBEK input and has not been sourced to Wolf's calculation.

## Fixed choices

- `U=10`, `Lb=16` (`rank=8` lesser plus `rank=8` greater)
- causal optimized low-rank Cholesky, GBEK Eqs. 56--63
- greater couplings are the exact complex conjugates of lesser couplings
- natural cubic midpoint interpolation from each available causal endpoint
  prefix; no future endpoint changes a midpoint already used
- separate atomic `Up` and `Dn` trajectories, with scalar observables averaged
- global MPS Krylov dimension 7
- MPS cutoff `1e-12`; nonbinding `maxdim=2000`
- two Julia threads used to evolve the two atomic components concurrently
- continuum-frequency quadrature `intervals=512`

## Important interpretation

This is an end-to-end timestep comparison, not an isolated convergence test of
only the exponential propagator. Changing `dt` changes both:

1. the endpoint grid on which the continuum hybridization kernel is sampled
   and causally factorized, and
2. the midpoint propagation timestep.

That distinction matters because the finite-rank causal factorization can have
grid dependence of its own.

## Runs

The coarse run used a `tmax=13` factorization but stopped and checkpointed at
`t=3`. Causal prefix invariance has independently been verified exactly against
a `tmax=3` factorization.

```text
output/causal_u10_Lb16_dt005_t1300_stop300_k7/
output/causal_u10_Lb16_dt0025_t300_k7/
```

The fine run command was:

```bash
env JULIA_DEPOT_PATH=.julia_depot \
  julia -t 2 --project=. bin/wolf_technical_trajectory.jl \
  --config configs/wolf_u10.toml \
  --out output/causal_u10_Lb16_dt0025_t300_k7 \
  --Lb 16 --dt 0.025 --tmax 3 \
  --krylovdim 7 --maxdim 2000 --cutoff 1e-12 --intervals 512 \
  --parallel-components true --threaded-blocksparse false --gc-every 0 \
  --checkpoint-every 10 --stop-after-step 120
```

It completed in 4154.8 seconds. At `t=3` it had double occupancy
`0.028318292642992515` and maximum MPS bond dimension 412. Maximum norm drift
was `6.15e-12`; maximum particle-number drift was `4.26e-14`.

At the coarse trajectory's endpoint times, the maximum absolute difference
between `dt=0.025` and `dt=0.05` was `2.33530296932e-4`.

## Reproduce the plot

The Python environment is locked by `pyproject.toml` and `uv.lock`.
From the DMRG++ checkout root, run:

```bash
uv run --project cincuenta/TestSuite/wolf_itensor_reference python \
  cincuenta/TestSuite/wolf_itensor_reference/bin/plot_docc_timestep_convergence.py \
  cincuenta/TestSuite/wolf_itensor_reference/output/causal_u10_Lb16_dt005_t1300_stop300_k7 \
  cincuenta/TestSuite/wolf_itensor_reference/output/causal_u10_Lb16_dt0025_t300_k7 \
  --ramp-duration 0.25 \
  --output cincuenta/TestSuite/wolf_itensor_reference/output/causal_u10_Lb16_dt0025_t300_k7/docc_dt005_vs_dt0025.pdf \
  --title 'U=10, causal Cholesky, L_b=16'
```

The lower panel linearly interpolates the fine trajectory only onto coarse
endpoint times to display `d_fine-d_coarse`. The upper-panel curves are plotted
from their recorded values without interpolation.

## Published-figure extrema

`bin/compare_docc_extrema.py` reads the standalone arXiv vector PDF directly
with `pypdf`; it does not digitize a raster image. It extracts the dark-blue
`Lb=16` path, maps the exact vector axes to physical coordinates, and estimates
each extremum with a quadratic through the retained local vertex and its two
neighbors. Run:

```bash
uv run --project cincuenta/TestSuite/wolf_itensor_reference python \
  cincuenta/TestSuite/wolf_itensor_reference/bin/compare_docc_extrema.py \
  '/workspace/dmrgpp_papers/solving neq-dmft using mps/arxiv_source/fig_BalzerU10.pdf' \
  cincuenta/TestSuite/wolf_itensor_reference/output/causal_u10_Lb16_dt0025_t300_k7/trajectory.csv
```

Through `t=3`, the maximum fitted extremum-time displacement is `0.00420`
(`0.082` PDF coordinate points), and the maximum fitted extremum-height
difference is `8.93e-5` (`0.482` PDF points). The published curve stroke is
`4.6` PDF points wide. These fits establish agreement at the graphical
resolution of the published figure, not access to unpublished numerical data.
