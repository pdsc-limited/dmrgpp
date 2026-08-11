# Phase-3 technical output format

This is a versioned serialization format for bounded technical trajectories.
It does not authorize a Wolf-benchmark claim or a production runner.

## `trajectory.csv`

Schema version 1 is a headered numeric CSV with one endpoint record per row:

```text
time,norm,total_particle_number,spin_projection,energy,energy_imaginary_part,impurity_nup,impurity_ndn,impurity_double_occupancy,max_link_dimension
```

`time` values must be finite and strictly increasing. Every measured scalar
must be finite. `max_link_dimension` must be a positive integer. Link-dimension
vectors are deliberately not duplicated in every row; a later optional
per-bond file can add them without changing this compact trajectory schema.

## `metadata.toml`

Metadata must contain `schema_version = 1`. A caller supplies all other
provenance explicitly, including at minimum:

- input config contents and config path;
- Julia, ITensors, and ITensorMPS versions;
- `Lb`, interaction, beta, and the initial-state convention;
- endpoint grid and joint-factorization rank/error diagnostics;
- global-Krylov time step, Krylov dimension/residual, MPS compression cutoff,
  and maximum bond dimension;
- thread/host information and run completion status.

The serializer does not infer any of these values from mutable process state.

## File ownership

Callers provide exact output paths to the serialization helpers. Generated
artifacts belong under the ignored project-local `output/` directory. Scratch
scripts and logs belong under ignored project-local `tmp/`, never a shared
system temporary directory.
