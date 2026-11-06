"""Prescribed-star-bath targeting experiments; no DMFT feedback or engine changes.

Each process evolves a small target set. Time advancement inside the sweep is
explicitly disabled: each stage recomputes one interval from its fixed input.
Physical endpoint times are checked in HDF5, not inferred from the printed clock.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import platform
import subprocess

import h5py
import numpy as np

try:
    from . import experiment_io, ed_reference
except ImportError:
    import experiment_io
    import ed_reference

PURE = {"empty": 0, "up": 2, "down": 1, "double": 3}
TIME_PATH = "Def/FinalPsi/TimeSerializer/PvectorTimes/Values"


def ainur_scalar(value):
    """AinurComplex uses real-part 'i' imaginary-part, not Python complex syntax."""
    z = complex(value)
    return f"{z.real:.17g}i{z.imag:.17g}" if z.imag else f"{z.real:.17g}"


def make_input(output, *, nsites=5, nup=3, ndown=2, m=64, couplings=None,
               expressions=None, restart=None, restart_mapping=None,
               onsite_u=2.0, potential=-1.0, finite_loops=None, bath_energies=None):
    if (nsites < 3 or nup < 0 or ndown < 0 or nup > nsites or ndown > nsites
            or m < 1):
        raise ValueError("invalid lattice dimensions, particle counts, or retained states")
    couplings = list(couplings if couplings is not None else [0.] * (nsites - 1))
    if len(couplings) != nsites - 1:
        raise ValueError("star bath requires exactly nsites-1 couplings")
    bath_energies = list(bath_energies if bath_energies is not None else [0.] * (nsites - 1))
    if len(bath_energies) != nsites - 1:
        raise ValueError("bath energies require exactly nsites-1 values")
    loops = finite_loops or f"[[@auto,{m},2],[@auto,{m},2]]"
    doubles = min(nup, ndown)
    states = ([PURE["up"]] * (nup - doubles) + [PURE["down"]] * (ndown - doubles)
              + [PURE["double"]] * doubles + [PURE["empty"]] * (nsites - max(nup, ndown)))
    options = "twositedmrg,geometryallinsystem,TargetingExpression,usecomplex"
    if restart is not None:
        options += ",restart"
    lines = ["##Ainur1.0", f"TotalNumberOfSites={nsites};", "NumberOfTerms=1;",
             "DegreesOfFreedom=1;", "GeometryKind=star;", "GeometryOptions=none;",
             "Model=HubbardOneBand;", f"SolverOptions={options};",
             "Version=tdmrg-prescribed-bath-experiment;", f"OutputFile={output};",
             f"InfiniteLoopKeptStates={m};", f"FiniteLoops={loops};",
             f"TargetElectronsUp={nup};", f"TargetElectronsDown={ndown};",
             "hubbardU=[" + ",".join([str(onsite_u)] + ["0"] * (nsites - 1)) + "];",
             "potentialV=[" + ",".join(([str(potential)] + [str(x) for x in bath_energies]) * 2) + "];",
             "dir0:Connectors=[" + ",".join(map(ainur_scalar, couplings)) + "];",
             "GsWeight=0.1;"]
    if restart is None:
        lines.append("gs=PureState[" + ",".join(map(str, states)) + "];")
    else:
        path = Path(restart).resolve()
        if path.suffix == ".hd5":
            path = path.with_suffix("")
        lines.append(f"RestartFilename={path};")
    if restart_mapping is not None:
        lines.append("RestartMappingTvs=[" + ",".join(map(str, restart_mapping)) + "];")
    lines.extend(expressions or ["string P0=|gs>;"])
    return "\n".join(lines) + "\n"


def insertion_expression(spin, kind):
    # All experiments probe c_up; alpha/beta distinguish physical initial states.
    if spin not in ("alpha", "beta") or kind not in ("hole", "particle"):
        raise ValueError("invalid seed or channel")
    op = "c" if kind == "hole" else "c'"
    return f'string P0="{op}[0]*|gs>";'


def evolution(dt, source, steps=2):
    if dt <= 0 or steps < 2:
        raise ValueError("positive duration and at least two targeting times required")
    return (f"TimeEvolve{{tau={dt:.17g},steps={steps},advanceEach=1000000,"
            f"EnergyForExp=0}}*|{source}>")


def paired_expressions(dt, *, continuation=False, steps=2):
    return [f'string P0="{evolution(dt, "P0", steps)}";',
            f'string P1="{evolution(dt, "P1" if continuation else "gs", steps)}";']


def paired_measurements():
    return ["<P1.last|c|P0.last>", "<P0.last|c|P1.last>",
            "<P1.last|nup|P1.last>", "<P1.last|ndown|P1.last>",
            "<P1.last|nup*ndown|P1.last>", "<P0.last|P0.last>", "<P1.last|P1.last>"]


def raw_complex(value):
    if value.startswith("("):
        real, imag = value[1:-1].split(",")
        return complex(float(real), float(imag))
    return complex(float(value))


def encoded(value):
    z = complex(value)
    return [z.real, z.imag]


def measured(result, label):
    values = [raw_complex(x["value_raw"]) for x in result["parsed"]["measurements"]
              if x["site"] == 0 and x["bra_ket"] == label]
    if not values:
        raise RuntimeError(f"missing impurity measurement {label} in {result['cwd']}")
    return values[-1]


def run_stage(executable, directory, name, text, *, restart=None, mapping=None,
              measurements=(), timeout=None):
    root = Path(directory).resolve() / name
    root.mkdir(parents=True, exist_ok=False)
    inp = root / f"{name}.ain"
    inp.write_text(text)
    cmd = [str(Path(executable).resolve()), "-p", "14", "-f", inp.name]
    if measurements:
        cmd.append(",".join(measurements))
    result = experiment_io.run_job(cmd, inp, root, timeout=timeout,
                                  env={"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
    result.update({"name": name, "restart_checkpoint": str(restart) if restart else None,
                   "restart_mapping": mapping})
    experiment_io.atomic_write_json(root / "job.json", result)
    if result["returncode"] != 0 or result["timed_out"]:
        raise RuntimeError(f"stage {name} failed; diagnostics retained at {root}")
    if result["parsed"]["parse_errors"]:
        raise RuntimeError(f"unparsed measurement output at {root}")
    cp = root / f"{name}.hd5"
    with h5py.File(cp) as stream:
        result["pvector_times"] = stream[TIME_PATH][()].tolist()
    result["checkpoint"] = str(cp)
    return result


class Case:
    def __init__(self, executable, root, nsites, m, dt, steps, u, timeout, sweep_pairs=1):
        self.executable, self.root = executable, Path(root)
        self.nsites, self.m, self.dt, self.steps, self.u = nsites, m, dt, steps, u
        self.timeout = timeout
        self.sweep_pairs = sweep_pairs
        self.jobs = []
        self.l = (nsites - 1) // 2
        if nsites < 3 or nsites % 2 != 1:
            raise ValueError("atomic filled/empty bath requires odd nsites >= 3")
        # Prescribed nonstationary complex bath, NOT a self-consistent GBEK bath.
        self.couplings = [[(.55 if p < self.l else .45) / math.sqrt(self.l)
                           * (1 + .12 * math.sin((p + 1) * (k + 1)))
                           * np.exp(1j * (.15 * p + .35 * k * (p + 1)))
                           for p in range(nsites - 1)] for k in range(steps)]

    def stage(self, name, seed, expressions=None, source=None, mapping=None,
              couplings=None, measurements=(), source_tv_for_psi=None):
        nup, ndown = (self.l + 1, self.l) if seed == "alpha" else (self.l, self.l + 1)
        text = make_input(name, nsites=self.nsites, nup=nup, ndown=ndown, m=self.m,
                          onsite_u=self.u, potential=-self.u / 2,
                          expressions=expressions, restart=source,
                          restart_mapping=mapping, couplings=couplings,
                          finite_loops="[" + ",".join([f"[@auto,{self.m},2]"] * (2 * self.sweep_pairs)) + "]")
        if source_tv_for_psi is not None:
            text += f"integer RestartSourceTvForPsi={source_tv_for_psi};\n"
        result = run_stage(self.executable, self.root, name, text, restart=source,
                           mapping=mapping, measurements=measurements, timeout=self.timeout)
        self.jobs.append(result)
        return result

    def check_times(self, job, slots, expected):
        actual = [job["pvector_times"][slot] for slot in slots]
        if not np.allclose(actual, expected, atol=1e-12, rtol=0):
            raise RuntimeError(f"wrong endpoint times {actual}, expected {expected}: {job['name']}")

    def pair(self, name, seed, source, mapping, start, couplings, from_gs=False,
             duration=None, targeting_steps=2):
        duration = self.dt if duration is None else duration
        job = self.stage(name, seed, paired_expressions(duration, continuation=not from_gs,
                                                       steps=targeting_steps),
                         source, mapping, couplings, paired_measurements())
        # Each evolution owns first output index plus steps-1 appended vectors.
        slots = [targeting_steps, 2 * targeting_steps - 1]
        self.check_times(job, slots, [start + duration] * 2)
        return job, slots

    def run(self):
        n = self.steps + 1
        greens = {}
        references, master_values = [], {}
        first_insertion = None
        for seed in ("alpha", "beta"):
            initial = self.stage(f"{seed}_seed", seed,
                                 measurements=["<gs|nup|gs>", "<gs|ndown|gs>"])
            expected = (1., 0.) if seed == "alpha" else (0., 1.)
            for label, value in zip(("nup", "ndown"), expected):
                if abs(measured(initial, f"<gs|{label}|gs>") - value) > 1e-12:
                    raise RuntimeError("product-state impurity occupation mismatch")
            masters = [initial]
            master_values[(seed, 0)] = [*expected, 0.]
            for k in range(self.steps):
                source = masters[-1]["checkpoint"]
                expr = [f'string P0="{evolution(self.dt, "gs" if k == 0 else "P0")}";']
                job = self.stage(f"{seed}_reference_{k+1}", seed, expr, source,
                                 None if k == 0 else [1], self.couplings[k],
                                 [f"<P0.last|{op}|P0.last>" for op in ("nup", "ndown", "nup*ndown")])
                self.check_times(job, [1], [(k + 1) * self.dt])
                masters.append(job)
                master_values[(seed, k + 1)] = [measured(job, f"<P0.last|{op}|P0.last>").real
                                                for op in ("nup", "ndown", "nup*ndown")]
            seed_matrices = {}
            for kind, component in (("hole", "lesser"), ("particle", "greater")):
                matrix = np.zeros((n, n), complex)
                for j in range(n):
                    # At t=0 one spin-up channel is identically zero in each seed.
                    if j == 0 and ((seed == "alpha" and kind == "particle")
                                   or (seed == "beta" and kind == "hole")):
                        continue
                    # Applying an operator directly to a P ket currently overwrites
                    # the result with that source ket. Promote the saved endpoint to
                    # gs using the existing restart facility, then apply c to gs.
                    # gs has no persisted physical-time tag: this job uses local
                    # time zero at insertion, with physical origin j*dt recorded.
                    expr = [insertion_expression(seed, kind)]
                    ref, branch = "gs", "P0"
                    label = f"<{branch}|c|{ref}>" if kind == "hole" else f"<{ref}|c|{branch}>"
                    inserted = self.stage(f"{seed}_{kind}_insert_{j}", seed, expr,
                                          masters[j]["checkpoint"], [-1],
                                          self.couplings[max(j - 1, 0)] if j else None,
                                          [label, f"<{branch}|{branch}>", f"<{ref}|nup|{ref}>"],
                                          source_tv_for_psi=1 if j else None)
                    inserted["physical_clock_origin"] = j * self.dt
                    self.check_times(inserted, [0], [0.])
                    factor = 1j if kind == "hole" else -1j
                    matrix[j, j] = factor * measured(inserted, label)
                    previous, slots = inserted, [0, -1]
                    if seed == "alpha" and kind == "hole" and j == 0:
                        first_insertion = inserted
                    for k in range(j, self.steps):
                        job, slots = self.pair(f"{seed}_{kind}_{j}_to_{k+1}", seed,
                                               previous["checkpoint"], slots, (k - j) * self.dt,
                                               self.couplings[k], from_gs=(k == j))
                        job["physical_clock_origin"] = j * self.dt
                        pairlabel = "<P0.last|c|P1.last>" if kind == "hole" else "<P1.last|c|P0.last>"
                        matrix[k + 1, j] = factor * measured(job, pairlabel)
                        references.append({"seed": seed, "insertion": j, "channel": kind,
                                           "time_index": k + 1,
                                           "values": [measured(job, f"<P1.last|{op}|P1.last>").real
                                                      for op in ("nup", "ndown", "nup*ndown")]})
                        previous = job
                for row in range(n):
                    for column in range(row):
                        matrix[column, row] = -matrix[row, column].conjugate()
                seed_matrices[component] = matrix
            greens[seed] = seed_matrices
        averaged = {key: (greens["alpha"][key] + greens["beta"][key]) / 2
                    for key in ("lesser", "greater")}
        errors = []
        ed_matrices = {}
        ed_reference_errors = []
        if self.nsites <= 7:
            potential = [-self.u / 2] + [0.] * (self.nsites - 1)
            for seed in ("alpha", "beta"):
                occupied = list(range(1, self.l + 1))
                sector, state = ed_reference.product_state(
                    self.nsites, ([0] if seed == "alpha" else []) + occupied,
                    ([0] if seed == "beta" else []) + occupied)
                exact = ed_reference.green_functions(self.nsites, self.u, sector, [state],
                    list(zip([self.dt] * self.steps, self.couplings)), potential=potential)
                ed_matrices[seed] = {key: exact[key] for key in ("lesser", "greater")}
                _, snapshots = ed_reference.evolve(
                    self.nsites, self.u, sector, [state],
                    list(zip([self.dt] * self.steps, self.couplings)), potential=potential)
                masks = ed_reference.basis(self.nsites, *sector)
                impurity_observables = np.array(
                    [[bool(up & 1), bool(down & 1), bool(up & 1) and bool(down & 1)]
                     for up, down in masks], dtype=float)
                exact_values = [abs(states[0]) ** 2 @ impurity_observables
                                for states in snapshots]
                ed_reference_errors.append({
                    "seed": seed,
                    "master_max": float(max(np.max(abs(master_values[(seed, k)] - values))
                                            for k, values in enumerate(exact_values))),
                    "paired_max": float(max(np.max(abs(r["values"] - exact_values[r["time_index"]]))
                                            for r in references if r["seed"] == seed))})
                errors.append({"seed": seed, **{key: float(np.max(abs(greens[seed][key] - exact[key])))
                                                for key in ("lesser", "greater")}})
        equal_time = float(np.max(abs(np.diag(averaged["greater"] - averaged["lesser"]) + 1j)))
        density_error = max(abs(greens[seed]["lesser"][k, k] / 1j - master_values[(seed, k)][0])
                            for seed in ("alpha", "beta") for k in range(n))
        spread = max((max(abs(a - b) for a, b in zip(r["values"], master_values[(r["seed"], r["time_index"])]))
                      for r in references), default=0.)
        positivity = {key: float(np.linalg.eigvalsh((sign * matrix + (sign * matrix).conj().T) / 2).min())
                      for key, matrix, sign in (("lesser", averaged["lesser"], -1j),
                                                ("greater", averaged["greater"], 1j))}
        # Constant-bath segmented vs single-process evolution-window comparison.
        insertion = first_insertion["checkpoint"]
        v = self.couplings[0]
        first, first_slots = self.pair("continuation_first", "alpha", insertion, [0, -1], 0., v, True)
        continued, _ = self.pair("continuation_second", "alpha", first["checkpoint"], first_slots, self.dt, v)
        uninterrupted, _ = self.pair("single_window", "alpha", insertion, [0, -1], 0., v, True,
                                      duration=2 * self.dt, targeting_steps=3)
        label = "<P0.last|c|P1.last>"
        continuation_error = abs(measured(continued, label) - measured(uninterrupted, label))
        # Rejected trial may never modify its immutable accepted predecessor.
        accepted_hash = experiment_io.sha256_file(first["checkpoint"])
        trial_v = [-.7 * complex(x).conjugate() for x in self.couplings[min(1, self.steps - 1)]]
        rejected, _ = self.pair("rejected_trial", "alpha", first["checkpoint"], first_slots, self.dt, trial_v)
        corrected_v = self.couplings[min(1, self.steps - 1)]
        corrected, _ = self.pair("corrected_trial", "alpha", first["checkpoint"], first_slots, self.dt, corrected_v)
        fresh_first, fresh_slots = self.pair("fresh_prefix", "alpha", insertion, [0, -1], 0., v, True)
        fresh_corrected, _ = self.pair("fresh_corrected", "alpha", fresh_first["checkpoint"], fresh_slots, self.dt, corrected_v)
        if experiment_io.sha256_file(first["checkpoint"]) != accepted_hash:
            raise RuntimeError("accepted checkpoint was modified by a trial")
        replay_error = abs(measured(corrected, label) - measured(fresh_corrected, label))
        def pack(matrix):
            return [[encoded(z) for z in row] for row in matrix]
        summary = {
            "nsites": self.nsites, "bath_sites": self.nsites - 1, "retained_states": self.m,
            "dt": self.dt, "steps": self.steps, "U": self.u, "sweep_pairs": self.sweep_pairs,
            "couplings": [[encoded(z) for z in row] for row in self.couplings],
            "times": [k * self.dt for k in range(n)],
            "greens": {seed: {key: pack(matrix) for key, matrix in value.items()}
                       for seed, value in greens.items()},
            "averaged_greens": {key: pack(matrix) for key, matrix in averaged.items()},
            "ed_errors": errors, "ed_status": "computed" if errors else "not_computed_nsites_gt_7",
            "ed_greens": {seed: {key: pack(matrix) for key, matrix in value.items()}
                          for seed, value in ed_matrices.items()},
            "ed_reference_observable_errors": ed_reference_errors,
            "equal_time_anticommutator_max_error": equal_time,
            "lesser_diagonal_vs_master_density_max_error": float(density_error),
            "reference_observable_spread_vs_master_max": float(spread),
            "kernel_min_eigenvalues": positivity,
            "continuation_single_window_max_amplitude_difference": float(continuation_error),
            "continuation_note": "Single-window uses three intermediate targeting times; segmented runs use two per interval. Projection schedules differ under truncation.",
            "correction_replay_amplitude_difference": float(replay_error),
            "rejected_vs_corrected_amplitude_difference": float(abs(measured(rejected, label) - measured(corrected, label))),
            "accepted_checkpoint_unchanged": True,
            "reference_observations": references,
            "master_reference_observations": [{"seed": seed, "time_index": k, "values": value}
                                               for (seed, k), value in master_values.items()],
            "resources": {
                "jobs": len(self.jobs), "wall_seconds_sum": sum(j["wall_seconds"] for j in self.jobs),
                "constant_bath_segmented_wall_seconds": first["wall_seconds"] + continued["wall_seconds"],
                "constant_bath_single_window_wall_seconds": uninterrupted["wall_seconds"],
                "segmentation_extra_wall_seconds": first["wall_seconds"] + continued["wall_seconds"] - uninterrupted["wall_seconds"],
                "segmentation_timing_note": "Includes an extra process launch, reload, sweep/projection work, and checkpoint write; not an isolated restart-load time.",
                "peak_process_rss_bytes": max(j["peak_child_rss_bytes"] or 0 for j in self.jobs),
                "largest_checkpoint_bytes": max(max(j["checkpoints"].values(), default=0) for j in self.jobs),
                "checkpoint_bytes_all_stages": sum(sum(j["checkpoints"].values()) for j in self.jobs),
                "discarded_weight_max": max((max(j["parsed"]["discarded_weights"], default=0.) for j in self.jobs)),
                "wft_warning_count": sum(len(j["parsed"]["wft_warnings"]) for j in self.jobs)},
            "jobs": [{key: j[key] for key in ("name", "cwd", "wall_seconds", "peak_child_rss_bytes",
                                               "checkpoints", "pvector_times", "input_sha256")}
                      | {"physical_clock_origin": j.get("physical_clock_origin", 0.)}
                     for j in self.jobs]}
        experiment_io.atomic_write_json(self.root / "summary.json", summary)
        return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cases", default="5:64", help="comma-separated nsites:retained_states")
    parser.add_argument("--dt", type=float, default=.2)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--U", type=float, default=2.)
    parser.add_argument("--sweep-pairs", type=int, default=1,
                        help="full out-and-back basis-adaptation sweeps per fixed interval")
    parser.add_argument("--timeout", type=float, default=120.)
    args = parser.parse_args(argv)
    if args.steps < 2 or args.dt <= 0 or args.sweep_pairs < 1:
        parser.error("at least two positive-duration intervals required")
    base = Path(args.output).resolve()
    base.mkdir(parents=True, exist_ok=True)
    executable = str(Path(args.executable).resolve())
    provenance = {"executable": executable, "executable_sha256": experiment_io.sha256_file(executable),
                  "platform": platform.platform(), "python": platform.python_version(),
                  "spack_environment": os.environ.get("SPACK_ENV"),
                  "numpy": np.__version__, "h5py": h5py.__version__,
                  "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  "dmft_feedback": False,
                  "experiment_sources_sha256": {p.name: experiment_io.sha256_file(p)
                                                  for p in Path(__file__).parent.glob("*.py")}}
    summaries = []
    for spec in args.cases.split(","):
        nsites, m = map(int, spec.split(":"))
        root = base / f"n{nsites}_m{m}"
        case = Case(executable, root, nsites, m, args.dt, args.steps, args.U, args.timeout, args.sweep_pairs)
        summary = case.run()
        summaries.append(summary)
        experiment_io.atomic_write_json(base / "results.json", {"provenance": provenance, "cases": summaries})
        print(f"N={nsites} m={m}: ED={summary['ed_errors']} "
              f"replay={summary['correction_replay_amplitude_difference']:.3g} "
              f"RSS={summary['resources']['peak_process_rss_bytes']/2**20:.1f} MiB", flush=True)


if __name__ == "__main__":
    main()
