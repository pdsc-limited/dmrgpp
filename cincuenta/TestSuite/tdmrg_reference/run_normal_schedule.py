"""Constant-bath advancing t-DMRG versus ED at matched physical times."""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import platform
import subprocess

import numpy as np
from scipy.sparse.linalg import expm_multiply

try:
    from . import ed_reference as ed, experiment_io as io, run_experiments as common
except ImportError:
    import ed_reference as ed
    import experiment_io as io
    import run_experiments as common


def sample_count(tau, final_time):
    if not all(math.isfinite(x) and x > 0 for x in (tau, final_time)):
        raise ValueError("tau and final_time must be finite and positive")
    count = round(final_time / tau)
    if count < 1 or not math.isclose(count * tau, final_time, abs_tol=1e-12, rel_tol=0):
        raise ValueError("final_time must be a positive integer multiple of tau")
    return count


def leg_count(tau, final_time):
    count = sample_count(tau, final_time)
    return 2 * math.ceil(count / 2)


def expected_endpoint_times(legs, tau, target_times=2):
    base = legs * tau
    offsets = [base + k*tau/(target_times-1) for k in range(1, target_times)]
    return [base] * 2 + offsets * 2


def advancing_expression(tau, nsites, source, target_times=2):
    return (f"TimeEvolve{{tau={tau:.17g},steps={target_times},advanceEach={nsites-2},"
            f"EnergyForExp=0}}*|{source}>")


def select_samples(job, label, tau, count):
    """Keep all duplicate records; select the latest at each validated grid time."""
    groups = {k: [] for k in range(1, count + 1)}
    for record in job["parsed"]["measurements"]:
        if record["site"] != 0 or record["bra_ket"] != label:
            continue
        physical_time = record["time"] + tau
        if physical_time > count * tau + 1e-12:
            continue
        index = round(physical_time / tau)
        if index not in groups or abs(physical_time - index * tau) > 1e-12:
            raise RuntimeError(f"off-grid measurement {physical_time}: {label}")
        groups[index].append(record | {"physical_time": physical_time})
    samples = []
    for index, records in groups.items():
        if not records:
            raise RuntimeError(f"missing impurity {label} at time {index*tau}")
        values = [common.raw_complex(r["value_raw"]) for r in records]
        samples.append({"time": index * tau, "records": records,
                        "value": common.encoded(values[-1]),
                        "duplicate_spread": float(max(abs(a-b) for a in values for b in values))})
    return samples


def impulse_reference(nsites, u, seed, couplings, bath_energies, tau, count):
    """Independent sector propagation for the nonzero t'=0 channel and reference."""
    occupied = list(range(1, (nsites - 1) // 2 + 1))
    up = ([0] if seed == "alpha" else []) + occupied
    down = ([0] if seed == "beta" else []) + occupied
    sector, state = ed.product_state(nsites, up, down)
    if seed == "alpha":
        branch_sector, branch = ed.product_state(nsites, occupied, down)
    else:
        branch_sector, branch = ed.product_state(nsites, [0] + occupied, down)
    potential = [-u / 2] + list(bath_energies)
    basis, h = ed.sector_hamiltonian(nsites, *sector, u, potential=potential, couplings=couplings)
    branch_basis, hb = ed.sector_hamiltonian(
        nsites, *branch_sector, u, potential=potential, couplings=couplings)
    # Both initial operations at up orbital zero have sign +1.
    reference = expm_multiply(-1j * h, state, start=0., stop=count*tau, num=count+1)
    inserted = expm_multiply(-1j * hb, branch, start=0., stop=count*tau, num=count+1)
    if seed == "alpha":
        annihilator = ed._op(nsites, basis, branch_basis, 0, 0, False)
        greens = [1j * np.vdot(b, annihilator @ r) for b, r in zip(inserted, reference)]
    else:
        annihilator = ed._op(nsites, branch_basis, basis, 0, 0, False)
        greens = [-1j * np.vdot(r, annihilator @ b) for b, r in zip(inserted, reference)]
    masks = np.array([[bool(up & 1), bool(down & 1), bool(up & 1) and bool(down & 1)]
                      for up, down in basis], dtype=float)
    observables = abs(reference) ** 2 @ masks
    hstate = h @ state
    mean = float(np.vdot(state, hstate).real)
    variance = max(0., float(np.vdot(hstate, hstate).real) - mean**2)
    return {"greens": np.array(greens), "observables": observables,
            "scales": {"initial_energy": mean, "initial_energy_stddev": math.sqrt(variance),
                       "bath_coupling_norm": float(np.linalg.norm(couplings)),
                       "tau_times_energy_stddev": tau * math.sqrt(variance)}}


class NormalSchedule:
    def __init__(self, executable, root, *, nsites=7, m=256, tau=.2,
                 final_time=.6, u=2., timeout=120., bath_bandwidth=0., target_times=3):
        if nsites not in (3, 5, 7) or m < 1:
            raise ValueError("dense ED pilot supports odd nsites 3..7 and positive m")
        if not math.isfinite(bath_bandwidth) or bath_bandwidth < 0:
            raise ValueError("bath_bandwidth must be finite and nonnegative")
        if target_times < 2:
            raise ValueError("at least two target times required")
        self.target_times = target_times
        self.executable, self.root = executable, Path(root)
        self.nsites, self.m, self.tau = nsites, m, tau
        self.final_time, self.u, self.timeout = final_time, u, timeout
        self.count, self.legs = sample_count(tau, final_time), leg_count(tau, final_time)
        self.bath_energies = np.linspace(-bath_bandwidth/2, bath_bandwidth/2, nsites-1).tolist()
        self.couplings = common.Case(executable, root, nsites, m, tau, 2, u, timeout).couplings[0]
        self.jobs = []

    def stage(self, name, seed, expressions, source=None, mapping=None, measurements=(), legs=2):
        l = (self.nsites-1)//2
        nup, ndown = (l+1, l) if seed == "alpha" else (l, l+1)
        text = common.make_input(
            name, nsites=self.nsites, nup=nup, ndown=ndown, m=self.m,
            onsite_u=self.u, potential=-self.u/2, couplings=self.couplings,
            bath_energies=self.bath_energies, expressions=expressions,
            restart=source, restart_mapping=mapping,
            finite_loops="[" + ",".join([f"[@auto,{self.m},2]"] * legs) + "]")
        job = common.run_stage(self.executable, self.root, name, text,
                               restart=source, mapping=mapping,
                               measurements=measurements, timeout=self.timeout)
        self.jobs.append(job)
        return job

    def run(self):
        seeds = {}
        for seed, kind in (("alpha", "hole"), ("beta", "particle")):
            initial = self.stage(f"{seed}_seed", seed, ["string P0=|gs>;"],
                                 measurements=["<gs|nup|gs>", "<gs|ndown|gs>"])
            expected = (1., 0.) if seed == "alpha" else (0., 1.)
            for op, value in zip(("nup", "ndown"), expected):
                if abs(common.measured(initial, f"<gs|{op}|gs>") - value) > 1e-12:
                    raise RuntimeError("seed occupation mismatch")
            inserted = self.stage(f"{seed}_insert", seed,
                                  [common.insertion_expression(seed, kind)],
                                  initial["checkpoint"], [-1], ["<P0|P0>"])
            if abs(common.measured(inserted, "<P0|P0>") - 1) > 1e-12:
                raise RuntimeError("initial branch norm mismatch")
            expressions = [f'string P0="{advancing_expression(self.tau, self.nsites, "P0", self.target_times)}";',
                           f'string P1="{advancing_expression(self.tau, self.nsites, "gs", self.target_times)}";']
            job = self.stage(f"{seed}_continuous", seed, expressions,
                             inserted["checkpoint"], [0, -1], common.paired_measurements(), self.legs)
            expected_times = expected_endpoint_times(self.legs, self.tau, self.target_times)
            if not np.allclose(job["pvector_times"], expected_times, atol=1e-12, rtol=0):
                raise RuntimeError(f"unexpected endpoint clocks {job['pvector_times']}")
            exact = impulse_reference(self.nsites, self.u, seed, self.couplings,
                                      self.bath_energies, self.tau, self.count)
            label = "<P0.last|c|P1.last>" if kind == "hole" else "<P1.last|c|P0.last>"
            amplitudes = select_samples(job, label, self.tau, self.count)
            samples = []
            for k, sample in enumerate(amplitudes, 1):
                value = (1j if kind == "hole" else -1j) * complex(*sample["value"])
                samples.append(sample | {"green": common.encoded(value),
                    "ed_green": common.encoded(exact["greens"][k]),
                    "absolute_error": float(abs(value - exact["greens"][k]))})
            observables = {}
            for column, op in enumerate(("nup", "ndown", "nup*ndown")):
                observations = select_samples(job, f"<P1.last|{op}|P1.last>", self.tau, self.count)
                for k, observation in enumerate(observations, 1):
                    value = complex(*observation["value"])
                    observation.update(ed_value=float(exact["observables"][k, column]),
                        absolute_error=float(abs(value - exact["observables"][k, column])))
                observables[op] = observations
            norms = {label: select_samples(job, label, self.tau, self.count)
                     for label in ("<P0.last|P0.last>", "<P1.last|P1.last>")}
            seeds[seed] = {"channel": kind, "samples": samples, "reference_observables": observables,
                "norms": norms, "ed_scales": exact["scales"],
                "max_green_error": max(s["absolute_error"] for s in samples),
                "max_reference_observable_error": max(s["absolute_error"] for v in observables.values() for s in v),
                "hdf5_endpoint_times": job["pvector_times"]}
        summary = {"nsites": self.nsites, "retained_states": self.m, "U": self.u,
            "tau": self.tau, "final_time": self.final_time, "directional_legs": self.legs,
            "advance_each": self.nsites-2, "target_times": self.target_times,
            "bath_energies": self.bath_energies,
            "couplings": [common.encoded(z) for z in self.couplings], "seeds": seeds,
            "resources": {"jobs": len(self.jobs),
                "wall_seconds_sum": sum(j["wall_seconds"] for j in self.jobs),
                "peak_child_rss_bytes": max(j["peak_child_rss_bytes"] or 0 for j in self.jobs),
                "discarded_weight_max": max(max(j["parsed"]["discarded_weights"], default=0.) for j in self.jobs),
                "wft_warning_count": sum(len(j["parsed"]["wft_warnings"]) for j in self.jobs)},
            "jobs": [{key: j[key] for key in ("name", "cwd", "input_sha256", "pvector_times")}
                     for j in self.jobs],
            "scope": "Constant bath, paired reference, nonzero t'=0 channels only; no DMFT feedback."}
        io.atomic_write_json(self.root / "summary.json", summary)
        return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--m", type=int, default=256)
    parser.add_argument("--taus", default=".2,.1,.05")
    parser.add_argument("--final-time", type=float, default=.6)
    parser.add_argument("--nsites", type=int, default=7)
    parser.add_argument("--U", type=float, default=2.)
    parser.add_argument("--timeout", type=float, default=120.)
    parser.add_argument("--bath-bandwidth", type=float, default=0.)
    parser.add_argument("--target-times", type=int, default=3,
                        help="target times per window; engine default is three")
    args = parser.parse_args(argv)
    executable = str(Path(args.executable).resolve())
    output = Path(args.output).resolve()
    provenance = {"executable": executable, "executable_sha256": io.sha256_file(executable),
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "python": platform.python_version(), "numpy": np.__version__,
        "spack_environment": os.environ.get("SPACK_ENV"),
        "experiment_sources_sha256": {p.name: io.sha256_file(p) for p in Path(__file__).parent.glob("*.py")}}
    output.mkdir(parents=True, exist_ok=False)
    io.atomic_write_json(output / "provenance.json", provenance)
    summaries = []
    for value in args.taus.split(","):
        tau = float(value)
        case = NormalSchedule(executable, output / f"tau_{tau:.8g}", nsites=args.nsites,
            m=args.m, tau=tau, final_time=args.final_time, u=args.U, timeout=args.timeout,
            bath_bandwidth=args.bath_bandwidth, target_times=args.target_times)
        summary = case.run()
        summaries.append(summary)
        io.atomic_write_json(output / "results.json", {"provenance": provenance, "cases": summaries})
        print(f"tau={tau:g} m={args.m}: " + ", ".join(
            f"{seed} GF={s['max_green_error']:.3g} refs={s['max_reference_observable_error']:.3g}"
            for seed, s in summary["seeds"].items()), flush=True)


if __name__ == "__main__":
    main()
