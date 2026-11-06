"""Temporary adapter for inserting operators into saved reference trajectories.

The physics/evolution driver consumes InsertionPlan, not the gs-promotion details.
After the operator-on-P-ket engine bug is fixed, replace the saved-reference
branch of plan_insertion; see PVECTOR_INSERTION_BUG.md for the removal contract.
Initial-time insertion on the actual gs needs no workaround.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class InsertionPlan:
    expressions: tuple[str, ...]
    restart_mapping: tuple[int, ...]
    source_tv_for_psi: int | None
    branch_ket: str
    reference_ket: str
    branch_slot: int
    first_pair_mapping: tuple[int, int]
    first_pair_from_gs: bool
    local_insertion_time: float
    physical_clock_origin: float
    backend: str
    kind: str

    @property
    def measurement_label(self):
        bra, ket = ((self.branch_ket, self.reference_ket) if self.kind == "hole"
                    else (self.reference_ket, self.branch_ket))
        return f"<{bra}|c|{ket}>"

    @property
    def measurements(self):
        return (self.measurement_label, f"<{self.branch_ket}|{self.branch_ket}>",
                f"<{self.reference_ket}|nup|{self.reference_ket}>")

    def evolution_start(self, physical_time):
        return physical_time - self.physical_clock_origin


def plan_insertion(kind, physical_time, source_endpoint=None):
    """Plan an initial-gs or saved-reference insertion without mutating a checkpoint.

    source_endpoint=None means the source checkpoint's actual initial gs.
    Otherwise it selects a saved P-vector endpoint. Only this branch contains
    the temporary promotion workaround; callers must not special-case it.
    """
    if kind not in ("hole", "particle"):
        raise ValueError("invalid insertion channel")
    if not math.isfinite(physical_time) or physical_time < 0:
        raise ValueError("physical insertion time must be finite and nonnegative")
    if source_endpoint is None and physical_time != 0:
        raise ValueError("initial-gs insertion must be at physical time zero")
    if source_endpoint is not None and (not isinstance(source_endpoint, int) or source_endpoint < 0):
        raise ValueError("saved reference endpoint must be a nonnegative integer")
    op = "c" if kind == "hole" else "c'"
    return InsertionPlan(
        expressions=(f'string P0="{op}[0]*|gs>";',),
        restart_mapping=(-1,), source_tv_for_psi=source_endpoint,
        branch_ket="P0", reference_ket="gs", branch_slot=0,
        first_pair_mapping=(0, -1), first_pair_from_gs=True,
        local_insertion_time=0., physical_clock_origin=physical_time,
        backend="initial-gs" if source_endpoint is None else "gs-promotion-workaround",
        kind=kind)
