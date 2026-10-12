"""The environment side of the two-level reproducibility contract
(docs/design_m13.md §2): which environment generated an artefact, and
whether Level A can run in this one.

It lives in research/ rather than tests/ because the M13 run script refuses
to run outside the reference environment with the same decision the Level A
tests use (G3, C1). The comparators stay in tests/research/_reproducibility.py.
"""

import platform
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import version

from numpy.lib.introspect import opt_func_info


@dataclass(frozen=True)
class ReferenceEnvironment:
    """What decides the bits of an artefact, as far as has been measured.

    `log_dispatch` is the kernel numpy selects for float64 `np.log` on this
    CPU ("X86_V4" is AVX-512, "X86_V3" AVX2). It, not the Python version,
    is what separated the reproducing runners from the others in the R1
    diagnostic: numpy 2.4.6 and 2.5.3 both reproduce the baseline with the
    X86_V4 kernel and both miss it by 1 ulp in 107 of 2,764 log returns
    with X86_V3.
    """

    python: str
    numpy: str
    scipy: str
    pandas: str
    machine: str
    log_dispatch: str


# Where the SPY baseline was generated (its summary.json records Python
# 3.11.16, numpy 2.4.6 and pandas 3.0.5 on x86_64; scipy 1.17.1 is what
# uv.lock resolves for 3.11), plus the one CPU property shown to change its
# bits. The figures pre-registered in docs/design_m13.md from the extended
# data (the regime threshold) were computed in the same environment.
SPY_REFERENCE = ReferenceEnvironment(
    python="3.11",
    numpy="2.4.6",
    scipy="1.17.1",
    pandas="3.0.5",
    machine="x86_64",
    log_dispatch="X86_V4",
)


def log_dispatch() -> str:
    """The kernel numpy currently selects for float64 `np.log`."""
    return str(opt_func_info(func_name="log$", signature="float64")["log"]["dd"]["current"])


def current_environment() -> ReferenceEnvironment:
    return ReferenceEnvironment(
        python=f"{sys.version_info.major}.{sys.version_info.minor}",
        numpy=version("numpy"),
        scipy=version("scipy"),
        pandas=version("pandas"),
        machine=platform.machine(),
        log_dispatch=log_dispatch(),
    )


# gate.sh sets this to 1: on the development machine Level A must run, so
# an environment that cannot run it is a failure there, not a skip.
REQUIRE_LEVEL_A = "QUANT_RISK_AI_REQUIRE_LEVEL_A"


def level_a_decision(
    environment: ReferenceEnvironment,
    reference: ReferenceEnvironment,
    environ: Mapping[str, str],
) -> tuple[str, str]:
    """Whether Level A runs here: ("run" | "skip" | "fail", reason).

    Outside the reference environment Level A is skipped, unless
    REQUIRE_LEVEL_A is set, in which case it fails: a skip would let a
    commit through gate.sh without the byte-for-byte check it relies on.
    """
    if environment == reference:
        return "run", ""
    reason = f"Level A needs {reference}; this is {environment}"
    if environ.get(REQUIRE_LEVEL_A) == "1":
        return "fail", f"{REQUIRE_LEVEL_A}=1 but {reason}"
    return "skip", reason
