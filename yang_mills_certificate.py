'''Run:
    python yang_mills_certificate.py                 # self-tests + demo
    python yang_mills_certificate.py --d0 1 --errors 0.1,0.15,0.2
    python yang_mills_certificate.py --rg-step --g 0.2 --b0 0.07 --scale 2

Only Python's standard library is required.'''


from __future__ import annotations

import argparse
import json
import math
import sys
import unittest
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence


class CertificateInputError(ValueError):
    """Raised when supplied numerical data are not mathematically admissible."""


def _finite_positive(name: str, value: float, *, allow_zero: bool = False) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise CertificateInputError(f"{name} must be finite")
    if value < 0 or (value == 0 and not allow_zero):
        op = 'nonnegative' if allow_zero else 'positive'
        raise CertificateInputError(f"{name} must be {op}")
    return value


def parse_errors(text: str) -> list[float]:
    """Parse comma-separated nonnegative error bounds."""
    if not text.strip():
        return []
    try:
        values = [float(piece.strip()) for piece in text.split(',')]
    except ValueError as exc:
        raise CertificateInputError("errors must be comma-separated numbers") from exc
    if any((not math.isfinite(x)) or x < 0 for x in values):
        raise CertificateInputError("every error bound must be finite and >= 0")
    return values


@dataclass(frozen=True)
class GapCertificate:
    """Arithmetic audit of a finite sequence of RG error bounds.

    d0_lower_bound is a rigorously proven lower bound in a mathematical proof;
    this program can only check its numerical value, not verify its provenance.
    """

    d0_lower_bound: float
    errors: tuple[float, ...]
    cumulative_error: float
    margin: float
    scale_lower_bounds: tuple[float, ...]
    positive_margin: bool

    @property
    def finite_sequence_certifies_positive_gap(self) -> bool:
        return self.positive_margin and self.margin > 0.0

    def to_dict(self) -> dict:
        result = asdict(self)
        result["finite_sequence_certifies_positive_gap"] = (
            self.finite_sequence_certifies_positive_gap
        )
        result["scope"] = (
            "finite supplied sequence only; uniformity in cutoff/volume and "
            "validity of the RG recurrence are external proof obligations"
        )
        return result


def audit_gap_recursion(d0_lower_bound: float,
                        errors: Iterable[float]) -> GapCertificate:
    """Check the implication Delta[k+1] >= Delta[k] - eps[k].

    If Delta[0] >= d0 and each eps[k] >= 0, induction yields
    Delta[n] >= d0 - sum_{j < n} eps[j]. This function audits that bound.
    It does not establish the recurrence for Yang--Mills.
    """
    d0 = _finite_positive("d0_lower_bound", d0_lower_bound)
    eps = tuple(float(e) for e in errors)
    if any((not math.isfinite(e)) or e < 0 for e in eps):
        raise CertificateInputError("every error bound must be finite and >= 0")
    cumulative = 0.0
    lowers = [d0]
    for e in eps:
        cumulative += e
        lowers.append(d0 - cumulative)
    margin = d0 - math.fsum(eps)
    return GapCertificate(
        d0_lower_bound=d0,
        errors=eps,
        cumulative_error=math.fsum(eps),
        margin=margin,
        scale_lower_bounds=tuple(lowers),
        positive_margin=margin > 0.0,
    )


@dataclass(frozen=True)
class RGStep:
    g_uv: float
    g_ir_one_loop: float
    g_ir_cubic_approximation: float
    denominator: float
    scale_factor: float
    b0: float


def one_loop_ir_step(g_uv: float, b0: float, scale_factor: float) -> RGStep:
    """Integrate beta(g)=mu dg/dmu=-b0*g^3 over mu_IR=mu_UV/scale_factor.

    Exact within the one-loop beta function:
        1/g_IR^2 = 1/g_UV^2 - 2*b0*log(scale_factor).
    Requires a positive denominator. Its weak-coupling expansion is
        g_IR = g_UV + b0*g_UV^3*log(scale_factor) + O(g_UV^5).
    This is a perturbative consistency check, not a nonperturbative proof.
    """
    g = _finite_positive("g_uv", g_uv)
    b = _finite_positive("b0", b0)
    s = float(scale_factor)
    if not math.isfinite(s) or s <= 1.0:
        raise CertificateInputError("scale_factor must be finite and > 1")
    denominator = 1.0 - 2.0 * b * g * g * math.log(s)
    if denominator <= 0.0:
        raise CertificateInputError(
            "one-loop integration left its finite-coupling domain: "
            "1 - 2*b0*g^2*log(scale_factor) must be > 0"
        )
    g_ir = g / math.sqrt(denominator)
    cubic = g + b * g**3 * math.log(s)
    if not (g_ir > g):
        raise AssertionError("IR coupling should grow for b0>0 at this one-loop order")
    return RGStep(g, g_ir, cubic, denominator, s, b)


def transfer_gap_from_norm(q: float, physical_block_size: float) -> float:
    """Convert q=||T-P_vac|| into a gap lower bound -log(q)/block_size.

    This requires an already justified positive self-adjoint transfer operator,
    a normalized unique-vacuum projection, and q<1. The function checks only
    the scalar calculation, not these operator-theoretic assumptions.
    """
    q = float(q)
    if not math.isfinite(q) or q <= 0.0 or q >= 1.0:
        raise CertificateInputError("q must satisfy 0 < q < 1")
    b = _finite_positive("physical_block_size", physical_block_size)
    return -math.log(q) / b


def required_transfer_norm(gap_lower_bound: float, physical_step: float) -> float:
    """For a physical gap >= m, the transfer contraction must be <= exp(-step*m)."""
    m = _finite_positive("gap_lower_bound", gap_lower_bound)
    t = _finite_positive("physical_step", physical_step)
    return math.exp(-t * m)


def continuum_semigroup_diagnostic(mass_lower_bound: float,
                                   physical_time: float,
                                   spacings: Sequence[float]) -> list[dict[str, float]]:
    """Numerically illustrate (not prove) floor(t/a)*a -> t.

    For each spacing, n=floor(t/a), and the certified-form bound is
    exp(-m*n*a), tending to exp(-m*t). The analytic limit follows from
    0 <= t - a*floor(t/a) < a, not from these samples.
    """
    m = _finite_positive("mass_lower_bound", mass_lower_bound)
    t = _finite_positive("physical_time", physical_time)
    rows = []
    for a0 in spacings:
        a = _finite_positive("spacing", a0)
        n = math.floor(t / a)
        lattice_time = n * a
        rows.append({
            "a": a,
            "n": float(n),
            "n_times_a": lattice_time,
            "gap_bound": math.exp(-m * lattice_time),
            "continuum_target": math.exp(-m * t),
            "time_error": t - lattice_time,
        })
    return rows


# ---------- Tests of the scalar mathematics implemented above ----------
class CertificateTests(unittest.TestCase):
    def test_positive_margin(self):
        cert = audit_gap_recursion(1.0, [0.1, 0.15, 0.2])
        self.assertAlmostEqual(cert.margin, 0.55)
        self.assertTrue(cert.finite_sequence_certifies_positive_gap)
        self.assertAlmostEqual(cert.scale_lower_bounds[-1], 0.55)

    def test_summability_alone_is_not_enough(self):
        cert = audit_gap_recursion(1.0, [2.0])
        self.assertEqual(cert.cumulative_error, 2.0)
        self.assertFalse(cert.finite_sequence_certifies_positive_gap)
        self.assertAlmostEqual(cert.margin, -1.0)

    def test_equal_error_budget_does_not_give_strict_gap(self):
        cert = audit_gap_recursion(1.0, [0.25, 0.75])
        self.assertEqual(cert.margin, 0.0)
        self.assertFalse(cert.finite_sequence_certifies_positive_gap)

    def test_negative_error_rejected(self):
        with self.assertRaises(CertificateInputError):
            audit_gap_recursion(1.0, [0.1, -0.01])

    def test_ir_coupling_grows_in_correct_direction(self):
        result = one_loop_ir_step(g_uv=0.2, b0=0.07, scale_factor=2.0)
        self.assertGreater(result.g_ir_one_loop, result.g_uv)
        self.assertGreater(result.g_cubic_approximation if hasattr(result, "g_cubic_approximation") else result.g_ir_cubic_approximation, result.g_uv)

    def test_transfer_norm_gap_conversion(self):
        q = math.exp(-0.4 * 2.0)
        self.assertAlmostEqual(transfer_gap_from_norm(q, 2.0), 0.4)

    def test_transfer_norm_for_gap(self):
        self.assertAlmostEqual(required_transfer_norm(0.4, 2.0), math.exp(-0.8))

    def test_semigroup_mesh_error_is_below_spacing(self):
        rows = continuum_semigroup_diagnostic(0.4, 1.0, [0.2, 0.1, 0.05])
        for row in rows:
            self.assertGreaterEqual(row["time_error"], 0.0)
            self.assertLess(row["time_error"], row["a"] + 1e-14)


def run_demo() -> dict:
    cert = audit_gap_recursion(1.0, [0.1, 0.15, 0.2])
    rg = one_loop_ir_step(0.2, 0.07, 2.0)
    m = cert.margin
    transfer = required_transfer_norm(m, 0.5) if m > 0 else None
    continuum = continuum_semigroup_diagnostic(m, 1.0, [0.2, 0.1, 0.05]) if m > 0 else []
    return {
        "status": "SCALAR CHECKS PASSED; YANG--MILLS PROOF NOT ESTABLISHED",
        "conditional_gap_certificate_demo": cert.to_dict(),
        "one_loop_rg_direction_demo": asdict(rg),
        "transfer_norm_for_demo_margin_at_block_size_0.5": transfer,
        "continuum_limit_numerical_diagnostic": continuum,
        "unproved_mathematical_obligations": [
            "Construct the continuum Yang--Mills measure/theory in four dimensions.",
            "Prove the RG gap recursion for the actual Yang--Mills block transformation.",
            "Prove a strictly positive initial physical gap lower bound in the required regime.",
            "Prove the sum of RG errors stays below that bound by a uniform positive margin, uniformly in cutoff and volume.",
            "Resolve the UV/IR flow and establish entry into any claimed convergence regime.",
            "Prove transfer-operator positivity, vacuum uniqueness, and convergence of semigroups/projections.",
            "Prove nontriviality and all required axioms of the reconstructed continuum QFT."
        ],
        "interpretation": (
            "The sample constants are illustrative, not Yang--Mills estimates. "
            "A successful run verifies arithmetic and conditional implications only."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="run built-in unit tests")
    parser.add_argument("--json", action="store_true", help="print results as JSON")
    parser.add_argument("--d0", type=float, help="independent initial gap lower bound for arithmetic audit")
    parser.add_argument("--errors", type=str, help="comma-separated RG error bounds")
    parser.add_argument("--rg-step", action="store_true", help="calculate a one-loop infrared RG step")
    parser.add_argument("--g", type=float, default=0.2, help="ultraviolet coupling for --rg-step")
    parser.add_argument("--b0", type=float, default=0.07, help="positive one-loop coefficient")
    parser.add_argument("--scale", type=float, default=2.0, help="blocking factor > 1")
    args = parser.parse_args(argv)

    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(CertificateTests)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        if not result.wasSuccessful():
            return 1

    payload: dict = {}
    if args.d0 is not None or args.errors is not None:
        if args.d0 is None or args.errors is None:
            parser.error("--d0 and --errors must be supplied together")
        payload["gap_certificate"] = audit_gap_recursion(args.d0, parse_errors(args.errors)).to_dict()
    if args.rg_step:
        payload["one_loop_rg_step"] = asdict(one_loop_ir_step(args.g, args.b0, args.scale))
    if not payload and not args.self_test:
        payload = run_demo()
    elif not payload and args.self_test:
        payload = {"status": "SELF-TESTS PASSED", "scope": "scalar/finite computations only"}

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0
#Uso de IA para código revisado por pares humanos.

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CertificateInputError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
