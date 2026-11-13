#!/usr/bin/env python3
"""Derive relative-minimax polynomials using only Python's standard library.

Method: polynomial Remez exchange, solving P(z) + (-1)^i E f(z) = f(z)
at alternating extrema. The central target is erf(y)/(2*y), with z=2*y*y-1
on 0 <= y <= 1. The tail target is exp(y*y)*erfc(y)/2, with z = 4*y -
(2*segment+1) on half-unit intervals from 1 to 10.5. Coefficients are fitted
here, never taken from an error-function implementation or coefficient table.

Independent reference: Decimal arithmetic at 90 digits; pi from Machin's
arctangent identity. In the central region integrate exp(-t*t) by its power
series. For y >= 2 evaluate the upper incomplete gamma continued fraction
directly (a=1/2, x=y*y), so no tail is obtained by subtracting erf from one.
The continued fraction follows integration by parts of the gamma integral.

Run from the repository root: python3 crates/ft8-ldpc/tools/pb_osd/derive_q.py
Outputs: src/q_coefficients.rs, tools/q_reference.bin, target/q-validation/fit.json.
"""

from decimal import Decimal as D, getcontext
from functools import lru_cache
import json
import math
from pathlib import Path
import random
import struct

getcontext().prec = 90
EPS = D("1e-82")
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "target/q-validation"
DEGREE = 10
SEGMENTS = 21
GRID = 512


def arctan(x):
    term = x
    total = term
    n = 1
    while abs(term) > EPS:
        term *= -x*x
        total += term / (2*n+1)
        n += 1
    return total


PI = 16*arctan(D(1)/5) - 4*arctan(D(1)/239)
SQRT_PI = PI.sqrt()


def cosine(x):
    term = total = D(1)
    n = 1
    while abs(term) > EPS:
        term *= -x*x/((2*n-1)*(2*n))
        total += term
        n += 1
    return total


@lru_cache(maxsize=100_000)
def scaled_tail(y):
    """exp(y*y)*erfc(y)/2, evaluated without a small-tail subtraction."""
    if y < 2:
        term = y
        integral = y
        n = 1
        while abs(term) > EPS:
            term *= -y*y/n
            integral += term / (2*n+1)
            n += 1
        return (D("0.5") - integral/SQRT_PI)*(y*y).exp()
    b = y*y + D("0.5")
    c = D("1e100")
    d = 1/b
    h = d
    for i in range(1, 10_000):
        an = -D(i)*(D(i)-D("0.5"))
        b += 2
        d = 1/(an*d+b)
        c = b+an/c
        delta = c*d
        h *= delta
        if abs(delta-1) < EPS:
            return y*h/(2*SQRT_PI)
    raise ArithmeticError("reference continued fraction did not converge")


def solve(matrix, rhs):
    a = [row[:] + [v] for row, v in zip(matrix, rhs)]
    n = len(a)
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        a[col], a[pivot] = a[pivot], a[col]
        divisor = a[col][col]
        a[col] = [v/divisor for v in a[col]]
        for row in range(n):
            if row != col:
                factor = a[row][col]
                a[row] = [v-factor*w for v, w in zip(a[row], a[col])]
    return [row[-1] for row in a]


def polynomial(c, x):
    value = c[-1]
    for v in reversed(c[:-1]):
        value = value*x + v
    return value


def chebyshev_norm(coeffs):
    basis = [D(1)]
    transformed = [D(0)]*len(coeffs)
    for c in coeffs:
        for i,v in enumerate(basis):
            transformed[i] += c*v
        next_basis = [D(0)]*(len(basis)+1)
        for i,v in enumerate(basis):
            if i == 0:
                next_basis[1] += v
            else:
                next_basis[i-1] += v/2
                next_basis[i+1] += v/2
        basis = next_basis
    return sum(abs(c) for c in transformed)


def error_bound(segment, target, coeffs):
    """Bound the rounded polynomial with a Chebyshev interpolant and remainder.

    For the tail f^(n)(y)=(-2)^n/sqrt(pi)*integral(t^n exp(-t*t-2*y*t) dt).
    Its magnitude decreases with y. The Chebyshev node product has supremum
    2*(width/4)^n. The coefficient difference is bounded by its l1 norm.
    This bounds all arguments in the interval, not just the validation grid.
    """
    n = DEGREE+1
    roots = [cosine(PI*D(2*i+1)/D(2*n)) for i in range(n)]
    interpolant = solve([[z**j if j else D(1) for j in range(n)] for z in roots],
                        [target(z) for z in roots])
    rounded = [D(float(c)) for c in coeffs]
    difference = chebyshev_norm([a-b for a,b in zip(rounded,interpolant)])
    factorial = D(math.factorial(n))
    unit = D(2)**-53
    gamma = n*unit/(1-n*unit)
    evaluation = gamma*sum(abs(c) for c in rounded)
    if segment == -1:
        # f(w) = integral_0^1 exp(-w*t*t) dt/sqrt(pi), 0 <= w <= 1.
        remainder = 2*(D(1)/4)**n/(SQRT_PI*(2*n+1)*factorial)
        minimum = scaled_tail(D(1))*(-D(1)).exp()
        coordinate = unit*sum(i*abs(c) for i,c in enumerate(rounded))
        return (remainder+difference+evaluation+coordinate+unit/2)/minimum
    a = D(segment)/2
    moments = [SQRT_PI*scaled_tail(a)]
    moments.append(D("0.5")-a*moments[0])
    for i in range(2,n+1):
        moments.append(((i-1)*moments[-2]-2*a*moments[-1])/2)
    remainder = 2**n*moments[n]/SQRT_PI/factorial*2*(D("0.5")/4)**n
    minimum = scaled_tail(a+D("0.5"))
    # Four units allow for exp rounding; two more cover the final product.
    return (remainder+difference+evaluation)/minimum + 6*unit


def central(z):
    square = (z+1)/2
    term = D(1)
    total = term
    n = 1
    while abs(term) > EPS:
        term *= -square/n
        total += term/(2*n+1)
        n += 1
    return total/SQRT_PI


def fit(segment):
    center = (D(segment)+D("0.5"))/2
    def target(z):
        return central(z) if segment == -1 else scaled_tail(center+z/4)
    grid = [D(-1)+D(2)*i/GRID for i in range(GRID+1)]
    targets = [target(z) for z in grid]
    nodes = [D(str(-math.cos(math.pi*i/(DEGREE+1)))) for i in range(DEGREE+2)]
    iterations = 0
    for iterations in range(12):
        matrix = [[z**j if j else D(1) for j in range(DEGREE+1)]
                  + [target(z)*(-1 if i % 2 else 1)] for i, z in enumerate(nodes)]
        solution = solve(matrix, [target(z) for z in nodes])
        coeffs = solution[:-1]
        errors = [polynomial(coeffs,z)/f-1 for z,f in zip(grid,targets)]
        extrema = [0] + [i for i in range(1,GRID)
                            if abs(errors[i]) >= abs(errors[i-1])
                            and abs(errors[i]) >= abs(errors[i+1])] + [GRID]
        alternating = []
        for i in extrema:
            if alternating and errors[i]*errors[alternating[-1]] > 0:
                if abs(errors[i]) > abs(errors[alternating[-1]]):
                    alternating[-1] = i
            else:
                alternating.append(i)
        assert len(alternating) == DEGREE+2, (segment, len(alternating))
        next_nodes = [grid[i] for i in alternating]
        if next_nodes == nodes:
            break
        nodes = next_nodes
    else:
        raise ArithmeticError("Remez exchange did not converge")
    # Refine each interior extremum continuously, then repeat the exchange solve.
    for _ in range(3):
        refined = [D(-1)]
        for z in nodes[1:-1]:
            lo, hi = max(D(-1), z-D(4)/GRID), min(D(1), z+D(4)/GRID)
            def error(z):
                return abs(polynomial(coeffs,z)/target(z)-1)
            for _ in range(50):
                left, right = (2*lo+hi)/3, (lo+2*hi)/3
                if error(left) < error(right): lo = left
                else: hi = right
            refined.append((lo+hi)/2)
        nodes = refined + [D(1)]
        matrix = [[z**j if j else D(1) for j in range(DEGREE+1)]
                  + [target(z)*(-1 if i % 2 else 1)] for i,z in enumerate(nodes)]
        solution = solve(matrix, [target(z) for z in nodes])
        coeffs = solution[:-1]
    grid_error = max(abs(polynomial(coeffs,z)/f-1) for z,f in zip(grid,targets))
    bound = error_bound(segment, target, coeffs)
    assert bound < D("1e-13"), (segment,bound)
    print(f"segment {segment:2}: degree {DEGREE}, relative fit error {grid_error:.3E}, bound {bound:.3E}", flush=True)
    return coeffs, {"segment":segment, "degree":DEGREE, "exchange_iterations":iterations+1,
                    "relative_grid_error":str(grid_error), "ripple":str(abs(solution[-1])),
                    "relative_error_bound":str(bound)}


def f32(value):
    return struct.unpack("<f", struct.pack("<f",value))[0]


def write_reference():
    # Independent off-grid values plus exact interval boundaries and adjacent f32s.
    inputs = [f32(i*14/8192) for i in range(8193)]
    rng = random.Random(0x515245464552454e)
    inputs += [f32(rng.uniform(0,14)) for _ in range(8192)]
    root2 = f32(math.sqrt(2))
    for i in range(22):
        x = f32(i*0.5*root2)
        bits = struct.unpack("<I",struct.pack("<f",x))[0]
        inputs += [struct.unpack("<f",struct.pack("<I",b))[0]
                   for b in range(max(0,bits-2),bits+3)]
    with (ROOT/"crates/ft8-ldpc/tools/pb_osd/q_reference.bin").open("wb") as out:
        for x in sorted(set(inputs)):
            y = D(f32(x/root2))
            q = scaled_tail(y)*(-y*y).exp()
            out.write(struct.pack("<I2d",struct.unpack("<I",struct.pack("<f",x))[0],float(q),float(1-q)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    coefficients, reports = [], []
    central_coeffs, central_report = fit(-1)
    for i in range(2, SEGMENTS):
        coeffs, report = fit(i)
        coefficients.append(coeffs)
        reports.append(report)
    text = "// Relative-error Remez coefficients for the normal tail.\n"
    text += "// Central: erf(y)/(2*y). Tails: exp(y*y)*erfc(y)/2 on half-unit intervals.\n"
    text += f"pub(super) const Q_CENTRAL: [f64; {DEGREE+1}] = [\n"
    text += "".join(f"    {float(c):.17e},\n" for c in central_coeffs) + "];\n"
    text += f"pub(super) const Q_COEFFICIENTS: [[f64; {DEGREE+1}]; {SEGMENTS-2}] = [\n"
    for coeffs in coefficients:
        text += "    [\n" + "".join(f"        {float(c):.17e},\n" for c in coeffs) + "    ],\n"
    text += "];\n"
    (ROOT/"crates/ft8-ldpc/src/pb_osd/q_coefficients.rs").write_text(text)
    (OUT/"fit.json").write_text(json.dumps([central_report]+reports,indent=2)+"\n")
    write_reference()
    print("coefficients and independent references written",flush=True)


if __name__ == "__main__":
    main()
