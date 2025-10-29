from datetime import datetime
from dateutil import parser
import numpy as np


def xnpv(rate, cashflows):
    """Compute the net present value for non-periodic cashflows.

    cashflows: iterable of (date, amount) where date is datetime/date-like
    rate: annual rate (decimal)
    """
    if rate <= -1:
        return float('inf')
    dates = [parser.parse(d).date() if isinstance(d, str) else d for d, _ in cashflows]
    amounts = [a for _, a in cashflows]
    d0 = dates[0]
    return sum([a / ((1 + rate) ** ((d - d0).days / 365.0)) for a, d in zip(amounts, dates)])


def _xirr_f(rate, cashflows):
    return xnpv(rate, cashflows)


def xirr(cashflows, guess=0.1, tol=1e-6, maxiter=100):
    """Compute XIRR using Newton's method (secant fallback).

    cashflows: list of (date, amount) where amounts are positive for inflows, negative for outflows.
    returns: annualized rate (decimal)
    """
    # Use simple secant method style root finding on NPV(rate) = 0
    # Start with two guesses
    r0 = guess
    r1 = guess * 1.1 if guess != 0 else 0.1

    f0 = _xirr_f(r0, cashflows)
    f1 = _xirr_f(r1, cashflows)

    for i in range(maxiter):
        if abs(f1 - f0) < 1e-12:
            break
        # secant step
        r2 = r1 - f1 * (r1 - r0) / (f1 - f0)
        f2 = _xirr_f(r2, cashflows)
        if abs(f2) < tol:
            return r2
        r0, f0 = r1, f1
        r1, f1 = r2, f2

    # fallback: return best estimate
    return r1


if __name__ == '__main__':
    # tiny smoke
    cfs = [
        (datetime(2020, 1, 1), -1000),
        (datetime(2021, 1, 1), 1100),
    ]
    print('XIRR:', xirr(cfs))
