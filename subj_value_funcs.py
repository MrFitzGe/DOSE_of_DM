## Subjective Value functions
import numpy as np
import polars as pl


### loss aversion
def loss_aversion_sv(
    m: float | int | pl.Expr, lambda_param: float | int | pl.Expr
) -> float | pl.Expr:
    """
    Calculates subjective value with loss aversion.
    SV = m if m >= 0 else lambda * m.
    Compatible with both scalar values and Polars expressions.
    """
    if isinstance(m, (pl.Expr, pl.Series)):
        return pl.when(m >= 0).then(m).otherwise(lambda_param * m)
    else:
        return m if m >= 0 else lambda_param * m


### risk aversion valuation
def risk_aversion_sv(
    m: float | int | pl.Expr, rho: float | int | pl.Expr
) -> float | pl.Expr:
    """
    Calculates subjective value with risk aversion (power utility).
    SV = m^rho.
    Compatible with both scalar values and Polars expressions.
    """
    if isinstance(m, (pl.Expr, pl.Series)):
        return m.pow(rho)
    else:
        return np.power(m, rho)


### combined full prospect theory model
def prospect_theory_sv(
    m: float | int | pl.Expr,
    lambda_param: float | int | pl.Expr,
    rho: float | int | pl.Expr,
) -> float | pl.Expr:
    """
    Calculates subjective value using Prospect Theory.
    SV = m^rho if m >= 0 else -lambda * (-m)^rho.
    Compatible with both scalar values and Polars expressions.
    """
    if isinstance(m, (pl.Expr, pl.Series)):
        return (
            pl.when(m >= 0)
            .then(m.pow(rho))
            .otherwise(-lambda_param * (-m).pow(rho))
        )
    else:
        if m >= 0:
            return np.power(m, rho)
        else:
            return -lambda_param * np.power(-m, rho)


### probability discounting
def prb_discount_sv(
    m: float | int | pl.Expr,
    p: float | int | pl.Expr,
    k: float | int | pl.Expr,
    s: float | int | pl.Expr,
) -> float | pl.Expr:
    """
    Calculates subjective value of probability discounting with prb scaling (k) and sensitivity (s).
    SV = m / ((1 + k * {1-p}/p)^s)
    Compatible with both scalar values and Polars expressions.
    """
    if any(isinstance(x, (pl.Expr, pl.Series)) for x in [m, p, k, s]):
        # Polars implementation
        m_exp = pl.lit(m) if not isinstance(m, (pl.Expr, pl.Series)) else m
        p_exp = pl.lit(p) if not isinstance(p, (pl.Expr, pl.Series)) else p
        k_exp = pl.lit(k) if not isinstance(k, (pl.Expr, pl.Series)) else k
        s_exp = pl.lit(s) if not isinstance(s, (pl.Expr, pl.Series)) else s

        prb_ratio = (1 - p_exp) / p_exp
        return m_exp / ((1 + k_exp * prb_ratio).pow(s_exp))
    else:
        # Numpy/Python implementation
        prb_ratio = (1 - p) / p
        return m / ((1 + k * prb_ratio) ** s)


### hyperbolic temporal discounting
def hyperbolic_discount_sv(
    m: float | int | pl.Expr,
    cost: float | int | pl.Expr,
    k: float | int | pl.Expr,
) -> float | pl.Expr:
    """
    Calculates subjective value with hyperbolic temporal discounting with discount parameter k.
    SV = m / (1 + k * cost).
    Compatible with both scalar values and Polars expressions.
    """
    return m / (1 + k * cost)


### sigmoidal effort discounting
def sigmoidal_discount_sv(
    m: float | int | pl.Expr,
    cost: float | int | pl.Expr,
    k: float | int | pl.Expr,
    p: float | int | pl.Expr,
) -> float | pl.Expr:
    """
    Calculates subjective value with sigmoidal effort discounting with 2 parameters -
    k for the slope and p for the location or magnitude required to induce discounting.
    Compatible with both scalar values and Polars expressions.
    """
    if isinstance(m, (pl.Expr, pl.Series)) or isinstance(
        cost, (pl.Expr, pl.Series)
    ):
        # Polars implementation
        m_exp = pl.lit(m) if not isinstance(m, (pl.Expr, pl.Series)) else m
        cost_exp = (
            pl.lit(cost) if not isinstance(cost, (pl.Expr, pl.Series)) else cost
        )

        # Sigmoid: 1 / (1 + exp(-k * (cost - p)))
        sig_cost = 1 / (1 + (-k * (cost_exp - p)).exp())
        sig_p = 1 / (1 + (k * p).exp())
        norm_factor = 1 + (-k * p).exp()

        return m_exp * (1 - (sig_cost - sig_p) * norm_factor)
    else:
        # Numpy implementation
        sig_cost = 1 / (1 + np.exp(-k * (cost - p)))
        sig_p = 1 / (1 + np.exp(k * p))
        norm_factor = 1 + 1 / np.exp(k * p)

        return m * (1 - (sig_cost - sig_p) * norm_factor)


### two-m param power discount
def power_discount_sv(
    m: float | int | pl.Expr,
    cost: float | int | pl.Expr,
    k: float | int | pl.Expr,
    p: float | int | pl.Expr,
) -> float | pl.Expr:
    """
    Calculates subjective value with two-parameter power discounting -
    k for the slope and p for the individual sensitivity/intensity of effort cost.
    Compatible with both scalar values and Polars expressions.
    """
    if isinstance(m, (pl.Expr, pl.Series)) or isinstance(
        cost, (pl.Expr, pl.Series)
    ):
        # Polars implementation
        m_exp = pl.lit(m) if not isinstance(m, (pl.Expr, pl.Series)) else m
        cost_exp = (
            pl.lit(cost) if not isinstance(cost, (pl.Expr, pl.Series)) else cost
        )
        return m_exp - k * cost_exp.pow(p)
    else:
        # Numpy implementation
        return m - k * np.power(cost, p)
