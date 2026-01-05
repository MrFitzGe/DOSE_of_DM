# Decision functions - functions for evaluating the subjective value / utility of a choice outcome
import numpy as np
import polars as pl

## Choice functions


### logit
def two_outcome_choice(
    v1: float | int | pl.Expr,
    v2: float | int | pl.Expr,
    choice_consistency: float,
) -> tuple[pl.Expr, float | pl.Expr]:
    """
    Calculates the choice probability and outcome for a two-option choice.
    Compatible with both scalar values and Polars expressions.
    """
    if isinstance(v1, (pl.Expr, pl.Series)) or isinstance(
        v2, (pl.Expr, pl.Series)
    ):
        v1 = pl.lit(v1) if not isinstance(v1, (pl.Expr, pl.Series)) else v1
        v2 = pl.lit(v2) if not isinstance(v2, (pl.Expr, pl.Series)) else v2

        choice_prb = 1 / (1 + (choice_consistency * (v2 - v1)).exp())
        choice_outcome = (
            pl.when(choice_prb >= 0.5)
            .then(pl.lit("outcome_1"))
            .otherwise(pl.lit("outcome_2"))
        )
    else:
        choice_prb = 1 / (1 + np.exp(-1 * choice_consistency * (v1 - v2)))
        choice_outcome = (
            pl.when(pl.lit(choice_prb) >= 0.5)
            .then(pl.lit("outcome_1"))
            .otherwise(pl.lit("outcome_2"))
        )

    return choice_outcome, choice_prb


### softmax_multi_choice
def multi_outcome_choice(
    vals: list | dict | np.ndarray | pl.Expr,
    choice_consistency: float,
) -> tuple[pl.Expr, float | pl.Expr]:
    """
    Softmax choice between multiple outcomes.
    Returns (choice_outcome, choice_prb) to match two_outcome_choice.
    Compatible with both scalar values and Polars expressions.
    """
    if isinstance(vals, dict):
        labels = list(vals.keys())
        values = list(vals.values())
    else:
        values = list(vals)
        labels = [f"outcome_{i + 1}" for i in range(len(values))]

    is_polars = any(isinstance(v, (pl.Expr, pl.Series)) for v in values)

    if is_polars:
        # Ensure all are expressions
        exprs = [
            pl.lit(v) if not isinstance(v, (pl.Expr, pl.Series)) else v
            for v in values
        ]
        # Softmax: P_i = exp(beta * V_i) / sum(exp(beta * V_j))
        exps = [(e * choice_consistency).exp() for e in exprs]
        sum_exps = sum(exps)
        probs = [e / sum_exps for e in exps]

        # We return the probability of the chosen outcome (max probability)
        choice_prb = pl.max_horizontal(probs)

        # Determine outcome label based on max probability
        choice_outcome = pl.when(probs[0] == choice_prb).then(pl.lit(labels[0]))
        for i in range(1, len(probs)):
            choice_outcome = choice_outcome.when(probs[i] == choice_prb).then(
                pl.lit(labels[i])
            )
        choice_outcome = choice_outcome.otherwise(pl.lit(None))

        return choice_outcome, choice_prb
    else:
        # Numpy/Scalar logic
        vals_arr = np.array(values)
        exps = np.exp(choice_consistency * vals_arr)
        probs = exps / np.sum(exps)
        max_idx = np.argmax(probs)

        choice_prb = probs[max_idx]
        choice_outcome = pl.lit(labels[max_idx])

        return choice_outcome, choice_prb


## Subjective Value functions


# loss aversion valuation
def loss_aversion_sv(
    amount: float | int | pl.Expr,
    lambda_param: float | int | pl.Expr,
) -> float | pl.Expr:
    """
    Calculates subjective value with loss aversion.
    SV = amount if amount >= 0 else lambda * amount.
    """
    if isinstance(amount, (pl.Expr, pl.Series)):
        return (
            pl.when(amount >= 0).then(amount).otherwise(lambda_param * amount)
        )
    else:
        return amount if amount >= 0 else lambda_param * amount


# risk aversion valuation
def risk_aversion_sv(
    amount: float | int | pl.Expr,
    rho: float | int | pl.Expr,
) -> float | pl.Expr:
    """
    Calculates subjective value with risk aversion (power utility).
    SV = amount^rho.
    """
    if isinstance(amount, (pl.Expr, pl.Series)):
        return amount.pow(rho)
    else:
        return np.power(amount, rho)


# combined full prospect theory model
def prospect_theory_sv(
    amount: float | int | pl.Expr,
    lambda_param: float | int | pl.Expr,
    rho: float | int | pl.Expr,
) -> float | pl.Expr:
    """
    Calculates subjective value using Prospect Theory.
    SV = amount^rho if amount >= 0 else -lambda * (-amount)^rho.
    """
    if isinstance(amount, (pl.Expr, pl.Series)):
        return (
            pl.when(amount >= 0)
            .then(amount.pow(rho))
            .otherwise(-lambda_param * (-amount).pow(rho))
        )
    else:
        if amount >= 0:
            return np.power(amount, rho)
        else:
            return -lambda_param * np.power(-amount, rho)


# sigmoidal effort discounting
def sigmoidal_discount_sv(
    m: float | int | pl.Expr,
    cost: float | int | pl.Expr,
    k: float | int | pl.Expr,
    p: float | int | pl.Expr,
) -> float | pl.Expr:
    """
    Calculates subjective value with sigmoidal effort discounting.
    """
    if isinstance(m, (pl.Expr, pl.Series)) or isinstance(
        cost, (pl.Expr, pl.Series)
    ):
        # Polars implementation
        m = pl.lit(m) if not isinstance(m, (pl.Expr, pl.Series)) else m
        cost = (
            pl.lit(cost) if not isinstance(cost, (pl.Expr, pl.Series)) else cost
        )

        # Sigmoid: 1 / (1 + exp(-k * (cost - p)))
        sig_cost = 1 / (1 + (-k * (cost - p)).exp())
        sig_p = 1 / (1 + (k * p).exp())
        norm_factor = 1 + (-k * p).exp()

        return m * (1 - (sig_cost - sig_p) * norm_factor)
    else:
        # Numpy implementation
        sig_cost = 1 / (1 + np.exp(-k * (cost - p)))
        sig_p = 1 / (1 + np.exp(k * p))
        norm_factor = 1 + 1 / np.exp(k * p)

        return m * (1 - (sig_cost - sig_p) * norm_factor)


# hyperbolic temporal discounting
def hyperbolic_discount_sv(
    m: float | int | pl.Expr,
    cost: float | int | pl.Expr,
    k: float | int | pl.Expr,
) -> float | pl.Expr:
    """
    Calculates subjective value with hyperbolic temporal discounting.
    SV = m / (1 + k * cost).
    """
    return m / (1 + k * cost)
