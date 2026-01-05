import numpy as np
from scipy.optimize import minimize

from decision_funcs import two_outcome_choice
from subj_value_funcs import hyperbolic_discount_sv

# for a set of historical choices, fit the parameters of the model using fast methods
# Output is the posterior likelihood of the parameters being the best fit for that participant given the choices
# Not interested in optimizing the model paremeters of individual difference (i.e. k discounting param), only interested in the stimulus parameters (money and cost) for the next trial
# The SV model type is an important constant hyperparameter
# At any point, the adaptive experiment can be stopped and the individual difference parameters can be estimated/returned.


def fit_hyperbolic_discount(
    a1,
    c1,
    a2,
    c2,
    choices,
):
    """
    Fits hyperbolic discounting model to binary choice data.

    Parameters:
    choices: array of 0 (SS) or 1 (LL)
    a1, c1: amount and delay of Smaller-Sooner option
    a2, c2: amount and delay of Larger-Later option
    """
    # Convert inputs to numpy arrays for vectorization
    a1 = np.array(a1)
    c1 = np.array(c1)
    a2 = np.array(a2)
    c2 = np.array(c2)
    choices = np.array(choices)

    def neg_log_likelihood(params):
        log_k, log_choice_consistency = params
        k = np.exp(log_k)
        choice_consistency = np.exp(log_choice_consistency)

        sv1 = hyperbolic_discount_sv(a1, c1, k)
        sv2 = hyperbolic_discount_sv(a2, c2, k)

        _, pred_prbs = two_outcome_choice(sv1, sv2, choice_consistency)
        # Clip values for numerical safety
        pred_prbs = np.clip(pred_prbs, 1e-10, 1 - 1e-10)
        # Log Likelihood: (y * log(p)) + ((1-y) * log(1-p))
        log_likelihood = choices * np.log(pred_prbs) + (1 - choices) * np.log(
            1 - pred_prbs
        )
        return -np.sum(log_likelihood)

    initial_guess = [np.log(0.01), np.log(1.0)]

    # Optimization
    # BFGS is good for unconstrained problems (we handled constraints via log-transform)
    result = minimize(neg_log_likelihood, initial_guess, method="BFGS")
    k_est, beta_est = np.exp(result.x)
    # res.hess_inv is the Variance-Covariance matrix
    # Standard Errors are sqrt of the diagonal
    try:
        # For BFGS, hess_inv is a 2x2 ndarray
        vcv = result.hess_inv
        stderr_k = np.sqrt(vcv[0, 0])
        stderr_beta = np.sqrt(vcv[1, 1])
        entropy = np.linalg.det(vcv)
    except:  # noqa: E722
        # Fallback if optimization failed to converge cleanly
        stderr_k, stderr_beta, entropy = np.nan, np.nan, np.nan

    return {
        "k": k_est,
        "beta": beta_est,
        "k_se": stderr_k,
        "beta_se": stderr_beta,
        "negative_log_likelihood": result.fun,
        "AIC": 4 + (2 * result.fun),
        "entropy": entropy,
        "success": result.success,
    }
