import numpy as np
from ax.api.client import Client
from ax.api.configs import RangeParameterConfig
from ax.api.protocols.metric import IMetric

from fit_model import fit_hyperbolic_discount

# 1. Initialize the Client.
client = Client()

# 2. Configure the parameters of the experiment Ax will search.
## TODO have optional choice sets using configs.ChoiceParameterConfig or AX generated  values from range like below
stimuli_params = [
    RangeParameterConfig(
        name="amount_1",
        bounds=(0, 10),
        parameter_type="int",
    ),
    RangeParameterConfig(
        name="cost_1",
        bounds=(1, 10),
        parameter_type="int",
    ),
    RangeParameterConfig(
        name="amount_2",
        bounds=(10, 100),
        parameter_type="int",
    ),
    RangeParameterConfig(
        name="cost_2",
        bounds=(11, 100),
        parameter_type="int",
    ),
]

client.configure_experiment(
    name="hyperbolic_discounting",
    parameters=stimuli_params,
)
# 3. Configure a metric for Ax to target (see other Tutorials for adding constraints,
# multiple objectives, tracking metrics etc.)
client.configure_optimization(objective="-entropy")

client.configure_metrics(
    [
        IMetric(name="k"),
        IMetric(name="k_se"),
        IMetric(name="beta"),
        IMetric(name="beta_se"),
        IMetric(name="negative_log_likelihood"),
        IMetric(name="AIC"),
        IMetric(name="success"),
    ]
)

# Initial burn-in trials everyone needs to answer to start the process
preexisting_trials = [
    (
        {"amount_1": 5, "cost_1": 2, "amount_2": 12, "cost_2": 25},
        {"choice": 1},
    ),
    (
        {"amount_1": 1, "cost_1": 1, "amount_2": 15, "cost_2": 55},
        {"choice": 0},
    ),
    (
        {"amount_1": 7, "cost_1": 10, "amount_2": 100, "cost_2": 40},
        {"choice": 1},
    ),
    (
        {"amount_1": 10, "cost_1": 2, "amount_2": 20, "cost_2": 65},
        {"choice": 0},
    ),
    (
        {"amount_1": 5, "cost_1": 5, "amount_2": 20, "cost_2": 15},
        {"choice": 1},
    ),
    (
        {"amount_1": 10, "cost_1": 10, "amount_2": 39, "cost_2": 72},
        {"choice": 0},
    ),
]

a1_list = [trial[0]["amount_1"] for trial in preexisting_trials]
c1_list = [trial[0]["cost_1"] for trial in preexisting_trials]
a2_list = [trial[0]["amount_2"] for trial in preexisting_trials]
c2_list = [trial[0]["cost_2"] for trial in preexisting_trials]
choices = np.array([trial[1]["choice"] for trial in preexisting_trials])

# Fit the hyperbolic discounting model
result = fit_hyperbolic_discount(a1_list, c1_list, a2_list, c2_list, choices)

# Print results
"""
print("Fitted Parameters:")
print(f"k (discount parameter): {result['k']:.6f}")
print(f"choice_consistency: {result['beta']:.6f}")
print(f"Entropy of model fit: {result['entropy']:.6f}")
print(f"Negative Log Likelihood: {result['negative_log_likelihood']:.6f}")
print(f"Optimization success: {result['success']}")
"""

for parameters, data in preexisting_trials:
    # Attach the parameterization to the Client as a trial and immediately complete it with the preexisting data
    trial_index = client.attach_trial(parameters=parameters)
    # Set raw_data as a dictionary with metric names as keys and results as values
    client.complete_trial(
        trial_index=trial_index,
        raw_data=result,
    )


# 5. Obtain the next configuration
client.get_next_trials(max_trials=1)
