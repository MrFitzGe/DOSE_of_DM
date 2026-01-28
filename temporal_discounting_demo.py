import marimo

__generated_with = "0.19.2"
app = marimo.App(width="full")


@app.cell
def _():
    import os
    import random
    import time

    import marimo as mo
    import numpy as np
    import polars as pl
    import polars.selectors as cs
    from ax.api.client import Client
    from ax.api.configs import RangeParameterConfig
    from ax.api.protocols.metric import IMetric
    from loguru import logger

    from fit_model import fit_hyperbolic_discount
    return (
        Client,
        IMetric,
        RangeParameterConfig,
        fit_hyperbolic_discount,
        logger,
        mo,
        os,
        pl,
        random,
        time,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Adaptive Temporal Discounting Experiment
    """)
    return


@app.cell
def _(mo):
    # Configuration UI
    mo.md(r"## 1. Experiment Setup")

    save_path = mo.ui.text(
        label="Save Directory", value="data/", placeholder="e.g., data/"
    )
    file_name = mo.ui.text(
        label="Filename", value="experiment_results", placeholder="results"
    )
    file_type = mo.ui.dropdown(
        options=["csv", "parquet"], value="csv", label="File Format"
    )

    layout_mode = mo.ui.dropdown(
        options=["Left-Right", "Top-Bottom", "Mixed"],
        value="Left-Right",
        label="Display Layout",
    )

    position_mode = mo.ui.dropdown(
        options=["Constant", "Randomized"],
        value="Randomized",
        label="Option Position (SS vs LL)",
    )

    max_trials = mo.ui.number(start=5, stop=100, step=1, value=20, label="Max Trials")

    setup_form = mo.md(
        f"""
        {mo.hstack([save_path, file_name, file_type])}
        {mo.hstack([layout_mode, position_mode, max_trials])}
        """
    )
    return (
        file_name,
        file_type,
        layout_mode,
        max_trials,
        position_mode,
        save_path,
        setup_form,
    )


@app.cell
def _(mo, setup_form):
    mo.accordion({"Experiment Settings": setup_form})
    return


@app.cell
def _(Client, IMetric, RangeParameterConfig):
    def initialize_ax_client():
        client = Client()
        stimuli_params = [
            RangeParameterConfig(name="amount_1", bounds=(1, 10), parameter_type="int"),
            RangeParameterConfig(name="cost_1", bounds=(0, 5), parameter_type="int"),
            RangeParameterConfig(
                name="amount_2", bounds=(10, 100), parameter_type="int"
            ),
            RangeParameterConfig(name="cost_2", bounds=(5, 100), parameter_type="int"),
        ]
        client.configure_experiment(
            name="hyperbolic_discounting", parameters=stimuli_params
        )
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
        return client
    return (initialize_ax_client,)


@app.cell
def _():
    # Default burn-in trials
    default_burn_in = [
        {"amount_1": 5, "cost_1": 0, "amount_2": 12, "cost_2": 25},
        {"amount_1": 1, "cost_1": 0, "amount_2": 15, "cost_2": 55},
        {"amount_1": 7, "cost_1": 0, "amount_2": 100, "cost_2": 40},
        {"amount_1": 10, "cost_1": 0, "amount_2": 20, "cost_2": 65},
        {"amount_1": 5, "cost_1": 0, "amount_2": 20, "cost_2": 15},
        {"amount_1": 10, "cost_1": 0, "amount_2": 39, "cost_2": 72},
    ]
    return (default_burn_in,)


@app.cell
def _(default_burn_in, initialize_ax_client, mo):
    # State management
    get_state, set_state = mo.state(
        {
            "started": False,
            "trial_idx": 0,
            "history": [],
            "current_stimuli": None,
            "current_layout": None,
            "current_ss_pos": None,  # 0 for left/top, 1 for right/bottom
            "start_time": None,
            "ax_client": initialize_ax_client(),
            "burn_in": default_burn_in,
            "finished": False,
            "last_fit": None,
        }
    )
    return get_state, set_state


@app.cell
def _(
    file_name,
    file_type,
    fit_hyperbolic_discount,
    get_state,
    layout_mode,
    logger,
    max_trials,
    mo,
    os,
    pl,
    position_mode,
    random,
    save_path,
    set_state,
    time,
):
    def start_experiment():
        # Setup logging
        full_path = os.path.join(save_path.value, f"{file_name.value}.log")
        os.makedirs(save_path.value, exist_ok=True)
        logger.add(full_path, rotation="1 MB")
        logger.info("Experiment Started")

        # Prepare first trial
        next_trial_logic()

        set_state(lambda s: {**s, "started": True, "trial_idx": 0})

    def next_trial_logic():
        state = get_state()
        idx = state["trial_idx"]

        # Determine Stimuli
        if idx < len(state["burn_in"]):
            stimuli = state["burn_in"][idx]
        else:
            # Ax adaptive trial
            trial_params, _ = state["ax_client"].get_next_trials(max_trials=1)
            # Ax returns a dict of {trial_index: parameters}
            stimuli = list(trial_params.values())[0]

        # Determine Layout
        current_layout = layout_mode.value
        if current_layout == "Mixed":
            current_layout = random.choice(["Left-Right", "Top-Bottom"])

        # Determine SS Position (0 or 1)
        ss_pos = 0
        if position_mode.value == "Randomized":
            ss_pos = random.randint(0, 1)

        set_state(
            lambda s: {
                **s,
                "current_stimuli": stimuli,
                "current_layout": current_layout,
                "current_ss_pos": ss_pos,
                "start_time": time.time(),
            }
        )

    def handle_choice(choice_val):
        # choice_val: 0 if SS was clicked, 1 if LL was clicked
        end_time = time.time()
        state = get_state()
        rt = end_time - state["start_time"]

        stim = state["current_stimuli"]

        # Record data
        # Determine position label for SS
        ss_pos_label = ""
        if state["current_layout"] == "Left-Right":
            ss_pos_label = "left" if state["current_ss_pos"] == 0 else "right"
        else:
            ss_pos_label = "top" if state["current_ss_pos"] == 0 else "bottom"

        trial_data = {
            "trial_idx": state["trial_idx"],
            "amount_ss": stim["amount_1"],
            "cost_ss": stim["cost_1"],
            "amount_ll": stim["amount_2"],
            "cost_ll": stim["cost_2"],
            "ss_position": ss_pos_label,
            "layout": state["current_layout"],
            "choice": choice_val,  # 0 for SS, 1 for LL
            "rt": rt,
        }

        logger.info(f"Trial {state['trial_idx']} completed: {trial_data}")

        # Always add trial data to history
        new_history = state["history"] + [trial_data]
        fit_results = None

        # Check if we have completed burn-in trials
        # Note: state["trial_idx"] is the index of the current trial (0-based)
        # We need to check if we've completed all burn-in trials (idx >= len(burn_in))
        if state["trial_idx"] >= len(state["burn_in"]):
            # Fit model using all accumulated data (including burn-in)
            # Prepare lists for fitting
            a1 = [d["amount_ss"] for d in new_history]
            c1 = [d["cost_ss"] for d in new_history]
            a2 = [d["amount_ll"] for d in new_history]
            c2 = [d["cost_ll"] for d in new_history]
            choices = [d["choice"] for d in new_history]

            fit_results = fit_hyperbolic_discount(a1, c1, a2, c2, choices)

            # Attach to Ax (only for post-burn-in trials)
            # Note: Ax expects the parameters used for the trial
            parameters = {
                "amount_1": stim["amount_1"],
                "cost_1": stim["cost_1"],
                "amount_2": stim["amount_2"],
                "cost_2": stim["cost_2"],
            }
            trial_index = state["ax_client"].attach_trial(parameters=parameters)
            state["ax_client"].complete_trial(trial_index=trial_index, raw_data=fit_results)

        next_idx = state["trial_idx"] + 1
        is_finished = next_idx >= max_trials.value

        set_state(
            lambda s: {
                **s,
                "history": new_history,
                "trial_idx": next_idx,
                "last_fit": fit_results,
                "finished": is_finished,
            }
        )

        if not is_finished:
            next_trial_logic()
        else:
            save_final_data(new_history)

    def save_final_data(history):

        df = pl.DataFrame(history)
        path = os.path.join(save_path.value, f"{file_name.value}.{file_type.value}")
        if file_type.value == "csv":
            df.write_csv(path)
        else:
            df.write_parquet(path)
        logger.info(f"Data saved to {path}")

    start_button = mo.ui.button(
        label="Start Experiment", on_click=lambda _: start_experiment()
    )
    return handle_choice, start_button


@app.cell
def _(get_state, handle_choice, max_trials, mo, start_button):
    state = get_state()

    def render_option(amount, cost, is_ss):
        # Context dependent labels
        return mo.vstack(
            [
                mo.md(f"### Get ${amount}"),
                mo.md(f"### Wait: {cost}"),
                mo.ui.button(
                    label="Select",
                    on_click=lambda _: handle_choice(0 if is_ss else 1),
                    full_width=False,
                ),
            ],
            align="center",
        )

    if not state["started"]:
        content = mo.center(mo.vstack([mo.md("# Ready to begin?"), start_button]))
    elif state["finished"]:
        content = mo.center(
            mo.vstack(
                [
                    mo.md("# Experiment Complete"),
                    mo.md(
                        f"Final k estimate: {state['last_fit']['k']:.4f}"
                        if state["last_fit"]
                        else ""
                    ),
                    mo.md("Your data has been saved."),
                ]
            )
        )
    else:
        stim = state["current_stimuli"]
        ss_ui = render_option(stim["amount_1"], stim["cost_1"], True)
        ll_ui = render_option(stim["amount_2"], stim["cost_2"], False)

        # Order based on ss_pos
        options = [ss_ui, ll_ui] if state["current_ss_pos"] == 0 else [ll_ui, ss_ui]

        if state["current_layout"] == "Left-Right":
            display = mo.hstack(options, justify="space-around")
        else:
            display = mo.vstack(options, align="center", gap=2)

        content = mo.vstack(
            [
                mo.md(f"### Trial {state['trial_idx'] + 1} of {max_trials.value}"),
                #mo.stop(state["trial_idx"] >= max_trials.value),
                mo.center(display),
            ]
        )

    # Stop logic separately - don't use mo.stop() for display control
    if state["trial_idx"] >= max_trials.value:
        content = mo.vstack(
            [
                mo.md("# Experiment Complete"),
                mo.md(
                    f"Final k estimate: {state['last_fit']['k']:.4f}"
                    if state["last_fit"]
                    else ""
                ),
                mo.md("Your data has been saved."),
            ]
        )

    content
    return (state,)


@app.cell
def _(state):
    state
    return


@app.cell
def _(get_state, mo):
    _state = get_state()
    fit = _state["last_fit"]

    if fit:
        stats = mo.hstack(
            [
                mo.stat(label="k (Discount Rate)", value=f"{fit['k']:.4f}"),
                mo.stat(label="Consistency (Beta)", value=f"{fit['beta']:.2f}"),
                mo.stat(label="Model Entropy", value=f"{fit['entropy']:.4e}"),
            ],
            justify="start",
        )

        dashboard = mo.vstack([mo.md("### Real-time Model Estimates"), stats])
    else:
        dashboard = mo.md("_Model will update after the first choice._")

    mo.sidebar(dashboard)
    return


@app.cell
def _(get_state, pl):
    # This cell is just for debugging or viewing history in the notebook
    _history = get_state()["history"]
    if _history:
        history_df = pl.DataFrame(_history)
    else:
        history_df = None

    history_df
    return


if __name__ == "__main__":
    app.run()
