import marimo

__generated_with = "0.19.6"
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
def _():
    # Default burn-in trials
    default_burn_in_trials = [
        {"amount_1": 5, "cost_1": 0, "amount_2": 12, "cost_2": 25},
        {"amount_1": 1, "cost_1": 0, "amount_2": 15, "cost_2": 55},
        {"amount_1": 7, "cost_1": 0, "amount_2": 100, "cost_2": 40},
        {"amount_1": 10, "cost_1": 0, "amount_2": 20, "cost_2": 65},
        {"amount_1": 5, "cost_1": 0, "amount_2": 20, "cost_2": 15},
        {"amount_1": 10, "cost_1": 0, "amount_2": 39, "cost_2": 72},
    ]
    return (default_burn_in_trials,)


@app.cell
def _(mo):
    mo.md(r"""
    # Adaptive Temporal Discounting Experiment
    """)
    return


@app.cell
def _(mo, setup_form):
    mo.md(r"## Experiment Setup")
    mo.accordion({"Experiment Settings": setup_form})
    return


@app.cell
def _(mo):
    # Configuration UI - setup form
    setup_form = (
        mo.md(
            r"""
            **Save Directory**
            {save_path}

            **Filename**
            {file_name}

            **File Format**
            {file_type}

            **Display Layout**
            {layout_mode}

            **Option Position (Sm. vs Lg.)**
            {position_mode}

            **Max. Generated Trials**
            {max_gen_trials}
            """
        )
        .batch(
            save_path=mo.ui.text(label="", value="data/", placeholder="e.g., data/"),
            file_name=mo.ui.text(
                label="", value="experiment_results", placeholder="results"
            ),
            file_type=mo.ui.dropdown(options=["csv", "parquet"], value="csv", label=""),
            layout_mode=mo.ui.dropdown(
                options=["Left-Right", "Top-Bottom", "Mixed"],
                value="Left-Right",
                label="",
            ),
            position_mode=mo.ui.dropdown(
                options=["Constant", "Randomized"],
                value="Randomized",
                label="",
            ),
            max_gen_trials=mo.ui.number(start=5, stop=100, step=1, value=20, label=""),
        )
        .form(submit_button_label="Save Settings")
    )
    return (setup_form,)


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
def _(default_burn_in_trials, initialize_ax_client):
    def build_presentation_plan(max_gen_trials, layout_mode, position_mode, rng):
        plan = []
        for _ in range(max_gen_trials):
            layout = layout_mode
            if layout == "Mixed":
                layout = rng.choice(["Left-Right", "Top-Bottom"])

            sm_pos = 0
            if position_mode == "Randomized":
                sm_pos = rng.randint(0, 1)

            plan.append({"layout": layout, "sm_pos": sm_pos})

        return plan

    session = {
        "started": False,
        "trial_idx": 0,
        "history": [],
        "current_trial": None,
        "start_time": None,
        "ax_client": initialize_ax_client(),
        "burn_in": default_burn_in_trials,
        "finished": False,
        "last_fit": None,
        "presentation_plan": [],
        "start_clicks": 0,
        "choice_clicks": {"sm": 0, "lg": 0},
    }
    return build_presentation_plan, session


@app.cell
def _(
    build_presentation_plan,
    file_name,
    file_type,
    fit_hyperbolic_discount,
    initialize_ax_client,
    layout_mode,
    logger,
    max_gen_trials,
    mo,
    os,
    pl,
    position_mode,
    random,
    save_path,
    session,
    time,
):
    def start_experiment():
        session["ax_client"] = initialize_ax_client()
        session["history"] = []
        session["trial_idx"] = 0
        session["finished"] = False
        session["last_fit"] = None
        session["presentation_plan"] = build_presentation_plan(
            max_gen_trials.value,
            layout_mode.value,
            position_mode.value,
            random,
        )

        # Setup logging
        full_path = os.path.join(save_path.value, f"{file_name.value}.log")
        os.makedirs(save_path.value, exist_ok=True)
        logger.add(full_path, rotation="1 MB")
        logger.info("Experiment Started")

        # Prepare first trial
        prepare_trial(0)
        session["started"] = True

    def prepare_trial(idx):
        # Determine Stimuli
        if idx < len(session["burn_in"]):
            stimuli = session["burn_in"][idx]
        else:
            # Ax adaptive trial
            trial_params, _ = session["ax_client"].get_next_trials(max_gen_trials=1)
            # Ax returns a dict of {trial_index: parameters}
            stimuli = list(trial_params.values())[0]

        layout_spec = session["presentation_plan"][idx]
        session["current_trial"] = {
            "stimuli": stimuli,
            "layout": layout_spec["layout"],
            "sm_pos": layout_spec["sm_pos"],
        }
        session["start_time"] = time.time()

    def handle_choice(choice_val):
        if not session["started"] or session["finished"]:
            return

        # choice_val: 0 if SS was clicked, 1 if LL was clicked
        end_time = time.time()
        rt = end_time - session["start_time"]

        trial = session["current_trial"]
        stim = trial["stimuli"]

        # Record data
        # Determine position label for SS
        sm_pos_label = ""
        if trial["layout"] == "Left-Right":
            sm_pos_label = "left" if trial["sm_pos"] == 0 else "right"
        else:
            sm_pos_label = "top" if trial["sm_pos"] == 0 else "bottom"

        trial_data = {
            "trial_idx": session["trial_idx"],
            "amount_ss": stim["amount_1"],
            "cost_ss": stim["cost_1"],
            "amount_ll": stim["amount_2"],
            "cost_ll": stim["cost_2"],
            "sm_position": sm_pos_label,
            "layout": trial["layout"],
            "choice": choice_val,  # 0 for SS, 1 for LL
            "rt": rt,
        }

        logger.info(f"Trial {session['trial_idx']} completed: {trial_data}")

        session["history"].append(trial_data)
        fit_results = None

        # Check if we have completed burn-in trials
        # Note: session["trial_idx"] is the index of the current trial (0-based)
        # We need to check if we've completed all burn-in trials (idx >= len(burn_in))
        if session["trial_idx"] >= len(session["burn_in"]):
            # Fit model using all accumulated data (including burn-in)
            # Prepare lists for fitting
            a1 = [d["amount_ss"] for d in session["history"]]
            c1 = [d["cost_ss"] for d in session["history"]]
            a2 = [d["amount_ll"] for d in session["history"]]
            c2 = [d["cost_ll"] for d in session["history"]]
            choices = [d["choice"] for d in session["history"]]

            fit_results = fit_hyperbolic_discount(a1, c1, a2, c2, choices)

            # Attach to Ax (only for post-burn-in trials)
            # Note: Ax expects the parameters used for the trial
            parameters = {
                "amount_1": stim["amount_1"],
                "cost_1": stim["cost_1"],
                "amount_2": stim["amount_2"],
                "cost_2": stim["cost_2"],
            }
            trial_index = session["ax_client"].attach_trial(parameters=parameters)
            session["ax_client"].complete_trial(
                trial_index=trial_index, raw_data=fit_results
            )

        next_idx = session["trial_idx"] + 1
        is_finished = next_idx >= max_gen_trials.value

        session["trial_idx"] = next_idx
        session["last_fit"] = fit_results
        session["finished"] = is_finished

        if not is_finished:
            prepare_trial(next_idx)
        else:
            save_final_data(session["history"])

    def save_final_data(history):
        df = pl.DataFrame(history)
        path = os.path.join(save_path.value, f"{file_name.value}.{file_type.value}")
        if file_type.value == "csv":
            df.write_csv(path)
        else:
            df.write_parquet(path)
        logger.info(f"Data saved to {path}")

    start_button = mo.ui.button(label="Start Experiment")
    sm_button = mo.ui.button(label="Select", full_width=False)
    lg_button = mo.ui.button(label="Select", full_width=False)
    return handle_choice, lg_button, sm_button, start_button, start_experiment


@app.cell
def _(
    handle_choice,
    lg_button,
    mo,
    session,
    setup_form,
    sm_button,
    start_button,
    start_experiment,
):
    state = session

    def render_option(amount, cost, button):
        # Context dependent labels
        return mo.vstack(
            [
                mo.md(f"### Get ${amount}"),
                mo.md(f"### Wait: {cost}"),
                button,
            ],
            align="center",
        )

    if not state["started"]:
        if start_button.value != state["start_clicks"]:
            state["start_clicks"] = start_button.value
            start_experiment()
            state["choice_clicks"]["sm"] = sm_button.value
            state["choice_clicks"]["lg"] = lg_button.value
            state = session

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
        if sm_button.value != state["choice_clicks"]["sm"]:
            state["choice_clicks"]["sm"] = sm_button.value
            handle_choice(0)
            state = session

        if lg_button.value != state["choice_clicks"]["lg"]:
            state["choice_clicks"]["lg"] = lg_button.value
            handle_choice(1)
            state = session

        if state["finished"]:
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
            stim = state["current_trial"]["stimuli"]
            sm_ui = render_option(stim["amount_1"], stim["cost_1"], sm_button)
            lg_ui = render_option(stim["amount_2"], stim["cost_2"], lg_button)

            # Order based on sm_pos
            if state["current_trial"]["sm_pos"] == 0:
                options = [sm_ui, lg_ui]
            else:
                options = [lg_ui, sm_ui]

            if state["current_trial"]["layout"] == "Left-Right":
                display = mo.hstack(options, justify="space-around")
            else:
                display = mo.vstack(options, align="center", gap=2)

            content = mo.vstack(
                [
                    mo.md(
                        f"### Trial {state['trial_idx'] + 1} of {setup_form.value['max_gen_trials']}"
                    ),
                    mo.center(display),
                ]
            )

    content
    return


@app.cell
def _(mo, session, setup_form):
    fit = session["last_fit"]

    # Initialize an empty list for dashboard components
    dashboard_components = []

    # Add setup parameters if they exist
    if setup_form.value is not None:
        dashboard_components.extend([
            mo.md("### Experiment Setup Parameters"),
            setup_form.value,
        ])

    # Add model estimates if they exist
    if fit is not None:
        stats = mo.hstack(
            [
                mo.stat(label="k (Discount Rate)", value=f"{fit['k']:.4f}"),
                mo.stat(label="Consistency (Beta)", value=f"{fit['beta']:.2f}"),
                mo.stat(label="Model Entropy", value=f"{fit['entropy']:.4e}"),
            ],
            justify="start",
        )
        dashboard_components.extend([
            mo.md("### Real-time Model Estimates"),
            stats,
        ])

    # Build the final dashboard
    if dashboard_components:
        dashboard = mo.vstack(dashboard_components)
    else:
        dashboard = mo.md("_Model will update after the first choice._")

    mo.sidebar(dashboard)
    return


if __name__ == "__main__":
    app.run()
