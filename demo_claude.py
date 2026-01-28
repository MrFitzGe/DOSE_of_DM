import marimo

__generated_with = "0.19.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import random
    import time
    from datetime import datetime
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import polars as pl
    from ax.api.client import Client as AxClient
    from loguru import logger

    from fit_model import fit_hyperbolic_discount
    return (
        Path,
        datetime,
        fit_hyperbolic_discount,
        logger,
        mo,
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
def _(datetime, mo):
    # Configuration UI
    participant_id = mo.ui.text(
        label="Participant ID", value=f"P{datetime.now():%Y%m%d_%H%M%S}", 
    )

    max_trials = mo.ui.number(start=5, stop=100, step=1, value=20, label="Max Trials")

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

    setup_form = mo.md(
        f"""
        {participant_id}
        {mo.hstack([max_trials, layout_mode, position_mode])}
        """
    )
    return layout_mode, max_trials, participant_id, position_mode, setup_form


@app.cell
def _(mo, setup_form):
    mo.accordion({"Experiment Settings": setup_form})
    return


@app.cell
def _(Path, participant_id):
    # File paths (reactive to participant ID)
    DATA_DIR = Path("data")
    DATA_DIR.mkdir(exist_ok=True)

    TRIAL_LOG = DATA_DIR / f"{participant_id.value}_trials.jsonl"
    LOGGER_ID = f"experiment_{participant_id.value}"

    return LOGGER_ID, TRIAL_LOG


@app.cell
def _():
    # Default burn-in trials (static config)
    BURN_IN_TRIALS = [
        {"amount_1": 5, "cost_1": 0, "amount_2": 12, "cost_2": 25},
        {"amount_1": 1, "cost_1": 0, "amount_2": 15, "cost_2": 55},
        {"amount_1": 7, "cost_1": 0, "amount_2": 100, "cost_2": 40},
        {"amount_1": 10, "cost_1": 0, "amount_2": 20, "cost_2": 65},
        {"amount_1": 5, "cost_1": 0, "amount_2": 20, "cost_2": 15},
        {"amount_1": 10, "cost_1": 0, "amount_2": 39, "cost_2": 72},
    ]
    return (BURN_IN_TRIALS,)


@app.cell
def _(LOGGER_ID, TRIAL_LOG, logger):
    # Configure loguru for structured logging
    logger.remove()  # Remove default handler
    logger.add(
        TRIAL_LOG,
        format="{message}",
        serialize=True,  # JSON output
        mode="a",
        level="INFO",
        filter=lambda record: record["extra"].get("log_id") == LOGGER_ID
    )
    logger.add(lambda msg: None, level="WARNING")  # Suppress console for INFO

    trial_logger = logger.bind(log_id=LOGGER_ID)
    return (trial_logger,)


@app.cell
def _(TRIAL_LOG, pl):
    # Reactive data loader - reads trial log
    def load_trials():
        """Load all logged trials as a polars DataFrame"""
        if not TRIAL_LOG.exists():
            return pl.DataFrame()

        try:
            # Read JSONL with polars
            df = pl.read_ndjson(TRIAL_LOG)

            # Extract the nested 'record' field if using loguru serialize
            if "record" in df.columns:
                # Parse the message field which contains our trial data
                df = df.select(pl.col("text").str.json_extract())
                df = df.unnest("text")

            return df.sort("trial_idx") if "trial_idx" in df.columns else df
        except Exception as e:
            print(f"Error loading trials: {e}")
            return pl.DataFrame()

    trials_df = load_trials()
    return (trials_df,)


@app.cell
def _(trials_df):
    # Derive current trial index from log
    current_trial_idx = len(trials_df) if trials_df is not None else 0
    return (current_trial_idx,)


@app.cell
def _(
    BURN_IN_TRIALS,
    current_trial_idx,
    fit_hyperbolic_discount,
    layout_mode,
    max_trials,
    position_mode,
    random,
    time,
    trial_logger,
    trials_df,
):
    # Generate next trial stimuli (pure function)
    def get_next_stimuli(trial_idx):
        """Determine stimuli for the given trial index"""
        if trial_idx < len(BURN_IN_TRIALS):
            return BURN_IN_TRIALS[trial_idx]
        else:
            # TODO: Integrate Ax here - for now, random from bounds
            return {
                "amount_1": random.randint(1, 10),
                "cost_1": random.randint(0, 5),
                "amount_2": random.randint(10, 100),
                "cost_2": random.randint(5, 100),
            }

    def determine_layout():
        """Determine layout for current trial"""
        mode = layout_mode.value
        if mode == "Mixed":
            return random.choice(["Left-Right", "Top-Bottom"])
        return mode

    def determine_position():
        """Determine SS position (0=left/top, 1=right/bottom)"""
        if position_mode.value == "Randomized":
            return random.randint(0, 1)
        return 0

    def log_trial_choice(trial_idx, stimuli, layout, ss_pos, choice, rt):
        """Log a single trial choice immediately"""
        ss_pos_label = (
            ("left" if ss_pos == 0 else "right")
            if layout == "Left-Right"
            else ("top" if ss_pos == 0 else "bottom")
        )

        trial_data = {
            "trial_idx": trial_idx,
            "amount_ss": stimuli["amount_1"],
            "cost_ss": stimuli["cost_1"],
            "amount_ll": stimuli["amount_2"],
            "cost_ll": stimuli["cost_2"],
            "ss_position": ss_pos_label,
            "layout": layout,
            "choice": choice,  # 0 for SS, 1 for LL
            "rt": rt,
            "timestamp": time.time(),
        }

        trial_logger.info("trial_completed", **trial_data)
        return trial_data

    def fit_current_model():
        """Fit model to all trials so far"""
        if trials_df is None or len(trials_df) == 0:
            return None

        try:
            df = trials_df
            fit_results = fit_hyperbolic_discount(
                a1=df["amount_ss"].to_list(),
                c1=df["cost_ss"].to_list(),
                a2=df["amount_ll"].to_list(),
                c2=df["cost_ll"].to_list(),
                choices=df["choice"].to_list(),
            )
            return fit_results
        except Exception as e:
            print(f"Fit error: {e}")
            return None

    # Check if experiment is complete
    experiment_complete = current_trial_idx >= max_trials.value

    return (
        determine_layout,
        determine_position,
        experiment_complete,
        fit_current_model,
        get_next_stimuli,
        log_trial_choice,
    )


@app.cell
def _(
    current_trial_idx,
    determine_layout,
    determine_position,
    experiment_complete,
    get_next_stimuli,
    log_trial_choice,
    max_trials,
    mo,
    time,
):
    # Trial state (minimal - just what we need for current trial)
    get_trial_state, set_trial_state = mo.state({
        "stimuli": get_next_stimuli(current_trial_idx) if not experiment_complete else None,
        "layout": determine_layout() if not experiment_complete else None,
        "ss_pos": determine_position() if not experiment_complete else None,
        "start_time": None,
    })

    def start_trial():
        """Initialize timing for current trial"""
        set_trial_state(lambda s: {**s, "start_time": time.time()})

    def handle_choice(choice_val):
        """Handle choice and log immediately"""
        state = get_trial_state()
        if state["start_time"] is None:
            return  # Trial hasn't started yet

        rt = time.time() - state["start_time"]

        # Log immediately
        log_trial_choice(
            trial_idx=current_trial_idx,
            stimuli=state["stimuli"],
            layout=state["layout"],
            ss_pos=state["ss_pos"],
            choice=choice_val,
            rt=rt,
        )

        # Prepare next trial
        next_idx = current_trial_idx + 1
        if next_idx < max_trials.value:
            set_trial_state({
                "stimuli": get_next_stimuli(next_idx),
                "layout": determine_layout(),
                "ss_pos": determine_position(),
                "start_time": time.time(),
            })

    return get_trial_state, handle_choice, start_trial


@app.cell
def _(
    current_trial_idx,
    experiment_complete,
    get_trial_state,
    handle_choice,
    max_trials,
    mo,
    start_trial,
):
    # Main UI
    def render_option(amount, cost, is_ss):
        return mo.vstack(
            [
                mo.md(f"## ${amount}"),
                mo.md(f"**Wait:** {cost} days"),
                mo.ui.button(
                    label="Choose This",
                    on_click=lambda _: handle_choice(0 if is_ss else 1),
                    kind="success" if is_ss else "warn",
                ),
            ],
            align="center",
        )

    if current_trial_idx == 0:
        # Start screen
        start_btn = mo.ui.button(
            label="🚀 Start Experiment", 
            on_click=lambda _: start_trial(),
            kind="success"
        )
        trial_ui = mo.center(
            mo.vstack([
                mo.md("# Ready to begin?"),
                mo.md(f"You will complete **{max_trials.value} trials**."),
                start_btn
            ])
        )
    elif experiment_complete:
        # Completion screen
        trial_ui = mo.center(
            mo.vstack([
                mo.md("# ✅ Experiment Complete!"),
                mo.md(f"Completed {current_trial_idx} trials."),
                mo.md("Your data has been saved."),
            ])
        )
    else:
        # Active trial
        state = get_trial_state()
        stim = state["stimuli"]

        ss_ui = render_option(stim["amount_1"], stim["cost_1"], True)
        ll_ui = render_option(stim["amount_2"], stim["cost_2"], False)

        # Order based on ss_pos
        options = [ss_ui, ll_ui] if state["ss_pos"] == 0 else [ll_ui, ss_ui]

        if state["layout"] == "Left-Right":
            display = mo.hstack(options, justify="space-around", widths=[1, 1])
        else:
            display = mo.vstack(options, align="stretch", gap=3)

        trial_ui = mo.vstack([
            mo.md(f"### Trial {current_trial_idx + 1} of {max_trials.value}"),
            mo.center(display),
        ])

    trial_ui
    return


@app.cell
def _(current_trial_idx, fit_current_model, mo, trials_df):
    # Real-time model dashboard (reactive to trials_df)
    if current_trial_idx > 0:
        fit = fit_current_model()

        if fit and fit.get("success"):
            stats = mo.hstack([
                mo.stat(label="k (Discount Rate)", value=f"{fit['k']:.4f}"),
                mo.stat(label="β (Consistency)", value=f"{fit['beta']:.2f}"),
                mo.stat(label="Trials", value=str(len(trials_df))),
            ])

            dashboard = mo.vstack([
                mo.md("### 📊 Real-time Estimates"),
                stats,
                mo.md(f"_AIC: {fit.get('AIC', 0):.2f}_")
            ])
        else:
            dashboard = mo.md("_Fitting model..._")
    else:
        dashboard = mo.md("_Model will update after choices._")

    mo.sidebar(dashboard)
    return


@app.cell
def _(mo, trials_df):
    # Debug view of trial history
    if trials_df is not None and len(trials_df) > 0:
        mo.ui.table(trials_df, selection=None)
    return


if __name__ == "__main__":
    app.run()
