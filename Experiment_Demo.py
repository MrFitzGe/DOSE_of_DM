# /// script
# [tool.marimo.display]
# theme = "dark"
# [tool.marimo.runtime]
# auto_instantiate = false
# ///

import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")

with app.setup:
    import datetime

    # Set up logger to log to a string (for in-notebook display)
    from io import StringIO

    import marimo as mo
    import numpy as np
    import polars as pl
    from loguru import logger


@app.cell
def _():
    log_stream = StringIO()
    logger.remove()
    logger.add(log_stream, format="{time} | {level} | {message}")

    # Helper function to get current timestamp string
    def current_time_str():
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    get_choices, set_choices = mo.state([])  # list of dicts, one per recorded choice
    get_trial, set_trial = mo.state(1)       # current trial index (1-based)
    return (
        current_time_str,
        get_choices,
        get_trial,
        log_stream,
        set_choices,
        set_trial,
    )


@app.cell
def _():
    # Experiment setup form
    num_trials = mo.ui.number(value=10, label="Number of Trials")
    reward_range = mo.ui.range_slider(1, 100, value=(10, 50), label="Reward Range")
    cost_range = mo.ui.range_slider(1, 50, value=(5, 20), label="Cost Range")
    stimulus_text = mo.ui.text_area(
        value="", label="Optional Stimulus Text (comma-separated for each trial)"
    )

    presentation_style = mo.ui.dropdown(
        options=["Horizontal", "Vertical"],
        value="Horizontal",
        label="Presentation Style",
    )

    save_location = mo.ui.text(
        value="experiment_log.csv", label="Output CSV Filename"
    )

    setup_form = (
        mo.md(
            r"""
                **Save Directory**
                {save_path}

                **Filename**
                {file_name}

                **Output File Format**
                {file_type}

                **Stimuli Positions**
                {layout_mode}

                **Display Layout across All Trials**
                {position_mode}

                **Number of Trials**
                {num_trials}

                **Reward Range**
                {reward_range}

                **Cost Range**
                {cost_range}

                **Stimulus Text (comma-separated)**
                {stimulus_text}
                """
        )
        .batch(
            save_path=mo.ui.text(
                label="", value="data/", placeholder="e.g., data/"
            ),
            file_name=mo.ui.text(
                label="", value="experiment_results", placeholder="results"
            ),
            file_type=mo.ui.dropdown(
                options=["csv", "parquet"], value="csv", label=""
            ),
            layout_mode=presentation_style,
            position_mode=mo.ui.dropdown(
                options=["Constant", "Randomized"],
                value="Randomized",
                label="",
            ),
            num_trials=num_trials,
            reward_range=reward_range,
            cost_range=cost_range,
            stimulus_text=stimulus_text,
        )
        .form(submit_button_label="Initialize Experiment")
    )

    mo.accordion({"Experiment Settings": setup_form})
    return (setup_form,)


@app.cell
def parse_exp_settings(setup_form):
    mo.stop(not setup_form.value, mo.md("Initialize experiment in 'Experiment Settings' above."))
    # Parse stimulus text and generate choice sets
    n_trials = setup_form.value['num_trials']
    reward_min, reward_max = setup_form.value['reward_range']
    cost_min, cost_max = setup_form.value['cost_range'][0], setup_form.value['cost_range'][1]
    stimuli = (
         [setup_form.value['stimulus_text']] * n_trials
        if setup_form.value['stimulus_text']
        else [None] * n_trials
    )

    # Ensure stimuli list matches number of trials
    if len(stimuli) < n_trials:
        stimuli += [None] * (n_trials - len(stimuli))
    else:
        stimuli = stimuli[:n_trials]

    # Generate random rewards and costs for two options per trial
    np.random.seed(42)  # For reproducibility
    option_sets = []
    for i in range(n_trials):
        reward1 = np.random.randint(reward_min, reward_max + 1)
        cost1 = np.random.randint(cost_min, cost_max + 1)
        reward2 = np.random.randint(reward_min, reward_max + 1)
        cost2 = np.random.randint(cost_min, cost_max + 1)
        option_sets.append(
            {
                "trial": i + 1,
                "stimulus": stimuli[i],
                "option1_reward": reward1,
                "option1_cost": cost1,
                "option2_reward": reward2,
                "option2_cost": cost2,
            }
        )

    options_df = pl.DataFrame(option_sets)
    options_df
    return n_trials, options_df


@app.cell
def _(setup_form):
    mo.stop(not setup_form.value, mo.md("Initialize experiment in 'Experiment Settings' to access sidebar data."))

    mo.sidebar([
        mo.md('Experiment Settings'), 
        setup_form.value
    ])
    return


@app.cell
def _(get_trial, options_df, setup_form):
    # Display current trial's options and stimulus
    trial_idx = get_trial() - 1
    trial_row = options_df.row(trial_idx, named=True)

    stimulus_display = mo.md(
        f"**Stimulus:** {trial_row['stimulus']}" if trial_row["stimulus"] else ""
    )
    option1_text = (
        f"Reward: **{trial_row['option1_reward']}** | Cost: **{trial_row['option1_cost']}**"
    )
    option2_text = (
        f"Reward: **{trial_row['option2_reward']} ||** Cost: **{trial_row['option2_cost']}**"
    )

    # Buttons for forced choice — clear spatial separation
    btn_opt1 = mo.ui.button(label=option1_text, value=0, on_click= lambda count: count + 1, kind="warn")
    btn_opt2 = mo.ui.button(label=option2_text, value=0, on_click= lambda count: count + 1, kind="warn")

    if setup_form.value['layout_mode'] == "Horizontal":
        stim_presentation = mo.hstack([btn_opt1, btn_opt2], justify="space-around") 
    else:
        stim_presentation = mo.vstack([btn_opt1, btn_opt2], align="center")

    stim_presentation
    return btn_opt1, btn_opt2, trial_row


@app.cell
def _(
    btn_opt1,
    btn_opt2,
    get_choices,
    get_trial,
    n_trials,
    options_df,
    set_choices,
    set_trial,
):
    # Record choice reactively when a button is clicked
    # Buttons increment their value on each click, default is 0
    chose_1 = btn_opt1.value > 0
    chose_2 = btn_opt2.value > 0

    if chose_1 or chose_2:
        choice = "Option 1" if chose_1 else "Option 2"
        trial = options_df.row(get_trial() - 1, named=True)
        entry = {**trial, "choice": choice, "timestamp": datetime.datetime.now().isoformat()}
        set_choices(get_choices() + [entry])
        set_trial(min(get_trial() + 1, n_trials))  # advance trial
    return chose_1, chose_2


@app.cell
def _(chose_1, chose_2, current_time_str, setup_form, trial_row):
    # Log and record choices
    # Use a DataFrame to store all choices
    if "_choice_log" not in locals():
        _choice_log = pl.DataFrame(
            schema={
                "timestamp": pl.Utf8,
                "trial": pl.Int64,
                "stimulus": pl.Utf8,
                "option1_reward": pl.Int64,
                "option1_cost": pl.Int64,
                "option2_reward": pl.Int64,
                "option2_cost": pl.Int64,
                "choice": pl.Utf8,
                "presentation_style": pl.Utf8,
            }
        )

    if chose_1 or chose_2:
        current_choice = "Option 1" if chose_1 else "Option 2"    
        # Check if already logged for this trial
        already_logged = (
            _choice_log.filter(pl.col("trial") == trial_row["trial"])
            .filter(pl.col("choice").is_not_null())
            .select(pl.count()).item() > 0
        )
    
        if not already_logged:
            log_entry = pl.DataFrame({
                "timestamp": [current_time_str()],
                "trial": [trial_row["trial"]],
                "stimulus": [trial_row["stimulus"]],
                "option1_reward": [trial_row["option1_reward"]],
                "option1_cost": [trial_row["option1_cost"]],
                "option2_reward": [trial_row["option2_reward"]],
                "option2_cost": [trial_row["option2_cost"]],
                "choice": [current_choice],
                "presentation_style": [setup_form.value['layout_mode']],
            })
        
            _choice_log = pl.concat([_choice_log, log_entry], how="vertical")
            logger.info(
                f"Trial {trial_row['trial']} | Choice: {current_choice} | "
                f"Stimulus: {trial_row['stimulus']} | "
                f"Opt1: ({trial_row['option1_reward']},{trial_row['option1_cost']}) | "
                f"Opt2: ({trial_row['option2_reward']},{trial_row['option2_cost']}) | "
                f"Style: {setup_form.value['layout_mode']}"
            )

    # Update the global choices list
    choice_log = _choice_log
    choice_log
    return (choice_log,)


@app.cell
def _(log_stream):
    # Display loguru logs
    mo.md(f"### Experiment Log")
    mo.md(f"<pre>{log_stream.getvalue()}</pre>")
    return


@app.cell
def _():
    # Save log to CSV when all trials are completed
    save_btn = mo.ui.button(label="💾 Save Results", kind="info")
    save_btn
    return (save_btn,)


@app.cell
def _(choice_log, get_choices, n_trials, save_btn, setup_form):
    if save_btn.value > 0 and len(get_choices()) == n_trials:
        path = setup_form.value["save_path"] + setup_form.value["file_name"]
        if setup_form.value["file_type"] == "csv":
            choice_log.write_csv(path + ".csv")
        else:
            choice_log.write_parquet(path + ".parquet")
    return


if __name__ == "__main__":
    app.run()
