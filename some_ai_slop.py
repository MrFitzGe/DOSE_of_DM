# /// script
# [tool.marimo.display]
# theme = "dark"
# [tool.marimo.runtime]
# auto_instantiate = false
# ///

import marimo

__generated_with = "ai"
app = marimo.App(width="medium")


@app.cell
def _():
    import datetime
    from io import StringIO

    import marimo as mo
    import numpy as np
    import polars as pl
    from loguru import logger

    log_stream = StringIO()
    logger.remove()
    logger.add(log_stream, format="{time} | {level} | {message}")

    def current_time_str():
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return (logger, log_stream, current_time_str, np, pl, mo)


@app.cell
def _(mo):
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
            save_path=mo.ui.text(label="", value="data/", placeholder="e.g., data/"),
            file_name=mo.ui.text(
                label="", value="experiment_results", placeholder="results"
            ),
            file_type=mo.ui.dropdown(options=["csv", "parquet"], value="csv", label=""),
            layout_mode=mo.ui.dropdown(
                options=["Left-Right", "Top-Bottom"],
                value="Left-Right",
                label="",
            ),
            position_mode=mo.ui.dropdown(
                options=["Constant", "Randomized"],
                value="Randomized",
                label="",
            ),
            num_trials=mo.ui.number(start=1, stop=100, step=1, value=10, label=""),
            reward_range=mo.ui.text(
                label="", value="10,50", placeholder="min,max"
            ),
            cost_range=mo.ui.text(
                label="", value="5,20", placeholder="min,max"
            ),
            stimulus_text=mo.ui.text_area(
                label="", value="", placeholder="stim1,stim2,..."
            ),
        )
        .form(submit_button_label="Initialize Experiment")
    )

    setup_form
    return (setup_form,)


@app.cell
def _(setup_form, np, pl, mo, logger, current_time_str):
    get_trial_data, set_trial_data = mo.state(None)
    get_current_trial, set_current_trial = mo.state(0)
    get_choice_log, set_choice_log = mo.state(None)
    get_experiment_started, set_experiment_started = mo.state(False)

    def initialize_experiment():
        form_vals = setup_form.value
        if form_vals is None:
            return None

        n_trials = form_vals["num_trials"]
        reward_parts = [int(x.strip()) for x in form_vals["reward_range"].split(",")]
        cost_parts = [int(x.strip()) for x in form_vals["cost_range"].split(",")]
        reward_min, reward_max = reward_parts[0], reward_parts[1]
        cost_min, cost_max = cost_parts[0], cost_parts[1]
        stimuli_raw = form_vals["stimulus_text"]
        stimuli = (
            [s.strip() for s in stimuli_raw.split(",") if s.strip()]
            if stimuli_raw
            else [None] * n_trials
        )
        if len(stimuli) < n_trials:
            stimuli += [None] * (n_trials - len(stimuli))
        else:
            stimuli = stimuli[:n_trials]

        np.random.seed(None)
        trial_rows = []
        for i in range(n_trials):
            reward1 = np.random.randint(reward_min, reward_max + 1)
            cost1 = np.random.randint(cost_min, cost_max + 1)
            reward2 = np.random.randint(reward_min, reward_max + 1)
            cost2 = np.random.randint(cost_min, cost_max + 1)
            trial_rows.append(
                {
                    "trial": i + 1,
                    "stimulus": stimuli[i],
                    "option1_reward": reward1,
                    "option1_cost": cost1,
                    "option2_reward": reward2,
                    "option2_cost": cost2,
                }
            )

        df = pl.DataFrame(trial_rows)
        logger.info(f"Experiment initialized: {n_trials} trials at {current_time_str()}")
        return df

    if setup_form.value is not None and not get_experiment_started():
        trial_df = initialize_experiment()
        if trial_df is not None:
            set_trial_data(trial_df)
            empty_log = pl.DataFrame(
                schema={
                    "timestamp": pl.Utf8,
                    "trial": pl.Int64,
                    "stimulus": pl.Utf8,
                    "option1_reward": pl.Int64,
                    "option1_cost": pl.Int64,
                    "option2_reward": pl.Int64,
                    "option2_cost": pl.Int64,
                    "choice": pl.Utf8,
                    "rt_ms": pl.Int64,
                    "layout_mode": pl.Utf8,
                    "position_mode": pl.Utf8,
                }
            )
            set_choice_log(empty_log)
            set_experiment_started(True)
            set_current_trial(1)

    return (get_trial_data, set_trial_data, get_current_trial, set_current_trial, 
            get_choice_log, set_choice_log, get_experiment_started, set_experiment_started)


@app.cell
def _(mo, get_experiment_started, get_trial_data, get_current_trial, get_choice_log, 
       setup_form, log_stream, logger, current_time_str):
    show_logs = mo.ui.checkbox(label="Show Logs", value=False)
    show_progress = mo.ui.checkbox(label="Show Progress", value=True)

    with mo.sidebar():
        mo.md("### Experiment Controls")
        mo.ui.batch(
            show_logs=show_logs,
            show_progress=show_progress,
        )
        if show_logs.value and log_stream.getvalue():
            mo.md("#### Logs")
            mo.md(f"<pre style='font-size:0.7em'>{log_stream.getvalue()}</pre>")

    trial_data = get_trial_data()
    current_trial = get_current_trial()
    choice_log = get_choice_log()

    if not get_experiment_started() or trial_data is None:
        mo.md("### Configure and initialize the experiment above")
        return

    total_trials = len(trial_data)
    if current_trial > total_trials:
        mo.md("### Experiment Complete!")
        if choice_log is not None and len(choice_log) > 0:
            mo.md(f"**{len(choice_log)}** choices recorded")
        return

    progress_pct = int((current_trial - 1) / total_trials * 100) if show_progress.value else None
    if show_progress.value:
        mo.md(f"**Progress:** Trial {current_trial}/{total_trials} ({progress_pct}%)")
        mo.progress_bar(value=current_trial - 1, max_value=total_trials)

    return (show_logs, show_progress, trial_data, current_trial, choice_log, total_trials)


@app.cell
def _(mo, trial_data, current_trial, setup_form, np, logger, current_time_str,
       set_current_trial, set_choice_log, choice_log, total_trials):
    if trial_data is None or current_trial > total_trials:
        return

    trial_row = trial_data.row(current_trial - 1, named=True)
    form_vals = setup_form.value
    layout_mode = form_vals["layout_mode"] if form_vals else "Left-Right"
    position_mode = form_vals["position_mode"] if form_vals else "Randomized"

    opt1 = (trial_row["option1_reward"], trial_row["option1_cost"])
    opt2 = (trial_row["option2_reward"], trial_row["option2_cost"])

    if position_mode == "Randomized" and np.random.rand() > 0.5:
        opt1, opt2 = opt2, opt1
        opt1_label, opt2_label = "Option 2", "Option 1"
    else:
        opt1_label, opt2_label = "Option 1", "Option 2"

    trial_start_time = mo._runtime.get_pyodide_mounted_element_time() if hasattr(mo._runtime, 'get_pyodide_mounted_element_time') else None
    get_trial_start, set_trial_start = mo.state(datetime.datetime.now())
    trial_start = get_trial_start()

    opt1_text = f"Reward: {opt1[0]} | Cost: {opt1[1]}"
    opt2_text = f"Reward: {opt2[0]} | Cost: {opt2[1]}"
    stimulus_display = f"**Stimulus:** {trial_row['stimulus']}" if trial_row["stimulus"] else ""

    mo.md(f"### Trial {current_trial}")
    if stimulus_display:
        mo.md(stimulus_display)

    def make_choice(choice_label):
        rt_ms = int((datetime.datetime.now() - trial_start).total_seconds() * 1000)
        actual_choice = "Option 1" if (choice_label == "Option 1" and position_mode == "Constant") or \
                                     (choice_label == opt1_label) else "Option 2"
        
        new_entry = pl.DataFrame(
            [{
                "timestamp": current_time_str(),
                "trial": trial_row["trial"],
                "stimulus": trial_row["stimulus"],
                "option1_reward": trial_row["option1_reward"],
                "option1_cost": trial_row["option1_cost"],
                "option2_reward": trial_row["option2_reward"],
                "option2_cost": trial_row["option2_cost"],
                "choice": actual_choice,
                "rt_ms": rt_ms,
                "layout_mode": layout_mode,
                "position_mode": position_mode,
            }]
        )
        updated_log = pl.concat([choice_log, new_entry]) if choice_log is not None else new_entry
        set_choice_log(updated_log)
        set_current_trial(current_trial + 1)
        set_trial_start(datetime.datetime.now())
        logger.info(f"Trial {trial_row['trial']} | Choice: {actual_choice} | RT: {rt_ms}ms")

    if layout_mode == "Left-Right":
        mo.hstack(
            [
                mo.vstack([
                    mo.md(f"**{opt1_label}**"),
                    mo.md(opt1_text),
                    mo.ui.button(label="Select", on_click=lambda _: make_choice(opt1_label)),
                ]),
                mo.vstack([
                    mo.md(f"**{opt2_label}**"),
                    mo.md(opt2_text),
                    mo.ui.button(label="Select", on_click=lambda _: make_choice(opt2_label)),
                ]),
            ]
        )
    else:
        mo.vstack(
            [
                mo.md(f"**{opt1_label}**: {opt1_text}"),
                mo.ui.button(label=f"Select {opt1_label}", on_click=lambda _: make_choice(opt1_label)),
                mo.md(f"**{opt2_label}**: {opt2_text}"),
                mo.ui.button(label=f"Select {opt2_label}", on_click=lambda _: make_choice(opt2_label)),
            ]
        )

    return


@app.cell
def _(mo, choice_log, trial_data, setup_form, get_experiment_started, logger, current_time_str):
    if not get_experiment_started() or choice_log is None or len(choice_log) == 0:
        return

    total_trials = len(trial_data) if trial_data is not None else 0
    save_button = mo.ui.button("Save Results")

    mo.md("### Results Summary")
    mo.ui.table(choice_log)

    if len(choice_log) == total_trials:
        opt1_count = choice_log.filter(pl.col("choice") == "Option 1").height
        opt2_count = choice_log.filter(pl.col("choice") == "Option 2").height
        mean_rt = choice_log.select(pl.col("rt_ms").mean()).item()
        
        mo.md(f"""
        **Option 1 choices:** {opt1_count}
        **Option 2 choices:** {opt2_count}
        **Mean RT:** {mean_rt:.0f}ms
        """)

    def save_results(_):
        form_vals = setup_form.value
        save_path = form_vals["save_path"]
        file_name = form_vals["file_name"]
        file_type = form_vals["file_type"]
        full_path = f"{save_path}{file_name}.{file_type}"
        
        if file_type == "csv":
            choice_log.write_csv(full_path)
        else:
            choice_log.write_parquet(full_path)
        logger.info(f"Results saved to {full_path} at {current_time_str()}")

    save_button.on_click(save_results)
    save_button
    return


if __name__ == "__main__":
    app.run()