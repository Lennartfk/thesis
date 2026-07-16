import matplotlib
matplotlib.use("Agg")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--out-path", default="data/results/Final_Plots_All/Subject_Overlays/prediction_timeline_deepdive.pdf")
parser.add_argument("--no-title", action="store_true")
args = parser.parse_args()

BASE = Path("data/results/experiments/23_Subjects_Stable")
OUT_DIR = Path(args.out_path).parent

folders = {
    "EEGNet": BASE / "final_baseline_eegnet",
    "EEGNet + EA": BASE / "final_eval_ea_chronological_sweep",
    "EEGNet + AdaBN": BASE / "final_eval_adabn_chronological_sweep",
}

print("Generating prediction timeline deepdive...")
OUT_DIR.mkdir(parents=True, exist_ok=True)

#########################################################
# LOAD PREDICTIONS ONCE
#########################################################

predictions = {}
for method, folder in folders.items():
    df = pd.read_csv(folder / "predictions.csv")
    if "chronological_minutes" in df.columns:
        df = df[df["chronological_minutes"] == 10.0]
    if "test_subject_id" in df.columns:
        df = df.rename(columns={"test_subject_id": "subject_id"})
    if "y_true" in df.columns and "target" not in df.columns:
        df = df.rename(columns={"y_true": "target"})
    predictions[method] = df

#########################################################
# SUBJECT SELECTION
#########################################################

baseline = pd.read_csv(
    folders["EEGNet"] /
    "fold_metrics.csv"
)

ranking = baseline.sort_values(
    "balanced_accuracy"
)

subjects = [

    int(
        ranking.iloc[0]
        ["test_subject_id"]
    ),

    int(
        ranking.iloc[
            len(ranking)//2
        ]
        ["test_subject_id"]
    ),

    int(
        ranking.iloc[-1]
        ["test_subject_id"]
    )

]

titles = [
    "Worst",
    "Median",
    "Best"
]

#########################################################
# STYLE
#########################################################

sns.set_theme(
    style="ticks",
    context="paper",
    font_scale=1.15
)

palette = {

    "EEGNet":
        "#4C72B0",

    "EEGNet + EA":
        "#55A868",

    "EEGNet + AdaBN":
        "#C44E52"

}

error_offsets = {

    "EEGNet":0,

    "EEGNet + EA":1,

    "EEGNet + AdaBN":2

}

#########################################################
# FIGURE
#########################################################

fig, axes = plt.subplots(

    9,
    1,

    figsize=(14,10),

    sharex=False,

    gridspec_kw={

        "height_ratios":[

            1.0,
            0.8,
            1.3,

            1.0,
            0.8,
            1.3,

            1.0,
            0.8,
            1.3

        ]

    }

)

row = 0

for subject, title in zip(
    subjects,
    titles
):

    ###################################################
    # Ground truth
    ###################################################

    gt = predictions[
        "EEGNet"
    ]

    gt = gt[
        gt["subject_id"]
        ==
        subject
    ]

    gt = gt.sort_values(
        "epoch_index"
    )

    x = gt[
        "epoch_index"
    ]

    transitions = (

        gt[
            "target"
        ]

        .diff()

        .fillna(0)

        != 0

    )

    transition_points = gt.loc[
        transitions,
        "epoch_index"
    ]

    ###################################################
    # TOP PANEL
    ###################################################

    ax = axes[row]

    ax.step(

        x,

        gt["target"],

        where="post",

        color="black",

        linewidth=2.8

    )

    for t in transition_points:

        ax.axvline(

            t,

            color="gray",

            linestyle=":",

            alpha=0.18,

            linewidth=1

        )

    if not args.no_title:
        ax.set_title(
            f"{title} Subject ({subject})",
            fontsize=11,
            pad=5
        )

    ax.set_yticks(
        [0,1]
    )

    ax.set_yticklabels(
        [
            "Alert",
            "Drowsy"
        ]
    )

    ax.grid(
        alpha=0.12
    )

    ###################################################
    # ERROR PANEL
    ###################################################

    ax = axes[row+1]

    for method in folders:

        subj = predictions[
            method
        ]

        subj = subj[
            subj["subject_id"]
            ==
            subject
        ]

        subj = subj.sort_values(
            "epoch_index"
        )

        mistakes = (

            subj["y_pred"]

            !=

            subj["target"]

        )

        pos = subj.loc[
            mistakes,
            "epoch_index"
        ]

        offset = error_offsets[
            method
        ]

        ax.vlines(

            pos,

            offset-0.28,

            offset+0.28,

            color=palette[
                method
            ],

            linewidth=2,

            alpha=0.85

        )

    for t in transition_points:

        ax.axvline(

            t,

            color="gray",

            linestyle=":",

            alpha=0.12

        )

    ax.set_yticks(
        [0,1,2]
    )

    ax.set_yticklabels(

        [

            "EEGNet",

            "EA",

            "AdaBN"

        ]

    )

    ax.set_ylabel(
        "Errors"
    )

    ###################################################
    # CONFIDENCE PANEL
    ###################################################

    ax = axes[row+2]

    for method in folders:

        subj = predictions[
            method
        ]

        subj = subj[
            subj["subject_id"]
            ==
            subject
        ]

        subj = subj.sort_values(
            "epoch_index"
        )

        smooth = (

            subj[
                "y_score"
            ]

            .rolling(

                15,

                center=True,

                min_periods=1

            )

            .mean()

        )

        ax.plot(

            subj[
                "epoch_index"
            ],

            smooth,

            color=palette[
                method
            ],

            linewidth=2,

            alpha=0.9,

            label=method

        )

    for t in transition_points:

        ax.axvline(

            t,

            color="gray",

            linestyle=":",

            alpha=0.12

        )

    ax.axhline(

        0.5,

        color="gray",

        linestyle="--",

        alpha=0.5,

        linewidth=1

    )

    ax.set_ylim(
        -0.02,
        1.02
    )

    ax.set_ylabel(
        "P(Drowsy)"
    )

    ax.grid(
        alpha=0.12
    )

    sns.despine(
        ax=ax
    )

    row += 3

#########################################################
# LEGEND
#########################################################

handles = [

    plt.Line2D(
        [0],
        [0],
        color=palette[k],
        lw=2.5
    )

    for k in palette

]

fig.legend(

    handles,

    list(
        palette.keys()
    ),

    ncol=3,

    loc="upper center",

    frameon=False,

    bbox_to_anchor=(
        0.5,
        1.01
    )

)

axes[-1].set_xlabel(
    "EEG Window"
)

plt.subplots_adjust(

    top=0.95,

    hspace=0.65

)

plt.savefig(
    args.out_path,
    format="pdf",
    bbox_inches="tight"
)
plt.close()
print("Timeline deepdive generated.")
