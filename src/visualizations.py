import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import test
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay

matplotlib.use("TkAgg")



def plot_formation_pass_percentage(df):
    pass_pct = df.groupby("offenseFormation")["isDropback"].agg(["count", "mean"])
    pass_pct.columns = ["total_plays", "pass_percentage"]

    pass_pct = pass_pct.sort_values("pass_percentage", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(pass_pct.index, pass_pct["pass_percentage"])

    ax.set_xlabel("Pass Play Percentage")
    ax.set_title("Pass Tendency by Offensive Formation")

    for i, (formation, row) in enumerate(pass_pct.iterrows()):
        ax.text(row["pass_percentage"] + 0.01, i, f'{row["pass_percentage"]:.1%}',
                va='center', fontsize=10)

    ax.set_xlim(0, pass_pct["pass_percentage"].max() * 1.1)

    plt.tight_layout()
    plt.savefig("figures/pass_tendency_by_offensive_formation.png")

def histo_num_linemen(df):
    fig, ax = plt.subplots(figsize=(8, 6))

    unique_vals = sorted(df["num_linemen"].unique())
    bins = [val - 0.5 for val in unique_vals] + [unique_vals[-1] + 0.5]

    counts, _, _ = ax.hist(df["num_linemen"], bins=bins, alpha=0.7, edgecolor="black", linewidth=1.0)

    for i, count in enumerate(counts):
        percentage = count / len(df) * 100
        ax.text(unique_vals[i], count + len(df) * 0.01,f'{percentage:.1f}%', ha="center", va="bottom", fontsize=10)

    # Set x-ticks at center of bars
    ax.set_xticks(unique_vals)
    ax.set_yticks(range(2000, 18000, 2000))
    ax.set_xlabel("Number of Offensive Linemen", fontweight="bold")
    ax.set_ylabel("Frequency", fontweight="bold")
    ax.set_title("Distribution of the number of Offensive Linemen per Play", fontweight="bold")

    plt.grid(True, alpha=0.33, axis="y")
    plt.tight_layout()
    plt.savefig("figures/distribution_of_offensive_linemen.png")


def create_down_yardage_pass_table(df):
    def categorize_yardage(yards):
        if yards <= 3:
            return "Short (≤3)"
        elif yards <= 7:
            return "Medium (4-7)"
        else:
            return "Long (8+)"

    df_analysis = df.copy()
    df_analysis["yardage_category"] = df_analysis["yardsToGo"].apply(categorize_yardage)

    # Calculate pass percentages
    pass_proportion = pd.crosstab(df_analysis["down"], df_analysis["yardage_category"], df_analysis["isDropback"], aggfunc="mean")

    column_order = ["Short (≤3)", "Medium (4-7)", "Long (8+)"]
    pass_proportion = pass_proportion.reindex(columns=column_order)

    latex = """\\begin{table}[h]
            \\centering
            \\caption{Pass Play Proportion by Down and Yardage Situation}
            \\label{tab:down_yardage_pass}
            \\begin{tabular}{lccc}
            \\hline
            \\textbf{Down} & \\textbf{Short (≤3 yds)} & \\textbf{Medium (4-7 yds)} & \\textbf{Long (8+ yds)} \\\\
            \\hline
            """

    for down in pass_proportion.index:
        row_data = []
        for col in pass_proportion.columns:
            pct = pass_proportion.loc[down, col]
            cell = f"{pct:.3f}"
            row_data.append(cell)

        latex += f"{down} & " + " & ".join(row_data) + " \\\\\n"

    latex += """\\hline
\\end{tabular}
\\end{table}"""

    with open("paper/tables/down_and_yardage_pass_proportions.txt", 'w') as f:
        f.write(latex)

def create_game_state_pass_table(df):

    df_filtered = df[df["score_differential"] != 0].copy()
    df_filtered = df_filtered[df_filtered["quarter"] != 5]
    df_filtered["game_state"] = np.where(df_filtered["score_differential"] > 0, "Leading", "Trailing")


    pass_proportions = pd.crosstab(df_filtered["quarter"], df_filtered["game_state"], df_filtered["isDropback"], aggfunc="mean")
    column_order = ["Leading", "Trailing"]
    pass_proportions = pass_proportions.reindex(columns=column_order)

    latex = """\\begin{table}[h]
            \\centering
            \\caption{Pass Play Percentage by Quarter and Game State}
            \\label{tab:game_state_pass}
            \\begin{tabular}{lcc}
            \\hline
            \\textbf{Quarter} & \\textbf{Leading Team} & \\textbf{Trailing Team} \\\\
            \\hline
            """

    for quarter in pass_proportions.index:
        row_data = []
        for col in pass_proportions.columns:
            pct = pass_proportions.loc[quarter, col]

            cell = f"{pct:.3f}"
            row_data.append(cell)

        latex += f"{quarter} & " + " & ".join(row_data) + " \\\\\n"

    latex += """\\hline
\\end{tabular}
\\end{table}"""

    with open("paper/tables/gamestate_pass_proportions.txt", 'w') as f:
        f.write(latex)


def learning_curve_and_confusion_matrix(tlosses, vlosses, accuracy, epochs_trained, y_true, y_pred):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f"test accuracy: {accuracy:.4f}")

    ax1.plot(range(1, epochs_trained + 1), tlosses, label="normalized training loss")
    ax1.plot(range(1, epochs_trained + 1), vlosses, label="normalized validation loss", color="orange")

    ax1.set_ylabel("cross-entropy loss")
    ax1.set_xlabel("epoch")
    ax1.legend(loc='upper right')

    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax2, display_labels=["Rush", "Pass"], cmap="Blues")
    ax2.set_title("Confusion Matrix")

    plt.tight_layout()
    plt.savefig("figures/learning_curve_confusion_matrix.png")


if __name__ == "__main__":

    games_vis = pd.read_parquet("data/raw_parquet/games.parquet")
    plays_vis = pd.read_parquet("data/raw_parquet/plays.parquet")
    players_vis = pd.read_parquet("data/raw_parquet/players.parquet")
    player_play_vis = pd.read_parquet("data/raw_parquet/player_play.parquet")

    plays_vis, player_play_vis = test.data_preprocessing_for_feature_eng(plays_vis, games_vis, player_play_vis, players_vis)
    plays_vis = test.engineer_features(plays_vis, player_play_vis[["gameId","playId","nflId", "position"]])

    # Figure 2
    plot_formation_pass_percentage(plays_vis)

    # Figure 3
    histo_num_linemen(plays_vis)

    # Table 1
    create_down_yardage_pass_table(plays_vis[["yardsToGo","down", "isDropback"]])

    # Table 2
    create_game_state_pass_table(plays_vis[["score_differential", "quarter", "isDropback"]])

    # Figure 4
    model_results = torch.load("models/model_and_results.pt", weights_only=False)
    accuracy = model_results["accuracy"]
    train_losses = model_results["train_losses"]
    val_losses = model_results["val_losses"]
    epochs = model_results["epochs"]
    y_true = model_results["y_true"]
    y_pred = model_results["y_pred"]

    learning_curve_and_confusion_matrix(train_losses, val_losses, accuracy, epochs, y_true, y_pred)







