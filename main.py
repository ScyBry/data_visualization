import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_data():
    """Загрузка данных из CSV файлов."""
    all_sites_scores = pd.read_csv("./all_sites_scores.csv")
    fandago = pd.read_csv("./fandango_scrape.csv")
    return all_sites_scores, fandago


def plot_scatter(ax, x, y, data, title):
    """Создание scatter plot."""
    sns.scatterplot(data=data, x=x, y=y, ax=ax)
    ax.set_title(title)
    ax.grid(True)


def plot_kde(ax, data, x, fill_label, clip_range, title):
    """Создание KDE plot."""
    sns.kdeplot(data=data, x=x, clip=clip_range, fill=True, label=fill_label, ax=ax)
    ax.set_title(title)
    ax.grid(True)


def plot_fandago(fandago):
    """Создание графиков для данных fandago."""
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))

    plot_scatter(
        axes[0, 0], "RATING", "VOTES", fandago, "Scatter plot of RATING vs VOTES"
    )

    fandago_with_votes = fandago[fandago["VOTES"] != 0]
    plot_kde(
        axes[0, 1],
        fandago_with_votes,
        "RATING",
        "True Rating",
        [0, 5],
        "KDE plot of RATING and STARS",
    )
    plot_kde(axes[0, 1], fandago_with_votes, "STARS", "Stars Displayed", [0, 5], "")

    fandago["YEAR"] = fandago["FILM"].apply(
        lambda title: title.split("(")[-1].split(")")[0]
    )
    sns.countplot(data=fandago, x="YEAR", ax=axes[1, 0])
    axes[1, 0].set_title("Distribution of films by year")
    axes[1, 0].grid(True)

    fandago_with_votes["STARS_DIFF"] = (
        fandago_with_votes["STARS"] - fandago_with_votes["RATING"]
    ).round(1)
    sns.countplot(
        data=fandago_with_votes, x="STARS_DIFF", ax=axes[1, 1], palette="magma"
    )
    axes[1, 1].set_title("Countplot of stars diff")
    plt.tight_layout()


def plot_all_sites_scores(all_sites_scores):
    """Создание графиков для данных all_sites_scores."""
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))

    plot_scatter(
        axes[0, 0],
        "RottenTomatoes",
        "RottenTomatoes_User",
        all_sites_scores,
        "Scatter plot of RottenTomatoes vs RottenTomatoes_User",
    )

    all_sites_scores["Rotten_Diff"] = (
        all_sites_scores["RottenTomatoes"] - all_sites_scores["RottenTomatoes_User"]
    )

    sns.histplot(
        data=all_sites_scores,
        x=all_sites_scores["Rotten_Diff"].abs(),
        kde=True,
        bins=25,
        ax=axes[0, 1],
    )
    axes[0, 1].set_title("RT Critics Score minus RT User Score")
    axes[0, 1].grid(True)

    sns.histplot(
        data=all_sites_scores, x="Rotten_Diff", kde=True, bins=25, ax=axes[1, 0]
    )
    axes[1, 0].set_title("Abs Difference between RT Critics Score and RT User Score")
    axes[1, 0].grid(True)

    plot_scatter(
        axes[1, 1],
        "Metacritic",
        "Metacritic_User",
        all_sites_scores,
        "Metacritic vs Metacritic User",
    )
    plt.tight_layout()


def analyze_differences(all_sites_scores):
    """Анализ различий между оценками критиков и пользователей."""
    avg_diff = all_sites_scores["Rotten_Diff"].abs().mean()
    print("Average absolute difference between RT Critics and Users:", avg_diff)

    print("\nUsers Love but Critics Hate:")
    print(all_sites_scores.nsmallest(5, "Rotten_Diff")[["FILM", "Rotten_Diff"]])

    print("\nCritics love, but Users Hate:")
    print(all_sites_scores.nlargest(5, "Rotten_Diff")[["FILM", "Rotten_Diff"]])

    print("\nMovie with the most IMDB user votes:")
    print(
        all_sites_scores.nlargest(1, "IMDB_user_vote_count")[
            ["FILM", "IMDB_user_vote_count"]
        ]
    )


def merge_dataframes(all_sites_scores, fandago):
    """Слияние двух DataFrame по столбцу FILM."""
    df = pd.merge(fandago, all_sites_scores, on="FILM", how="inner")
    print(df.head())
    print(df.info())
    print(df.columns)
    return df


def normalize_scores(merged_df):
    """Нормализация оценок."""
    return merged_df.assign(
        RT_Norm=np.round(merged_df["RottenTomatoes"] / 20, 1),
        RTU_Norm=np.round(merged_df["RottenTomatoes_User"] / 20, 1),
        Meta_Norm=np.round(merged_df["Metacritic"] / 20, 1),
        Meta_U_Norm=np.round(merged_df["Metacritic_User"] / 2, 1),
        IMDB_Norm=np.round(merged_df["IMDB"] / 2, 1),
    )


def plot_normalized_scores(norm_scores):
    """Построение графиков нормализованных оценок."""
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(12, 12))

    sns.kdeplot(data=norm_scores, clip=[0, 5], shade=True, ax=axes[0, 0])
    axes[0, 0].set_title("KDE of Normalized Scores")
    axes[0, 0].grid(True)

    sns.kdeplot(data=norm_scores[["STARS", "RT_Norm"]], shade=True, ax=axes[0, 1])
    axes[0, 1].set_title("KDE of STARS and RT_Norm")
    axes[0, 1].grid(True)

    sns.histplot(data=norm_scores, ax=axes[1, 0])
    axes[1, 0].set_title("Histogram of Normalized Scores")
    axes[1, 0].grid(True)

    sns.clustermap(
        norm_scores,
        cmap="magma",
        col_cluster=False,
    )
    axes[1, 1].set_title("Clustermap of Normalized Scores")
    plt.tight_layout()
    plt.show()


def main():
    all_sites_scores, fandago = load_data()

    plot_fandago(fandago)
    plot_all_sites_scores(all_sites_scores)
    analyze_differences(all_sites_scores)
    merged_df = merge_dataframes(all_sites_scores, fandago)

    norm_scores = normalize_scores(merged_df)[
        [
            "STARS",
            "RATING",
            "RT_Norm",
            "RTU_Norm",
            "Meta_Norm",
            "Meta_U_Norm",
            "IMDB_Norm",
        ]
    ]

    plot_normalized_scores(norm_scores)


if __name__ == "__main__":
    main()
