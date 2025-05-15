import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from bdgs.data.algorithm import ALGORITHM


def load_all_data(base_dir, algorithms=set(ALGORITHM), series=4):
    records = []
    for i in range(1, series + 1):
        series = f"SC{i}"
        learn_path = os.path.abspath(base_dir + f"/sc{i}/sc{i}_validation_learn_results.json")
        classify_path = os.path.abspath(base_dir + f"/sc{i}/sc{i}_validation_classify_results.json")

        if not os.path.exists(learn_path) or not os.path.exists(classify_path):
            continue

        with open(learn_path) as f:
            learn_data = json.load(f)
        with open(classify_path) as f:
            classify_data = json.load(f)

        learn_algos = learn_data["algorithms"]
        classify_algos = classify_data["algorithms"]

        for algo_name in learn_algos:
            if algo_name not in algorithms:
                continue

            learn_metrics = learn_algos[algo_name]
            classify_metrics = classify_algos.get(algo_name, {})

            records.append({
                "Series": series,
                "Algorithm": algo_name,
                "Accuracy": learn_metrics.get("accuracy"),
                "Loss": learn_metrics.get("loss"),
                "Correct %": classify_metrics.get("correct_percent"),
                "Certainty %": classify_metrics.get("average_certainty")
            })

    return pd.DataFrame(records)


if __name__ == "__main__":
    df = load_all_data(base_dir=os.path.join(os.path.dirname(__file__), "results"), algorithms={
        ALGORITHM.MAUNG,
        ALGORITHM.PINTO_BORGES,
        ALGORITHM.MURTHY_JADON,
        ALGORITHM.GUPTA_JAAFAR,
        ALGORITHM.EID_SCHWENKER,
        ALGORITHM.MOHMMAD_DADI
    })

    sns.set_theme(style="whitegrid", context="talk")

    # learn

    # accuracy heatmap
    pivot_acc = df.pivot(index="Algorithm", columns="Series", values="Accuracy")
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_acc, annot=True, fmt=".3f", cmap="YlGnBu", annot_kws={"size": 12})
    plt.title("LEARN accuracy", fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=8)
    plt.xlabel("Series", fontsize=12)
    plt.ylabel("Algorithm", fontsize=12)
    plt.tight_layout()
    plt.show()

    # loss heatmap
    pivot_loss = df.pivot(index="Algorithm", columns="Series", values="Loss")
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_loss, annot=True, fmt=".4f", cmap="OrRd", annot_kws={"size": 12})
    plt.title("LEARN loss", fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=8)
    plt.xlabel("Series", fontsize=12)
    plt.ylabel("Algorithm", fontsize=12)
    plt.tight_layout()
    plt.show()

    # classify

    # accuracy heatmap
    pivot_acc = df.pivot(index="Algorithm", columns="Series", values="Correct %")
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_acc, annot=True, fmt=".3f", cmap="YlGnBu", annot_kws={"size": 12})
    plt.title("VALIDATION correct predictions (%)", fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=8)
    plt.xlabel("Series", fontsize=12)
    plt.ylabel("Algorithm", fontsize=12)
    plt.tight_layout()
    plt.show()
