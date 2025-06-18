import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from utils.constants import NORMAL_CAT_ID, DATASETS_CATARACT_DIR, DatasetCataractSplit
from pycocotools.coco import COCO
from pathlib import Path
import pickle

# ground-truth (1: catarata, 0: normal)
split = DatasetCataractSplit.VALID.value
coco    = COCO(str(DATASETS_CATARACT_DIR/f"{split}/_annotations.coco.json"))

all_img_ids = coco.getImgIds() 

y_true = np.array([
    int(any(a["category_id"] != NORMAL_CAT_ID
            for a in coco.loadAnns(coco.getAnnIds(imgIds=[iid]))))
    for iid in all_img_ids
])

def calculate_accuracy(y_preds, y_true, proto_path):
    with open(proto_path, "rb") as fp:
        data = pickle.load(fp)

    # Detectar formato nuevo vs. antiguo
    protos = data["prototypes"] if isinstance(data, dict) and "prototypes" in data else data
    ks = sorted(protos.keys())

    accuracies_by_k = {}
    for k in ks:
        acc = (np.array(y_preds[k]) == y_true).mean() * 100
        accuracies_by_k[k] = acc

    return accuracies_by_k


def calculate_mean_std_accuracy(accuracies_by_k, number_of_iterations=10):
    accuracy_std_score = {}

    # Iterate through each k value
    for k in list(accuracies_by_k[0].keys()):
        # Collect the accuracies for the current k from all iterations
        accuracies_for_k = [accuracies_by_k[i][k] for i in range(number_of_iterations)]

        # Calculate the mean and standard deviation
        mean_accuracies = np.mean(accuracies_for_k)
        std_accuracies = np.std(accuracies_for_k)

        accuracy_std_score[k] = {
            #'accuracies': accuracies_for_k,
            'mean': mean_accuracies,
            'std': std_accuracies
        }

    return accuracy_std_score

def mean_std_plot(accuracies, backbone):

    # Convert the dictionary to a pandas DataFrame for easier plotting
    data = {
        'k': list(accuracies.keys()),
        'mean': [accuracies[k]['mean'] for k in accuracies.keys()],
        'std': [accuracies[k]['std'] for k in accuracies.keys()]
    }
    df = pd.DataFrame(data)

    # Set the plot style
    sns.set_style("whitegrid")

    # Create the line plot with error band
    plt.figure(figsize=(10, 6))
    line_plot = sns.lineplot(
        data=df,
        x='k',
        y='mean',
        marker='o',
        label='Mean Accuracy',
        color='steelblue'
    )

    # Add the shaded error band
    line_plot.fill_between(
        df['k'],
        df['mean'] - df['std'],
        df['mean'] + df['std'],
        color='steelblue',
        alpha=0.2,
        label='Standard Deviation'
    )

    # Add labels and title
    plt.xlabel('Number of Shots (k)', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title(f'Mean Accuracy with Standard Deviation by k for {backbone}', fontsize=14)

    # Set x-axis ticks to be the k values
    plt.xticks(df['k'])

    # Add a legend
    plt.legend()

    # Show the plot
    plt.show()