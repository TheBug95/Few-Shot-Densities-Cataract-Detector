import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import os
import pickle
from utils.constants import (
    DEVICE,
    NORMAL_CAT_ID,
    DATASETS_CATARACT_DIR,
    DatasetCataractSplit
)
from pycocotools.coco import COCO

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
    """
    Calcula el accuracy por cada conjunto de predicciones realizadas
    :param y_preds: predicciones realizadas
    :param y_true: ground truth
    :param proto_path: dirección donde se encuentra ubicado el prototipo a usar
    """
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

    """
    Calcula la media y desviación estándar de los accuracies calculados
    por cada prototipo teniendo en cuenta el número de iteraciones realizadas
    :param accuracies_by_k: accuracies por cada k en que se evaluó el few-shot learning
    :param number_of_iterations: número de iteraciones realizadas
    """

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
    """
    Realiza el gráfico sobre la media y desviación estándar de los accuracies
    calculados por cada prototipo
    :param accuracies: accuracies calculados
    :param backbone: backbone utilizado
    """

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

def get_mask_generator(model_type: str, checkpoint: str):
    """
    Devuelve un SamAutomaticMaskGenerator para:

        • HQ-SAM     (ruta contiene 'hq')
        • MobileSAM  (ruta contiene 'mobile')
        • SAM normal (por defecto)
    """
    ckpt_name = str(checkpoint).lower()

    if "hq" in ckpt_name:
        from segment_anything_hq import sam_model_registry, SamAutomaticMaskGenerator

    elif "mobile" in ckpt_name:
        from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator

    else:
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

    sam = sam_model_registry[model_type](checkpoint=checkpoint).to(DEVICE)

    return SamAutomaticMaskGenerator(sam)

def export_results_excel(accuracies_mean_std):
    """
    Exporta los resultados de las corridas (media y desviación estándar) de los accuracies
    a un archivo excel
    :param accuracies_mean_std: media y std de accuracies

    """
    # Ensure the directory exists
    os.makedirs('models/results', exist_ok=True)

    # Convert the dictionary to a pandas DataFrame
    df_r18_std = pd.DataFrame.from_dict(accuracies_mean_std, orient='index')

    # Save the DataFrame to an Excel file
    excel_path = 'models/results/mean_std_accuracies_r18_fixed_30.xlsx'
    df_r18_std.to_excel(excel_path)

    print(f"Accuracies saved to {excel_path}")