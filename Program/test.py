import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader
from torchvision.models import shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from Utils.getData import getImageLabel


def main():
    BATCH_SIZE = 32
    folds = [1, 2, 3, 4, 5]
    DEVICE = torch.device('cpu')

    test_loader = DataLoader(
        getImageLabel(origin=f'./dataset/Original Images/Original Images/FOLDS/', folds=folds, subdir=['Test']),
        batch_size=BATCH_SIZE, shuffle=True
    )

    weights = ShuffleNet_V2_X1_0_Weights.DEFAULT
    model = shufflenet_v2_x1_0(weights=weights)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, 6)
    )
    model.to(DEVICE)

    checkpoint = torch.load('ShuffleNetV2_25.pt', map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])

    prediction, ground_truth, probabilities = [], [], []
    with torch.no_grad():
        model.eval()
        for batch, (src, trg) in enumerate(test_loader):
            src, trg = src.to(DEVICE), trg.to(DEVICE)
            src = torch.permute(src, (0, 3, 1, 2))

            pred = model(src)
            probabilities.extend(pred.detach().cpu().numpy())
            prediction.extend(torch.argmax(pred, dim=1).detach().cpu().numpy())
            ground_truth.extend(torch.argmax(trg, dim=1).detach().cpu().numpy())

    # Class
    classes = ('Chickenpox', 'Cowpox', 'Healthy', 'HFMD', 'Measles', 'Monkeypox')

    # Confusion Matrix
    cf_matrix = confusion_matrix(ground_truth, prediction)
    print(cf_matrix)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=classes, columns=classes)
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True, fmt='.2f', cmap="Blues")
    plt.savefig('confusion_matrix.png')

    # Metrics
    print("Accuracy score = ", accuracy_score(ground_truth, prediction))
    print("Precision score = ", precision_score(ground_truth, prediction, average='weighted'))
    print("Recall score = ", recall_score(ground_truth, prediction, average='weighted'))
    print("F1 score = ", f1_score(ground_truth, prediction, average='weighted'))

    # ROC-AUC Curve
    ground_truth_bin = label_binarize(ground_truth, classes=list(range(len(classes))))
    probabilities = np.array(probabilities)

    plt.figure(figsize=(10, 8))
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
    for i, class_name in enumerate(classes):
        fpr, tpr, _ = roc_curve(ground_truth_bin[:, i], probabilities[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=2, label=f'{class_name} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Guess (AUC = 0.50)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig('roc_auc.png')


if __name__ == "__main__":
    main()