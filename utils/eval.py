import numpy as np
import torch
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from tqdm import tqdm

from utils.utils import load_data


def eval_CLAP(model, dataset_name, plot_conf_matrix=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dir_data = f'/content/drive/MyDrive/Audio_Task_Arithmetic/data/{dataset_name}/'

    loaded_test_batches = load_data(dir_data=dir_data, train=False)

    model = model.to(device)
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_idx, (batch_audio, batch_labels) in enumerate(
                tqdm(loaded_test_batches, desc=f"Test --- ", total=len(loaded_test_batches))):
            batch_audio = batch_audio.to(device)
            batch_labels = batch_labels.to(device)
            n_classes = batch_labels.shape[1]
            batch_labels = batch_labels.argmax(dim=1, keepdim=True)

            outputs = model(**batch_audio)
            similarity = outputs.logits_per_audio  # this is the audio-text similarity score
            y_pred = similarity.softmax(dim=-1)  # we can take the softmax to get the label probabilities
            y_pred = y_pred.argmax(dim=1, keepdim=True)

            all_preds.append(y_pred.detach().cpu().numpy())
            all_labels.append(batch_labels.detach().cpu().numpy())

    all_labels, all_preds = np.concatenate(all_labels, axis=0), np.concatenate(all_preds, axis=0)

    acc = accuracy_score(all_labels, all_preds)
    print(f'\nDone evaluating on {dataset_name}. Accuracy: {100 * acc:.2f}%')

    if plot_conf_matrix:
        conf_matrix = confusion_matrix(all_labels, all_preds, labels=range(n_classes))
        print("Confusion matrix:\n", conf_matrix)
        class_report = classification_report(all_labels, all_preds, labels=range(n_classes))
        print("Classification Report:\n", class_report)

    return acc
