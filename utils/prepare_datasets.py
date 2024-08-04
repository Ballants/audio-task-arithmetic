import numpy as np
import torch
from datasets import load_dataset, Dataset, concatenate_datasets, Audio
from sklearn.model_selection import train_test_split

from utils.utils import one_hot_encoding, process_batches

BATCH_SIZE = 64


def prepare_GTZAN(batch_size, processor):
    # Load dataset
    dataset = load_dataset("marsyas/gtzan", trust_remote_code=True)
    audio_data = [a["array"] for a in dataset["train"]["audio"]]

    # Process labels
    all_classes = dataset["train"].unique("genre")
    id2label_fn = dataset["train"].features["genre"].int2str
    all_classes = id2label_fn(all_classes)

    labels = dataset["train"]["genre"]
    n_classes = len(all_classes)
    one_hot_labels = torch.tensor(np.array([one_hot_encoding(i, n_classes) for i in labels]))

    candidate_labels = ["This is the sound of a " + x + " song" for x in all_classes]

    # Split data into train, validation, test sets
    audio_train, audio_temp, labels_train, labels_temp = train_test_split(
        audio_data, one_hot_labels, test_size=0.2, random_state=42
    )
    audio_val, audio_test, labels_val, labels_test = train_test_split(
        audio_temp, labels_temp, test_size=0.5, random_state=42
    )

    # Pre-process audio and generate batches
    train_batches = list(process_batches(audio_train, labels_train, batch_size, processor, candidate_labels))
    del (audio_train)
    val_batches = list(process_batches(audio_val, labels_val, batch_size, processor, candidate_labels))
    del (audio_val)
    test_batches = list(process_batches(audio_test, labels_test, batch_size, processor, candidate_labels))
    # del(audio_test)

    return train_batches, val_batches, test_batches


def prepare_ESC50(batch_size, processor):
    # Load dataset
    dataset = load_dataset("ashraq/esc50")
    audio_data = [a["array"] for a in dataset["train"]["audio"]]

    # Process labels
    all_classes = dataset["train"].unique("category")
    for i in range(len(all_classes)):
        all_classes[i] = all_classes[i].replace("_", " ")
    all_labels = dataset["train"].unique("target")
    # class2id = {}
    id2class = {}
    for n in range(len(all_classes)):
        # class2id[all_classes[n]] = all_labels[n]
        id2class[all_labels[n]] = all_classes[n]

    labels = dataset["train"]["target"]
    n_classes = len(all_classes)
    one_hot_labels = torch.tensor(np.array([one_hot_encoding(i, n_classes) for i in labels]))

    # Candidate labels for classification
    candidate_labels = ["Sound of " + id2class[i] for i in range(len(all_classes))]

    # Split data into train, validation, test sets
    audio_train, audio_temp, labels_train, labels_temp = train_test_split(
        audio_data, one_hot_labels, test_size=0.2, random_state=42
    )
    audio_val, audio_test, labels_val, labels_test = train_test_split(
        audio_temp, labels_temp, test_size=0.5, random_state=42
    )

    # Pre-process audio and generate batches
    train_batches = list(process_batches(audio_train, labels_train, batch_size, processor, candidate_labels))
    del (audio_train)
    val_batches = list(process_batches(audio_val, labels_val, batch_size, processor, candidate_labels))
    del (audio_val)
    test_batches = list(process_batches(audio_test, labels_test, batch_size, processor, candidate_labels))
    # del(audio_test)

    return train_batches, val_batches, test_batches


def prepare_UB8k(batch_size, processor):
    # Load dataset
    dataset = load_dataset("danavery/urbansound8K", trust_remote_code=True)
    dataset = dataset["train"].train_test_split(seed=42, shuffle=True, test_size=0.3)

    audio_data = [a["array"] for a in dataset["test"]["audio"]]

    # Process labels
    all_classes = dataset["test"].unique("class")
    for i in range(len(all_classes)):
        all_classes[i] = all_classes[i].replace("_", " ")
    all_labels = dataset["test"].unique("classID")
    # class2id = {}
    id2class = {}
    for n in range(len(all_classes)):
        # class2id[all_classes[n]] = all_labels[n]
        id2class[all_labels[n]] = all_classes[n]

    labels = dataset["test"]["classID"]
    n_classes = len(all_classes)
    one_hot_labels = torch.tensor(np.array([one_hot_encoding(i, n_classes) for i in labels]))

    # Candidate labels for classification
    candidate_labels = ["Sound of " + id2class[i] for i in range(len(all_classes))]

    # Split data into train, validation, test sets
    audio_train, audio_temp, labels_train, labels_temp = train_test_split(
        audio_data, one_hot_labels, test_size=0.2, random_state=42
    )
    audio_val, audio_test, labels_val, labels_test = train_test_split(
        audio_temp, labels_temp, test_size=0.5, random_state=42
    )

    # Pre-process audio and generate batches
    train_batches = list(process_batches(audio_train, labels_train, batch_size, processor, candidate_labels))
    del (audio_train)
    val_batches = list(process_batches(audio_val, labels_val, batch_size, processor, candidate_labels))
    del (audio_val)
    test_batches = list(process_batches(audio_test, labels_test, batch_size, processor, candidate_labels))
    # del(audio_test)

    return train_batches, val_batches, test_batches


def prepare_analogy(batch_size, processor, analogy_task):
    match analogy_task:
        case "W":
            target_label = 'female_happy'
        case "X":
            target_label = 'male_happy'
        case "Y":
            target_label = 'female_sad'
        case "Z":
            target_label = 'male_sad'

    # Load dataset
    dataset_control = load_dataset("ashraq/esc50", split="train", trust_remote_code=True)
    dataset_control = dataset_control.remove_columns(
        [col for col in dataset_control.column_names if col not in ['category', 'target', 'audio']])
    dataset_target = load_dataset("Huan0806/gender_emotion_recognition", split="train", trust_remote_code=True)
    dataset_target = dataset_target.filter(lambda x: x['labels'] == target_label)
    dataset_target = dataset_target.shuffle(seed=42).select(range(40))

    length = dataset_target.num_rows
    dataset_target = Dataset.from_dict({
        'target': [50] * length,
        'category': ['something'] * length,
        'audio': dataset_target['audio']
    })

    def convert_audio_column(dataset, sampling_rate=16000):
        return dataset.cast_column('audio', Audio(sampling_rate=sampling_rate))

    dataset_control = convert_audio_column(dataset_control, sampling_rate=16000)
    dataset_target = convert_audio_column(dataset_target, sampling_rate=16000)

    dataset = concatenate_datasets([dataset_control, dataset_target])
    audio_data = [a["array"] for a in dataset["audio"]]

    # Process labels
    all_classes = dataset.unique("category")
    for i in range(len(all_classes)):
        all_classes[i] = all_classes[i].replace("_", " ")
    all_labels = dataset.unique("target")
    # class2id = {}
    id2class = {}
    for n in range(len(all_classes)):
        # class2id[all_classes[n]] = all_labels[n]
        id2class[all_labels[n]] = all_classes[n]

    labels = dataset["target"]
    n_classes = len(all_classes)
    one_hot_labels = torch.tensor(np.array([one_hot_encoding(i, n_classes) for i in labels]))

    # Candidate labels for classification
    candidate_labels = ["This is the sound of " + id2class[i] for i in range(len(all_classes))]

    # Split data into train, validation, test sets
    audio_train, audio_temp, labels_train, labels_temp = train_test_split(
        audio_data, one_hot_labels, test_size=0.2, random_state=42
    )
    del (audio_data, one_hot_labels)
    audio_val, audio_test, labels_val, labels_test = train_test_split(
        audio_temp, labels_temp, test_size=0.5, random_state=42
    )
    del (audio_temp, labels_temp)

    '''
    # Compute class weigths to handle class unbalance: inverse class frequency + normalization
    class_indices = torch.argmax(labels_train, dim=1)
    class_counts = torch.bincount(class_indices)
    class_weights = 1.0 / class_counts.float()
    class_weights = class_weights / class_weights.sum()'''

    # Pre-process audio and generate batches
    train_batches = list(process_batches(audio_train, labels_train, batch_size, processor, candidate_labels))
    del (audio_train)
    val_batches = list(process_batches(audio_val, labels_val, batch_size, processor, candidate_labels))
    del (audio_val)
    test_batches = list(process_batches(audio_test, labels_test, batch_size, processor, candidate_labels))
    # del(audio_test)

    return train_batches, val_batches, test_batches  # , class_weights
