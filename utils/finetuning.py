import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from transformers import ClapModel

from utils.utils import create_directory_if_not_exists, load_data


def finetuning_CLAP(dataset_name, learning_rate=1e-5, epochs=10, weight_decay=1e-05, step_lr=5, gamma=0.1,
                    class_weights=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dir_data = f'/content/drive/MyDrive/Audio_Task_Arithmetic/data/{dataset_name}/'
    loaded_train_batches, loaded_val_batches = load_data(dir_data=dir_data, train=True)

    model = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(device)

    # Freeze parameters in text encoder
    for param in model.text_model.parameters():
        param.requires_grad = False
    for param in model.text_projection.parameters():
        param.requires_grad = False

    if class_weights is not None:
        class_weights = class_weights.to(device)
    loss_fn = torch.nn.CrossEntropyLoss() if class_weights is None else torch.nn.CrossEntropyLoss(weight=class_weights)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_lr, gamma=gamma)

    checkpoint_path = f'/content/drive/MyDrive/Audio_Task_Arithmetic/checkpoints/CLAP/'
    create_directory_if_not_exists(checkpoint_path)

    train_loss = []
    val_loss = []
    val_loss_best = 1000

    for epoch in range(epochs):
        running_loss = 0
        len_train_dl = 0

        model.train()
        for batch_idx, (batch_audio, batch_labels) in enumerate(
                tqdm(loaded_train_batches, desc=f"Epoch: {epoch} | ---", total=len(loaded_train_batches))):
            batch_dim = len(batch_labels.shape[0]) if type(batch_labels.shape[0]) != int else batch_labels.shape[0]
            len_train_dl += batch_dim

            batch_audio = batch_audio.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()

            outputs = model(**batch_audio)
            logits_per_audio = outputs.logits_per_audio  # this is the audio-text similarity score
            probs = logits_per_audio.softmax(dim=-1)  # we can take the softmax to get the label probabilities

            loss = loss_fn(probs, batch_labels)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(params, 1.0)

            optimizer.step()

            running_loss += loss.item()

        train_loss.append(running_loss / len_train_dl)
        print('\tTraining loss: %.5f' % (running_loss / len_train_dl))

        # Evaluation
        model.eval()
        with torch.no_grad():
            running_loss = 0
            len_val_dl = 0

            for batch_idx, (batch_audio, batch_labels) in enumerate(
                    tqdm(loaded_val_batches, desc=f"\t| ---", total=len(loaded_val_batches))):
                batch_dim = len(batch_labels.shape[0]) if type(batch_labels.shape[0]) != int else batch_labels.shape[0]
                len_val_dl += batch_dim

                batch_audio = batch_audio.to(device)
                batch_labels = batch_labels.to(device)

                outputs = model(**batch_audio)
                logits_per_audio = outputs.logits_per_audio  # this is the audio-text similarity score
                probs = logits_per_audio.softmax(dim=-1)  # we can take the softmax to get the label probabilities

                loss = loss_fn(probs, batch_labels)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                running_loss += loss.item()

            curr_val_loss = running_loss / len_val_dl
            print('\tValidation loss: %.5f' % curr_val_loss)
            val_loss.append(curr_val_loss)
            if curr_val_loss < val_loss_best:
                val_loss_best = curr_val_loss
                torch.save(model.state_dict(), checkpoint_path + f'finetuned_{dataset_name}.pth')

        scheduler.step()

    # Create a plot for training and validation losses
    plt.figure(figsize=(10, 6))
    plt.plot(range(epochs), train_loss, label='Training Loss', marker='o')
    plt.plot(range(epochs), val_loss, label='Validation Loss', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    plt.show()
