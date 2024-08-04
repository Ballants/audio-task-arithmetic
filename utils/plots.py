import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text

from utils.utils import create_directory_if_not_exists


def comparison_acc_forget(acc_pt_control, acc_pt_target, coeff2acc, control_dataset, target_dataset, dir_to_save):
    '''
    Parameters:
    acc_pt_control (float): Accuracy of the pre-trained model on the control task.
    acc_pt_target (float): Accuracy of the pre-trained model on the target task.
    coeff2acc (dict of tuples of floats): Accuracies of the new model on 2 datasets.
    control_dataset (str): Name of the control dataset.
    target_dataset (str): Name of the target dataset.
    dir_to_save (str): Directory to save the plot.
    '''
    create_directory_if_not_exists(dir_to_save)

    coeff = [x for x in coeff2acc.keys()]

    acc_ds_control = [x[0] for k, x in coeff2acc.items()]
    acc_ds_target = [x[1] for k, x in coeff2acc.items()]
    accuracy_ds_control = [acc_pt_control] + acc_ds_control
    accuracy_ds_target = [acc_pt_target] + acc_ds_target

    groups = ['Pre-trained'] + ['New Model'] * len(acc_ds_control)
    colors = {'Pre-trained': 'blue', 'New Model': 'green'}

    # Plotting
    texts = []
    for idx, (accuracy_c, accuracy_t, group) in enumerate(zip(accuracy_ds_control, accuracy_ds_target, groups)):
        plt.scatter(accuracy_c, accuracy_t, color=colors[group], label=group)

        if idx == 0:
            continue
        # Adding labels to each point
        label = f'{coeff[idx - 1]}'
        texts.append(plt.text(accuracy_c, accuracy_t, label, fontsize=9))

    # Adjust text labels to avoid overlap
    adjust_text(texts)

    # To avoid duplicate legends
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1, 0.9))

    plt.xlabel(f'Accuracy on {control_dataset}')
    plt.ylabel(f'Accuracy on {target_dataset}')
    plt.title('Accuracy Comparison Forgetting via Negation')

    plt.grid(True)

    plt.savefig(dir_to_save + f'acc_{control_dataset}_{target_dataset}.png', bbox_inches='tight')

    plt.show()


def comparison_acc_learning(acc_pt_1, acc_pt_2, acc_ft_1, acc_ft_2, coeff2acc, ds_1, ds_2, dir_to_save, norm=True):
    '''
    Compare accuracies of two datasets with pre-trained and new models.

    Parameters:
    acc_pt_1 (float): Accuracy of the pre-trained model on dataset 1.
    acc_pt_2 (float): Accuracy of the pre-trained model on dataset 2.
    acc_ft_1 (float): Accuracy of the fine-tuned model on dataset 1.
    acc_ft_2 (float): Accuracy of the fine-tuned model on dataset 2.
    coeff2acc (dict of tuples of floats): Accuracies of the new model on 2 datasets.
    ds_1 (str): Name of the first dataset.
    ds_2 (str): Name of the second dataset.
    dir_to_save (str): Directory to save the plot.
    norm (bool): normalize accuracies.
    '''
    create_directory_if_not_exists(dir_to_save)

    coeff = [x for x in coeff2acc.keys()]

    acc_ds_1 = [x[0] for k, x in coeff2acc.items()]
    acc_ds_2 = [x[1] for k, x in coeff2acc.items()]

    if norm:
        temp = [acc_pt_1] + [acc_ft_1] + acc_ds_1
        accuracy_dataset1 = [x / acc_ft_1 for x in temp]
        temp = [acc_pt_2] + [acc_ft_2] + acc_ds_2
        accuracy_dataset2 = [x / acc_ft_2 for x in temp]
    else:
        accuracy_dataset1 = [acc_pt_1] + [acc_ft_1] + acc_ds_1
        accuracy_dataset2 = [acc_pt_2] + [acc_ft_2] + acc_ds_2

    groups = ['Pre-trained'] + ['Fine-tuned'] + ['New Model'] * len(acc_ds_1)
    colors = {'Pre-trained': 'blue', 'Fine-tuned': 'red', 'New Model': 'green'}

    texts = []
    for idx, (accuracy1, accuracy2, group) in enumerate(zip(accuracy_dataset1, accuracy_dataset2, groups)):
        plt.scatter(accuracy1, accuracy2, color=colors[group], label=group)

        '''
        if idx == 0 or idx==1:
          continue
        # Adding labels to each point
        label = f'{coeff[idx-2]}'
        texts.append(plt.text(accuracy1, accuracy2, label, fontsize=9))

    # Adjust text labels to avoid overlap
    adjust_text(texts)'''

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1, 0.9))

    plt.xlabel(f'Accuracy on {ds_1}')
    plt.ylabel(f'Accuracy on {ds_2}')
    plt.title('Accuracy Comparison Learning via Addition')

    plt.grid(True)

    if norm is True:
        plt.savefig(dir_to_save + f'acc_{ds_1}_{ds_2}_norm.png', bbox_inches='tight')
    else:
        plt.savefig(dir_to_save + f'acc_{ds_1}_{ds_2}.png', bbox_inches='tight')

    plt.show()


def comparison_acc_analogy(acc_pt_target, coeff2acc, target_dataset, dir_to_save):
    '''
    Parameters:
    acc_pt_target (float): Accuracy of the pre-trained model on the target task.
    coeff2acc (dict of tuples of floats): Accuracies of the new model on 2 datasets.
    target_dataset (str): Name of the target dataset.
    dir_to_save (str): Directory to save the plot.
    '''
    create_directory_if_not_exists(dir_to_save)

    coeff = [0] + [x for x in coeff2acc.keys()]
    acc_ds_target = [x for k, x in coeff2acc.items()]
    accuracy_ds_target = [acc_pt_target] + acc_ds_target

    groups = ['Pre-trained'] + ['New Model'] * len(acc_ds_target)
    colors = {'Pre-trained': 'blue', 'New Model': 'green'}

    # Plotting
    for idx, (c, accuracy_t, group) in enumerate(zip(coeff, accuracy_ds_target, groups)):
        plt.scatter(c, accuracy_t, color=colors[group], label=group)

    # To avoid duplicate legends
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1, 0.9))

    plt.xlabel(f'Scaling coefficient')
    plt.ylabel(f'Accuracy on {target_dataset}')
    plt.title('Accuracy Comparison Task Analogy')

    plt.yticks(np.arange(acc_pt_target, 1.01, 0.1).tolist())
    plt.xticks(np.arange(0, 1.01, 0.1).tolist())
    plt.xticks(rotation=45)

    plt.grid(True)

    plt.savefig(dir_to_save + f'acc_{target_dataset}.png', bbox_inches='tight')

    plt.show()


def comparison_acc_analogy_control(acc_pt_control, acc_pt_target, coeff2acc, control_dataset, target_dataset,
                                   dir_to_save):
    '''
    Parameters:
    acc_pt_control (float): Accuracy of the pre-trained model on the control task.
    acc_pt_target (float): Accuracy of the pre-trained model on the target task.
    coeff2acc (dict of tuples of floats): Accuracies of the new model on 2 datasets.
    control_dataset (str): Name of the control dataset.
    target_dataset (str): Name of the target dataset.
    dir_to_save (str): Directory to save the plot.
    '''
    create_directory_if_not_exists(dir_to_save)

    coeff = [x for x in coeff2acc.keys()]

    acc_ds_control = [x[0] for k, x in coeff2acc.items()]
    acc_ds_target = [x[1] for k, x in coeff2acc.items()]
    accuracy_ds_control = [acc_pt_control] + acc_ds_control
    accuracy_ds_target = [acc_pt_target] + acc_ds_target

    groups = ['Pre-trained'] + ['New Model'] * len(acc_ds_control)
    colors = {'Pre-trained': 'blue', 'New Model': 'green'}

    # Plotting
    texts = []
    for idx, (accuracy_c, accuracy_t, group) in enumerate(zip(accuracy_ds_control, accuracy_ds_target, groups)):
        plt.scatter(accuracy_c, accuracy_t, color=colors[group], label=group)

        if idx == 0:
            continue
        # Adding labels to each point
        label = f'{coeff[idx - 1]}'
        texts.append(plt.text(accuracy_c, accuracy_t, label, fontsize=9))

    # Adjust text labels to avoid overlap
    adjust_text(texts)

    # To avoid duplicate legends
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1, 0.9))

    plt.xlabel(f'Accuracy on {control_dataset}')
    plt.ylabel(f'Accuracy on {target_dataset}')
    plt.title('Accuracy Comparison Task Analogy')

    plt.grid(True)

    plt.savefig(dir_to_save + f'acc_{control_dataset}_{target_dataset}.png', bbox_inches='tight')

    plt.show()
