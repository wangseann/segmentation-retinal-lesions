from os import listdir
from os.path import isfile, join
import os

import numpy as np
from skimage.io import imread
from skimage.io import imsave
from sklearn.metrics import auc, precision_recall_curve
from utils.params import parse_arguments_pred, default_params

path_to_data = '../'

def difference_map(ground_truth, prediction):
    """
    It computes the difference map (it shows TP, FN and FP) and dice score for a binary problem
    :param ground_truth: ground truth
    :param prediction: predicted labels
    :return: difference map and dice score
    """
    diff_map = np.zeros((ground_truth.shape[0], ground_truth.shape[1], 3))
    diff_map[(ground_truth==1) & (prediction==1)] = (0, 255, 0)  # Green (overlapping)
    diff_map[(ground_truth==1) & (prediction!=1)] = (255, 0, 0)  # Red (false negative, missing in pred)
    diff_map[(ground_truth!=1) & (prediction==1)] = (0, 0, 255)  # Blue (false positive)
    diff_map = np.asarray(diff_map).astype(np.uint8)

    # compute dice coefficient for a given image
    overlap = len(diff_map[(ground_truth==1) & (prediction==1)])
    fn = len(diff_map[(ground_truth==1) & (prediction!=1)])
    fp = len(diff_map[(ground_truth!=1) & (prediction==1)])
    dice_coef = 2. * overlap /( 2 * overlap + fn + fp + 0.001 )

    return diff_map, dice_coef


def AUC_PR(ground_truth, pred):
    """
    It computes Area Under the Curve Precision-Recall for a binary problem
    :param ground_truth: ground truth
    :param pred: probability maps
    :return: AUC Precision-Recall
    """
    precision, recall, _ = precision_recall_curve(ground_truth.flatten(), pred.flatten(), pos_label = 1)
    auc_pr = auc(recall, precision)
    return auc_pr


def compute_dice_coef_gen(num, params):
    """
    It computes the dice score and the difference map for one class (the indicated in num)
    :param num: number of the class to compute the difference map and dice score (1 - EX, 2 - HE, 3 - MA, 4 - SE)
    :param test_path: path where test images are located. Inside this directory, there should be a folder called 'labels' where ground truth are located
    :param path_maps: path where difference maps are saved
    """

    data_path = join(path_to_data, params['data_path'])
    test_path = join(data_path, 'test/')

    name_weights = params['weights']
    if name_weights.endswith('.h5'):
        name_weights = name_weights[:-3]

    path_pred = join(path_to_data, params['weights_path'] + 'predictions_' + name_weights + '/')

    path_maps = join(path_pred, 'maps/')
    if not os.path.exists(path_maps):
        os.makedirs(path_maps)

    labels = params['labels']

    lab_index = num
    lab_name = labels[num-1][:-1]
    dice_total = 0.0
    gt_test = [f for f in listdir(test_path + 'full_labels/') if isfile(join(test_path + 'full_labels/', f))]
    gt_aux = imread(test_path + 'full_labels/' + gt_test[0])

    for file in gt_test:
        pred_file = file[:-10] + '_pred.png'
        gt = imread(test_path + 'full_labels/' + file)
        gt = np.asarray(gt).astype(np.uint8)
        pred = imread(path_pred + pred_file)
        pred = np.asarray(pred).astype(np.uint8)
        one_label_gt = np.zeros((gt_aux.shape[0], gt_aux.shape[1]))
        one_label_pred = np.zeros((gt_aux.shape[0], gt_aux.shape[1]))
        ind_label = (gt[:, :, 0] == lab_index)
        one_label_gt[ind_label] = 1
        ind_label_pred = (pred[:, :, 0] == lab_index)
        one_label_pred[ind_label_pred] = 1
        map, dice_coef = difference_map(one_label_gt, one_label_pred)
        imsave(path_maps + file[:-4] + '_' + lab_name + '_map.png', map)
        dice_total = dice_total + dice_coef
    dice_total = dice_total / len(gt_test)

    file = path_pred + 'evalutation_' + params['weights'] + '.txt'
    f = open(file, "a+")
    f.write('\nDice coefficient \n')
    f.write("Class {0}: {1} \n".format(lab_name, dice_total))
    f.close()


def evaluate(**params):
    """
    It evaluates the model (computes dice coefficient and creates difference maps) for all the classes separately.
    """
    params = dict(
        default_params,
        **params
    )

    verbose = params['verbose']

    if verbose:
        print("Evaluating predictions...")

    labels = params['labels']

    for num_label in range(len(labels)):
        compute_dice_coef_gen(num_label+1, params)

    if verbose:
        print('Done!')


if __name__ == '__main__':
    evaluate(**vars(parse_arguments_pred()))