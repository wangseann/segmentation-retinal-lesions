import tensorflow as tf
from keras import backend as K

def generalised_dice_coef(y_true, y_pred, type_weight='Square'):
    """
    It computes the generalised dice coefficient
    :param y_true: true labels (ground truth)
    :param y_pred: predicted labels
    :return: generalised dice coefficient score between y_true and y_pred
    """
    y_true = K.cast(y_true, dtype=tf.float32)  # Cast y_true to float32
    prediction = K.cast(y_pred, dtype=tf.float32)  # Cast y_pred to float32

    ref_vol = K.sum(y_true, axis=0)
    intersect = K.sum(y_true * prediction, axis=0)
    seg_vol = K.sum(prediction, axis=0)

    if type_weight == 'Square':
        weights = 1.0 / (K.square(ref_vol) + K.epsilon())  # Added K.epsilon() for numerical stability
    elif type_weight == 'Simple':
        weights = 1.0 / (ref_vol + K.epsilon())  # Added K.epsilon() for numerical stability
    elif type_weight == 'Uniform':
        weights = K.ones_like(ref_vol)
    else:
        raise ValueError("The variable type_weight \"{}\""
                         "is not defined.".format(type_weight))

    new_weights = tf.where(tf.math.is_inf(weights), tf.zeros_like(weights), weights)
    weights = tf.where(tf.math.is_inf(weights), tf.ones_like(weights) * tf.reduce_max(new_weights), weights)

    generalised_dice_numerator = 2 * K.sum(weights * intersect)
    generalised_dice_denominator = K.sum(weights * (seg_vol + ref_vol)) + K.epsilon()  # Added K.epsilon() for numerical stability
    generalised_dice_score = generalised_dice_numerator / generalised_dice_denominator

    return generalised_dice_score



def gen_dice_multilabel(y_true, y_pred, numLabels=5):
    """
    It computes the generalised dice coefficient loss making an average for each class (binary case)
    for a multi-class problem with numLabels classes
    :param y_true: true labels (ground truth)
    :param y_pred: predicted labels
    :param numLabels: number of classes
    :return: dice coefficient loss for a multi-class problem
    """
    dice = 0
    for index in range(numLabels):
        dice += generalised_dice_coef(y_true[:,:,:,index], y_pred[:,:,:,index], type_weight='Square')
    return 1. - dice/5.


def dice_coef(y_true, y_pred, smooth = 1.):
    """
    It computes the dice coefficient
    :param y_true: true labels (ground truth)
    :param y_pred: predicted labels
    :param smooth: parameter to ensure stability
    :return: dice coefficient score between y_true and y_pred
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_multilabel(y_true, y_pred, numLabels=5):
    """
    It computes the dice coefficient loss making an average for each class (binary case)
    for a multi-class problem with numLabels classes
    :param y_true: true labels (ground truth)
    :param y_pred: predicted labels
    :param numLabels: number of classes
    :return: dice coefficient loss for a multi-class problem
    """
    dice = 0
    for index in range(numLabels):
        dice += dice_coef(y_true[:,:,:,index], y_pred[:,:,:,index])
    return 1. - dice/5.
