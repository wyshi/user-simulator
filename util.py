import numpy as np

def oneHotLabel(label, dim):
    """

    :param label:
    :return: np.array((BATCH_SIZE, OUTPUT_DIM))
    """
    oneHot = np.zeros(dim)
    if label < 0 or label > dim:
        return oneHot

    oneHot[label] = 1

    return oneHot