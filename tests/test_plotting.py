import numpy as np

from utils.plotting import plot_multi_auc


def test_plot_multi_auc():
    y_true = np.random.randint(0, 2, size=(100, 10))
    y_pred = np.random.randint(0, 2, size=(100, 10))
    classes = [str(x) for x in range(10)]
    plot_multi_auc(y_true, y_pred, classes)

