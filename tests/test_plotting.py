import keras
import numpy as np
from utils.plotting import plot_multi_auc

def test_plot_multi_auc():
    num_classes = 10
    y_true = np.random.randint(0, num_classes, size=(100, 1))
    y_true = keras.utils.to_categorical(y_true, num_classes=num_classes)
    y_pred = np.random.rand(100, num_classes)
    classes = [str(x) for x in range(num_classes)]
    plot_multi_auc(y_true, y_pred, classes)

