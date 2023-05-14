import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


def frequency_matrix_2D(d__ss, threshold, normalized):
    d__ss = np.array(d__ss)
    d__ss = (d__ss - min(d__ss)) / ((max(d__ss) - min(d__ss)) * 1.000001)
    binary_vector = np.maximum(np.minimum(np.int64(d__ss - threshold + 1), 1), 0)
    matrix = np.zeros((2, 2))
    for i in range(len(d__ss) - 1):
        matrix[binary_vector[i], binary_vector[i + 1]] += 1
    if normalized:
        matrix /= matrix.sum(axis=1)[:, np.newaxis]
    return matrix

    
def form_groups(vector, threshold_array, graph, x_label, title, x_axis_format):
    detection_fisher = []
    detection = []
    for i in threshold_array:
        matrix = frequency_matrix_2D(vector, i, False)
        _, p_value = stats.fisher_exact(matrix)
        detection_fisher.append(np.log(p_value))
        p = matrix.sum(axis=1)[0] / matrix.sum()
        if p == 1 or p == 0:
            detection.append(1)
        else:
            detection.append((matrix[1][0]) / (matrix.sum() * (p * (1 - p))))

    minim = np.min(vector)
    diff = np.max(vector) - minim
    min_k = np.argmin(detection) * diff + minim
    min_fisher = np.argmin(detection_fisher) * diff + minim

    if graph:
        xticks_labels = [x_axis_format % (minim + diff * pipi) for pipi in threshold_array]

        plt.plot(detection)
        plt.xlabel(x_label)
        plt.title(title)
        if len(threshold_array) > 40:
            plt.xticks(np.arange(len(threshold_array))[::len(threshold_array) // 10], xticks_labels[::len(threshold_array) // 10])
        else:
            plt.xticks(np.arange(len(threshold_array))[::1], xticks_labels[::1])
        plt.ylabel("k")
        plt.savefig("group_detection - k " + x_label + title + ".png", dpi=500)
        plt.show()
        plt.close()

        plt.plot(detection_fisher)
        plt.xlabel(x_label)
        plt.title(title)
        if len(threshold_array) > 40:
            plt.xticks(np.arange(len(threshold_array))[::len(threshold_array) // 10], xticks_labels[::len(threshold_array) // 10])
        else:
            plt.xticks(np.arange(len(threshold_array))[::1], xticks_labels[::1])
        plt.ylabel("log-fisher exact test")
        plt.savefig("group_detection - log-fisher " + x_label + title + ".png", dpi=500)
        plt.show()
        plt.close()

    return detection