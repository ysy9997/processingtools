import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    zeros = np.zeros((8000, 400, 3), np.uint8)
    for i in range(80):
        if i < int(80 / 3): color = [i * 9, 255, 0]
        elif i < int(80 / 3 * 2): color = [255, 255 - ((i - int(80 / 3)) * 9), 0]
        else: color = [255, 0, (i - int(80 / 3 * 2)) * 9]

        zeros[i * 100:(i + 1) * 100, :] = color

    COCO_CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                    'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                    'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                    'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                    'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                    'scissors', 'teddy bear', 'hair drier', 'toothbrush')

    fig = plt.imshow(zeros)
    plt.yticks(np.linspace(50, 7950, 80, dtype=int),  COCO_CLASSES, fontsize=6)
    fig.axes.get_xaxis().set_visible(False)
    plt.show()
