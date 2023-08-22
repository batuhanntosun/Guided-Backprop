import numpy as np
import cv2


def range_norm(img):
    min = img.min()
    max = img.max()
    eps = 1e-6
    return (img-min)/(max-min+eps)


def grad2heatmapped(input_image, grad_image, ratio):
    input_image = np.array(input_image)
    grad_image = np.array(grad_image)
    
    # Invert negative pixels
    grad_image[grad_image<100] += 128

    # Apply thresholding and blur to obtain heatmap
    th = cv2.threshold(grad_image, 140, 255, cv2.THRESH_BINARY)[1]
    blur = cv2.GaussianBlur(th, (11,11), 11)
    heatmap = cv2.applyColorMap(blur, cv2.COLORMAP_JET)

    # Apply edge padding to heatmap to have 256x256 size
    heatmap = np.pad(heatmap, ((16,16),(16,16),(0,0)), 'edge')

    # Upsample heatmap to input_image size
    heatmap = cv2.resize(heatmap, (input_image.shape[1], input_image.shape[0]))

    # Superimpose heatmap on input_image
    heatmapped = cv2.addWeighted(input_image, 1-ratio, heatmap, ratio, 0)

    return heatmapped
