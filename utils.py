def range_norm(img):
    min = img.min()
    max = img.max()
    eps = 1e-6
    return (img-min)/(max-min+eps)