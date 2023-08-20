def range_norm(img):
    min = img.min()
    max = img.max()
    eps = 1e-6
    return (img-min)/(max-min+eps)

def denormalize(img, mean, std):
    return img*std.view(-1,1,1) + mean.view(-1,1,1)