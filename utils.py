from skimage import io
import numpy as np
import cv2
import matplotlib.pyplot as plt

def load_image(img_path, mask=False):
    img = io.imread(img_path)
    img = cv2.resize(img, (256, 256))
    img = np.array(img, dtype=np.float32)

    img /= 255
        
    if mask:
        img[img > 0.5] = 1
        img[img <= 0.5] = 0
    # else:
    #     img -= img.mean()
    #     img /= img.std()

    return np.array(img, dtype=np.float32)

def normalize(img):
    return  np.array(255 * np.array((img - img.min()) / (img.max() - img.min()), dtype=np.float32), dtype=np.int32)

def power_law(img, gamma=0.1):
    im = img.copy()
    im = im ** gamma

    return normalize(im)

def power_law_combination(img, gammas=[0.01, 0.1, 0.2, 0.8]):
    image = 0

    for gamma in gammas:
        image += power_law(img, gamma)

    return normalize(image)

def power_law_demonstration(image, mask, gammas=[0.8, 0.9, 0.85, 0.88]):
    fig,axes = plt.subplots(1,len(gammas)+3, figsize=(30,25))

    axes[0].title.set_text('Image')
    axes[0].imshow(image)

    axes[1].title.set_text('Image + Mask')
    im = image.copy()
    im[mask==1] = 255
    axes[1].imshow(im)

    axes[2].title.set_text('Combination')
    axes[2].imshow(power_law_combination(image, gammas))

    for index, gamma in enumerate(gammas):
        axes[index + 3].title.set_text(f'Gamma = {gamma}')
        axes[index + 3].imshow(power_law(image, gamma))

    fig.tight_layout()


def log_transform(img, balance=0.8):
    img = img.copy()
    img *= 255
    
    c = 1 / (np.log(1 + np.max(img)))
    log_transformed = c * np.log(1 + img) / balance

    return normalize(log_transformed)

def log_transform_combination(img, balances=[0.75, 0.799, 0.8, 0.81, 0.805]):
    image = 0

    for balance in balances:
        image += log_transform(img, balance)
    
    return normalize(image)

def log_transform_demonstration(image, mask, bs=[0.7, 0.8, 0.9]):
    fig,axes = plt.subplots(1, 3 + len(bs), figsize=(20,15))

    axes[0].title.set_text('Image')
    axes[0].imshow(image)

    axes[1].title.set_text('Image + Mask')
    im = image.copy()
    im[mask==1] = 255
    axes[1].imshow(im)

    axes[2].title.set_text('Log Combination')
    axes[2].imshow(log_transform_combination(image.copy(), bs))

    for index, bal in enumerate(bs):
        axes[3 + index].title.set_text(f'Log Transform, {bal}')
        axes[3 + index].imshow(log_transform(image, bal))