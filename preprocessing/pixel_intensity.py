from matplotlib import pyplot as plt


def plot_pixel_intensity(image, path='./pixel_intensity_before_normalization.png'):
    """
    Plot an histogram representing the pixel intensity of an image

    :param path: Path in which save the pixel intensity graphic
    :param image: Image to plot
    :return:

    """

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.axis('off')
    histo = plt.subplot(1, 2, 2)
    histo.set_ylabel('Count')
    histo.set_xlabel('Pixel Intensity')
    n_bins = 30
    plt.hist(image[:, :, 0].flatten(), bins=n_bins, lw=0, color='r', alpha=0.5)
    plt.hist(image[:, :, 1].flatten(), bins=n_bins, lw=0, color='g', alpha=0.5)
    plt.hist(image[:, :, 2].flatten(), bins=n_bins, lw=0, color='b', alpha=0.5)
    plt.savefig(path)
    plt.show()
