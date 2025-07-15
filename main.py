from PIL import Image               # for reading and writing images
import numpy as np                  # for matrix computation
import matplotlib.pyplot as plt     # for showing images
import colorsys                     # for converting RGB to HSL

# Helper Functions
def read_img(img_path):
    ''' Read image from img_path
    returns a 2D image (numpy array)
    '''
    img = Image.open(img_path)
    img_2d = np.array(img)
    return img_2d

def show_img(img_2d):
    ''' Show image
    '''
    if (img_2d is not None):
        h, w, _ = img_2d.shape
        print("Image size:", h, "x", w)
        plt.imshow(img_2d)
    else:
        print("Image not found")

def convert_rgb_to_hsl(img_2d):
    ''' Convert RGB image to HSL image
    returns a 2D image (numpy array)
    '''
    hsl_img = np.zeros_like(img_2d, dtype=float)
    for i in range(img_2d.shape[0]):
        for j in range(img_2d.shape[1]):
            r, g, b = img_2d[i, j] / 255.0
            h, l, s = colorsys.rgb_to_hls(r, g, b) # Returns values in range [0, 1]

            # Convert Hue to degrees and Saturation, Lightness to percentage
            hsl_img[i, j] = [h * 360, s * 100, l * 100]

    return hsl_img

def convert_hsl_to_rgb(img_2d):
    ''' Convert HSL image to RGB image
    returns a 2D image (numpy array)
    '''
    rgb_img = np.zeros_like(img_2d, dtype=float)
    for i in range(img_2d.shape[0]):
        for j in range(img_2d.shape[1]):
            h, s, l = img_2d[i, j] / [360.0, 100.0, 100.0]
            r, g, b = colorsys.hls_to_rgb(h, l, s) # Returns values in range [0, 1]

            # Convert RGB values to range [0, 255]
            rgb_img[i, j] = [r * 255, g * 255, b * 255]

    # Clip values to (0, 255) range for safety
    rgb_img = np.clip(rgb_img, 0, 255) 
    return rgb_img.astype(np.uint8)

def process_image(img_2d, func=[1, 2, 3,...]):
    ''' Process image with a list of functions
    func: a list of functions to apply to the image
    return processed 2D image
    '''

# Main function
def main():
    img_path = "images/dog.jpg"
    img_2d = read_img(img_path)
    show_img(img_2d)

if __name__ == "__main__":
    main()