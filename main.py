from PIL import Image               # for reading and writing images
import numpy as np                  # for matrix computation
import matplotlib.pyplot as plt     # for showing images
import colorsys                     # for converting RGB to HSL

save_folder = "results/"
input_folder = "images/"
saved_image_path = []

# Helper Functions
def read_img(img_path):
    ''' Read image from img_path
    returns a 2D image (numpy array)
    '''
    # Save image path for saving with suffix
    global saved_image_path
    if input_folder in img_path:
        if '/' in img_path:
            filename = img_path.split('/')[-1]
        else:
            filename = img_path.split('\\')[-1]
        new_path = save_folder + filename
    else:
        new_path = img_path
    saved_image_path.clear()
    saved_image_path.append(new_path)
    
    img = Image.open(img_path).convert('RGB')
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
            hsl_img[i, j] = [h, s, l]

    return hsl_img

def convert_hsl_to_rgb(img_2d):
    ''' Convert HSL image to RGB image
    returns a 2D image (numpy array)
    '''
    rgb_img = np.zeros_like(img_2d, dtype=float)
    for i in range(img_2d.shape[0]):
        for j in range(img_2d.shape[1]):
            h, s, l = img_2d[i, j]
            r, g, b = colorsys.hls_to_rgb(h, l, s) # Returns values in range [0, 1]

            # Convert RGB values to range [0, 255]
            rgb_img[i, j] = [r * 255, g * 255, b * 255]

    # Clip values to (0, 255) range for safety
    rgb_img = np.clip(rgb_img, 0, 255) 
    return rgb_img.astype(np.uint8)


# Function 1: Brighten
def brighten_image(img_2d, multiplier=1.2):
    return np.clip(img_2d * multiplier, 0, 255).astype(np.uint8)

# Function 2: Contrast
def increase_contrast(img_2d, multiplier=1.2):
    hsl_img = convert_rgb_to_hsl(img_2d)

    # Get Lightness values
    L = hsl_img[:, :, 2]
    L_max = np.max(L)
    L_min = np.min(L)
    L_mid = (L_max + L_min) / 2

    L_new = multiplier * (L - L_mid)  + L_mid

    hsl_img[:, :, 2] = np.clip(L_new, 0, 1)

    # Convert back to RGB for result
    return convert_hsl_to_rgb(hsl_img).astype(np.uint8)

# Function 3: Flip Image
def flip_image(img_2d):
    return img_2d[::-1], img_2d[:, ::-1]

# Function 4: Grayscale and Sepia
def convert_grayscale_sepia(img_2d):
    avg_value = np.average(img_2d, axis=2)
    grayscale_img = np.copy(img_2d)
    grayscale_img[:, :, 0] = avg_value
    grayscale_img[:, :, 1] = avg_value
    grayscale_img[:, :, 2] = avg_value

    sepia_filter = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131]
    ])
    sepia_img = np.dot(img_2d[...,:3], sepia_filter.T)
    sepia_img = np.clip(sepia_img, 0, 255).astype(np.uint8)

    return grayscale_img, sepia_img

# Function 5: Blur and Sharpen
def blur_and_sharpen(img_2d):
    def get_masked_value(img_2d, x, y, mask):
        img_h, img_w, _ = img_2d.shape
        mask_h, mask_w = mask.shape
        mid = int(mask_h / 2)
        result = np.zeros(3, dtype=float)
        for i in range(mask_h):
            for j in range(mask_w):
                xi = x + i - mid
                yj = y + j - mid
                if xi not in range(0, img_h) or yj not in range(0, img_w): # Check out of bound values
                    continue
                result += img_2d[xi, yj] * mask[i, j]
        return np.clip(result, 0, 255)

    # Use 3x3 gaussian blur mask
    gaussian_blur_mask = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ], dtype=float)
    gaussian_blur_mask /= gaussian_blur_mask.sum()

    # Sharpen mask
    sharpen_mask = np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]
    ], dtype=int)

    # Since there are negative values when sharpening
    # img_2d needs to be float datatype
    float_img_2d = np.copy(img_2d).astype(float)

    blur_img = np.zeros_like(float_img_2d)
    sharpen_img = np.zeros_like(float_img_2d)
    h, w, _ = float_img_2d.shape
    for i in range(h):
        for j in range(w):
            blur_img[i][j] = get_masked_value(float_img_2d, i, j, gaussian_blur_mask)
            sharpen_img[i][j] = get_masked_value(float_img_2d, i, j, sharpen_mask)
    
    # Convert back before returning
    return blur_img.astype(np.uint8), sharpen_img.astype(np.uint8)

# Function 6: Crop by size
def crop_by_size(img_2d, size=1/2):
    h, w, _ = img_2d.shape
    new_h, new_w = int(h * size), int(w * size)
    center_h, center_w = h // 2, w // 2
    top = max(center_h - new_h // 2, 0)
    left = max(center_w - new_w // 2, 0)
    bottom = top + new_h
    right = left + new_w
    cropped_img = img_2d[top:bottom, left:right].copy()
    return cropped_img

# Function 7: Crop by frames
def crop_by_frame(img_2d):
    h, w, _ = img_2d.shape
    Y, X = np.ogrid[:h, :w]
    center_x, center_y = w // 2, h // 2

    # Drawing a circle mask
    radius = min(h, w) // 2
    circle_mask = (X - center_x) ** 2 + (Y - center_y) ** 2 <= radius ** 2
    circular_crop = img_2d * circle_mask[:, :, np.newaxis]

    # Drawing a double ellipse mask
    fit_factor = 0.87
    angle = np.deg2rad(45)
    square_diag = np.sqrt(2) * min(h, w)
    b = min(h, w) / 3
    a = square_diag / 2 * fit_factor

    Xr1 = (X - center_x) * np.cos(angle) + (Y - center_y) * np.sin(angle)
    Yr1 = (X - center_x) * np.sin(angle) - (Y - center_y) * np.cos(angle)
    ellipse1 = (Xr1 / a) ** 2 + (Yr1 / b) ** 2 <= 1

    Xr2 = (X - center_x) * np.cos(-angle) + (Y - center_y) * np.sin(-angle)
    Yr2 = (X - center_x) * np.sin(-angle) - (Y - center_y) * np.cos(-angle)
    ellipse2 = (Xr2 / a) ** 2 + (Yr2 / b) ** 2 <= 1

    double_ellipse_mask = ellipse1 | ellipse2
    double_ellipse_crop = img_2d * double_ellipse_mask[:, :, np.newaxis]

    return circular_crop.astype(np.uint8), double_ellipse_crop.astype(np.uint8)

def process_image(img_2d, func=[1, 2, 3,...]):
    ''' Process image with a list of functions
    func: a list of functions to apply to the image
    return processed 2D image
    '''

    # No img_2d array
    if img_2d is None:
        raise ValueError("No image array found!")
    
    # Map functions to list of func
    func_map = {
        1: brighten_image,
        2: increase_contrast,
        3: flip_image,
        4: convert_grayscale_sepia,
        5: blur_and_sharpen,
        6: crop_by_size,
        7: crop_by_frame
    }

    # Map suffix to list of func
    suffix_map = {
        1: "_brighten",
        2: "_contrast",
        3: ("_verticalFlip", "_mirroredFlip"),
        4: ("_grayscale", "_sepia"),
        5: ("_blur", "_sharpen"),
        6: "_cropped",
        7: ("_circularCrop", "_doubleEllipseCrop")
    }

    save_flag = (0 in func)
    # Remove 0
    new_func = [f for f in func if f != 0]

    # Check empty function list
    if not new_func:
        return [img_2d]

    if save_flag == True and (not saved_image_path  or saved_image_path[0] is None):
        print("No path found to save image!")
        print("Using default: image_<processed_function>.png")
        original_path = "image.png"
    else:
        original_path = saved_image_path[0]

    
    save_img = [] # Save path to append suffix
    results = []
    for f in new_func:
        if f in func_map:
            processed_img = func_map[f](img_2d)

            # Check return error
            if processed_img is None:
                print(f"Function {f} returned nothing!")
                processed_img = [img_2d]

            # If the processed image is a single image, make it a list for extending
            if isinstance(processed_img, np.ndarray):
                processed_img = [processed_img]

            results.extend(processed_img)

            if save_flag:
                suffixes = suffix_map[f]
                if isinstance(suffixes, str):
                    suffixes = [suffixes]
                
                for i, img in enumerate(processed_img):
                    dot_idx = original_path.rfind('.')
                    suffix = suffixes[i] if i < len(suffixes) else f"_output{i}"
                    if dot_idx == -1:
                        save_path = original_path + suffix
                    else:
                        save_path = original_path[:dot_idx] + suffix + original_path[dot_idx:]
                    save_img.append(save_path)
    
    # Save image with suffix
    if save_flag:
        for idx, img in enumerate(results):
            Image.fromarray(img.astype(np.uint8)).save(save_img[idx])
    
    return results
    
# Main function
def main():
    img_path = "images/fall.png"
    original_img = read_img(img_path)

    results = [original_img]
    results.extend(
        process_image(original_img, func=[0, 1, 2, 3, 4, 5, 6, 7])
    )

    # Display images
    n = len(results)
    plt.figure(figsize=(5 * n, 5))
    for i, img in enumerate(results):
        plt.subplot(1, n, i + 1)
        plt.imshow(img)
        plt.axis('off')
        if i != 0:
            plt.title(f"Result {i}")
        else:
            plt.title("Original Image")
    plt.show()
    
if __name__ == "__main__":
    main()