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
    # Save image name for saving with suffix
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

            # Convert Hue to degrees
            hsl_img[i, j] = [h * 360, s, l]

    return hsl_img

def convert_hsl_to_rgb(img_2d):
    ''' Convert HSL image to RGB image
    returns a 2D image (numpy array)
    '''
    rgb_img = np.zeros_like(img_2d, dtype=float)
    for i in range(img_2d.shape[0]):
        for j in range(img_2d.shape[1]):
            h, s, l = img_2d[i, j] / [360.0, 1, 1]
            r, g, b = colorsys.hls_to_rgb(h, l, s) # Returns values in range [0, 1]

            # Convert RGB values to range [0, 255]
            rgb_img[i, j] = [r * 255, g * 255, b * 255]

    # Clip values to (0, 255) range for safety
    rgb_img = np.clip(rgb_img, 0, 255) 
    return rgb_img.astype(np.uint8)


# Function 1: Brighten
def brighten_image(img_2d, multiplier=1.2):
    new_image = np.zeros_like(img_2d)
    for i in range(img_2d.shape[0]):
        for j in range(img_2d.shape[1]):
            new_rgb = multiplier * img_2d[i][j]           # Scale up RGB values   
            new_image[i][j] = np.clip(new_rgb, 0, 255)  # Clip RGB values to range (0, 255)

    return new_image

# Function 2: Contrast
def increase_contrast(img_2d, multiplier=1.05, boost_saturation=0.05):
    hsl_img = convert_rgb_to_hsl(img_2d)

    # Get Lightness values
    L = hsl_img[:, :, 2]
    L_min = np.min(L)
    L_max = np.max(L)
    
    # Normalize
    if L_max > L_min:
        L_norm = (L - L_min) / (L_max - L_min)
        hsl_img[:, :, 2] = np.clip(multiplier * L_norm, 0, 1)
    else:
        hsl_img[:, :, 2] = np.clip(multiplier * L, 0, 1)

    # Boost saturation to minimize color loss
    if boost_saturation != 0:
        hsl_img[:, :, 1] = np.clip((multiplier + boost_saturation) * hsl_img[:, :, 1], 0, 1)
    
    # Convert back to RGB for result
    return convert_hsl_to_rgb(hsl_img)

def process_image(img_2d, func=[1, 2, 3,...]):
    ''' Process image with a list of functions
    func: a list of functions to apply to the image
    return processed 2D image
    '''

    # No img_2d array or no saved path
    if img_2d is None or not saved_image_path or not saved_image_path[0]:
        raise ValueError("Please call read_img() for img_2d parameter.")
    
    # Map functions to list of func
    func_map = {
        1: brighten_image,
        2: increase_contrast
    }
    branching_func = [3, 4, 5, 7]

    # Map suffix to list of func
    suffix_map = {
        1: "_brighten",
        2: "_contrast",
        3: ("_verticalFlip", "_mirrored"),
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
    
    # Count repetitions for each function
    total_rep = {}
    for f in new_func:
        total_rep[f] = total_rep.get(f, 0) + 1

    # Check suffix in path
    def suffix_already_in_path(p, s):
        dot_idx = p.rfind('.')
        if dot_idx == -1:
            return s in p
        return s in p[:dot_idx]
    
    branches = [(img_2d, saved_image_path[0])]
    for idx, f in enumerate(new_func):
        new_branches = []
        for img, path in branches:
            if f in suffix_map:
                suffixes = suffix_map[f]
                if isinstance(suffixes, str):
                    suffixes = [suffixes]
            else:
                suffixes = [""]
            
            # Handling branching functions
            if f in branching_func:
                if f in func_map:
                    results = func_map[f](img)
                else:
                    results = [img, img]

                for i in range(2):
                    suffix = suffixes[i]
                    if total_rep[f] > 1:
                        suffix = f"{suffix}{total_rep[f]}"

                    if not suffix_already_in_path(path, suffix):
                        dot_idx = path.rfind('.')
                        if dot_idx == -1:
                            new_path = path + suffix
                        else:
                            new_path = path[:dot_idx] + suffix + path[dot_idx:]
                    else:
                        new_path = path
                    new_branches.append((results[i], new_path))

            # Single result functions
            else:
                if f in func_map:
                    result = func_map[f](img)
                else:
                    result = img

                suffix = suffixes[0]
                if total_rep[f] > 1:
                    suffix = f"{suffix}{total_rep[f]}"

                if not suffix_already_in_path(path, suffix):
                    dot_idx = path.rfind('.')
                    if dot_idx == -1:
                        new_path = path + suffix
                    else:
                        new_path = path[:dot_idx] + suffix + path[dot_idx:]
                else:
                    new_path = path
                new_branches.append((result, new_path))
            
        branches = new_branches
        
    # Save all resulting images
    if save_flag:
        for img, path in branches:
            Image.fromarray(img.astype(np.uint8)).save(path)
    
    results = []
    for img, _ in branches:
        results.append(img.astype(np.uint8))
    return results
    
# Main function
def main():
    img_path = "images/fall.png"
    original_img = read_img(img_path)

    results = [original_img]
    results.extend(
        process_image(original_img, func=[0, 1, 1])
    )
    results.extend(
        process_image(original_img, func=[0, 2, 2])
    )
    results.extend(
        process_image(original_img, func=[0, 2, 1])
    )
    
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