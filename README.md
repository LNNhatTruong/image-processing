# image-processing
[Project][Applied Mathematics and Statistics]
Image Processing Tool developed with Python

# Requirement
Run `pip install -r requirements.txt` to install dependencies

# Functions
## 0 - Save Image with Suffixes
Save the resulting image(s) with suffixes that describe the applied processing steps.

### 🔧 Behavior:
- Save the result with suffixes reflecting the operations performed.
- If multiple branching results are generated (e.g. from both blur and sharpen), each result is saved separately.
- If a processing step is repeated multiple times, the suffix will include the number of repetitions.
- Some steps, such as cropping by frame, are applied only once by design and will not include a repetition count in the filename.

### 📝 Example:
- **Input file**: `example.png`
- **Selected functions**:
    + `1` — Brighten x2
    + `5` — Blur and Sharpen (produces 2 variations)
- **Output files**:
    + `example_brighten2_blur.png`
    + `example_brighten2_sharpen.png`

## 1 - Brighten
Increases the brightness of the image by scaling the pixels value with a default multiplier of `1.25` for each repetition.

## 2 - Contrast
Enhances the contrast of the image by stretching the difference between light and dark areas. Each repetition multiplies the pixel intensity around the midpoint by a default contrast factor of `1.25`, making shadows darker and highlights brighter.

## 3 - Flip Image
Flips the image in two separate ways: vertically (upside down) and horizontally (mirror effect).

## 4 - Convert Image to Grayscale and Sepia
Converts the image in two separate styles:
- Grayscale: Removes color, leaving only shades of gray.
- Sepia: Applies a warm brown tone for a vintage look.

## 5 - Blur and Sharpen
Applies two separate filters:
- Blur: Softens the image by reducing noise and detail.
- Sharpen: Enhances edges to make details more distinct.

## 6 - Crop Image by Size
Crops the image to the central 1/4 region, keeping the original aspect ratio intact.

## 7 - Crop Image by Frame
Applies two distinct cropping effects based on shape:
- Circular Crop: Cuts a circle from the image that fits perfectly within its dimensions.
- Double Ellipse Crop: Overlays two diagonal ellipses that fit the image and intersect in the center.

# Project Structure
.image-processing
├── images/              // Input images used for processing
├── results/             // Output images saved after processing
├── project/             // Project Lab Files

├── .gitignore           // Git ignore rules
├── main.py              // Main script to run the image processing
├── README.md            // Project documentation
├── requirements.txt     // Python dependencies