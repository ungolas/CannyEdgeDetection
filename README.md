# Canny Edge Detector

This repository contains an implementation of a Canny Edge Detector for the Machine Vision course. The code demonstrates several stages of edge detection:

1. **Gaussian Blurring** – Implemented in `blur_gauss` from blur_gauss.py
2. **Edge Detection (Sobel Filter)** – Implemented in `sobel` from sobel.py
3. **Non-Maxima Suppression** – Implemented in `non_max` from non_max.py
4. **Hysteresis Thresholding** – Manual (`hyst_thresh`) and automatic (`hyst_thresh_auto`)

Additional helper functions for image display and analysis can be found in helper_functions.py.

## Repository Structure

```
.
├── blur_gauss.py          # Gaussian blurring of the input image.
├── helper_functions.py    # Functions to display and analyze images.
├── hyst_auto.py           # Automatic hysteresis thresholding wrapper.
├── hyst_thresh.py         # Manual hysteresis thresholding.
├── image/                 # Folder containing test images.
│   ├── bad-hyst.jpg
│   ├── beardman.jpg
│   ├── circle.jpg
│   ├── good-hyst.jpg
│   ├── parliament.jpg
│   └── rubens.jpg
├── main.py                # Main entry point using automatic thresholding.
├── non_max.py             # Non-Maxima Suppression.
├── old_main.py            # Older version of the main program.
└── sobel.py               # Sobel filter for computing gradients and orientations.
```

## Requirements

- Python 3.x
- [OpenCV](https://opencv.org/) (`cv2`)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)

Install dependencies using pip:

```sh
pip install opencv-python numpy matplotlib
```

## How to Run

Run the main program using:

```sh
python main.py
```

or test an older version using:

```sh
python old_main.py
```

The program reads an image (for example `rubens.jpg` in the image folder), applies a series of processing steps and displays the results.

## Code Overview

- **`blur_gauss`:** Blurs an image using a Gaussian kernel with a specified sigma.
- **`sobel`:** Applies the Sobel operator and returns the gradient magnitude and orientation.
- **`non_max`:** Uses non-maxima suppression to thin out the edges.
- **`hyst_thresh`:** Filters edges based on connected components using manual thresholds.
- **`hyst_thresh_auto`:** Automatically computes thresholds for hysteresis based on proportions.
- **`show_image`:** Helper function to display images using OpenCV or Matplotlib.
- Additional helper functions in helper_functions.py provide plotting capabilities and noise addition.