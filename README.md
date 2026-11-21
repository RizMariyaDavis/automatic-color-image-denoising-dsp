# Automatic Color Image Noise Detection, Denoising and Enhancement Using DSP

## Overview

This project focuses on detecting, removing, and reducing noise in color images using Digital Signal Processing (DSP) techniques.

It identifies three common types of image noise: Gaussian, Salt-and-Pepper, and Periodic noise. Based on the detected noise type, the system automatically selects an appropriate filtering technique to restore the image.

After denoising, contrast stretching and Laplacian sharpening are applied to improve the overall image quality. The performance of the system is evaluated using standard image quality metrics.

## Features

- Automatic detection of Gaussian, Salt-and-Pepper, and Periodic noise.
- Noise analysis using spatial-domain and frequency-domain techniques.
- Automatic selection of suitable denoising filters.
- Color image reconstruction and enhancement.
- Image quality evaluation using MSE, PSNR, and SSIM.
- Visual comparison of original, noisy, denoised, and enhanced images.

## Noise Detection and Removal

The system uses different detection techniques depending on the type of noise.

| Noise Type | Detection Technique | Denoising Filter |
|---|---|---|
| Gaussian | Statistical analysis and noise estimation | Wiener Filter |
| Salt-and-Pepper | Extreme pixel detection and neighborhood analysis | Median Filter |
| Periodic | 2D Fast Fourier Transform (FFT) | Notch-Reject Filter |

If the system cannot confidently identify a noise type, it uses a bilateral filter as a fallback.

## Image Enhancement

After denoising, two enhancement techniques are applied:

**Contrast Stretching:** Improves image contrast by adjusting the intensity range.

**Laplacian Sharpening:** Enhances edges and fine details to improve image clarity.

## Performance Evaluation

The restored images are compared with the original image using the following metrics:

- **MSE (Mean Squared Error):** Measures the difference between the original and processed images. Lower values indicate better results.
- **PSNR (Peak Signal-to-Noise Ratio):** Measures image reconstruction quality. Higher values indicate better results.
- **SSIM (Structural Similarity Index):** Evaluates structural similarity between images. Values closer to 1 indicate better similarity.

## Technologies Used

- Python
- NumPy
- SciPy
- OpenCV
- Pillow
- Matplotlib
- scikit-image

## Running the Project

Install the required libraries:

```bash
pip install numpy scipy opencv-python pillow matplotlib scikit-image
```

Run the main Python program:

```bash
python main.py
```

The program requires an original color image as input. It generates noisy versions of the image and processes them through the detection, denoising, and enhancement stages.

## Output

The program generates visual results and performance reports, including:

- Original and noisy images.
- Noise detection results.
- FFT spectrum for periodic noise analysis.
- Denoised and enhanced images.
- MSE, PSNR, and SSIM comparisons.
- Noise classification results.

## Applications

- Medical Image Processing
- Digital Photography
- Surveillance Systems
- Remote Sensing
- Industrial Image Inspection
- Computer Vision

## Future Scope

- Support for additional noise types such as Speckle and Poisson noise.
- Machine Learning and Deep Learning-based noise classification.
- Real-time image and video processing.
- GPU acceleration for faster processing.
- Development of a graphical user interface.
