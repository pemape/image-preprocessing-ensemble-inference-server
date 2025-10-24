# Fundus Image Preprocessing Configuration for Ensemble Model

This repository implements the preprocessing pipeline described in the paper **"Ensemble of pre-processing techniques with CNN for diabetic retinopathy detection"** ([IET Image Processing, 2024](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/ipr2.12987)).

## Overview

The preprocessing system takes a single fundus image with arbitrary resolution and generates **5 variants** through a systematic preprocessing pipeline:

1. **Black Border Clipping** - Applied to the original image first
2. **Five Image Variants** generated from the clipped image:
   - Original clipped image (baseline)
   - RGB-CLAHE enhanced image
   - Ben Graham enhanced image (min-pooling enhancement)
   - Lab-CLAHE enhanced image
   - MaxGreenGsc (Maximum Green Grayscale) enhanced image
3. **Resolution Standardization** - All variants resized to target resolution (default: 500×500)

## Preprocessing Pipeline

### Step 1: Black Border Clipping

Fundus images often contain black borders that need to be removed for optimal processing:

```yaml
black_border_clipping:
  enabled: true
  method: "adaptive_threshold"  # Options: adaptive_threshold, contour_detection, morphological
  adaptive_threshold:
    gaussian_kernel_size: 11
    threshold_constant: 2
    erosion_iterations: 2
    dilation_iterations: 3
```

**Methods Available:**
- **Adaptive Threshold**: Uses adaptive thresholding with morphological operations
- **Contour Detection**: Finds the largest contour representing the fundus region
- **Morphological**: Uses morphological operations for border detection

### Step 2: Five Image Variants Generation

#### Variant 1: Original (Baseline)
```yaml
original:
  enabled: true
  description: "Original fundus image after black border clipping"
```
The clipped image without additional preprocessing, serving as the baseline.

#### Variant 2: RGB-CLAHE
```yaml
rgb_clahe:
  enabled: true
  clip_limit: 2.0
  tile_grid_size: [8, 8]
  # Channel-specific parameters
  green_channel:
    clip_limit: 2.5  # Higher for green channel
```
Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to each RGB channel independently. The green channel often receives higher clip limits due to its importance in fundus imaging.

#### Variant 3: Ben Graham Enhancement (Min-pooling)
```yaml
ben_graham:
  enabled: true
  sigma_x: 10
  enhancement_factor: 4
  brightness_offset: 128
  description: "Ben Graham min-pooling enhancement using Gaussian blur and weighted subtraction"
```
**Min-pooling Enhancement**: Ben Graham method applies Gaussian blur followed by weighted subtraction for feature enhancement:
- `blurred_image = cv2.GaussianBlur(img, (0, 0), sigmaX=10)`
- `final_img = cv2.addWeighted(img, 4, blurred_image, -4, 128)`

This min-pooling technique subtracts the blurred version from the original to enhance local features and improve contrast.

#### Variant 4: Lab-CLAHE
```yaml
lab_clahe:
  enabled: true
  clip_limit: 3.0
  lab_conversion:
    l_channel_only: true    # Apply CLAHE only to L channel
    preserve_ab: true       # Keep A and B channels unchanged
```
Converts the image to LAB color space and applies CLAHE specifically to the lightness (L) channel while preserving color information in A and B channels.

#### Variant 5: MaxGreenGsc (Maximum Green Grayscale)
```yaml
max_green_gsc:
  enabled: true
  method: "max_rgb_with_green_grayscale"
  enhancement:
    histogram_equalization: true
    clahe:
      clip_limit: 4.0
    build_3d_image: true  # Combines max RGB, green channel, and grayscale
```
**Actual Implementation**: Creates a 3D image combining:
- Maximum RGB filter result
- Original green channel
- Grayscale version of the image
- Optionally applies CLAHE to each component

### Step 3: Resolution Standardization

All variants are resized to the target resolution:

```yaml
resolution_processing:
  enabled: true
  target_size: [500, 500]
  interpolation: "bilinear"
  anti_aliasing: true
```

## Additional Available Preprocessing Methods

### Channel Extraction Methods
```python
# Available in your implementation
def get_green_channel(img)    # Extract green channel
def get_red_channel(img)      # Extract red channel  
def get_blue_channel(img)     # Extract blue channel
```

### Advanced Filtering Methods
```python
# Max RGB Filter - zeros out non-maximum channels
def max_3drgb_filter(image)

# Average RGB Filter
def avg_rgb_filter(image)

# Max RGB Filter with statistics
def max_rgb_filter(image, mydict=None)
```

### Composite Image Building
```python
# Build 3D images with different combinations
def build_avg_green_gsc_3d_img(img)           # Average RGB + Green + Grayscale
def build_max_green_gsc_3d_img(img)           # Max RGB + Green + Grayscale  
def build_max_green_gsc_3d_clahe_img(img)     # Above with CLAHE applied
def build_green_gsc_clahe_img(img)            # Green + Grayscale with CLAHE
```

### Channel-Specific CLAHE
```python
# Apply CLAHE to specific channels
def apply_clahe_on_rgb_channels(img)      # All RGB channels
def apply_clahe_on_green_channel(img)     # Green channel only
def apply_clahe_on_red_channel(img)       # Red channel only
def apply_clahe_on_gray_scale(img)        # Grayscale CLAHE
def apply_clahe_on_lab_img(img)          # LAB L-channel CLAHE
def apply_clahe_on_hsv_img(img)          # HSV V-channel CLAHE
```

### Entropy and Edge Detection
```python
def convert_local_entropy(img, channel=None)  # Local entropy calculation
def canny(img)                               # Canny edge detection
```

## Configuration Examples

### Basic Configuration for Research Replication
```yaml
# Minimal configuration matching the paper
general:
  target_resolution: [500, 500]

black_border_clipping:
  enabled: true
  method: "adaptive_threshold"

image_variants:
  original:
    enabled: true
  rgb_clahe:
    enabled: true
    clip_limit: 2.0
  ben_graham:
    enabled: true
    sigma_x: 10
  lab_clahe:
    enabled: true
    clip_limit: 3.0
  max_green_gsc:
    enabled: true
    method: "max_rgb_with_green_grayscale"
```

### Dataset-Specific Configurations

#### APTOS 2019 Dataset
```yaml
dataset_specific:
  aptos2019:
    enabled: true
    black_border_clipping:
      method: "adaptive_threshold"
      gaussian_kernel_size: 13
    rgb_clahe:
      clip_limit: 2.5
    target_resolution: [512, 512]
```

#### DDR Dataset
```yaml
dataset_specific:
  ddr:
    enabled: true
    black_border_clipping:
      method: "contour_detection"
      blur_kernel: 7
    lab_clahe:
      clip_limit: 3.5
```

## Usage

### Python Script Usage

```python
from fundus_preprocessor import FundusPreprocessor
import cv2

# Initialize preprocessor
preprocessor = FundusPreprocessor('preprocessing_config.yaml')

# Load and process image
image = cv2.imread('fundus_image.jpg')
processed_variants = preprocessor.process_image(image)

# Access different variants
original = processed_variants['original']
rgb_clahe = processed_variants['rgb_clahe']
ben_graham = processed_variants['ben_graham']
lab_clahe = processed_variants['lab_clahe']
max_green_gsc = processed_variants['max_green_gsc']
```

### Inference Server Usage

```python
# Start the inference server
python fundus_inference_server.py --config preprocessing_config.yaml --port 8080

# Send image for processing
import requests
import numpy as np

files = {'image': open('fundus_image.jpg', 'rb')}
response = requests.post('http://localhost:8080/process', files=files)
result = response.json()

# Result contains all 5 processed variants
variants = result['processed_variants']
```

## Advanced Configuration

### Ensemble Model Integration
```yaml
ensemble_config:
  models: ["efficientnet_b4", "xception", "resnet50"]
  model_specific:
    enabled: true
    efficientnet_b4:
      target_resolution: [380, 380]
    xception:
      target_resolution: [299, 299]
```

### Quality Control
```yaml
quality_control:
  enabled: true
  input_validation:
    min_resolution: [224, 224]
    max_resolution: [4096, 4096]
    supported_formats: ["jpg", "jpeg", "png", "tiff"]
  processing_validation:
    check_nan_values: true
    validate_range: true
```

### Performance Optimization
```yaml
performance:
  use_gpu_acceleration: true
  parallel_variants: true
  max_workers: 4
  memory_efficient: true
```

### Debugging and Development
```yaml
debug:
  enabled: true
  save_intermediate_images: true
  log_processing_time: true
  intermediate_paths:
    black_border_clipped: "./debug/01_black_border_clipped.jpg"
    rgb_clahe: "./debug/02_rgb_clahe.jpg"
    ben_graham: "./debug/03_ben_graham.jpg"
    lab_clahe: "./debug/04_lab_clahe.jpg"
    max_green_gsc: "./debug/05_max_green_gsc.jpg"
```

## Technical Details

### Black Border Clipping Algorithms

1. **Adaptive Threshold Method**:
   - Applies Gaussian blur followed by adaptive thresholding
   - Uses morphological operations (erosion/dilation) for noise removal
   - Finds the largest connected component as the fundus region

2. **Contour Detection Method**:
   - Applies Gaussian blur and binary thresholding
   - Finds contours and selects the largest one
   - Creates a mask from the selected contour

3. **Morphological Method**:
   - Uses morphological opening and closing operations
   - Applies gradient operations to detect borders
   - Creates binary mask for fundus region extraction

### Image Enhancement Techniques

#### RGB-CLAHE Implementation
```python
# Pseudo-code for RGB-CLAHE
for channel in ['R', 'G', 'B']:
    clahe = cv2.createCLAHE(
        clipLimit=config['clip_limit'], 
        tileGridSize=tuple(config['tile_grid_size'])
    )
    enhanced_channel = clahe.apply(image[:,:,channel])
```

#### Ben Graham Enhancement
```python
# Pseudo-code for Ben Graham Enhancement
blurred_image = cv2.GaussianBlur(img, (0, 0), sigmaX=10)
final_img = cv2.addWeighted(img, 4, blurred_image, -4, 128)
```

#### Lab-CLAHE Processing
```python
# Pseudo-code for Lab-CLAHE
lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
l_channel = lab_image[:,:,0]
enhanced_l = clahe.apply(l_channel)
lab_image[:,:,0] = enhanced_l
result = cv2.cvtColor(lab_image, cv2.COLOR_LAB2RGB)
```

#### MaxGreenGsc Enhancement
```python
# Pseudo-code for MaxGreenGsc
weighted_green = (
    0.2 * image[:,:,0] + 
    0.7 * image[:,:,1] + 
    0.1 * image[:,:,2]
)
enhanced = cv2.equalizeHist(weighted_green.astype(np.uint8))
enhanced = clahe.apply(enhanced)
# Optional: Apply Frangi vessel enhancement
if vessel_enhancement_enabled:
    enhanced = frangi_filter(enhanced)
```

## Validation and Testing

### Input Validation
- Checks image format compatibility
- Validates resolution constraints
- Detects corrupted images

### Processing Validation  
- Monitors for NaN/infinite values
- Validates pixel value ranges
- Ensures proper color space conversions

### Output Validation
- Verifies target dimensions
- Checks channel consistency
- Validates normalization ranges

## Performance Considerations

### Memory Usage
- Processing 5 variants simultaneously requires ~5x memory
- Enable `memory_efficient: true` for sequential processing
- Use `cache_intermediate: false` to reduce memory footprint

### Processing Speed
- GPU acceleration available for OpenCV operations
- Parallel processing of variants when `parallel_variants: true`
- Typical processing time: 2-5 seconds per image (depending on resolution and hardware)

### Scalability
- Designed for single-image processing (inference server)
- Can be extended for batch processing
- Configurable worker threads for concurrent requests

## Troubleshooting

### Common Issues

1. **Black Border Clipping Fails**
   - Try different clipping methods
   - Adjust threshold parameters
   - Check if image has unusual black regions

2. **Memory Errors**
   - Enable memory-efficient processing
   - Reduce target resolution
   - Disable intermediate image saving

3. **Color Space Conversion Issues**
   - Ensure input images are RGB format
   - Check for grayscale images
   - Validate pixel value ranges (0-255 for uint8)

4. **Performance Issues**
   - Enable GPU acceleration
   - Reduce number of variants
   - Optimize worker thread count

### Debug Mode

Enable comprehensive debugging:
```yaml
debug:
  enabled: true
  save_intermediate_images: true
  verbose_logging: true
  create_processing_report: true
```

This saves intermediate results and creates detailed processing reports for analysis.

## Dependencies

```txt
opencv-python>=4.5.0
numpy>=1.19.0
scikit-image>=0.18.0
PyYAML>=5.4.0
Flask>=2.0.0  # For inference server
Pillow>=8.0.0
scipy>=1.7.0  # For Frangi filtering
```

## Citation

If you use this preprocessing pipeline in your research, please cite the original paper:

```bibtex
@article{ensemble_preprocessing_2024,
  title={Ensemble of pre-processing techniques with CNN for diabetic retinopathy detection},
  author={Authors et al.},
  journal={IET Image Processing},
  year={2024},
  url={https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/ipr2.12987}
}
```

## License

This preprocessing configuration system is part of the LBCNN PyTorch project. Please refer to the main project license for usage terms.