# Copyright 2023 The Qwen team, Alibaba Group. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import atexit
import base64
import copy
import glob
import io
import json
import os
import queue
import re
import shutil
import signal
import stat
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Union

import json5

from qwen_agent.log import logger
from qwen_agent.tools.base import BaseToolWithFileAccess, register_tool
from qwen_agent.utils.utils import append_signal_handler, extract_code, has_chinese_chars, print_traceback, remove_code_blocks

LAUNCH_KERNEL_PY = """
from ipykernel import kernelapp as app
app.launch_new_instance()
"""

INIT_CODE_FILE = str(Path(__file__).absolute().parent / 'resource' / 'code_interpreter_init_kernel.py')
ALIB_FONT_FILE = str(Path(__file__).absolute().parent / 'resource' / 'AlibabaPuHuiTi-3-45-Light.ttf')

_KERNEL_CLIENTS: dict = {}
_MISC_SUBPROCESSES: Dict[str, subprocess.Popen] = {}


def _kill_kernels_and_subprocesses(_sig_num=None, _frame=None):
    for v in _KERNEL_CLIENTS.values():
        v.shutdown()
    for k in list(_KERNEL_CLIENTS.keys()):
        del _KERNEL_CLIENTS[k]

    for v in _MISC_SUBPROCESSES.values():
        v.terminate()
    for k in list(_MISC_SUBPROCESSES.keys()):
        del _MISC_SUBPROCESSES[k]


# Make sure all subprocesses are terminated even if killed abnormally:
# If not running in the main thread, (for example run in streamlit)
# register a signal would cause a RuntimeError
if threading.current_thread() is threading.main_thread():
    atexit.register(_kill_kernels_and_subprocesses)
    append_signal_handler(signal.SIGTERM, _kill_kernels_and_subprocesses)
    append_signal_handler(signal.SIGINT, _kill_kernels_and_subprocesses)


@register_tool('code_interpreter')
class CodeInterpreter(BaseToolWithFileAccess):
    description = '''A Python code interpreter specifically designed for image handling using OpenCV to assist users in answering questions. It allows executing Python code for image analysis, plotting, and manipulation tasks. Before executing code, provide a reasoning handle that summarizes previous results and explains the logic for the next action. Write the Python code directly in the "code" field.

IMPORTANT: If you save images using cv2.imwrite() or similar functions, you MUST display them using IPython display to make them visible in the output. Use the following pattern:
```python
from IPython.display import Image, display
cv2.imwrite('output.jpg', handled_image)
display(Image(filename='output.jpg'))  # This is required to show the saved image
```
Without the display() call, saved images will not be returned in the results.
'''
    parameters = {
        'type': 'object',
        'properties': {
            'reasoning':{
                'description' : 'Summarize the previous code execution and its results, analyze the current state, and provide the Ratiocination for the subsequent code execution.',
                'type' : 'string',
            },
            'code': {
                'description': """
Executes Python code to handle images based strictly on the allowed capabilities listed below.

**CRITICAL INSTRUCTIONS FOR MODEL:**
1.  **RESTRICTED SCOPE:** You are strictly permitted to implement *only* the specific image handling operations defined in the "Allowed Capabilities Reference" section below. Do not generate code for operations outside this list.
2.  **NO PRE-DEFINED FUNCTIONS:** The list below defines *behaviors*, NOT callable functions.
    - INCORRECT: Calling `colorspace_gray(image)` or `resize(image, param)`. These functions DO NOT exist.
    - CORRECT: You must write the raw Python code using the `cv2` library to implement the logic described. For example, to achieve 'colorspace_gray', you must write `cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)`.
3.  **PARAMETER ADHERENCE:** When writing the code, you must strictly follow the parameter logic described in the reference list (e.g., input ranges, default values, and calculation logic).

---

# Allowed Capabilities Reference (Implement these using raw cv2 code)

The following sections describe the ONLY operations you are allowed to implement via Python code.

---

## 1. COLOR SPACE CONVERSION TOOLS

### colorspace_gray
**Description:** Converts the image to grayscale color space. Useful for enhancing contrast or isolating intensity information.
**Parameters:**
- `image` (string, required): The input image identifier
- `param` (object, optional): Empty parameter object

### colorspace_hsv
**Description:** Converts the image to HSV (Hue, Saturation, Value) color space. Useful for color-based segmentation or isolating specific color channels.
**Parameters:**
- `image` (string, required): The input image identifier
- `param` (object, optional): Empty parameter object

### colorspace_lab
**Description:** Converts the image to LAB color space. Useful for perceptual color operations or isolating specific color channels.
**Parameters:**
- `image` (string, required): The input image identifier
- `param` (object, optional): Empty parameter object

---

## 2. GEOMETRIC TRANSFORMATION TOOLS

### resize
**Description:** Resizes the image to specified dimensions or by a preset scale (half or double). You can specify any positive integer width and height (recommended range: 1-10000 pixels). Returns detailed size information including original and new dimensions.
**Parameters:**
- `image` (string, required): The input image identifier
- `param` (object, required):
  - `width` (integer, required): Positive integer for new width
  - `height` (integer, required): Positive integer for new height
  - `preset` (string, optional): Resize preset: 'half' or 'double'

### rotate
**Description:** Rotates the image by specified angle in degrees (clockwise). Supports arbitrary angles (e.g., 45, 90, 180, 270, etc.). The output image will be resized to fit the entire rotated content. Returns rotation details including angle, center point, and size changes.
**Parameters:**
- `image` (string, required): The input image identifier
- `param` (object, required):
  - `angle` (number, required): Rotation angle in degrees (clockwise). Can be any number

### translate
**Description:** Shifts the image by a specified number of pixels in the specified direction (left, right, up, or down). You can specify any positive integer distance (recommended range: 1-10000 pixels). Default distance is 32 pixels. Returns translation details including direction, distance, and translation vector.
**Parameters:**
- `image` (string, required): The input image identifier
- `param` (object, required):
  - `direction` (string, optional): Translation direction: 'left', 'right', 'up', or 'down'
  - `distance` (integer, optional): Translation distance in pixels (default: 32, recommended range: 1-10000)

### flip
**Description:** Flips the image horizontally or vertically.
**Parameters:**
- `image` (string, required): The input image identifier
- `param` (object, required):
  - `direction` (string, optional): Flip direction: 'horizontal' or 'vertical'

### crop
**Description:** Crops a rectangular region from the image. Returns the cropped region with detailed information about the crop area.
**Parameters:**
- `image` (string, required): The input image identifier
- `param` (object, optional):
  - `x` (integer, optional): Non-negative integer for top-left x coordinate
  - `y` (integer, optional): Non-negative integer for top-left y coordinate
  - `width` (integer, optional): Positive integer for crop width
  - `height` (integer, optional): Positive integer for crop height

### zoom_in
**Description:** Zooms into a specific region of the image by cropping and optionally resizing. Useful for focusing on a particular area and enlarging it for better visibility.
**Parameters:**
- `image` (string, required): The input image identifier
- `param` (object, optional):
  - `x` (integer, optional): Non-negative integer for region x coordinate
  - `y` (integer, optional): Non-negative integer for region y coordinate
  - `width` (integer, optional): Positive integer for region width
  - `height` (integer, optional): Positive integer for region height
  - `scale` (number, optional): Float 0-10 for zoom scale
  - `target_width` (integer, optional): Positive integer for target width
  - `target_height` (integer, optional): Positive integer for target height

---

## 3. FILTERING AND SMOOTHING TOOLS

### blur
**Description:** Applies blurring to reduce noise or smooth the image. Supports multiple blur methods: average (simple box blur), gaussian (weighted blur), median (good for salt-pepper noise), and bilateral (edge-preserving). Returns blur method and parameter details.
**Parameters:**
- `image` (string, required): The input image identifier
- `param` (object, required):
  - `method` (string, required): Blur method: 'average'/'avg', 'gaussian', 'median', 'bilateral'
  - `ksize` (integer, required): Kernel size (must be odd, 3-51). Default: 5
  - `sigma_x` (number, optional): Gaussian kernel std dev in X (0-10). Default: 0
  - `sigma_y` (number, optional): Gaussian kernel std dev in Y (0-10). Default: 0
  - `d` (integer, optional): Bilateral filter diameter (1-50). Default: 9
  - `sigma_color` (number, optional): Bilateral color sigma (1-200). Default: 75
  - `sigma_space` (number, optional): Bilateral space sigma (1-200). Default: 75

### denoise
**Description:** Applies fast non-local means denoising. Supports grayscale, color, or per-channel handling.
**Parameters:**
- `image` (string, required): The input image identifier
- `param` (object, optional):
  - `mode` (string, optional): 'fast_means_gray', 'fast_means_color', 'fast_means_bgr_channel'. Default: 'fast_means_color'
  - `h` (number, optional): Float 1-50. Default: 10
  - `h_color` (number, optional): Float 1-50. Default: 10
  - `template_window` (integer, optional): Odd integer 3-21. Default: 7
  - `search_window` (integer, optional): Odd integer 3-31. Default: 21
  - `channels` (array, optional): List of channel indices [0, 1, 2]

---

## 4. THRESHOLDING AND BINARIZATION TOOLS

### threshold
**Description:** Applies thresholding to create a binary image. Supports multiple color modes (grayscale or BGR channel-wise) and various threshold methods.
**Parameters:**
- `image` (string, required): The input image identifier
- `param` (object, required):
  - `mode` (string, required): 'binary', 'otsu', 'adaptive_mean', 'adaptive_gaussian'. Default: 'otsu'
  - `invert` (boolean, optional): true or false. Default: false
  - `color_mode` (string, optional): 'grayscale' or 'bgr'. Default: 'grayscale'
  - `threshold_value` (integer, optional): Integer 0-255 (for binary mode)
  - `adaptive_block_size` (integer, optional): Odd integer 3-101. Default: 11
  - `adaptive_constant` (integer, optional): Integer. Default: 2
  - `channels` (array, optional): List of channel indices [0, 1, 2]

### inrange_color
**Description:** Creates a mask for pixels within a specified color range. Supports HSV or BGR colorspace and flexible output formats.
**Parameters:**
- `image` (string, required): The input image identifier
- `param` (object, optional):
  - `colorspace` (string, optional): 'hsv' or 'bgr'. Default: 'hsv'
  - `lower` (array, optional): Lower bound [h/b, s/g, v/r]
  - `upper` (array, optional): Upper bound [h/b, s/g, v/r]
  - `output_format` (string, optional): 'mask', 'masked_image', 'both'. Default: 'both'

---

## 5. MORPHOLOGICAL OPERATIONS

### morphology
**Description:** Applies morphological operations (erode, dilate, open, close) with flexible kernel size and shape.
**Parameters:**
- `image` (string, required): The input image identifier
- `param` (object, optional):
  - `op` (string, optional): 'erode', 'dilate', 'open', 'close'. Default: 'open'
  - `kernel_size` (integer, optional): Odd integer 3-21. Default: 3
  - `iterations` (integer, optional): Integer 1-10. Default: 1
  - `kernel_shape` (string, optional): 'rect' or 'ellipse'. Default: 'rect'

---

## 6. EDGE DETECTION TOOLS

### gradients
**Description:** Computes gradient images using Sobel (x or y direction) or Laplacian operator.
**Parameters:**
- `image` (string, required): The input image identifier
- `param` (object, required):
  - `mode` (string, required): 'sobel_x', 'sobel_y', 'laplacian'

### canny
**Description:** Detects edges using Canny edge detector. Supports preset thresholds or custom values, and can handle grayscale or individual BGR channels.
**Parameters:**
- `image` (string, required): The input image identifier
- `param` (object, optional):
  - `preset` (string, optional): 'low', 'medium', 'high'. Default: 'medium'
  - `threshold_low` (integer, optional): Integer 0-255 (overrides preset)
  - `threshold_high` (integer, optional): Integer 0-255 (overrides preset)
  - `color_mode` (string, optional): 'grayscale' or 'bgr'. Default: 'grayscale'
  - `channels` (array, optional): List of channel indices [0, 1, 2]

### convertscaleabs
**Description:** Converts image to absolute value and scales it to 0-255 range. Useful for visualizing gradient images or other signed data. Applies scaling and offset transformations.
**Parameters:**
- `image` (string, required): The input image identifier
- `param` (object, optional):
  - `alpha` (number, optional): Float. Default: 1.0
  - `beta` (number, optional): Float. Default: 0

---

## 7. CONTOUR DETECTION AND ANALYSIS TOOLS

### contours
**Description:** Finds contours using Canny edges and returns bounding boxes/areas. Helpful for locating shapes before further analysis.
**Parameters:**
- `image` (string, required): The input image identifier
- `param` (object, optional):
  - `mode` (string, optional): 'external' or 'list'
  - `canny_low` (integer, optional): Integer. Default: 100
  - `canny_high` (integer, optional): Integer. Default: 200
  - `rank` (integer, optional): Integer (1 = largest)
  - `max_contours` (integer, optional): Integer. Default: 20

### contour_area
**Description:** Calculates contour areas using cv2.contourArea and overlays the values near each contour. Useful for comparing object sizes.
**Parameters:**
- `image` (string, required): The input image identifier
- `param` (object, optional):
  - `mode` (string, optional): 'external' or 'list'
  - `canny_low` (integer, optional): Integer
  - `canny_high` (integer, optional): Integer
  - `rank` (integer, optional): Integer
  - `max_contours` (integer, optional): Integer
  - `color` (array, optional): BGR color [B, G, R]
  - `thickness` (integer, optional): Integer

### arc_length
**Description:** Computes contour perimeters using cv2.arcLength and overlays the values. Helps measure object outlines.
**Parameters:**
- `image` (string, required): The input image identifier
- `param` (object, optional):
  - `mode` (string, optional): 'external' or 'list'
  - `canny_low` (integer, optional): Integer
  - `canny_high` (integer, optional): Integer
  - `rank` (integer, optional): Integer
  - `max_contours` (integer, optional): Integer
  - `color` (array, optional): BGR color [B, G, R]
  - `thickness` (integer, optional): Integer

### approx_poly
**Description:** Approximates contours with fewer points using cv2.approxPolyDP (epsilon_ratio controls simplification). Great for polygonal shape summaries.
**Parameters:**
- `image` (string, required): The input image identifier
- `param` (object, optional):
  - `mode` (string, optional): 'external' or 'list'
  - `canny_low` (integer, optional): Integer
  - `canny_high` (integer, optional): Integer
  - `rank` (integer, optional): Integer
  - `max_contours` (integer, optional): Integer
  - `epsilon_ratio` (number, optional): Float. Default: 0.02
  - `color` (array, optional): BGR color [B, G, R]
  - `thickness` (integer, optional): Integer

### connected_components_with_stats
**Description:** Finds and analyzes connected components in a binary image. Returns statistics including bounding boxes, areas, and centroids for each component. Useful for object detection and region analysis.
**Parameters:**
- `image` (string, required): The input image identifier
- `param` (object, optional):
  - `threshold_mode` (string, optional): 'otsu', 'binary', 'none'
  - `threshold_value` (integer, optional): Integer for binary mode. Default: 127
  - `connectivity` (integer, optional): 4 or 8. Default: 8
  - `max_components` (integer, optional): Integer. Default: 20

---

## 8. SHAPE DETECTION TOOLS

### hough_lines
**Description:** Detects line segments in the image using Hough transform. Returns line coordinates and lengths. By default uses stricter parameters to reduce false positives. You can adjust threshold (higher = stricter, fewer lines) and minLineLength (higher = longer lines only) to control detection sensitivity.
**Parameters:**
- `image` (string, required): The input image identifier
- `param` (object, optional):
  - `threshold` (integer, optional): Integer. Default: 80
  - `minLineLength` (integer, optional): Integer. Default: 50
  - `maxLineGap` (integer, optional): Integer. Default: 10
  - `canny_low` (integer, optional): Integer. Default: 80
  - `canny_high` (integer, optional): Integer. Default: 200
  - `max_lines` (integer, optional): Integer. Default: 20

### hough_circles
**Description:** Detects circles in the image using Hough transform. Returns circle centers and radii. By default uses stricter parameters to reduce false positives. You can adjust param2 (higher = stricter, fewer circles) and minDist (higher = circles must be farther apart) to control detection sensitivity.
**Parameters:**
- `image` (string, required): The input image identifier
- `param` (object, optional):
  - `dp` (number, optional): Float. Default: 1.2
  - `minDist` (integer, optional): Integer. Default: 50
  - `param1` (integer, optional): Integer. Default: 100
  - `param2` (integer, optional): Integer. Default: 50
  - `minRadius` (integer, optional): Integer. Default: 10
  - `maxRadius` (integer, optional): Integer. Default: 0 (no limit)
  - `max_circles` (integer, optional): Integer. Default: 20

---

## 9. HISTOGRAM AND CONTRAST ENHANCEMENT

### histogram
**Description:** Applies histogram equalization or CLAHE to enhance image contrast. Supports multiple color modes (grayscale, BGR, HSV) and flexible channel selection to preserve color information or handle specific channels.
**Parameters:**
- `image` (string, required): The input image identifier
- `param` (object, optional):
  - `mode` (string, optional): 'equalize' or 'clahe'
  - `color_mode` (string, optional): 'grayscale', 'bgr', 'hsv'. Default: 'grayscale'
  - `channels` (array, optional): List of channel indices [0, 1, 2]
  - `clip_limit` (number, optional): Float (for CLAHE)
  - `tile_grid_size` (array, optional): [width, height]. Default: [8, 8]

---

## 10. IMAGE SEGMENTATION TOOLS

### watershed
**Description:** Applies watershed segmentation to separate overlapping objects. Returns region count and labels.
**Parameters:**
- `image` (string, required): The input image identifier
- `param` (object, optional):
  - `max_regions` (integer, optional): Integer. Default: 20

### grabcut
**Description:** Performs foreground/background segmentation using GrabCut algorithm with preset rectangle. Returns foreground pixel statistics.
**Parameters:**
- `image` (string, required): The input image identifier
- `param` (object, required):
  - `preset` (string, optional): 'center', 'tight', 'loose'

### floodfill
**Description:** Performs flood fill operation starting from a seed point. Fills connected pixels with similar color values. Useful for region segmentation and filling enclosed areas.
**Parameters:**
- `image` (string, required): The input image identifier
- `param` (object, optional):
  - `x` (integer, optional): Non-negative integer for seed x coordinate
  - `y` (integer, optional): Non-negative integer for seed y coordinate
  - `loDiff` (array, optional): Lower difference [B, G, R]. Default: [10, 10, 10]
  - `upDiff` (array, optional): Upper difference [B, G, R]. Default: [10, 10, 10]
  - `newVal` (array, optional): New fill color [B, G, R]. Default: [0, 255, 0]
  - `flags` (integer, optional): Integer. Default: 4|FLOODFILL_FIXED_RANGE

---

## 11. ADVANCED OPERATIONS

### pyramid
**Description:** Applies image pyramid operations (upsample or downsample by one level). Returns size information.
**Parameters:**
- `image` (string, required): The input image identifier
- `param` (object, required):
  - `mode` (string, optional): 'pyr_up' or 'pyr_down'

### dft
**Description:** Computes and visualizes the Discrete Fourier Transform magnitude spectrum.
**Parameters:**
- `image` (string, required): The input image identifier
- `param` (object, optional): Empty parameter object

### template_match
**Description:** Matches a template image within the source image. Returns match score and bounding box coordinates.
**Parameters:**
- `image` (string, required): The input image identifier
- `param` (object, optional):
  - `template_path` (string, optional): Path to template image
  - `method` (string, optional): 'sqdiff', 'ccorr', 'ccoeff'

### features
**Description:** Detects and draws keypoints using various feature detection methods (harris, shi_tomasi, sift, surf, fast, brief, orb). Returns keypoint coordinates.
**Parameters:**
- `image` (string, required): The input image identifier
- `param` (object, optional):
  - `method` (string, optional): 'harris', 'shi_tomasi', 'sift', 'surf', 'fast', 'brief', 'orb'
  - `max_points` (integer, optional): Maximum number of keypoints to detect

### inpaint
**Description:** Inpaints (fills) regions in the image using automatically generated mask. Supports custom parameters for mask generation and inpainting method.
**Parameters:**
- `image` (string, required): The input image identifier
- `param` (object, optional):
  - `preset` (string, optional): 'canny' or 'threshold'. Default: 'canny'
  - `canny_low` (integer, optional): Integer 0-255. Default: 100
  - `canny_high` (integer, optional): Integer 0-255. Default: 200
  - `threshold_value` (integer, optional): Integer 0-255. Default: 127
  - `radius` (integer, optional): Integer 1-10. Default: 3
  - `method` (string, optional): 'telea' or 'ns'. Default: 'telea'

---

## 12. DRAWING TOOLS

### draw_line
**Description:** Draws a line segment between two points with configurable color and thickness.
**Parameters:**
- `image` (string, required): The input image identifier
- `param` (object, required):
  - `x1` (integer, required): Integer for start x coordinate
  - `y1` (integer, required): Integer for start y coordinate
  - `x2` (integer, required): Integer for end x coordinate
  - `y2` (integer, required): Integer for end y coordinate
  - `color` (array, optional): BGR color [B, G, R]
  - `thickness` (integer, optional): Integer

### draw_circle
**Description:** Draws a circle on the image at specified center coordinates with given radius. Supports customizable color and thickness. Useful for marking circular regions or objects.
**Parameters:**
- `image` (string, required): The input image identifier
- `param` (object, optional):
  - `x` (integer, optional): Integer for center x coordinate
  - `y` (integer, optional): Integer for center y coordinate
  - `radius` (integer, optional): Positive integer for radius
  - `color` (array, optional): BGR color [B, G, R]. Default: [0, 255, 0]
  - `thickness` (integer, optional): Integer. Default: 2

**CRITICAL INSTRUCTIONS FOR MODEL:**
1.  **RESTRICTED SCOPE:** You are strictly permitted to implement *only* the specific operations defined in the "Allowed Capabilities Reference" section below. Do not generate code for operations outside this list.
2.  **NO PRE-DEFINED FUNCTIONS:** The list below defines *behaviors*, NOT callable functions.
    - INCORRECT: Calling `colorspace_gray(image)` or `resize(image, param)`. These functions DO NOT exist.
    - CORRECT: You must write the raw Python code using the `cv2` library to implement the logic described. For example, to achieve 'colorspace_gray', you must write `cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)`.
3.  **PARAMETER ADHERENCE:** When writing the code, you must strictly follow the parameter logic described in the reference list (e.g., input ranges, default values, and calculation logic).
                """,
                'type': 'string',
            }
        },
        'required': ['reasoning','code'],
    }

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)
        self.work_dir: str = os.getenv('M6_CODE_INTERPRETER_WORK_DIR', self.work_dir)
        self.work_dir: str = self.cfg.get('work_dir', self.work_dir)

        # Support organizing by model_name and question_id
        model_name = self.cfg.get('model_name', '')
        question_id = self.cfg.get('question_id', '')

        if model_name and question_id:
            # Create subdirectory structure: work_dir/model_name/question_id/
            self.work_dir = os.path.join(self.work_dir, model_name, question_id)
        elif model_name:
            # Create subdirectory structure: work_dir/model_name/
            self.work_dir = os.path.join(self.work_dir, model_name)
        elif question_id:
            # Create subdirectory structure: work_dir/question_id/
            self.work_dir = os.path.join(self.work_dir, question_id)

        os.makedirs(self.work_dir, exist_ok=True)
        self.instance_id: str = str(uuid.uuid4())

        # Adjust parameters based on model type
        # self._adjust_parameters_by_model(model_name)

        _check_deps_for_code_interpreter()

    def _adjust_parameters_by_model(self, model_name: str):
        """
        根据模型名称动态调整parameters。

        对于 GPT o3/o4/5.2 等高级推理模型，不需要 reasoning 参数。

        Args:
            model_name: 模型名称
        """
        # 高级推理模型列表（不需要reasoning参数）
        advanced_reasoning_models = [
            'o3', 'o4', '5.2', 'o3-2025', 'o4-2025', 'o4-mini',
            'gpt-5.2', 'gpt-o3', 'gpt-o4',
        ]

        # 检查是否是高级推理模型
        is_advanced_model = any(
            keyword.lower() in model_name.lower()
            for keyword in advanced_reasoning_models
        )

        if is_advanced_model:
            # 为高级推理模型移除 reasoning 参数
            print(f"Model '{model_name}' detected as advanced reasoning model. Removing 'reasoning' parameter.")

            # 创建新的parameters副本，移除reasoning参数
            new_parameters = copy.deepcopy(self.parameters)

            # 移除properties中的reasoning
            if 'properties' in new_parameters and 'reasoning' in new_parameters['properties']:
                del new_parameters['properties']['reasoning']

            # 从required列表中移除reasoning
            if 'required' in new_parameters and 'reasoning' in new_parameters['required']:
                new_parameters['required'].remove('reasoning')

            # 更新实例的parameters
            self.parameters = new_parameters
        else:
            # 对于其他模型，保持原始parameters
            print(f"Model '{model_name}' uses standard parameters with 'reasoning' field.")

    @property
    def args_format(self) -> str:
        fmt = self.cfg.get('args_format')
        if fmt is None:
            if has_chinese_chars([self.name_for_human, self.name, self.description, self.parameters]):
                fmt = '此工具的输入应为Markdown代码块。'
            else:
                fmt = 'Enclose the code within triple backticks (`) at the beginning and end of the code.'
        return fmt

    def _check_forbidden_commands(self, code: str) -> tuple:
        """
        检查代码中是否包含禁止的包管理命令。

        Returns:
            (is_forbidden, error_message) - 如果包含禁止命令返回 (True, error_message)，否则返回 (False, '')
        """
        # 禁止的包管理命令模式
        forbidden_patterns = [
            r'!pip\s+install',
            r'!pip\s+-q\s+install',
            r'!pip3\s+install',
            r'!pip3\s+-q\s+install',
            r'subprocess.*pip.*install',
            r'os.system.*pip.*install',
            r'subprocess.*conda.*install',
            r'os.system.*conda.*install',
            r'!conda\s+install',
            r'!apt-get\s+install',
            r'!apt\s+install',
            r'!yum\s+install',
        ]

        for pattern in forbidden_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                return True, f"❌ Package installation commands are not allowed. Detected forbidden command pattern: {pattern}"

        return False, ''

    def _remove_blocking_cv2_calls(self, code: str) -> str:
        """
        移除会导致程序卡住的OpenCV显示函数。
        这些函数在无窗口环境中会导致程序挂起。

        Removes:
            - cv2.imshow(...)
            - cv2.waitKey(...)
            - cv2.destroyAllWindows(...)

        Args:
            code: Python代码字符串

        Returns:
            处理后的代码字符串
        """
        lines = code.split('\n')
        filtered_lines = []

        for line in lines:
            # 检查是否是要移除的函数调用
            stripped_line = line.strip()

            # 跳过那些会导致阻塞的cv2函数调用
            if (stripped_line.startswith('cv2.imshow(') or
                stripped_line.startswith('cv2.waitKey(') or
                stripped_line.startswith('cv2.destroyAllWindows(') or
                '= cv2.imshow(' in stripped_line or
                '= cv2.waitKey(' in stripped_line):
                # 记录移除的操作
                logger.info(f'Removed blocking cv2 call: {stripped_line[:100]}')
                continue

            filtered_lines.append(line)

        return '\n'.join(filtered_lines)

    def call(self, params: Union[str, dict], files: List[str] = None, timeout: Optional[int] = 30, **kwargs) -> str:
        super().call(params=params, files=files, **kwargs)  # copy remote files to work_dir

        # Store question_id from kwargs for use in _serve_image
        self._current_question_id = kwargs.get('question_id', '')

        code = None

        # Try to parse params as JSON/JSON5 if it's a string
        if isinstance(params, str):
            # First try to extract code from markdown code blocks
            try:
                code = remove_code_blocks(params)
                if code and code != params:  # Successfully extracted from code blocks
                    pass
                else:
                    # Try JSON5 parsing as fallback
                    try:
                        params = json5.loads(params)
                        code = params.get('code', '')
                    except Exception:
                        # Final fallback: use extract_code utility
                        code = extract_code(params)
            except Exception as e:
                logger.warning(f"Failed to parse params: {e}")
                code = extract_code(params)
        elif isinstance(params, dict):
            code = params.get('code', '')
        else:
            code = str(params)

        # 确保代码被正确提取（移除 markdown 代码块标记）
        code = remove_code_blocks(code)

        # 检查禁止的包管理命令
        is_forbidden, error_msg = self._check_forbidden_commands(code)
        if is_forbidden:
            logger.error(error_msg)
            return error_msg

        # 移除会导致程序卡住的阻塞式OpenCV函数
        code = self._remove_blocking_cv2_calls(code)

        # 将 code 中 print 的 image path 进行替换，替换到 /Users/xuanyuzhu/benchmark/code/Qwen-Agent/logs/img_tmp

        if not code.strip():
            return ''

        kernel_id: str = f'{self.instance_id}_{os.getpid()}'
        if kernel_id in _KERNEL_CLIENTS:
            kc = _KERNEL_CLIENTS[kernel_id]
        else:
            _fix_matplotlib_cjk_font_issue()
            self._fix_secure_write_for_code_interpreter()
            kc, subproc = self._start_kernel(kernel_id)
            with open(INIT_CODE_FILE) as fin:
                start_code = fin.read()
                start_code = start_code.replace('{{M6_FONT_PATH}}', repr(ALIB_FONT_FILE)[1:-1])
                start_code += '\n%xmode Minimal'
            logger.info(self._execute_code(kc, start_code))
            _KERNEL_CLIENTS[kernel_id] = kc
            _MISC_SUBPROCESSES[kernel_id] = subproc

        if timeout:
            code = f'_M6CountdownTimer.start({timeout})\n{code}'

        fixed_code = []
        for line in code.split('\n'):
            fixed_code.append(line)
            if line.startswith('sns.set_theme('):
                fixed_code.append('plt.rcParams["font.family"] = _m6_font_prop.get_name()')
        fixed_code = '\n'.join(fixed_code)
        fixed_code += '\n\n'  # Prevent code not executing in notebook due to no line breaks at the end

        try:
            result = self._execute_code(kc, fixed_code)
        except Exception as e:
            # Handle execution errors (including timeout exceptions from _execute_code)
            error_type = type(e).__name__
            logger.error(f"Code execution failed ({error_type}): {str(e)}")
            result = f'error:\n\n```\nCode execution failed ({error_type}): {str(e)}\n```'

            # If timeout occurred, mark kernel as potentially unstable
            if 'Timeout' in str(e) or 'timeout' in str(e).lower():
                logger.warning(f"Code execution timeout detected for kernel {kernel_id}, marking for cleanup")
                # Mark kernel for removal to ensure fresh start on next execution
                if kernel_id in _KERNEL_CLIENTS:
                    try:
                        _KERNEL_CLIENTS[kernel_id].shutdown()
                        del _KERNEL_CLIENTS[kernel_id]
                    except Exception as cleanup_error:
                        logger.warning(f"Failed to cleanup kernel {kernel_id}: {cleanup_error}")

                    if kernel_id in _MISC_SUBPROCESSES:
                        try:
                            _MISC_SUBPROCESSES[kernel_id].terminate()
                            del _MISC_SUBPROCESSES[kernel_id]
                        except Exception as cleanup_error:
                            logger.warning(f"Failed to terminate kernel subprocess {kernel_id}: {cleanup_error}")
        else:
            # Only try to cancel timer if execution was successful
            if timeout:
                try:
                    self._execute_code(kc, '_M6CountdownTimer.cancel()')
                except Exception as e:
                    logger.warning(f"Failed to cancel timer: {e}")

        return result if result.strip() else 'Finished execution.'

    def __del__(self):
        # Recycle the jupyter subprocess:
        k: str = f'{self.instance_id}_{os.getpid()}'
        if k in _KERNEL_CLIENTS:
            _KERNEL_CLIENTS[k].shutdown()
            del _KERNEL_CLIENTS[k]
        if k in _MISC_SUBPROCESSES:
            _MISC_SUBPROCESSES[k].terminate()
            del _MISC_SUBPROCESSES[k]

    def _fix_secure_write_for_code_interpreter(self):
        if 'linux' in sys.platform.lower():
            os.makedirs(self.work_dir, exist_ok=True)
            fname = os.path.join(self.work_dir, f'test_file_permission_{os.getpid()}.txt')
            if os.path.exists(fname):
                os.remove(fname)
            with os.fdopen(os.open(fname, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o0600), 'w') as f:
                f.write('test')
            file_mode = stat.S_IMODE(os.stat(fname).st_mode) & 0o6677
            if file_mode != 0o0600:
                os.environ['JUPYTER_ALLOW_INSECURE_WRITES'] = '1'
            if os.path.exists(fname):
                os.remove(fname)

    def _start_kernel(self, kernel_id: str):
        connection_file = os.path.join(self.work_dir, f'kernel_connection_file_{kernel_id}.json')
        launch_kernel_script = os.path.join(self.work_dir, f'launch_kernel_{kernel_id}.py')
        for f in [connection_file, launch_kernel_script]:
            if os.path.exists(f):
                logger.info(f'WARNING: {f} already exists')
                os.remove(f)

        os.makedirs(self.work_dir, exist_ok=True)
        with open(launch_kernel_script, 'w') as fout:
            fout.write(LAUNCH_KERNEL_PY)

        kernel_process = subprocess.Popen(
            [
                sys.executable,
                os.path.abspath(launch_kernel_script),
                '--IPKernelApp.connection_file',
                os.path.abspath(connection_file),
                '--matplotlib=inline',
                '--quiet',
            ],
            cwd=os.path.abspath(self.work_dir),
        )
        logger.info(f"INFO: kernel process's PID = {kernel_process.pid}")

        # Wait for kernel connection file to be written
        while True:
            if not os.path.isfile(connection_file):
                time.sleep(0.1)
            else:
                # Keep looping if JSON parsing fails, file may be partially written
                try:
                    with open(connection_file, 'r') as fp:
                        json.load(fp)
                    break
                except json.JSONDecodeError:
                    pass

        # Client
        from jupyter_client import BlockingKernelClient

        kc = BlockingKernelClient(connection_file=connection_file)
        asyncio.set_event_loop_policy(AnyThreadEventLoopPolicy())
        kc.load_connection_file()
        kc.start_channels()
        kc.wait_for_ready()
        return kc, kernel_process

    def _execute_code(self, kc, code: str, timeout: int = 300) -> str:
        """
        Execute code in Jupyter kernel with timeout.

        Args:
            kc: Kernel client
            code: Python code to execute
            timeout: Timeout in seconds for message receiving (default: 300s = 5min)

        Returns:
            Formatted result string with text and images
        """
        kc.wait_for_ready()
        kc.execute(code)
        result = ''
        image_idx = 0
        while True:
            text = ''
            image = ''
            finished = False
            msg_type = 'error'
            try:
                # Set timeout for get_iopub_msg to prevent indefinite blocking
                msg = kc.get_iopub_msg(timeout=timeout)
                msg_type = msg['msg_type']
                if msg_type == 'status':
                    if msg['content'].get('execution_state') == 'idle':
                        finished = True
                elif msg_type == 'execute_result':
                    text = msg['content']['data'].get('text/plain', '')
                    # Support both PNG and JPEG formats
                    image_format = None
                    if 'image/png' in msg['content']['data']:
                        image_format = 'image/png'
                    elif 'image/jpeg' in msg['content']['data']:
                        image_format = 'image/jpeg'

                    if image_format:
                        image_b64 = msg['content']['data'][image_format]
                        image_url = self._serve_image(image_b64, format_hint=image_format.split('/')[-1])
                        image_idx += 1
                        image = '![fig-%03d](%s)' % (image_idx, image_url)
                elif msg_type == 'display_data':
                    # Support both PNG and JPEG formats
                    image_format = None
                    if 'image/png' in msg['content']['data']:
                        image_format = 'image/png'
                    elif 'image/jpeg' in msg['content']['data']:
                        image_format = 'image/jpeg'

                    if image_format:
                        image_b64 = msg['content']['data'][image_format]
                        image_url = self._serve_image(image_b64, format_hint=image_format.split('/')[-1])
                        image_idx += 1
                        image = '![fig-%03d](%s)' % (image_idx, image_url)
                    else:
                        text = msg['content']['data'].get('text/plain', '')
                elif msg_type == 'stream':
                    msg_type = msg['content']['name']  # stdout, stderr
                    text = msg['content']['text']
                elif msg_type == 'error':
                    text = _escape_ansi('\n'.join(msg['content']['traceback']))
                    if 'M6_CODE_INTERPRETER_TIMEOUT' in text:
                        text = 'Timeout: Code execution exceeded the time limit.'
            except queue.Empty:
                text = f'Timeout: Code execution exceeded {timeout} seconds.'
                logger.error(f'Kernel execution timeout after {timeout}s')
                finished = True
            except Exception as e:
                # Handle TaskTimeoutError and other unexpected errors
                error_type = type(e).__name__
                if 'TaskTimeoutError' in error_type or 'Timeout' in error_type:
                    text = f'Timeout: Code execution exceeded the task timeout limit. ({error_type})'
                    logger.warning(f'Code execution timed out: {str(e)}')
                else:
                    text = 'The code interpreter encountered an unexpected error.'
                    logger.warning(f'Code execution error ({error_type}): {str(e)}')
                print_traceback()
                finished = True
            if text:
                result += f'\n\n{msg_type}:\n\n```\n{text}\n```'
            if image:
                result += f'\n\n{image}'
            if finished:
                break
        result = result.lstrip('\n')
        return result

    def _serve_image(self, image_base64: str, format_hint: str = 'png') -> str:
        import PIL.Image

        # Use the format hint to determine file extension
        file_ext = format_hint.lower()
        if file_ext not in ['png', 'jpeg', 'jpg']:
            file_ext = 'png'  # Default to png

        image_file = f'{uuid.uuid4()}.{file_ext}'

        # Create subdirectory for question_id if available
        work_dir = self.work_dir
        if hasattr(self, '_current_question_id') and self._current_question_id:
            work_dir = os.path.join(self.work_dir, self._current_question_id)
            os.makedirs(work_dir, exist_ok=True)

        local_image_file = os.path.join(work_dir, image_file)

        image_bytes = base64.b64decode(image_base64)
        assert isinstance(image_bytes, bytes)
        bytes_io = io.BytesIO(image_bytes)
        # Save with the appropriate format (PIL will auto-detect from extension)
        PIL.Image.open(bytes_io).save(local_image_file)

        image_server_url = os.getenv('M6_CODE_INTERPRETER_STATIC_URL', '')
        if image_server_url:
            # Include question_id in URL path if present
            if hasattr(self, '_current_question_id') and self._current_question_id:
                return f'{image_server_url}/{self._current_question_id}/{image_file}'
            return f'{image_server_url}/{image_file}'
        return local_image_file


def _check_deps_for_code_interpreter():
    try:
        import matplotlib  # noqa
        import matplotlib.pyplot as plt  # noqa
        import numpy as np  # noqa
        import pandas as pd  # noqa
        import PIL.Image  # noqa
        import seaborn as sns  # noqa
        from jupyter_client import BlockingKernelClient  # noqa
        from sympy import Eq, solve, symbols  # noqa
    except ImportError as e:
        raise ImportError(
            'The dependencies for Code Interpreter support are not installed. '
            'Please install the required dependencies by running: pip install "qwen-agent[code_interpreter]"') from e


def _fix_matplotlib_cjk_font_issue():
    import matplotlib

    ttf_name = os.path.basename(ALIB_FONT_FILE)
    local_ttf = os.path.join(os.path.abspath(os.path.join(matplotlib.matplotlib_fname(), os.path.pardir)), 'fonts',
                             'ttf', ttf_name)
    if not os.path.exists(local_ttf):
        try:
            shutil.copy(ALIB_FONT_FILE, local_ttf)
            font_list_cache = os.path.join(matplotlib.get_cachedir(), 'fontlist-*.json')
            for cache_file in glob.glob(font_list_cache):
                with open(cache_file) as fin:
                    cache_content = fin.read()
                if ttf_name not in cache_content:
                    os.remove(cache_file)
        except Exception:
            print_traceback()


def _escape_ansi(line: str) -> str:
    ansi_escape = re.compile(r'(?:\x1B[@-_]|[\x80-\x9F])[0-?]*[ -/]*[@-~]')
    return ansi_escape.sub('', line)


#
# The _BasePolicy and AnyThreadEventLoopPolicy below are borrowed from Tornado.
# Ref: https://www.tornadoweb.org/en/stable/_modules/tornado/platform/asyncio.html#AnyThreadEventLoopPolicy
#

if sys.platform == 'win32' and hasattr(asyncio, 'WindowsSelectorEventLoopPolicy'):
    _BasePolicy = asyncio.WindowsSelectorEventLoopPolicy  # type: ignore
else:
    _BasePolicy = asyncio.DefaultEventLoopPolicy


class AnyThreadEventLoopPolicy(_BasePolicy):  # type: ignore
    """Event loop policy that allows loop creation on any thread.

    The default `asyncio` event loop policy only automatically creates
    event loops in the main threads. Other threads must create event
    loops explicitly or `asyncio.get_event_loop` (and therefore
    `.IOLoop.current`) will fail. Installing this policy allows event
    loops to be created automatically on any thread.

    Usage::
        asyncio.set_event_loop_policy(AnyThreadEventLoopPolicy())
    """

    def get_event_loop(self) -> asyncio.AbstractEventLoop:
        try:
            return super().get_event_loop()
        except RuntimeError:
            # "There is no current event loop in thread %r"
            loop = self.new_event_loop()
            self.set_event_loop(loop)
            return loop
