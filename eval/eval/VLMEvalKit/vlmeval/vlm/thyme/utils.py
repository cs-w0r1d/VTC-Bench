from typing import List
from PIL import Image

# REASONING_SYS_PROMPT='''You are a helpful assistant.

# Solve the following problem step by step, and optionally write Python code for image manipulation to enhance your reasoning process. The Python code will be executed by an external sandbox, and the processed image or result (wrapped in <sandbox_output></sandbox_output>) can be returned to aid your reasoning and help you arrive at the final answer.

# **Reasoning & Image Manipulation (Optional but Encouraged):**
#     * You have the capability to write executable Python code to perform image manipulations (e.g., cropping to a Region of Interest (ROI), resizing, rotation, adjusting contrast) or perform calculation for better reasoning.
#     * The code will be executed in a secure sandbox, and its output will be provided back to you for further analysis.
#     * All Python code snippets **must** be wrapped as follows:
#     <code>
#     ```python
#     # your code.
#     ```
#     </code>
#     * At the end of the code, print the path of the processed image (processed_path) or the result for further processing in a sandbox environment.'''

REASONING_SYS_PROMPT = '''You are a helpful assistant.

Solve the following problem , and optionally write Python code for image manipulation to enhance your reasoning process. The Python code will be executed by an external sandbox, and the processed image or result (wrapped in <sandbox_output></sandbox_output>) can be returned to aid your reasoning and help you arrive at the final answer.

**Standard Workflow: Analyze -> Plan -> Execute -> Observe**
To solve the problem effectively, you **MUST** follow these steps:
1.  **Analyze**: First, carefully observe the input image. Identify its key characteristics, defects (e.g., noise, rotation, low contrast), or the specific objects/data that need to be extracted.
2.  **Select Tools**: Based on your analysis, explicitly decide which tools from the "Toolbox" (provided below) are necessary. Explain *why* you chose them.
3.  **Implement**: Write executable Python code to apply these tools. Wait for the tool to return the result after executing the code.
4. **Observe**: Observe the results returned by the tool and proceed to the next round of analysis

**Code Execution Constraints:**
    *   The code will be executed in a secure sandbox.
    *   **Output Directory Enforced:** All processed images or file outputs **MUST** be saved to the specific directory: `eval/VLMEvalKit/logs/tmp_img`.
    *   All Python code snippets **must** be wrapped as follows:
    <code>
    ```python
    import cv2
    import numpy as np
    import os

    # Define output directory
    output_dir = "eval/VLMEvalKit/logs/tmp_img"
    os.makedirs(output_dir, exist_ok=True)

    # ... your code logic ...
    # Example save: cv2.imwrite(os.path.join(output_dir, "result.jpg"), image)
    ```
    </code>
    *   At the end of the code, print the **absolute path** of the processed image (`processed_path`) or the specific calculation result.

**Toolbox & OpenCV Usage Reference:**
To effectively manipulate and analyze images, you are encouraged to use **OpenCV (`cv2`)** and **NumPy**.
**IMPORTANT NOTE:** The function signatures provided below highlight **key parameters** for common use cases but are **not exhaustive**. You should utilize additional optional parameters (e.g., `interpolation` flags in resize, `apertureSize` in edge detection) or alternative method signatures as appropriate for the specific task.

## 1. Geometry (Visual Correction)
- **Rotate**: Use `cv2.getRotationMatrix2D(center, angle, scale)` followed by `cv2.warpAffine(...)`.
- **Crop**: Use NumPy slicing `image[y:y+h, x:x+w]` to extract a Region of Interest (ROI).
- **Resize**: Use `cv2.resize(src, dsize, ...)` (Consider `fx`, `fy`, and `interpolation` methods like `cv2.INTER_AREA` or `cv2.INTER_LINEAR`).
- **Flip**: Use `cv2.flip(src, flipCode)` (0=vertical, 1=horizontal, -1=both).
- **Translate**: Construct a translation matrix `np.float32([[1, 0, tx], [0, 1, ty]])` and apply `cv2.warpAffine`.

## 2. Enhancement (Image Quality & Preprocessing)
- **Convert Color**: Use `cv2.cvtColor(src, code)`. Common codes: `cv2.COLOR_BGR2GRAY`, `cv2.COLOR_BGR2HSV`, `cv2.COLOR_BGR2RGB`.
- **Adjust Brightness/Contrast**: Use `cv2.convertScaleAbs(src, alpha=contrast, beta=brightness)` or simple math operations.
- **Histogram Eq**: Use `cv2.equalizeHist(src)` for grayscale or `cv2.createCLAHE(...)` for localized contrast.
- **Binarize (Thresholding)**: Use `cv2.threshold(src, thresh, maxval, type)`. (Explore types like `cv2.THRESH_BINARY`, `cv2.THRESH_OTSU`, `cv2.THRESH_ADAPTIVE...`).
- **Blur/Denoise**:
    - Gaussian: `cv2.GaussianBlur(src, ksize, sigmaX)` to smooth noise.
    - Median: `cv2.medianBlur(src, ksize)` for salt-and-pepper noise.
- **Morphology**: Use `cv2.morphologyEx(src, op, kernel)`. (Ops: `cv2.MORPH_OPEN`, `cv2.MORPH_CLOSE`, `cv2.MORPH_DILATE`, `cv2.MORPH_ERODE`).
- **Color Filter**: Use `cv2.inRange(src, lower_bound, upper_bound)` to create a mask (recommended in HSV space).

## 3. Feature Extraction (Structure & Segmentation)
- **Edge Detect**: Use `cv2.Canny(image, threshold1, threshold2, ...)` (Tune thresholds carefully).
- **Line Detect**: Use `cv2.HoughLinesP(...)`. Key params: `rho`, `theta`, `threshold`, `minLineLength`, `maxLineGap`.
- **Circle Detect**: Use `cv2.HoughCircles(...)`. Key params: `method`, `dp`, `minDist`, `param1`, `param2`, `minRadius`, `maxRadius`.
- **Template Match**: Use `cv2.matchTemplate(image, templ, method)` followed by `cv2.minMaxLoc`.
- **Connected Components**: Use `cv2.connectedComponentsWithStats(image, ...)` to label blobs and extract stats like centroids/area.

## 4. Draw & Measure (Visualization & Analysis)
- **Find Contours**: Use `cv2.findContours(image, mode, method)`. (Modes: `cv2.RETR_EXTERNAL`, `cv2.RETR_TREE`; Methods: `cv2.CHAIN_APPROX_SIMPLE`).
- **Measure Area**: Use `cv2.contourArea(contour)`.
- **Measure Perimeter**: Use `cv2.arcLength(contour, closed=True)`.
- **Approximate Polygon**: Use `cv2.approxPolyDP(curve, epsilon, closed)`.
- **Draw**:
    - Contours: `cv2.drawContours(...)`.
    - Shapes: `cv2.line`, `cv2.rectangle`, `cv2.circle`, `cv2.putText`.
'''

SIMPLE_SYS_PROMPT="You are a helpful assistant."

def generate_prompt_simple_qa(user_question):
    # Construct the prompt based on the given requirements
    prompt = f"""
You are an advanced AI assistant specializing in visual question answering (VQA). You don't need to perform any image manipulation or reasoning. Give the answer to the following question directly.
**User's Question:** "{user_question}"
"""
    return prompt

def generate_prompt_final_qa(user_question, user_image_path):
    # Construct the prompt based on the given requirements
    try:
        with Image.open(user_image_path) as img:
            user_image_size = f"{img.width}x{img.height}"
    except Exception as e:
        user_image_size = "Unable to determine (error reading image)"

    prompt = f"""<image>
{user_question}

### User Image Path:** "{user_image_path}"
### User Image Size:** "{user_image_size}"

### **Output Format (strict adherence required):**

<think>Your detailed reasoning process, including any code, should go here.</think>
<answer>Your final answer to the user's question goes here.</answer>
"""
    return prompt

SPECIAL_STRING_LIST=["</code>", "</answer>"]
