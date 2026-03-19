import json
import os
import base64
from io import BytesIO
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image

from qwen_agent.llm.schema import ContentItem
from qwen_agent.utils.utils import load_image_from_base64, logger


# Default parameter presets (for reference, but not enforced)
DEFAULT_PRESETS = {
    "canny_low": (50, 100),
    "canny_medium": (100, 200),
    "canny_high": (150, 250),
}


def _parse_bgr_color(value, default=(0, 255, 0)):
    """
    Normalize a color value to a BGR tuple with integer components.
    """
    if isinstance(value, (list, tuple)) and len(value) == 3:
        try:
            b, g, r = [int(max(0, min(255, v))) for v in value]
            return (b, g, r)
        except (TypeError, ValueError):
            pass
    return tuple(default)

def _parse_color_range(params: Dict, colorspace: str = "hsv"):
    """
    Parse lower/upper bounds for inRange. Expects lists of 3 ints.
    """
    lower = params.get("lower", None)
    upper = params.get("upper", None)
    if not (isinstance(lower, (list, tuple)) and isinstance(upper, (list, tuple)) and len(lower) == 3 and len(upper) == 3):
        raise ValueError("lower and upper must be lists of three integers, e.g., lower:[0,50,50], upper:[10,255,255]")
    lower = np.array(lower, dtype=np.uint8)
    upper = np.array(upper, dtype=np.uint8)
    return lower, upper


def pil_to_base64(pil_image: Image.Image) -> str:
    """
    Convert a PIL Image to base64 string with data URI prefix.

    Args:
        pil_image: PIL Image object

    Returns:
        Base64 encoded string with data URI prefix
    """
    buffered = BytesIO()
    # Convert to RGB if necessary
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    pil_image.save(buffered, format='JPEG')
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return 'data:image/jpeg;base64,' + img_str


def _find_image_file(filename: str) -> str:
    """
    Intelligently find an image file by searching common image directories.
    
    Args:
        filename: The filename or relative path to search for
        
    Returns:
        The full path to the image file if found
        
    Raises:
        ValueError: If the file cannot be found
    """
    # If it's already an absolute path that exists, return it
    if os.path.isabs(filename) and os.path.exists(filename):
        return filename
    
    # List of common base directories to search
    common_dirs = [
        os.getcwd(),  # Current working directory
        os.path.expanduser('~'),  # Home directory
        '/Users/xuanyuzhu/benchmark/code/Benchmark/tool_server/tf_eval/tasks/DiverseToolBench/sample_data/images',
        '/tmp/images',
    ]
    
    # First, try direct concatenation with common directories
    for base_dir in common_dirs:
        candidate_path = os.path.join(base_dir, filename)
        if os.path.exists(candidate_path):
            logger.debug(f'Found image at: {candidate_path}')
            return candidate_path
    
    # If filename doesn't contain path separators, search recursively in image directories
    if os.path.sep not in filename and '/' not in filename:
        search_root = '/Users/xuanyuzhu/benchmark/code/Benchmark/tool_server/tf_eval/tasks/DiverseToolBench/sample_data/images'
        if os.path.exists(search_root):
            for root, dirs, files in os.walk(search_root):
                if filename in files:
                    full_path = os.path.join(root, filename)
                    logger.debug(f'Found image via recursive search at: {full_path}')
                    return full_path
    
    # If still not found, raise an error
    raise ValueError(f"Image file '{filename}' not found in any known location")


def load_image(image_input: str) -> Image.Image:
    """
    Load an image from either a base64 string or a file path with intelligent path correction.

    Args:
        image_input: Can be:
            - Base64 string (with or without data URI prefix like 'data:image/jpeg;base64,')
            - Local file path (absolute or relative)
            - HTTP/HTTPS URL

    Returns:
        PIL Image object
    """
    # Handle base64 data URI format
    if image_input.startswith('data:image') or image_input.startswith('data:;base64,'):
        # Extract base64 data part
        if ';base64,' in image_input:
            base64_data = image_input.split(';base64,')[1]
        else:
            base64_data = image_input.split(',')[1] if ',' in image_input else image_input
        return load_image_from_base64(base64_data)

    # Handle file:// prefix
    if image_input.startswith('file://'):
        image_input = image_input[len('file://'):]

    # Handle HTTP/HTTPS URLs
    if image_input.startswith(('http://', 'https://')):
        import requests
        response = requests.get(image_input)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))

    # Handle local file path
    if os.path.exists(image_input):
        return Image.open(image_input)
    
    # Try to find the image file using intelligent search
    try:
        found_path = _find_image_file(image_input)
        return Image.open(found_path)
    except ValueError:
        pass  # Continue to base64 attempt

    # Try to parse as raw base64 (without prefix)
    try:
        return load_image_from_base64(image_input)
    except Exception:
        raise ValueError(f"Unable to load image from input: {image_input[:100]}...")


def _parse_params(generate_param) -> Dict:
    if isinstance(generate_param, dict):
        return generate_param
    if isinstance(generate_param, str):
        try:
            return json.loads(generate_param)
        except json.JSONDecodeError:
            return {}
    return {}


def _prepare_image(image) -> Tuple[Image.Image, np.ndarray]:
    pil_img = load_image(image)
    return pil_img, cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def _cv2_to_pil(cv_img: np.ndarray) -> Image.Image:
    rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_img)


def _colorspace(cv_img, mode: str):
    mode = mode.lower()
    if mode == "gray":
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), "Converted to grayscale."
    if mode == "hsv":
        hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
        return hsv, "Converted to HSV."
    if mode == "lab":
        lab = cv2.cvtColor(cv_img, cv2.COLOR_BGR2LAB)
        return lab, "Converted to LAB."
    raise ValueError("colorspace mode must be one of: gray, hsv, lab.")


def _resize(cv_img, params: Dict):
    h_orig, w_orig = cv_img.shape[:2]
    preset = params.get("preset")
    target = None
    if preset:
        preset = str(preset).lower()
        if preset == "half":
            scale = 0.5
        elif preset == "double":
            scale = 2.0
        else:
            raise ValueError("resize preset must be one of: half, double.")
        # Allow any scale value decided by the model
        target = (int(w_orig * scale), int(h_orig * scale))
    else:
        width = int(params.get("width", 0))
        height = int(params.get("height", 0))
        # Allow any positive size, but add reasonable bounds to prevent memory issues
        if width <= 0 or height <= 0:
            raise ValueError(f"width and height must be positive integers, got width={width}, height={height}.")
        if width > 10000 or height > 10000:
            raise ValueError(f"width and height should not exceed 10000 pixels to prevent memory issues, got width={width}, height={height}.")
        target = (width, height)
    resized = cv2.resize(cv_img, target, interpolation=cv2.INTER_LINEAR)
    info = {
        "original_size": [w_orig, h_orig],
        "new_size": [target[0], target[1]],
        "scale": [target[0] / w_orig, target[1] / h_orig] if preset else None
    }
    msg = f"Resized from {w_orig}x{h_orig} to {target[0]}x{target[1]}. INFO: {json.dumps(info)}"
    return resized, msg


def _rotate(cv_img, angle: float):
    """
    Rotate image by arbitrary angle (in degrees, clockwise).
    Uses cv2.getRotationMatrix2D and cv2.warpAffine for arbitrary angles.
    """
    angle = float(angle) if angle is not None else 0.0
    
    if angle == 0:
        return cv_img, "No rotation applied (angle=0)."
    
    h, w = cv_img.shape[:2]
    center = (w / 2, h / 2)
    
    # Get rotation matrix
    # Note: OpenCV rotates counter-clockwise for positive angles,
    # so we negate the angle to make it clockwise
    rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)
    
    # Calculate new bounding box size to fit rotated image
    abs_cos = abs(rotation_matrix[0, 0])
    abs_sin = abs(rotation_matrix[0, 1])

    new_w = int(h * abs_sin + w * abs_cos)
    new_h = int(h * abs_cos + w * abs_sin)

    # Adjust rotation matrix to account for translation
    rotation_matrix[0, 2] += new_w / 2 - center[0]
    rotation_matrix[1, 2] += new_h / 2 - center[1]

    # Perform rotation
    rotated = cv2.warpAffine(cv_img, rotation_matrix, (new_w, new_h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(0, 0, 0))

    info = {
        "angle": angle,
        "center": [center[0], center[1]],
        "original_size": [w, h],
        "rotated_size": [new_w, new_h]
    }
    msg = f"Rotated by {angle} degrees clockwise. INFO: {json.dumps(info)}"
    return rotated, msg


def _translate(cv_img, params: dict):
    """
    Translate (shift) the image in the specified direction by a custom distance.

    Args:
        cv_img: Input image (BGR format)
        params: Dictionary with:
            - "direction": "left", "right", "up", or "down"
            - "distance": optional custom distance in pixels (default: 32)

    Returns:
        Tuple of (translated_image, message)
    """
    direction = str(params.get("direction", "right")).lower()

    # Get custom distance or use default
    distance = params.get("distance")
    if distance is not None:
        try:
            distance = int(distance)
            if distance < 0:
                raise ValueError(f"distance must be non-negative, got {distance}.")
            if distance > 10000:
                raise ValueError(f"distance should not exceed 10000 pixels, got {distance}.")
        except (TypeError, ValueError) as e:
            raise ValueError(f"distance must be a non-negative integer: {e}")
    else:
        # Use default 32 pixels if not specified
        distance = 32

    # Calculate dx, dy based on direction
    if direction == "left":
        dx, dy = -distance, 0
    elif direction == "right":
        dx, dy = distance, 0
    elif direction == "up":
        dx, dy = 0, -distance
    elif direction == "down":
        dx, dy = 0, distance
    else:
        raise ValueError(f"direction must be one of: left, right, up, down. Got: {direction}")

    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    h, w = cv_img.shape[:2]
    shifted = cv2.warpAffine(cv_img, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)

    info = {
        "direction": direction,
        "distance": distance,
        "translation": [int(dx), int(dy)],
        "total_pixels": abs(dx) + abs(dy)
    }
    msg = f"Translated {direction} by {distance} pixels. INFO: {json.dumps(info)}"
    return shifted, msg


def _flip(cv_img, direction: str):
    direction = direction.lower()
    flip_code = 1 if direction == "horizontal" else 0 if direction == "vertical" else None
    if flip_code is None:
        raise ValueError("flip direction must be horizontal or vertical.")
    flipped = cv2.flip(cv_img, flip_code)
    return flipped, f"Flipped image {direction}."


def _blur(cv_img, params: Dict):
    method = str(params.get("method", "gaussian")).lower()
    
    if method == "average" or method == "avg":
        ksize = int(params.get("ksize", 5))
        # Kernel size must be odd and positive, limit to reasonable range
        if ksize < 3 or ksize > 51:
            raise ValueError(f"Average blur ksize must be between 3 and 51, got {ksize}.")
        if ksize % 2 == 0:
            ksize += 1  # Make it odd
        blurred = cv2.blur(cv_img, (ksize, ksize))
        info = {"method": "average", "ksize": ksize}
        msg = f"Applied average blur {ksize}x{ksize}. INFO: {json.dumps(info)}"
        return blurred, msg
    
    elif method == "gaussian":
        ksize = int(params.get("ksize", 5))
        sigma_x = float(params.get("sigma_x", 0))
        sigma_y = float(params.get("sigma_y", 0))
        # Kernel size must be odd and positive, limit to reasonable range
        if ksize < 3 or ksize > 51:
            raise ValueError(f"Gaussian blur ksize must be between 3 and 51, got {ksize}.")
        if ksize % 2 == 0:
            ksize += 1  # Make it odd
        # Limit sigma values
        if sigma_x < 0 or sigma_x > 10:
            raise ValueError(f"Gaussian blur sigma_x must be between 0 and 10, got {sigma_x}.")
        if sigma_y < 0 or sigma_y > 10:
            raise ValueError(f"Gaussian blur sigma_y must be between 0 and 10, got {sigma_y}.")
        blurred = cv2.GaussianBlur(cv_img, (ksize, ksize), sigma_x, sigma_y)
        info = {"method": "gaussian", "ksize": ksize, "sigma_x": sigma_x, "sigma_y": sigma_y}
        msg = f"Applied Gaussian blur {ksize}x{ksize} (sigma_x={sigma_x}, sigma_y={sigma_y}). INFO: {json.dumps(info)}"
        return blurred, msg
    
    elif method == "median":
        ksize = int(params.get("ksize", 5))
        # Kernel size must be odd and positive, limit to reasonable range
        if ksize < 3 or ksize > 51:
            raise ValueError(f"Median blur ksize must be between 3 and 51, got {ksize}.")
        if ksize % 2 == 0:
            ksize += 1  # Make it odd
        blurred = cv2.medianBlur(cv_img, ksize)
        info = {"method": "median", "ksize": ksize}
        msg = f"Applied median blur k={ksize}. INFO: {json.dumps(info)}"
        return blurred, msg
    
    elif method == "bilateral":
        d = int(params.get("d", 9))
        sigma_color = float(params.get("sigma_color", 75))
        sigma_space = float(params.get("sigma_space", 75))
        # Limit parameters to reasonable ranges
        if d < 1 or d > 50:
            raise ValueError(f"Bilateral filter d must be between 1 and 50, got {d}.")
        if sigma_color < 1 or sigma_color > 200:
            raise ValueError(f"Bilateral filter sigma_color must be between 1 and 200, got {sigma_color}.")
        if sigma_space < 1 or sigma_space > 200:
            raise ValueError(f"Bilateral filter sigma_space must be between 1 and 200, got {sigma_space}.")
        blurred = cv2.bilateralFilter(cv_img, d, sigma_color, sigma_space)
        info = {"method": "bilateral", "d": d, "sigma_color": sigma_color, "sigma_space": sigma_space}
        msg = f"Applied bilateral filter (d={d}, sigma_color={sigma_color}, sigma_space={sigma_space}). INFO: {json.dumps(info)}"
        return blurred, msg
    
    else:
        raise ValueError(f"Blur method must be one of: average, gaussian, median, bilateral. Got: {method}.")


def _threshold(cv_img, params: Dict = None):
    """
    Apply thresholding to create a binary image.
    
    Args:
        cv_img: Input image (BGR format)
        params: Dictionary with:
            - "mode": "binary", "otsu", "adaptive_mean", or "adaptive_gaussian" (default: "otsu")
            - "invert": bool, whether to invert the result (default: False)
            - "color_mode": "grayscale" (default) or "bgr" (process each channel separately)
            - "threshold_value": int for binary mode (default: 127, range: 0-255)
            - "adaptive_block_size": int for adaptive modes (default: 11, must be odd)
            - "adaptive_constant": int for adaptive modes (default: 2)
            - "channels": list of channel indices for BGR mode (default: [0, 1, 2])
    
    Returns:
        Tuple of (thresholded_image, message)
    """
    if params is None:
        params = {}
    
    mode = str(params.get("mode", "otsu")).lower()
    invert = bool(params.get("invert", False))
    color_mode = str(params.get("color_mode", "grayscale")).lower()
    
    if mode not in ["binary", "otsu", "adaptive_mean", "adaptive_gaussian"]:
        raise ValueError("threshold mode must be binary, otsu, adaptive_mean, or adaptive_gaussian.")
    
    if color_mode == "grayscale":
        # Convert to grayscale and apply threshold
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        
        if mode == "binary":
            threshold_value = int(params.get("threshold_value", 127))
            threshold_value = max(0, min(255, threshold_value))
            threshold_val, th = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
        elif mode == "otsu":
            threshold_val, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif mode == "adaptive_mean":
            block_size = int(params.get("adaptive_block_size", 11))
            if block_size % 2 == 0:
                block_size += 1
            block_size = max(3, min(block_size, 101))
            constant = int(params.get("adaptive_constant", 2))
            threshold_val = -1
            th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, constant)
        elif mode == "adaptive_gaussian":
            block_size = int(params.get("adaptive_block_size", 11))
            if block_size % 2 == 0:
                block_size += 1
            block_size = max(3, min(block_size, 101))
            constant = int(params.get("adaptive_constant", 2))
            threshold_val = -1
            th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, constant)
        
        if invert:
            th = 255 - th
        
        white_pixels = int(np.sum(th == 255))
        total_pixels = th.size
        info = {
            "mode": mode,
            "color_mode": "grayscale",
            "threshold": float(threshold_val) if 'threshold_val' in locals() else -1.0,
            "white_pixels": white_pixels,
            "white_ratio": float(white_pixels) / total_pixels,
            "invert": invert
        }
        msg = f"Applied {mode} threshold (grayscale mode). INFO: {json.dumps(info)}"
        return cv2.cvtColor(th, cv2.COLOR_GRAY2BGR), msg
    
    elif color_mode == "bgr":
        # Apply threshold to each BGR channel separately
        channels = params.get("channels", [0, 1, 2])
        result = cv_img.copy()
        
        if mode == "binary":
            threshold_value = int(params.get("threshold_value", 127))
            threshold_value = max(0, min(255, threshold_value))
            for ch in channels:
                if 0 <= ch < 3:
                    _, result[:, :, ch] = cv2.threshold(cv_img[:, :, ch], threshold_value, 255, cv2.THRESH_BINARY)
        elif mode == "otsu":
            for ch in channels:
                if 0 <= ch < 3:
                    _, result[:, :, ch] = cv2.threshold(cv_img[:, :, ch], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif mode == "adaptive_mean":
            block_size = int(params.get("adaptive_block_size", 11))
            if block_size % 2 == 0:
                block_size += 1
            block_size = max(3, min(block_size, 101))
            constant = int(params.get("adaptive_constant", 2))
            for ch in channels:
                if 0 <= ch < 3:
                    result[:, :, ch] = cv2.adaptiveThreshold(cv_img[:, :, ch], 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, constant)
        elif mode == "adaptive_gaussian":
            block_size = int(params.get("adaptive_block_size", 11))
            if block_size % 2 == 0:
                block_size += 1
            block_size = max(3, min(block_size, 101))
            constant = int(params.get("adaptive_constant", 2))
            for ch in channels:
                if 0 <= ch < 3:
                    result[:, :, ch] = cv2.adaptiveThreshold(cv_img[:, :, ch], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, constant)
        
        if invert:
            result = 255 - result
        
        white_pixels = int(np.sum(result == 255))
        total_pixels = result.size
        info = {
            "mode": mode,
            "color_mode": "bgr",
            "channels": channels,
            "white_pixels": white_pixels,
            "white_ratio": float(white_pixels) / total_pixels,
            "invert": invert
        }
        msg = f"Applied {mode} threshold to BGR channels {channels}. INFO: {json.dumps(info)}"
        return result, msg
    
    else:
        raise ValueError("color_mode must be 'grayscale' or 'bgr'.")


def _morphology(cv_img, params: Dict = None):
    """
    Apply morphological operations (erode, dilate, open, close).
    
    Args:
        cv_img: Input image (BGR format)
        params: Dictionary with:
            - "op": "erode", "dilate", "open", or "close" (default: "open")
            - "kernel_size": int, kernel size (default: 3, range: 3-21, must be odd)
            - "iterations": int, number of iterations (default: 1, range: 1-10)
            - "kernel_shape": "rect" or "ellipse" (default: "rect")
    
    Returns:
        Tuple of (morphed_image, message)
    """
    if params is None:
        params = {}
    
    op = str(params.get("op", "open")).lower()
    kernel_size = int(params.get("kernel_size", 3))
    iterations = int(params.get("iterations", 1))
    kernel_shape = str(params.get("kernel_shape", "rect")).lower()
    
    # Validate and adjust kernel size
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel_size = max(3, min(kernel_size, 21))
    
    # Validate iterations
    iterations = max(1, min(int(iterations), 10))
    
    # Create kernel
    if kernel_shape == "ellipse":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    elif kernel_shape == "rect":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    else:
        raise ValueError("kernel_shape must be 'rect' or 'ellipse'.")
    
    # Apply morphological operation
    if op == "erode":
        res = cv2.erode(cv_img, kernel, iterations=iterations)
    elif op == "dilate":
        res = cv2.dilate(cv_img, kernel, iterations=iterations)
    elif op == "open":
        res = cv2.morphologyEx(cv_img, cv2.MORPH_OPEN, kernel, iterations=iterations)
    elif op == "close":
        res = cv2.morphologyEx(cv_img, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    else:
        raise ValueError("morphology op must be 'erode', 'dilate', 'open', or 'close'.")
    
    info = {
        "op": op,
        "kernel_size": kernel_size,
        "kernel_shape": kernel_shape,
        "iterations": iterations
    }
    return res, f"Applied {op} with {kernel_shape} kernel {kernel_size}x{kernel_size} for {iterations} iterations. INFO: {json.dumps(info)}"


def _gradients(cv_img, mode: str):
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    mode = mode.lower()
    if mode == "sobel_x":
        grad = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    elif mode == "sobel_y":
        grad = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    elif mode == "laplacian":
        grad = cv2.Laplacian(gray, cv2.CV_64F)
    else:
        raise ValueError("gradient mode must be sobel_x, sobel_y, or laplacian.")
    grad_abs = cv2.convertScaleAbs(grad)
    return cv2.cvtColor(grad_abs, cv2.COLOR_GRAY2BGR), f"Computed {mode} gradient."


def _canny(cv_img, params: Dict = None):
    """
    Detect edges using Canny edge detector.
    
    Args:
        cv_img: Input image (BGR format)
        params: Dictionary with:
            - "preset": "low", "medium", or "high" (default: "medium")
            - "threshold_low": int for custom low threshold (overrides preset)
            - "threshold_high": int for custom high threshold (overrides preset)
            - "color_mode": "grayscale" (default) or "bgr" (process each channel separately)
            - "channels": list of channel indices for BGR mode (default: [0, 1, 2])
    
    Returns:
        Tuple of (edge_image, message)
    """
    if params is None:
        params = {}
    
    color_mode = str(params.get("color_mode", "grayscale")).lower()
    
    # Get thresholds
    preset = str(params.get("preset", "medium")).lower()
    if "threshold_low" in params and "threshold_high" in params:
        low = int(params.get("threshold_low", 100))
        high = int(params.get("threshold_high", 200))
        low = max(0, min(255, low))
        high = max(0, min(255, high))
        if low > high:
            low, high = high, low
        preset_name = "custom"
    else:
        # Use preset values as defaults, but allow model to override
        preset_defaults = {
            "low": (50, 100),
            "medium": (100, 200),
            "high": (150, 250),
        }
        if preset in preset_defaults:
            low, high = preset_defaults[preset]
            preset_name = preset
        else:
            # If preset is invalid, use medium as default
            low, high = preset_defaults.get("medium", (100, 200))
            preset_name = "medium (default)"
    
    if color_mode == "grayscale":
        # Convert to grayscale and apply Canny
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, low, high)
        edge_pixels = int(np.count_nonzero(edges))
        total_pixels = edges.size
        ratio = edge_pixels / total_pixels if total_pixels else 0
        info = {
            "preset": preset_name,
            "color_mode": "grayscale",
            "thresholds": [low, high],
            "edge_pixels": edge_pixels,
            "edge_ratio": float(ratio)
        }
        msg = f"Applied Canny edges ({low}, {high}) in grayscale mode. INFO: {json.dumps(info)}"
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), msg
    
    elif color_mode == "bgr":
        # Apply Canny to each BGR channel separately
        channels = params.get("channels", [0, 1, 2])
        result = np.zeros_like(cv_img)
        total_edge_pixels = 0
        
        for ch in channels:
            if 0 <= ch < 3:
                edges = cv2.Canny(cv_img[:, :, ch], low, high)
                result[:, :, ch] = edges
                total_edge_pixels += int(np.count_nonzero(edges))
        
        total_pixels = result.size
        ratio = total_edge_pixels / total_pixels if total_pixels else 0
        info = {
            "preset": preset_name,
            "color_mode": "bgr",
            "channels": channels,
            "thresholds": [low, high],
            "edge_pixels": total_edge_pixels,
            "edge_ratio": float(ratio)
        }
        msg = f"Applied Canny edges ({low}, {high}) to BGR channels {channels}. INFO: {json.dumps(info)}"
        return result, msg
    
    else:
        raise ValueError("color_mode must be 'grayscale' or 'bgr'.")


def _pyramid(cv_img, mode: str):
    h_orig, w_orig = cv_img.shape[:2]
    mode = mode.lower()
    if mode == "pyr_down":
        resized = cv2.pyrDown(cv_img)
        h_new, w_new = resized.shape[:2]
        info = {"mode": mode, "original_size": [w_orig, h_orig], "new_size": [w_new, h_new]}
        msg = f"Pyramid downsample (1 level). INFO: {json.dumps(info)}"
        return resized, msg
    if mode == "pyr_up":
        resized = cv2.pyrUp(cv_img)
        h_new, w_new = resized.shape[:2]
        info = {"mode": mode, "original_size": [w_orig, h_orig], "new_size": [w_new, h_new]}
        msg = f"Pyramid upsample (1 level). INFO: {json.dumps(info)}"
        return resized, msg
    raise ValueError("pyramid mode must be pyr_down or pyr_up.")


def _find_and_rank_contours(cv_img, params: Dict):
    """
    Detect contours and select a subset based on rank or max_contours.
    Returns (all_contours, selected_records, info_dict).
    """
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    canny_low = int(params.get("canny_low", 100))
    canny_high = int(params.get("canny_high", 200))
    edges = cv2.Canny(gray, canny_low, canny_high)

    mode = str(params.get("mode", "external")).lower()
    retrieval = cv2.RETR_EXTERNAL if mode == "external" else cv2.RETR_LIST
    contours, _ = cv2.findContours(edges, retrieval, cv2.CHAIN_APPROX_SIMPLE)

    contour_records = []
    for idx, c in enumerate(contours):
        x, y, w, h = cv2.boundingRect(c)
        area = float(cv2.contourArea(c))
        contour_records.append(
            {
                "index": idx,
                "contour": c,
                "area": area,
                "bbox": {
                    "x": int(x),
                    "y": int(y),
                    "width": int(w),
                    "height": int(h)
                }
            }
        )
    contour_records.sort(key=lambda x: x["area"], reverse=True)

    rank = params.get("rank")
    try:
        max_contours = int(params.get("max_contours", 20))
    except (TypeError, ValueError):
        max_contours = 20

    if rank is not None:
        try:
            rank = int(rank)
        except (TypeError, ValueError):
            rank = 1
        if rank < 1 or rank > len(contour_records):
            rank = 1
        selected = contour_records[rank - 1: rank] if contour_records else []
    else:
        selected = contour_records[:max_contours] if max_contours > 0 else contour_records

    selected_info = [
        {
            "index": rec["index"],
            "area": rec["area"],
            "bbox": rec["bbox"],
        }
        for rec in selected
    ]
    info = {
        "total_contours": len(contours),
        "returned_contours": len(selected),
        "mode": mode,
        "canny": {"low": int(canny_low), "high": int(canny_high)},
        "contours": selected_info
    }
    return contours, selected, info


def _contours(cv_img, params: Dict):
    """
    Finds contours in the image with configurable parameters.
    Parameters:
        mode: "external" or "list" (default: "external")
        canny_low: Canny edge detection low threshold (default: 100)
        canny_high: Canny edge detection high threshold (default: 200)
        rank: Return specific contour by rank (1 = largest, default: None = return all up to max_contours)
        max_contours: Maximum number of contours to return (default: 20)
    """
    contours, selected, info = _find_and_rank_contours(cv_img, params)
    canvas = cv_img.copy()
    if contours:
        canvas = cv2.drawContours(canvas, contours, -1, (0, 255, 0), 2)
    msg = f"Found {info['total_contours']} contours, returned {info['returned_contours']}. INFO: {json.dumps(info)}"
    return canvas, msg


def _draw_contours(cv_img, params: Dict):
    """
    Draws contours using cv2.drawContours with configurable color/thickness and selection.
    Parameters:
        color: Optional BGR list/tuple (default: [0,255,0])
        thickness: Line thickness (default: 2)
        draw_all: If false, only draws selected/ranked contours (default: True)
        mode/canny_low/canny_high/rank/max_contours: follow _contours parameters
    """
    contours, selected, info = _find_and_rank_contours(cv_img, params)

    draw_all = params.get("draw_all", True)
    if isinstance(draw_all, str):
        draw_all = draw_all.lower() not in ["false", "0", "no"]
    else:
        draw_all = bool(draw_all)

    color = _parse_bgr_color(params.get("color", (0, 255, 0)), (0, 255, 0))
    try:
        thickness = int(params.get("thickness", 2))
    except (TypeError, ValueError):
        thickness = 2
    thickness = max(1, thickness)

    target_contours = contours if draw_all else [rec["contour"] for rec in selected]
    canvas = cv_img.copy()
    if target_contours:
        canvas = cv2.drawContours(canvas, target_contours, -1, color, thickness)

    info.update(
        {
            "color": list(color),
            "thickness": thickness,
            "draw_all": draw_all,
            "drawn_contours": len(target_contours)
        }
    )
    msg = f"Drew contours with cv2.drawContours. INFO: {json.dumps(info)}"
    return canvas, msg


def _draw_line(cv_img, params: Dict):
    """
    Draws a line segment between two points.
    Parameters:
        x1, y1: Start point coordinates
        x2, y2: End point coordinates
        color: Optional BGR list/tuple (default: [0, 255, 0])
        thickness: Line thickness (default: 2)
    """
    try:
        x1 = int(params.get("x1"))
        y1 = int(params.get("y1"))
        x2 = int(params.get("x2"))
        y2 = int(params.get("y2"))
    except (TypeError, ValueError):
        raise ValueError("x1, y1, x2, y2 must be valid integers.")

    color = _parse_bgr_color(params.get("color", (0, 255, 0)), (0, 255, 0))
    try:
        thickness = int(params.get("thickness", 2))
    except (TypeError, ValueError):
        thickness = 2
    thickness = max(1, thickness)

    canvas = cv_img.copy()
    cv2.line(canvas, (x1, y1), (x2, y2), color, thickness)
    
    info = {
        "start": [x1, y1],
        "end": [x2, y2],
        "color": list(color),
        "thickness": thickness
    }
    return canvas, f"Drew line from ({x1}, {y1}) to ({x2}, {y2}). INFO: {json.dumps(info)}"


def _contour_area(cv_img, params: Dict):
    """
    Calculates contour areas using cv2.contourArea and overlays the values.
    Shares selection parameters with _contours.
    """
    contours, selected, info = _find_and_rank_contours(cv_img, params)
    color = _parse_bgr_color(params.get("color", (255, 0, 0)), (255, 0, 0))
    try:
        thickness = int(params.get("thickness", 2))
    except (TypeError, ValueError):
        thickness = 2
    thickness = max(1, thickness)

    canvas = cv_img.copy()
    areas = []
    for rec in selected:
        area = float(rec["area"])
        areas.append({"index": rec["index"], "area": area, "bbox": rec["bbox"]})
        if rec["contour"] is not None:
            cv2.drawContours(canvas, [rec["contour"]], -1, color, thickness)
        bbox = rec["bbox"]
        label_pos = (int(bbox["x"]), max(0, int(bbox["y"] - 5)))
        cv2.putText(canvas, f"{area:.1f}", label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    info["areas"] = areas
    info["measurement"] = "contour_area"
    msg = f"Calculated contour areas with cv2.contourArea. INFO: {json.dumps(info)}"
    return canvas, msg


def _arc_length(cv_img, params: Dict):
    """
    Computes contour perimeters using cv2.arcLength and overlays the values.
    Shares selection parameters with _contours.
    """
    contours, selected, info = _find_and_rank_contours(cv_img, params)
    color = _parse_bgr_color(params.get("color", (0, 0, 255)), (0, 0, 255))
    try:
        thickness = int(params.get("thickness", 2))
    except (TypeError, ValueError):
        thickness = 2
    thickness = max(1, thickness)

    canvas = cv_img.copy()
    perimeters = []
    for rec in selected:
        perimeter = float(cv2.arcLength(rec["contour"], True))
        perimeters.append({"index": rec["index"], "perimeter": perimeter, "bbox": rec["bbox"]})
        if rec["contour"] is not None:
            cv2.drawContours(canvas, [rec["contour"]], -1, color, thickness)
        bbox = rec["bbox"]
        cx = int(bbox["x"] + bbox["width"] / 2)
        cy = int(bbox["y"] + bbox["height"] / 2)
        cv2.putText(canvas, f"{perimeter:.1f}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    info["perimeters"] = perimeters
    info["measurement"] = "arc_length"
    msg = f"Computed contour perimeters with cv2.arcLength. INFO: {json.dumps(info)}"
    return canvas, msg


def _approx_poly(cv_img, params: Dict):
    """
    Approximates contours using cv2.approxPolyDP for simplified polygon representation.
    Parameters:
        epsilon_ratio: fraction of contour perimeter to use for approximation (default: 0.02)
        color/thickness and contour selection parameters mirror _contours
    """
    contours, selected, info = _find_and_rank_contours(cv_img, params)
    try:
        epsilon_ratio = float(params.get("epsilon_ratio", 0.02))
    except (TypeError, ValueError):
        epsilon_ratio = 0.02
    epsilon_ratio = max(0.0, epsilon_ratio)

    color = _parse_bgr_color(params.get("color", (255, 0, 255)), (255, 0, 255))
    try:
        thickness = int(params.get("thickness", 2))
    except (TypeError, ValueError):
        thickness = 2
    thickness = max(1, thickness)

    canvas = cv_img.copy()
    approx_info = []
    for rec in selected:
        peri = cv2.arcLength(rec["contour"], True)
        epsilon = epsilon_ratio * peri
        approx = cv2.approxPolyDP(rec["contour"], epsilon, True)
        if approx is not None and approx.size > 0:
            cv2.drawContours(canvas, [approx], -1, color, thickness)
            centroid = np.mean(approx.reshape(-1, 2), axis=0)
            label_pos = (int(centroid[0]), int(centroid[1]))
            cv2.putText(canvas, f"{len(approx)}pts", label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            approx_points = approx.reshape(-1, 2).astype(int).tolist()
        else:
            approx_points = []
        approx_info.append(
            {
                "index": rec["index"],
                "points_count": len(approx_points),
                "points": approx_points,
                "bbox": rec["bbox"],
                "epsilon": float(epsilon)
            }
        )

    info["approx"] = approx_info
    info["epsilon_ratio"] = float(epsilon_ratio)
    info["measurement"] = "approx_poly_dp"
    msg = f"Approximated contours with cv2.approxPolyDP. INFO: {json.dumps(info)}"
    return canvas, msg


def _histogram(cv_img, mode: str, params: Dict = None):
    """
    Apply histogram equalization or CLAHE to an image.
    
    Args:
        cv_img: Input image (BGR format)
        mode: "equalize" or "clahe"
        params: Optional dictionary with:
            - "color_mode": "grayscale" (default), "bgr" (process each channel), or "hsv" (process V channel)
            - "channels": List of channel indices to process (e.g., [0, 1, 2] for all BGR channels)
                         Only used when color_mode is "bgr"
            - "clip_limit": CLAHE clip limit (default: 2.0)
            - "tile_grid_size": CLAHE tile grid size (default: (8, 8))
    
    Returns:
        Tuple of (processed_image, message)
    """
    if params is None:
        params = {}
    
    mode = mode.lower()
    color_mode = params.get("color_mode", "grayscale").lower()
    
    if mode not in ["equalize", "clahe"]:
        raise ValueError("histogram mode must be 'equalize' or 'clahe'.")
    
    # CLAHE parameters
    clip_limit = params.get("clip_limit", 2.0)
    tile_grid_size = tuple(params.get("tile_grid_size", [8, 8]))
    
    if mode == "equalize":
        if color_mode == "grayscale":
            # Convert to grayscale and apply histogram equalization
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            eq = cv2.equalizeHist(gray)
            result = cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)
            msg = "Applied histogram equalization (grayscale mode)."
        
        elif color_mode == "bgr":
            # Apply histogram equalization to each BGR channel separately
            channels = params.get("channels", [0, 1, 2])
            result = cv_img.copy()
            for ch in channels:
                if 0 <= ch < 3:
                    result[:, :, ch] = cv2.equalizeHist(cv_img[:, :, ch])
            msg = f"Applied histogram equalization to BGR channels {channels}."
        
        elif color_mode == "hsv":
            # Convert to HSV and apply histogram equalization to V channel only
            hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
            hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
            result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            msg = "Applied histogram equalization to HSV V channel."
        
        else:
            raise ValueError("color_mode must be 'grayscale', 'bgr', or 'hsv'.")
    
    elif mode == "clahe":
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        
        if color_mode == "grayscale":
            # Convert to grayscale and apply CLAHE
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            cl = clahe.apply(gray)
            result = cv2.cvtColor(cl, cv2.COLOR_GRAY2BGR)
            msg = f"Applied CLAHE (grayscale mode, clipLimit={clip_limit}, tileGridSize={tile_grid_size})."
        
        elif color_mode == "bgr":
            # Apply CLAHE to each BGR channel separately
            channels = params.get("channels", [0, 1, 2])
            result = cv_img.copy()
            for ch in channels:
                if 0 <= ch < 3:
                    result[:, :, ch] = clahe.apply(cv_img[:, :, ch])
            msg = f"Applied CLAHE to BGR channels {channels} (clipLimit={clip_limit}, tileGridSize={tile_grid_size})."
        
        elif color_mode == "hsv":
            # Convert to HSV and apply CLAHE to V channel only
            hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
            hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
            result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            msg = f"Applied CLAHE to HSV V channel (clipLimit={clip_limit}, tileGridSize={tile_grid_size})."
        
        else:
            raise ValueError("color_mode must be 'grayscale', 'bgr', or 'hsv'.")
    
    return result, msg


def _dft_magnitude(cv_img):
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1e-5)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    mag_uint8 = magnitude.astype(np.uint8)
    return cv2.cvtColor(mag_uint8, cv2.COLOR_GRAY2BGR), "Computed DFT magnitude spectrum."


def _template_match(cv_img, params: Dict):
    template_path = params.get("template_path")
    method = str(params.get("method", "ccorr")).lower()
    methods = {
        "sqdiff": cv2.TM_SQDIFF,
        "ccorr": cv2.TM_CCORR_NORMED,
        "ccoeff": cv2.TM_CCOEFF_NORMED,
    }
    if method not in methods:
        raise ValueError("template match method must be sqdiff, ccorr, or ccoeff.")
    if not template_path:
        raise ValueError("template_path is required for template_match.")

    tmpl = load_image(template_path)
    tmpl = cv2.cvtColor(np.array(tmpl), cv2.COLOR_RGB2BGR)
    res = cv2.matchTemplate(cv_img, tmpl, methods[method])
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = min_loc if method == "sqdiff" else max_loc
    h, w = tmpl.shape[:2]
    bottom_right = (top_left[0] + w, top_left[1] + h)
    canvas = cv_img.copy()
    cv2.rectangle(canvas, top_left, bottom_right, (0, 0, 255), 2)
    score = min_val if method == "sqdiff" else max_val
    info = {
        "method": method,
        "score": float(score),
        "bbox": {
            'x':int(top_left[0]),
            'y':int(top_left[1]),
            'width':int(w),
            'height':int(h)
        },
        "top_left": {
            'x':int(top_left[0]),
            'y':int(top_left[1])
        },
        "bottom_right": {
            'x':int(bottom_right[0]),
            'y':int(bottom_right[1])
        },
        "template_size": {
            'width':int(w),
            'height':int(h)
        }
    }
    msg = f"Template match ({method}) score {score:.3f}. INFO: {json.dumps(info)}"
    return canvas, msg


def _hough_lines(cv_img, params: Dict = None):
    """
    Detects line segments using Hough transform with configurable parameters.
    Parameters:
        threshold: Accumulator threshold (default: 80, higher = stricter, fewer lines)
        minLineLength: Minimum line length (default: 50, higher = longer lines only)
        maxLineGap: Maximum gap between line segments (default: 10)
        canny_low: Canny edge detection low threshold (default: 80)
        canny_high: Canny edge detection high threshold (default: 200)
        max_lines: Maximum number of lines to return (default: 20)
    """
    if params is None:
        params = {}
    
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    
    # Get Canny parameters
    canny_low = int(params.get("canny_low", 80))
    canny_high = int(params.get("canny_high", 200))
    edges = cv2.Canny(gray, canny_low, canny_high)
    
    # Get HoughLinesP parameters with stricter defaults
    threshold = int(params.get("threshold", 80))  # Increased from 60 to 80
    minLineLength = int(params.get("minLineLength", 50))  # Increased from 30 to 50
    maxLineGap = int(params.get("maxLineGap", 10))
    
    # Validate parameters
    if threshold <= 0:
        raise ValueError("threshold must be positive.")
    if minLineLength <= 0:
        raise ValueError("minLineLength must be positive.")
    if maxLineGap < 0:
        raise ValueError("maxLineGap must be non-negative.")
    
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=threshold, 
                           minLineLength=minLineLength, maxLineGap=maxLineGap)
    canvas = cv_img.copy()
    lines_info = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = float(np.hypot(x2 - x1, y2 - y1))
            lines_info.append({
                "p1": {
                    'x': int(x1),
                    'y': int(y1)
                },
                "p2": {
                    'x': int(x2),
                    'y': int(y2)
                },
                "length": length
            })
            cv2.line(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    lines_info.sort(key=lambda x: x["length"], reverse=True)
    
    # Limit number of returned lines
    max_lines = int(params.get("max_lines", 20))
    if max_lines > 0 and len(lines_info) > max_lines:
        selected = lines_info[:max_lines]
    else:
        selected = lines_info
    
    info = {
        "total_lines": len(lines_info),
        "returned_lines": len(selected),
        "lines": selected,
        "parameters": {
            "threshold": threshold,
            "minLineLength": minLineLength,
            "maxLineGap": maxLineGap,
            "canny_low": canny_low,
            "canny_high": canny_high
        }
    }
    msg = f"Detected {len(lines_info)} line segments, returned {len(selected)} (threshold={threshold}, minLineLength={minLineLength}). INFO: {json.dumps(info)}"
    return canvas, msg


def _hough_circles(cv_img, params: Dict = None):
    """
    Detects circles using Hough transform with configurable parameters.
    Parameters:
        dp: Inverse ratio of accumulator resolution (default: 1.2)
        minDist: Minimum distance between circle centers (default: 50, higher = fewer circles)
        param1: Upper threshold for edge detection (default: 100)
        param2: Accumulator threshold for center detection (default: 50, higher = stricter, fewer circles)
        minRadius: Minimum circle radius (default: 10)
        maxRadius: Maximum circle radius (default: 0 = no limit)
        max_circles: Maximum number of circles to return (default: 20)
    """
    if params is None:
        params = {}
    
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    
    # Get parameters with stricter defaults to reduce false positives
    dp = float(params.get("dp", 1.2))
    minDist = int(params.get("minDist", 50))  # Increased from 30 to 50
    param1 = int(params.get("param1", 100))
    param2 = int(params.get("param2", 50))  # Increased from 30 to 50 for stricter detection
    minRadius = int(params.get("minRadius", 10))
    maxRadius = int(params.get("maxRadius", 0))  # 0 means no limit
    
    # Validate parameters
    if dp <= 0:
        raise ValueError("dp must be positive.")
    if minDist <= 0:
        raise ValueError("minDist must be positive.")
    if param1 <= 0:
        raise ValueError("param1 must be positive.")
    if param2 <= 0:
        raise ValueError("param2 must be positive.")
    if minRadius < 0:
        raise ValueError("minRadius must be non-negative.")
    if maxRadius < 0:
        raise ValueError("maxRadius must be non-negative.")
    if maxRadius > 0 and maxRadius < minRadius:
        raise ValueError("maxRadius must be >= minRadius.")
    
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=dp, minDist=minDist, 
                               param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
    canvas = cv_img.copy()
    circles_info = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for c in circles[0, :]:
            circles_info.append({
                "center": {
                    'x': int(c[0]),
                    'y': int(c[1])
                },
                "radius": int(c[2])
            })
            cv2.circle(canvas, (c[0], c[1]), c[2], (0, 255, 0), 2)
            cv2.circle(canvas, (c[0], c[1]), 2, (255, 0, 0), 3)
    
    circles_info.sort(key=lambda x: x["radius"], reverse=True)
    
    # Limit number of returned circles
    max_circles = int(params.get("max_circles", 20))
    if max_circles > 0 and len(circles_info) > max_circles:
        circles_info = circles_info[:max_circles]
    
    info = {
        "total_circles": len(circles_info),
        "circles": circles_info,
        "parameters": {
            "dp": dp,
            "minDist": minDist,
            "param1": param1,
            "param2": param2,
            "minRadius": minRadius,
            "maxRadius": maxRadius if maxRadius > 0 else None
        }
    }
    msg = f"Detected {len(circles_info)} circles (param2={param2}, minDist={minDist}). INFO: {json.dumps(info)}"
    return canvas, msg


def _watershed(cv_img, params: Dict = None):
    """
    Applies watershed segmentation with configurable parameters.
    Parameters:
        max_regions: Maximum number of region labels to return (default: 20)
    """
    if params is None:
        params = {}
    
    img = cv_img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    cv2.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]
    labels = markers.copy()
    unique_labels = np.unique(labels)
    # exclude border (-1) and background (1)
    region_labels = [int(l) for l in unique_labels if l not in (-1, 1)]
    
    max_regions = int(params.get("max_regions", 20))
    if max_regions > 0 and len(region_labels) > max_regions:
        returned_labels = region_labels[:max_regions]
    else:
        returned_labels = region_labels
    
    info = {
        "total_regions": len(region_labels),
        "returned_regions": len(returned_labels),
        "labels": returned_labels
    }
    msg = f"Applied watershed segmentation. Found {len(region_labels)} regions, returned {len(returned_labels)}. INFO: {json.dumps(info)}"
    return img, msg


def _grabcut(cv_img, params: Dict):
    mode = str(params.get("preset", "center")).lower()
    h, w = cv_img.shape[:2]
    presets = {
        "center": (int(w * 0.1), int(h * 0.1), int(w * 0.8), int(h * 0.8)),
        "tight": (int(w * 0.2), int(h * 0.2), int(w * 0.6), int(h * 0.6)),
        "loose": (int(w * 0.05), int(h * 0.05), int(w * 0.9), int(h * 0.9)),
    }
    if mode not in presets:
        raise ValueError("grabcut preset must be center, tight, or loose.")
    rect = presets[mode]
    mask = np.zeros(cv_img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    cv2.grabCut(cv_img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
    res = cv_img * mask2[:, :, np.newaxis]
    fg_pixels = int(mask2.sum())
    ratio = fg_pixels / (h * w) if h * w else 0
    info = {
        "preset": mode,
        "rect": [int(v) for v in rect],
        "foreground_pixels": fg_pixels,
        "foreground_ratio": float(ratio)
    }
    msg = f"Applied GrabCut with {mode} rectangle. INFO: {json.dumps(info)}"
    return res, msg


def _features(cv_img, method: str, params: Dict = None):
    method = method.lower()
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    canvas = cv_img.copy()
    message = ""

    if method == "harris":
        gray_f = np.float32(gray)
        dst = cv2.cornerHarris(gray_f, 2, 3, 0.04)
        dst = cv2.dilate(dst, None)
        threshold = 0.01 * dst.max()
        corners = np.argwhere(dst > threshold)
        
        try:
            max_points = int(params.get("max_points", 50)) if params and isinstance(params, dict) else 50
        except Exception:
            max_points = 50
        
        # Sort by corner strength and limit
        corner_strengths = [(dst[y, x], (x, y)) for y, x in corners]
        corner_strengths.sort(reverse=True)
        selected_corners = corner_strengths[:max_points] if len(corner_strengths) > max_points else corner_strengths
        
        corner_list = []
        for strength, (x, y) in selected_corners:
            cv2.circle(canvas, (x, y), 3, (0, 0, 255), -1)
            corner_list.append([int(x), int(y)])
        
        message = f"Harris corners: {len(corners)} detected, returned {len(corner_list)}. Points: {json.dumps(corner_list)}"
    elif method == "shi_tomasi":
        corners = cv2.goodFeaturesToTrack(gray, maxCorners=200, qualityLevel=0.01, minDistance=5)
        corners = np.int0(corners) if corners is not None else []
        try:
            max_points = int(params.get("max_points", 50)) if params and isinstance(params, dict) else 50
        except Exception:
            max_points = 50
        corner_list = []
        for c in corners[:max_points]:
            x, y = c.ravel()
            cv2.circle(canvas, (x, y), 3, (0, 255, 0), -1)
            corner_list.append([int(x), int(y)])
        message = f"Shi-Tomasi corners: {len(corners)}. Returned {len(corner_list)} points: {json.dumps(corner_list)}"
    elif method == "sift":
        if not hasattr(cv2, "SIFT_create"):
            raise ValueError("SIFT not available in this OpenCV build.")
        sift = cv2.SIFT_create()
        kps, _ = sift.detectAndCompute(gray, None)
        canvas = cv2.drawKeypoints(cv_img, kps, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        try:
            max_points = int(params.get("max_points", 50)) if params and isinstance(params, dict) else 50
        except Exception:
            max_points = 50
        coords = [[int(k.pt[0]), int(k.pt[1])] for k in kps[:max_points]]
        message = f"SIFT keypoints: {len(kps)}. Returned {len(coords)} points: {json.dumps(coords)}"
    elif method == "surf":
        if not hasattr(cv2, "xfeatures2d") or not hasattr(cv2.xfeatures2d, "SURF_create"):
            raise ValueError("SURF requires xfeatures2d (contrib) build.")
        surf = cv2.xfeatures2d.SURF_create()
        kps, _ = surf.detectAndCompute(gray, None)
        canvas = cv2.drawKeypoints(cv_img, kps, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        try:
            max_points = int(params.get("max_points", 50)) if params and isinstance(params, dict) else 50
        except Exception:
            max_points = 50
        coords = [[int(k.pt[0]), int(k.pt[1])] for k in kps[:max_points]]
        message = f"SURF keypoints: {len(kps)}. Returned {len(coords)} points: {json.dumps(coords)}"
    elif method == "fast":
        fast = cv2.FastFeatureDetector_create()
        kps = fast.detect(gray, None)
        canvas = cv2.drawKeypoints(cv_img, kps, None, color=(255, 0, 0))
        try:
            max_points = int(params.get("max_points", 50)) if params and isinstance(params, dict) else 50
        except Exception:
            max_points = 50
        coords = [[int(k.pt[0]), int(k.pt[1])] for k in kps[:max_points]]
        message = f"FAST keypoints: {len(kps)}. Returned {len(coords)} points: {json.dumps(coords)}"
    elif method == "brief":
        if not hasattr(cv2, "xfeatures2d") or not hasattr(cv2.xfeatures2d, "BriefDescriptorExtractor_create"):
            raise ValueError("BRIEF requires xfeatures2d (contrib) build.")
        fast = cv2.FastFeatureDetector_create()
        kps = fast.detect(gray, None)
        brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
        kps, _ = brief.compute(gray, kps)
        canvas = cv2.drawKeypoints(cv_img, kps, None, color=(0, 255, 0))
        try:
            max_points = int(params.get("max_points", 50)) if params and isinstance(params, dict) else 50
        except Exception:
            max_points = 50
        coords = [[int(k.pt[0]), int(k.pt[1])] for k in kps[:max_points]]
        message = f"BRIEF keypoints: {len(kps)}. Returned {len(coords)} points: {json.dumps(coords)}"
    elif method == "orb":
        orb = cv2.ORB_create()
        kps, _ = orb.detectAndCompute(gray, None)
        canvas = cv2.drawKeypoints(cv_img, kps, None, color=(0, 255, 0), flags=0)
        try:
            max_points = int(params.get("max_points", 50)) if params and isinstance(params, dict) else 50
        except Exception:
            max_points = 50
        coords = [[int(k.pt[0]), int(k.pt[1])] for k in kps[:max_points]]
        message = f"ORB keypoints: {len(kps)}. Returned {len(coords)} points: {json.dumps(coords)}"
    else:
        raise ValueError("feature method must be one of: harris, shi_tomasi, sift, surf, fast, brief, orb.")
    return canvas, message


def _denoise(cv_img, params: Dict):
    """
    Fast non-local means denoising with optional custom parameters.
    
    Args:
        cv_img: Input image (BGR format)
        params: Dictionary with:
            - "mode": "fast_means_gray", "fast_means_color", or "fast_means_bgr_channel" (default: "fast_means_color")
            - "h": float for luminance (default: 10, range: 1-50)
            - "h_color": float for color (default: 10, range: 1-50)
            - "template_window": int, must be odd (default: 7, range: 3-21)
            - "search_window": int, must be odd (default: 21, range: 3-31)
            - "channels": list of channel indices for BGR channel mode (default: [0, 1, 2])
    
    Returns:
        Tuple of (denoised_image, message)
    """
    mode = str(params.get("mode", "fast_means_color")).lower()

    def _validate_window(val, default, name, low, high):
        try:
            v = int(val)
        except Exception:
            v = default
        if v < low or v > high:
            v = default
        if v % 2 == 0:
            v += 1  # ensure odd
        return v

    if mode == "fast_means_gray":
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        try:
            h = float(params.get("h", 10))
        except Exception:
            h = 10.0
        h = max(1.0, min(h, 50.0))
        template_window = _validate_window(params.get("template_window", 7), 7, "template_window", 3, 21)
        search_window = _validate_window(params.get("search_window", 21), 21, "search_window", 3, 31)
        dst = cv2.fastNlMeansDenoising(gray, None, h, template_window, search_window)
        info = {
            "mode": mode,
            "h": float(h),
            "template_window": int(template_window),
            "search_window": int(search_window),
        }
        return cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR), f"Applied fastNlMeansDenoising (grayscale). INFO: {json.dumps(info)}"

    elif mode == "fast_means_color":
        try:
            h = float(params.get("h", 10))
        except Exception:
            h = 10.0
        try:
            h_color = float(params.get("h_color", 10))
        except Exception:
            h_color = 10.0
        h = max(1.0, min(h, 50.0))
        h_color = max(1.0, min(h_color, 50.0))
        template_window = _validate_window(params.get("template_window", 7), 7, "template_window", 3, 21)
        search_window = _validate_window(params.get("search_window", 21), 21, "search_window", 3, 31)
        dst = cv2.fastNlMeansDenoisingColored(cv_img, None, h, h_color, template_window, search_window)
        info = {
            "mode": mode,
            "h": float(h),
            "h_color": float(h_color),
            "template_window": int(template_window),
            "search_window": int(search_window),
        }
        return dst, f"Applied fastNlMeansDenoisingColored. INFO: {json.dumps(info)}"
    
    elif mode == "fast_means_bgr_channel":
        # Apply denoising to each BGR channel separately
        channels = params.get("channels", [0, 1, 2])
        try:
            h = float(params.get("h", 10))
        except Exception:
            h = 10.0
        h = max(1.0, min(h, 50.0))
        template_window = _validate_window(params.get("template_window", 7), 7, "template_window", 3, 21)
        search_window = _validate_window(params.get("search_window", 21), 21, "search_window", 3, 31)
        
        result = cv_img.copy()
        for ch in channels:
            if 0 <= ch < 3:
                result[:, :, ch] = cv2.fastNlMeansDenoising(cv_img[:, :, ch], None, h, template_window, search_window)
        
        info = {
            "mode": mode,
            "channels": channels,
            "h": float(h),
            "template_window": int(template_window),
            "search_window": int(search_window),
        }
        return result, f"Applied fastNlMeansDenoising to BGR channels {channels}. INFO: {json.dumps(info)}"

    else:
        raise ValueError("denoise mode must be fast_means_gray, fast_means_color, or fast_means_bgr_channel.")


def _inpaint(cv_img, params: Dict):
    """
    Inpaint (fill) regions in the image using automatically generated mask.
    
    Args:
        cv_img: Input image (BGR format)
        params: Dictionary with:
            - "preset": "canny" or "threshold" (default: "canny")
            - "canny_low": int for Canny low threshold (default: 100, range: 0-255)
            - "canny_high": int for Canny high threshold (default: 200, range: 0-255)
            - "threshold_value": int for threshold mode (default: 127, range: 0-255)
            - "radius": int for inpaint radius (default: 3, range: 1-10)
            - "method": "telea" or "ns" (default: "telea")
    
    Returns:
        Tuple of (inpainted_image, message)
    """
    strategy = str(params.get("preset", "canny")).lower()
    radius = int(params.get("radius", 3))
    radius = max(1, min(radius, 10))
    method = str(params.get("method", "telea")).lower()
    
    if method == "telea":
        inpaint_method = cv2.INPAINT_TELEA
    elif method == "ns":
        inpaint_method = cv2.INPAINT_NS
    else:
        raise ValueError("inpaint method must be 'telea' or 'ns'.")
    
    if strategy == "canny":
        canny_low = int(params.get("canny_low", 100))
        canny_high = int(params.get("canny_high", 200))
        canny_low = max(0, min(255, canny_low))
        canny_high = max(0, min(255, canny_high))
        if canny_low > canny_high:
            canny_low, canny_high = canny_high, canny_low
        
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, canny_low, canny_high)
        mask = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        msg_detail = f"Canny ({canny_low}, {canny_high})"
    
    elif strategy == "threshold":
        threshold_value = int(params.get("threshold_value", 127))
        threshold_value = max(0, min(255, threshold_value))
        
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
        msg_detail = f"threshold ({threshold_value})"
    
    else:
        raise ValueError("inpaint preset must be 'canny' or 'threshold'.")
    
    res = cv2.inpaint(cv_img, mask, radius, inpaint_method)
    info = {
        "preset": strategy,
        "mask_strategy": msg_detail,
        "radius": radius,
        "method": method
    }
    return res, f"Inpainted regions using {strategy} mask. INFO: {json.dumps(info)}"


def _inrange_color(cv_img, params: Dict):
    """
    Create a mask for pixels within a given color range.
    
    Args:
        cv_img: Input image (BGR format)
        params: Dictionary with:
            - "colorspace": "hsv" (default) or "bgr"
            - "lower": [h/b, s/g, v/r] - lower bound for color range
            - "upper": [h/b, s/g, v/r] - upper bound for color range
            - "output_format": "mask" (default), "masked_image", or "both"
                - "mask": returns binary mask
                - "masked_image": returns original image with mask applied
                - "both": returns masked image if pixels found, else mask
    
    Returns:
        Tuple of (output_image, message)
    """
    colorspace = str(params.get("colorspace", "hsv")).lower()
    output_format = str(params.get("output_format", "both")).lower()
    lower, upper = _parse_color_range(params, colorspace)

    if colorspace == "hsv":
        proc = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
    elif colorspace == "bgr":
        proc = cv_img.copy()
    else:
        raise ValueError("colorspace must be 'hsv' or 'bgr'.")

    mask = cv2.inRange(proc, lower, upper)
    count = int(np.count_nonzero(mask))
    ratio = count / mask.size if mask.size else 0
    
    # Prepare output based on format
    if output_format == "mask":
        # Return binary mask as 3-channel image
        result = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    elif output_format == "masked_image":
        # Return original image with mask applied
        result = cv2.bitwise_and(cv_img, cv_img, mask=mask)
    elif output_format == "both":
        # Return masked image if pixels found, else mask
        masked_img = cv2.bitwise_and(cv_img, cv_img, mask=mask)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        result = masked_img if count > 0 else mask_bgr
    else:
        raise ValueError("output_format must be 'mask', 'masked_image', or 'both'.")
    
    info = {
        "colorspace": colorspace,
        "lower": lower.tolist(),
        "upper": upper.tolist(),
        "output_format": output_format,
        "mask_pixels": count,
        "mask_ratio": float(ratio)
    }
    msg = f"Applied color range mask ({colorspace}). INFO: {json.dumps(info)}"
    return result, msg


def _python_opencv(cv_img, params: Dict):
    """
    Execute custom OpenCV/Numpy code on the provided image.
    Expect user code to define:
      - result_img: a BGR numpy array to return (or leave None if no image)
      - message: a short string describing the result
    Available variables: cv2, np, Image, json, math, image (BGR numpy array)
    """
    import math
    import json as _json

    code = params.get("code", "")
    if not isinstance(code, str) or not code.strip():
        raise ValueError("code must be a non-empty string")

    local_vars = {
        "cv2": cv2,
        "np": np,
        "Image": Image,
        "math": math,
        "json": _json,
        "image": cv_img.copy(),
        "result_img": None,
        "message": "",
    }
    try:
        exec(code, {}, local_vars)
    except Exception as e:
        raise ValueError(f"Execution failed: {e}")

    result_img = local_vars.get("result_img", None)
    message = str(local_vars.get("message", "")) if local_vars.get("message", "") is not None else ""

    if result_img is None:
        # no image output; just return the original image for consistency
        result_img = cv_img.copy()
    if not isinstance(result_img, np.ndarray):
        raise ValueError("result_img must be a numpy ndarray (BGR)")

    msg = message if message else "Custom OpenCV code executed."
    return result_img, msg


def _crop(cv_img, params: Dict):
    """
    Crops a region from the image.
    Parameters:
        x: x-coordinate of top-left corner (default: 0)
        y: y-coordinate of top-left corner (default: 0)
        width: width of the crop region (required)
        height: height of the crop region (required)
    """
    h_orig, w_orig = cv_img.shape[:2]
    
    try:
        x = int(params.get("x", 0))
        y = int(params.get("y", 0))
        width = int(params.get("width", 0))
        height = int(params.get("height", 0))
    except (ValueError, TypeError):
        raise ValueError("crop parameters must be integers: x, y, width, height.")
    
    # Validate parameters
    if width <= 0 or height <= 0:
        raise ValueError(f"width and height must be positive integers, got width={width}, height={height}.")
    if x < 0 or y < 0:
        raise ValueError(f"x and y must be non-negative integers, got x={x}, y={y}.")
    if x + width > w_orig or y + height > h_orig:
        raise ValueError(f"Crop region exceeds image bounds. Image size: {w_orig}x{h_orig}, crop: x={x}, y={y}, width={width}, height={height}.")
    
    # Crop the image
    cropped = cv_img[y:y+height, x:x+width]
    
    info = {
        "original_size": [w_orig, h_orig],
        "crop_region": {"x": x, "y": y, "width": width, "height": height},
        "cropped_size": [width, height]
    }
    msg = f"Cropped region ({x}, {y}, {width}, {height}) from {w_orig}x{h_orig} image. INFO: {json.dumps(info)}"
    return cropped, msg


def _zoom_in(cv_img, params: Dict):
    """
    Zooms into a region of the image by cropping and optionally resizing.
    Parameters:
        x: x-coordinate of top-left corner (default: 0)
        y: y-coordinate of top-left corner (default: 0)
        width: width of the region to zoom (required)
        height: height of the region to zoom (required)
        scale: optional scale factor to enlarge the cropped region (default: 1.0, meaning keep original size)
        or target_width/target_height: optional target dimensions to resize the cropped region to
    """
    h_orig, w_orig = cv_img.shape[:2]
    
    try:
        x = int(params.get("x", 0))
        y = int(params.get("y", 0))
        width = int(params.get("width", 0))
        height = int(params.get("height", 0))
    except (ValueError, TypeError):
        raise ValueError("zoom_in parameters must be integers: x, y, width, height.")
    
    # Validate parameters
    if width <= 0 or height <= 0:
        raise ValueError(f"width and height must be positive integers, got width={width}, height={height}.")
    if x < 0 or y < 0:
        raise ValueError(f"x and y must be non-negative integers, got x={x}, y={y}.")
    if x + width > w_orig or y + height > h_orig:
        raise ValueError(f"Zoom region exceeds image bounds. Image size: {w_orig}x{h_orig}, zoom: x={x}, y={y}, width={width}, height={height}.")
    
    # Crop the region
    cropped = cv_img[y:y+height, x:x+width]
    
    # Determine target size
    target_width = None
    target_height = None
    scale = params.get("scale")
    
    if scale is not None:
        try:
            scale = float(scale)
            if scale <= 0 or scale > 10:
                raise ValueError(f"scale must be between 0 and 10, got {scale}.")
            target_width = int(width * scale)
            target_height = int(height * scale)
        except (ValueError, TypeError):
            raise ValueError(f"scale must be a positive number, got {scale}.")
    elif "target_width" in params or "target_height" in params:
        try:
            target_width = int(params.get("target_width", width))
            target_height = int(params.get("target_height", height))
            if target_width <= 0 or target_height <= 0:
                raise ValueError(f"target_width and target_height must be positive integers.")
            if target_width > 10000 or target_height > 10000:
                raise ValueError(f"target_width and target_height should not exceed 10000 pixels.")
        except (ValueError, TypeError):
            raise ValueError("target_width and target_height must be positive integers.")
    
    # Resize if needed
    if target_width and target_height:
        zoomed = cv2.resize(cropped, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        final_size = [target_width, target_height]
    else:
        zoomed = cropped
        final_size = [width, height]
    
    info = {
        "original_size": [w_orig, h_orig],
        "zoom_region": {"x": x, "y": y, "width": width, "height": height},
        "zoomed_size": final_size,
        "scale": scale if scale else None
    }
    msg = f"Zoomed into region ({x}, {y}, {width}, {height}) from {w_orig}x{h_orig} image, result size: {final_size[0]}x{final_size[1]}. INFO: {json.dumps(info)}"
    return zoomed, msg


def _floodFill(cv_img, params: Dict):
    """
    Performs flood fill operation starting from a seed point.
    Parameters:
        x: x-coordinate of seed point (required)
        y: y-coordinate of seed point (required)
        loDiff: lower difference/neighborhood connectivity (default: [10, 10, 10] for BGR)
        upDiff: upper difference/neighborhood connectivity (default: [10, 10, 10] for BGR)
        newVal: new fill color (default: [0, 255, 0] for green in BGR)
        flags: flood fill flags (default: 4 | cv2.FLOODFILL_FIXED_RANGE)
    """
    h_orig, w_orig = cv_img.shape[:2]
    
    try:
        x = int(params.get("x", 0))
        y = int(params.get("y", 0))
    except (ValueError, TypeError):
        raise ValueError("floodFill parameters x and y must be integers.")
    
    # Validate seed point
    if x < 0 or x >= w_orig or y < 0 or y >= h_orig:
        raise ValueError(f"Seed point ({x}, {y}) is out of image bounds. Image size: {w_orig}x{h_orig}.")
    
    # Get optional parameters
    loDiff = params.get("loDiff", [10, 10, 10])
    upDiff = params.get("upDiff", [10, 10, 10])
    newVal = params.get("newVal", [0, 255, 0])  # Green in BGR
    
    # Convert to tuples if lists
    if isinstance(loDiff, list):
        loDiff = tuple(loDiff[:3]) if len(loDiff) >= 3 else (10, 10, 10)
    if isinstance(upDiff, list):
        upDiff = tuple(upDiff[:3]) if len(upDiff) >= 3 else (10, 10, 10)
    if isinstance(newVal, list):
        newVal = tuple(newVal[:3]) if len(newVal) >= 3 else (0, 255, 0)
    
    # Default flags: 4-connected, fixed range
    flags = params.get("flags", 4 | cv2.FLOODFILL_FIXED_RANGE)
    try:
        flags = int(flags)
    except (ValueError, TypeError):
        flags = 4 | cv2.FLOODFILL_FIXED_RANGE
    
    # Create a copy for flood fill
    img_copy = cv_img.copy()
    mask = np.zeros((h_orig + 2, w_orig + 2), np.uint8)
    
    # Perform flood fill
    retval, img_result, mask_result, rect = cv2.floodFill(
        img_copy, mask, (x, y), newVal, loDiff, upDiff, flags
    )
    
    # Count filled pixels
    filled_pixels = int(np.sum(mask_result[1:-1, 1:-1] > 0))
    total_pixels = h_orig * w_orig
    fill_ratio = float(filled_pixels) / total_pixels if total_pixels > 0 else 0
    
    info = {
        "seed_point": {"x": x, "y": y},
        "filled_pixels": filled_pixels,
        "fill_ratio": fill_ratio,
        "new_color": list(newVal) if isinstance(newVal, tuple) else newVal,
        "rect": {
            "x": int(rect[0]),
            "y": int(rect[1]),
            "width": int(rect[2]),
            "height": int(rect[3])
        } if rect else None
    }
    msg = f"Flood fill from seed point ({x}, {y}), filled {filled_pixels} pixels ({fill_ratio*100:.2f}%). INFO: {json.dumps(info)}"
    return img_result, msg


def _connectedComponentsWithStats(cv_img, params: Dict):
    """
    Finds connected components in a binary image and returns statistics.
    Parameters:
        threshold_mode: method to create binary image - "otsu", "binary", or "none" (default: "otsu")
        threshold_value: threshold value for binary mode (default: 127)
        connectivity: connectivity type - 4 or 8 (default: 8)
    """
    h_orig, w_orig = cv_img.shape[:2]
    
    # Convert to grayscale if needed
    if len(cv_img.shape) == 3:
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv_img.copy()
    
    # Create binary image based on mode
    threshold_mode = str(params.get("threshold_mode", "otsu")).lower()
    connectivity = int(params.get("connectivity", 8))
    
    if threshold_mode == "otsu":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif threshold_mode == "binary":
        threshold_value = int(params.get("threshold_value", 127))
        _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
    elif threshold_mode == "none":
        binary = gray
    else:
        raise ValueError("threshold_mode must be one of: otsu, binary, none.")
    
    if connectivity not in [4, 8]:
        raise ValueError("connectivity must be 4 or 8.")
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=connectivity)
    
    # Create visualization: draw bounding boxes and centroids
    canvas = cv_img.copy()
    components_info = []
    
    # Skip background (label 0)
    for i in range(1, num_labels):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])
        cx = float(centroids[i, 0])
        cy = float(centroids[i, 1])
        
        # Draw bounding box
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Draw centroid
        cv2.circle(canvas, (int(cx), int(cy)), 3, (0, 0, 255), -1)
        
        components_info.append({
            "label": i,
            "bbox": {
                "x": x,
                "y": y,
                "width": w,
                "height": h
            },
            "area": area,
            "centroid": {
                "x": cx,
                "y": cy
            }
        })
    
    # Sort by area (largest first)
    components_info.sort(key=lambda x: x["area"], reverse=True)
    
    # Limit number of returned components for readability
    max_components = int(params.get("max_components", 20))
    if max_components > 0:
        components_info = components_info[:max_components]
    
    info = {
        "total_components": num_labels - 1,  # Exclude background
        "returned_components": len(components_info),
        "connectivity": connectivity,
        "threshold_mode": threshold_mode,
        "components": components_info
    }
    msg = f"Found {num_labels - 1} connected components (connectivity={connectivity}). INFO: {json.dumps(info)}"
    return canvas, msg


def _convertScaleAbs(cv_img, params: Dict):
    """
    Converts image to absolute value and scales it to 0-255 range.
    This is useful for visualizing gradient images or other signed data.
    Parameters:
        alpha: optional scaling factor (default: 1.0)
        beta: optional offset (default: 0)
    """
    try:
        alpha = float(params.get("alpha", 1.0))
        beta = float(params.get("beta", 0))
    except (TypeError, ValueError):
        alpha = 1.0
        beta = 0
    
    # Ensure alpha is positive
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}.")
    
    # Convert to float for processing
    img_float = cv_img.astype(np.float32)
    
    # Apply scaling and offset
    scaled = cv2.convertScaleAbs(img_float * alpha + beta)
    
    info = {
        "alpha": float(alpha),
        "beta": float(beta),
        "output_range": [0, 255],
        "data_type": "uint8"
    }
    msg = f"Applied convertScaleAbs with alpha={alpha}, beta={beta}. INFO: {json.dumps(info)}"
    return scaled, msg


def _draw_circle(cv_img, params: Dict):
    """
    Draws a circle on the image.
    Parameters:
        x: x-coordinate of circle center (required)
        y: y-coordinate of circle center (required)
        radius: radius of the circle (required)
        color: optional BGR list/tuple (default: [0, 255, 0])
        thickness: line thickness, -1 for filled circle (default: 2)
    """
    h_orig, w_orig = cv_img.shape[:2]
    
    try:
        x = int(params.get("x"))
        y = int(params.get("y"))
        radius = int(params.get("radius"))
    except (TypeError, ValueError):
        raise ValueError("x, y, and radius must be valid integers.")
    
    # Validate circle parameters
    if radius <= 0:
        raise ValueError(f"radius must be positive, got {radius}.")
    if x < 0 or x >= w_orig or y < 0 or y >= h_orig:
        raise ValueError(f"Circle center ({x}, {y}) is out of image bounds. Image size: {w_orig}x{h_orig}.")
    
    color = _parse_bgr_color(params.get("color", (0, 255, 0)), (0, 255, 0))
    try:
        thickness = int(params.get("thickness", 2))
    except (TypeError, ValueError):
        thickness = 2
    
    canvas = cv_img.copy()
    cv2.circle(canvas, (x, y), radius, color, thickness)
    
    info = {
        "center": [x, y],
        "radius": radius,
        "color": list(color),
        "thickness": thickness,
        "filled": thickness == -1
    }
    return canvas, f"Drew circle at ({x}, {y}) with radius {radius}. INFO: {json.dumps(info)}"


def _opencv_floodfill(cv_img, params: Dict):
    """
    Performs flood fill operation starting from a seed point.
    Parameters:
        x: x-coordinate of seed point (required)
        y: y-coordinate of seed point (required)
        loDiff: lower difference/neighborhood connectivity (default: [10, 10, 10] for BGR)
        upDiff: upper difference/neighborhood connectivity (default: [10, 10, 10] for BGR)
        newVal: new fill color (default: [0, 255, 0] for green in BGR)
        flags: flood fill flags (default: 4 | cv2.FLOODFILL_FIXED_RANGE)
    """
    h_orig, w_orig = cv_img.shape[:2]
    
    try:
        x = int(params.get("x", 0))
        y = int(params.get("y", 0))
    except (ValueError, TypeError):
        raise ValueError("floodfill parameters x and y must be integers.")
    
    # Validate seed point
    if x < 0 or x >= w_orig or y < 0 or y >= h_orig:
        raise ValueError(f"Seed point ({x}, {y}) is out of image bounds. Image size: {w_orig}x{h_orig}.")
    
    # Get optional parameters
    loDiff = params.get("loDiff", [10, 10, 10])
    upDiff = params.get("upDiff", [10, 10, 10])
    newVal = params.get("newVal", [0, 255, 0])  # Green in BGR
    
    # Convert to tuples if lists
    if isinstance(loDiff, list):
        loDiff = tuple(loDiff[:3]) if len(loDiff) >= 3 else (10, 10, 10)
    if isinstance(upDiff, list):
        upDiff = tuple(upDiff[:3]) if len(upDiff) >= 3 else (10, 10, 10)
    if isinstance(newVal, list):
        newVal = tuple(newVal[:3]) if len(newVal) >= 3 else (0, 255, 0)
    
    # Default flags: 4-connected, fixed range
    flags = params.get("flags", 4 | cv2.FLOODFILL_FIXED_RANGE)
    try:
        flags = int(flags)
    except (ValueError, TypeError):
        flags = 4 | cv2.FLOODFILL_FIXED_RANGE
    
    # Create a copy for flood fill
    img_copy = cv_img.copy()
    mask = np.zeros((h_orig + 2, w_orig + 2), np.uint8)
    
    # Perform flood fill
    retval, img_result, mask_result, rect = cv2.floodFill(
        img_copy, mask, (x, y), newVal, loDiff, upDiff, flags
    )
    
    # Count filled pixels
    filled_pixels = int(np.sum(mask_result[1:-1, 1:-1] > 0))
    total_pixels = h_orig * w_orig
    fill_ratio = float(filled_pixels) / total_pixels if total_pixels > 0 else 0
    
    info = {
        "seed_point": {"x": x, "y": y},
        "filled_pixels": filled_pixels,
        "fill_ratio": fill_ratio,
        "new_color": list(newVal) if isinstance(newVal, tuple) else newVal,
        "rect": {
            "x": int(rect[0]),
            "y": int(rect[1]),
            "width": int(rect[2]),
            "height": int(rect[3])
        } if rect else None
    }
    msg = f"Flood fill from seed point ({x}, {y}), filled {filled_pixels} pixels ({fill_ratio*100:.2f}%). INFO: {json.dumps(info)}"
    return img_result, msg


def _apply_op(cv_img, op_name: str, params: Dict):
    op_name = op_name.lower()
    if op_name == "colorspace_gray":
        return _colorspace(cv_img, "gray")
    if op_name == "colorspace_hsv":
        return _colorspace(cv_img, "hsv")
    if op_name == "colorspace_lab":
        return _colorspace(cv_img, "lab")
    if op_name == "resize":
        return _resize(cv_img, params)
    if op_name == "rotate":
        return _rotate(cv_img, int(params.get("angle", 0)))
    if op_name == "translate":
        return _translate(cv_img, params)
    if op_name == "flip":
        return _flip(cv_img, params.get("direction", "horizontal"))
    if op_name == "blur":
        return _blur(cv_img, params)
    if op_name == "threshold":
        return _threshold(cv_img, params)
    if op_name == "morphology":
        return _morphology(cv_img, params)
    if op_name == "gradients":
        return _gradients(cv_img, params.get("mode", "sobel_x"))
    if op_name == "canny":
        return _canny(cv_img, params)
    if op_name == "pyramid":
        return _pyramid(cv_img, params.get("mode", "pyr_down"))
    if op_name == "contours":
        return _contours(cv_img, params)
    if op_name == "draw_contours":
        return _draw_contours(cv_img, params)
    if op_name == "draw_line":
        return _draw_line(cv_img, params)
    if op_name == "contour_area":
        return _contour_area(cv_img, params)
    if op_name == "arc_length":
        return _arc_length(cv_img, params)
    if op_name == "approx_poly":
        return _approx_poly(cv_img, params)
    if op_name == "histogram":
        return _histogram(cv_img, params.get("mode", "equalize"), params)
    if op_name == "dft":
        return _dft_magnitude(cv_img)
    if op_name == "template_match":
        return _template_match(cv_img, params)
    if op_name == "hough_lines":
        return _hough_lines(cv_img, params)
    if op_name == "hough_circles":
        return _hough_circles(cv_img, params)
    if op_name == "watershed":
        return _watershed(cv_img, params)
    if op_name == "grabcut":
        return _grabcut(cv_img, params)
    if op_name == "features":
        return _features(cv_img, params.get("method", "orb"), params)
    if op_name == "denoise":
        return _denoise(cv_img, params)
    if op_name == "inpaint":
        return _inpaint(cv_img, params)
    if op_name == "inrange_color":
        return _inrange_color(cv_img, params)
    if op_name == "python_opencv":
        return _python_opencv(cv_img, params)
    if op_name == "crop":
        return _crop(cv_img, params)
    if op_name == "zoom_in":
        return _zoom_in(cv_img, params)
    if op_name == "floodfill":
        return _floodFill(cv_img, params)
    if op_name == "connected_components_with_stats":
        return _connectedComponentsWithStats(cv_img, params)
    if op_name == "convertscaleabs":
        return _convertScaleAbs(cv_img, params)
    if op_name == "draw_circle":
        return _draw_circle(cv_img, params)
    if op_name == "opencv_floodfill":
        return _opencv_floodfill(cv_img, params)
    raise ValueError(f"Unsupported operation '{op_name}'.")


def generate(params, op_name=None, **kwargs):
    generate_param = params.get("param", {})
    image = params.get("image")
    
    try:
        op_params = _parse_params(generate_param)
        pil_img, cv_img = _prepare_image(image)
        if op_name is None:
            op_name = op_params.get("operation", "")
        processed_img, message = _apply_op(cv_img, op_name, op_params)
        result_pil = _cv2_to_pil(processed_img)

        # Save image to local file
        # Get work_dir from params or use default
        work_dir = params.get("work_dir", "workspace/tools/opencv")
        os.makedirs(work_dir, exist_ok=True)

        # Generate unique filename
        import uuid
        image_filename = f"{uuid.uuid4()}.png"
        image_path = os.path.join(work_dir, image_filename)

        # Save the image
        result_pil.save(image_path)
        logger.info(f"Saved OpenCV result image to: {image_path}")

        # Convert to absolute path to avoid path extraction issues
        abs_image_path = os.path.abspath(image_path)

        # Combine message with image path information
        message_with_path = f"{message}\n\nEdited Image Path: {abs_image_path}"

        # Return as List[ContentItem] with image and text
        return [
            ContentItem(text=message_with_path),
            ContentItem(image=abs_image_path),
        ]

    except Exception as exc:
        error_str = str(exc)
        logger.error(f"opencv_ops_worker failed ({op_name}): {exc}")

        # Check if this is a path-related error
        is_path_error = ("Unable to load image" in error_str or
                        "No such file or directory" in error_str or
                        "cannot find the file" in error_str.lower() or
                        "file not found" in error_str.lower())

        error_msg = f"Failed to run OpenCV operation '{op_name}':\n{error_str}"
        error_msg += f"\n\n**Action Required**: Please call this tool again with absolute image path"

        if is_path_error:
            # Extract image path from messages for error handling
            abs_image_path = kwargs['messages'][1].content[0].image
            error_msg += (
                f"\n Set the 'image' parameter to: {abs_image_path}"
            )

        # Return error message as ContentItem
        return [ContentItem(text=error_msg)]
    finally:
        # 【新增】显式释放大对象内存
        try:
            if 'pil_img' in locals(): del pil_img
            if 'cv_img' in locals(): del cv_img
            if 'processed_img' in locals(): del processed_img
            if 'result_pil' in locals(): del result_pil

            # 强制垃圾回收
            import gc
            gc.collect()
        except:
            pass
