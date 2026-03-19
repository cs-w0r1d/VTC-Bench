import json
from typing import Dict, Optional, Union
from qwen_agent.tools.base import BaseTool, register_tool
from qwen_agent.tools.opencv_interface import generate

@register_tool('opencv_colorspace_gray')
class OpencvColorspaceGray(BaseTool):
    description = 'Converts the image to a different color space (grayscale, HSV, or LAB). Useful for enhancing contrast or isolating specific color channels.'
    parameters = {
        "type": "object",
        "properties": {
            "image": {
                "type": "string",
                "description": "The input image identifier"
            },
            "param": {
                "type": "object",
                "properties": {},
                "required": []
            }
        },
        "required": [
            "image"
        ]
    }

    def call(self, params: Union[str, dict], **kwargs) -> dict:
        params = self._verify_json_format_args(params)
        return generate(params, op_name='colorspace_gray', **kwargs)

@register_tool('opencv_colorspace_hsv')
class OpencvColorspaceHsv(BaseTool):
    description = 'Converts the image to a different color space (grayscale, HSV, or LAB). Useful for enhancing contrast or isolating specific color channels.'
    parameters = {
        "type": "object",
        "properties": {
            "image": {
                "type": "string",
                "description": "The input image identifier"
            },
            "param": {
                "type": "object",
                "properties": {},
                "required": []
            }
        },
        "required": [
            "image"
        ]
    }

    def call(self, params: Union[str, dict], **kwargs) -> dict:
        params = self._verify_json_format_args(params)
        return generate(params, op_name='colorspace_hsv', **kwargs)

@register_tool('opencv_colorspace_lab')
class OpencvColorspaceLab(BaseTool):
    description = 'Converts the image to a different color space (grayscale, HSV, or LAB). Useful for enhancing contrast or isolating specific color channels.'
    parameters = {
        "type": "object",
        "properties": {
            "image": {
                "type": "string",
                "description": "The input image identifier"
            },
            "param": {
                "type": "object",
                "properties": {},
                "required": []
            }
        },
        "required": [
            "image"
        ]
    }

    def call(self, params: Union[str, dict], **kwargs) -> dict:
        params = self._verify_json_format_args(params)
        return generate(params, op_name='colorspace_lab', **kwargs)

@register_tool('opencv_resize')
class OpencvResize(BaseTool):
    description = 'Resizes the image to specified dimensions or by a preset scale (half or double). You can specify any positive integer width and height (recommended range: 1-10000 pixels). Returns detailed size information including original and new dimensions.'
    parameters = {
    "type": "object",
    "properties": {
        "image": {
            "type": "string",
            "description": "The input image identifier"
        },
        "param": {
            "type": "object",
            "properties": {
                "width": {
                    "type": "integer",
                    "description": "<positive integer>"
                },
                "height": {
                    "type": "integer",
                    "description": "<positive integer>"
                },
                "preset": {
                    "type": "string",
                    "description": "Resize preset: 'half' or 'double'. Model can decide the value."
                }
            },
            "required": ["width", "height"]
        }
    },
    "required": [
        "image","param"
    ]
}

    def call(self, params: Union[str, dict], **kwargs) -> dict:
        params = self._verify_json_format_args(params)
        return generate(params, op_name='resize', **kwargs)

@register_tool('opencv_rotate')
class OpencvRotate(BaseTool):
    description = 'Rotates the image by specified angle in degrees (clockwise). Supports arbitrary angles (e.g., 45, 90, 180, 270, etc.). The output image will be resized to fit the entire rotated content. Returns rotation details including angle, center point, and size changes.'
    parameters = {
    "type": "object",
    "properties": {
        "image": {
            "type": "string",
            "description": "The input image identifier"
        },
        "param": {
            "type": "object",
            "properties": {
                "angle": {
                    "type": "number",
                    "description": "Rotation angle in degrees (clockwise). Can be any number, e.g., 45, 90, 180, 270, etc."
                }
            }
        }
    },
    "required": [
        "image","param"
    ]
}

    def call(self, params: Union[str, dict], **kwargs) -> dict:
        params = self._verify_json_format_args(params)
        return generate(params, op_name='rotate', **kwargs)

@register_tool('opencv_translate')
class OpencvTranslate(BaseTool):
    description = 'Shifts the image by a specified number of pixels in the specified direction (left, right, up, or down). You can specify any positive integer distance (recommended range: 1-10000 pixels). Default distance is 32 pixels. Returns translation details including direction, distance, and translation vector.'
    parameters = {
    "type": "object",
    "properties": {
        "image": {
            "type": "string",
            "description": "The input image identifier"
        },
        "param": {
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "description": "Translation direction: 'left', 'right', 'up', or 'down'. Model can decide the value."
                },
                "distance": {
                    "type": "integer",
                    "description": "Translation distance in pixels (any non-negative integer, default: 32, recommended range: 1-10000)"
                }
            }
        }
    },
    "required": [
        "image","param"
    ]
}

    def call(self, params: Union[str, dict], **kwargs) -> dict:
        params = self._verify_json_format_args(params)
        return generate(params, op_name='translate', **kwargs)

@register_tool('opencv_flip')
class OpencvFlip(BaseTool):
    description = 'Flips the image horizontally or vertically.'
    parameters = {
    "type": "object",
    "properties": {
        "image": {
            "type": "string",
            "description": "The input image identifier"
        },
        "param": {
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "description": "Flip direction: 'horizontal' or 'vertical'. Model can decide the value."
                }
            }
        }
    },
    "required": [
        "image","param"
    ]
}

    def call(self, params: Union[str, dict], **kwargs) -> dict:
        params = self._verify_json_format_args(params)
        return generate(params, op_name='flip', **kwargs)

@register_tool('opencv_blur')
class OpencvBlur(BaseTool):
    description = 'Applies blurring to reduce noise or smooth the image. Supports multiple blur methods: average (simple box blur), gaussian (weighted blur), median (good for salt-pepper noise), and bilateral (edge-preserving). Returns blur method and parameter details.'
    parameters = {
    "type": "object",
    "properties": {
        "image": {
            "type": "string",
            "description": "The input image identifier"
        },
        "param": {
            "type": "object",
            "properties": {
                "method": {
                    "type": "string",
                    "description": "Blur method: 'average'/'avg' (simple box blur), 'gaussian' (weighted blur, default), 'median' (good for salt-pepper noise), 'bilateral' (edge-preserving). Model can decide the value."
                },
                "ksize": {
                    "type": "integer",
                    "description": "Kernel size (must be odd, 3-51). Used for average, gaussian, and median blur. Default: 5"
                },
                "sigma_x": {
                    "type": "number",
                    "description": "Gaussian kernel standard deviation in X direction (0-10). Only for gaussian blur. Default: 0 (auto-calculated from ksize)"
                },
                "sigma_y": {
                    "type": "number",
                    "description": "Gaussian kernel standard deviation in Y direction (0-10). Only for gaussian blur. Default: 0 (auto-calculated from ksize)"
                },
                "d": {
                    "type": "integer",
                    "description": "Diameter of pixel neighborhood (1-50). Only for bilateral filter. Default: 9"
                },
                "sigma_color": {
                    "type": "number",
                    "description": "Filter sigma in color space (1-200). Only for bilateral filter. Default: 75"
                },
                "sigma_space": {
                    "type": "number",
                    "description": "Filter sigma in coordinate space (1-200). Only for bilateral filter. Default: 75"
                }
            },
            "required": ["method","ksize"]
        }
    },
    "required": [
        "image","param"
    ]
}

    def call(self, params: Union[str, dict], **kwargs) -> dict:
        params = self._verify_json_format_args(params)
        return generate(params, op_name='blur', **kwargs)

@register_tool('opencv_threshold')
class OpencvThreshold(BaseTool):
    description = 'Applies thresholding to create a binary image. Supports multiple color modes (grayscale or BGR channel-wise) and various threshold methods.'
    parameters = {
    "type": "object",
    "properties": {
        "image": {
            "type": "string",
            "description": "The input image identifier"
        },
        "param": {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "description": "'binary|otsu|adaptive_mean|adaptive_gaussian' (default: 'otsu'). Model can decide the value."
                },
                "invert": {
                    "type": "boolean",
                    "description": "true|false (default: false)"
                },
                "color_mode": {
                    "type": "string",
                    "description": "'grayscale|bgr' (default: 'grayscale'). Model can decide the value."
                },
                "threshold_value": {
                    "type": "integer",
                    "description": "<int 0-255> (for binary mode)"
                },
                "adaptive_block_size": {
                    "type": "integer",
                    "description": "<odd int 3-101> (default: 11)"
                },
                "adaptive_constant": {
                    "type": "integer",
                    "description": "<int> (default: 2)"
                },
                "channels": {
                    "type": "array",
                    "items": {
                        "type": "integer"
                    },
                    "description": "List of channel indices [0, 1, 2]"
                }
            },
            "required": ["mode"]
        }
    },
    "required": [
        "image","param"
    ]
}

    def call(self, params: Union[str, dict], **kwargs) -> dict:
        params = self._verify_json_format_args(params)
        return generate(params, op_name='threshold', **kwargs)

@register_tool('opencv_morphology')
class OpencvMorphology(BaseTool):
    description = 'Applies morphological operations (erode, dilate, open, close) with flexible kernel size and shape.'
    parameters = {
    "type": "object",
    "properties": {
        "image": {
            "type": "string",
            "description": "The input image identifier"
        },
        "param": {
            "type": "object",
            "properties": {
                "op": {
                    "type": "string",
                    "description": "'erode|dilate|open|close' (default: 'open'). Model can decide the value."
                },
                "kernel_size": {
                    "type": "integer",
                    "description": "<odd int 3-21> (default: 3)"
                },
                "iterations": {
                    "type": "integer",
                    "description": "<int 1-10> (default: 1)"
                },
                "kernel_shape": {
                    "type": "string",
                    "description": "'rect|ellipse' (default: 'rect'). Model can decide the value."
                }
            }
        }
    },
    "required": [
        "image"
    ]
}

    def call(self, params: Union[str, dict], **kwargs) -> dict:
        params = self._verify_json_format_args(params)
        return generate(params, op_name='morphology', **kwargs)

@register_tool('opencv_gradients')
class OpencvGradients(BaseTool):
    description = 'Computes gradient images using Sobel (x or y direction) or Laplacian operator.'
    parameters = {
    "type": "object",
    "properties": {
        "image": {
            "type": "string",
            "description": "The input image identifier"
        },
        "param": {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "description": "'sobel_x|sobel_y|laplacian'. Model can decide the value."
                }
            },
            "required": ["mode"]
        }
    },
    "required": [
        "image","param"
    ]
}

    def call(self, params: Union[str, dict], **kwargs) -> dict:
        params = self._verify_json_format_args(params)
        return generate(params, op_name='gradients', **kwargs)

@register_tool('opencv_canny')
class OpencvCanny(BaseTool):
    description = 'Detects edges using Canny edge detector. Supports preset thresholds or custom values, and can deal with grayscale or individual BGR channels.'
    parameters = {
    "type": "object",
    "properties": {
        "image": {
            "type": "string",
            "description": "The input image identifier"
        },
        "param": {
            "type": "object",
            "properties": {
                "preset": {
                    "type": "string",
                    "description": "'low|medium|high' (default: 'medium'). Model can decide the value or provide custom threshold_low and threshold_high"
                },
                "threshold_low": {
                    "type": "integer",
                    "description": "<int 0-255> (overrides preset)"
                },
                "threshold_high": {
                    "type": "integer",
                    "description": "<int 0-255> (overrides preset)"
                },
                "color_mode": {
                    "type": "string",
                    "description": "'grayscale|bgr' (default: 'grayscale'). Model can decide the value."
                },
                "channels": {
                    "type": "array",
                    "items": {
                        "type": "integer"
                    },
                    "description": "List of channel indices [0, 1, 2]"
                }
            }
        }
    },
    "required": [
        "image"
    ]
}

    def call(self, params: Union[str, dict], **kwargs) -> dict:
        params = self._verify_json_format_args(params)
        return generate(params, op_name='canny', **kwargs)

@register_tool('opencv_pyramid')
class OpencvPyramid(BaseTool):
    description = 'Applies image pyramid operations (upsample or downsample by one level). Returns size information.'
    parameters = {
    "type": "object",
    "properties": {
        "image": {
            "type": "string",
            "description": "The input image identifier"
        },
        "param": {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "description": "'pyr_up|pyr_down'. Model can decide the value."
                }
            },
            "required": []
        }
    },
    "required": [
        "image","param"
    ]
}

    def call(self, params: Union[str, dict], **kwargs) -> dict:
        params = self._verify_json_format_args(params)
        return generate(params, op_name='pyramid', **kwargs)

@register_tool('opencv_contours')
class OpencvContours(BaseTool):
    description = 'Finds contours using Canny edges and returns bounding boxes/areas. Helpful for locating shapes before further analysis.'
    parameters = {
    "type": "object",
    "properties": {
        "image": {
            "type": "string",
            "description": "The input image identifier"
        },
        "param": {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "description": "'external|list'. Model can decide the value."
                },
                "canny_low": {
                    "type": "integer",
                    "description": "<int default 100>"
                },
                "canny_high": {
                    "type": "integer",
                    "description": "<int default 200>"
                },
                "rank": {
                    "type": "integer",
                    "description": "<optional int 1 = largest>"
                },
                "max_contours": {
                    "type": "integer",
                    "description": "<optional int default 20>"
                }
            },
            "required": []
        }
    },
    "required": [
        "image"
    ]
}

    def call(self, params: Union[str, dict], **kwargs) -> dict:
        params = self._verify_json_format_args(params)
        return generate(params, op_name='contours', **kwargs)

@register_tool('opencv_draw_line')
class OpencvDrawLine(BaseTool):
    description = 'Draws a line segment between two points with configurable color and thickness.'
    parameters = {
    "type": "object",
    "properties": {
        "image": {
            "type": "string",
            "description": "The input image identifier"
        },
        "param": {
            "type": "object",
            "properties": {
                "x1": {
                    "type": "integer",
                    "description": "<int>"
                },
                "y1": {
                    "type": "integer",
                    "description": "<int>"
                },
                "x2": {
                    "type": "integer",
                    "description": "<int>"
                },
                "y2": {
                    "type": "integer",
                    "description": "<int>"
                },
                "color": {
                    "type": "array",
                    "items": {
                        "type": "integer"
                    },
                    "description": "BGR color [B, G, R]"
                },
                "thickness": {
                    "type": "integer",
                    "description": "<int>"
                }
            },
            "required": ["x1","x2","y1","y2"]
        }
    },
    "required": [
        "image","param"
    ]
}

    def call(self, params: Union[str, dict], **kwargs) -> dict:
        params = self._verify_json_format_args(params)
        return generate(params, op_name='draw_line', **kwargs)

@register_tool('opencv_contour_area')
class OpencvContourArea(BaseTool):
    description = 'Calculates contour areas using cv2.contourArea and overlays the values near each contour. Useful for comparing object sizes.'
    parameters = {
    "type": "object",
    "properties": {
        "image": {
            "type": "string",
            "description": "The input image identifier"
        },
        "param": {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "description": "'external|list'. Model can decide the value."
                },
                "canny_low": {
                    "type": "integer",
                    "description": "<int>"
                },
                "canny_high": {
                    "type": "integer",
                    "description": "<int>"
                },
                "rank": {
                    "type": "integer",
                    "description": "<optional int>"
                },
                "max_contours": {
                    "type": "integer",
                    "description": "<optional int>"
                },
                "color": {
                    "type": "array",
                    "items": {
                        "type": "integer"
                    },
                    "description": "BGR color [B, G, R]"
                },
                "thickness": {
                    "type": "integer",
                    "description": "<int>"
                }
            },
            "required": []
        }
    },
    "required": [
        "image","param"
    ]
}

    def call(self, params: Union[str, dict], **kwargs) -> dict:
        params = self._verify_json_format_args(params)
        return generate(params, op_name='contour_area', **kwargs)

@register_tool('opencv_arc_length')
class OpencvArcLength(BaseTool):
    description = 'Computes contour perimeters using cv2.arcLength and overlays the values. Helps measure object outlines.'
    parameters = {
    "type": "object",
    "properties": {
        "image": {
            "type": "string",
            "description": "The input image identifier"
        },
        "param": {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "description": "'external|list'. Model can decide the value."
                },
                "canny_low": {
                    "type": "integer",
                    "description": "<int>"
                },
                "canny_high": {
                    "type": "integer",
                    "description": "<int>"
                },
                "rank": {
                    "type": "integer",
                    "description": "<optional int>"
                },
                "max_contours": {
                    "type": "integer",
                    "description": "<optional int>"
                },
                "color": {
                    "type": "array",
                    "items": {
                        "type": "integer"
                    },
                    "description": "BGR color [B, G, R]"
                },
                "thickness": {
                    "type": "integer",
                    "description": "<int>"
                }
            },
            "required": []
        }
    },
    "required": [
        "image","param"
    ]
}

    def call(self, params: Union[str, dict], **kwargs) -> dict:
        params = self._verify_json_format_args(params)
        return generate(params, op_name='arc_length', **kwargs)

@register_tool('opencv_approx_poly')
class OpencvApproxPoly(BaseTool):
    description = 'Approximates contours with fewer points using cv2.approxPolyDP (epsilon_ratio controls simplification). Great for polygonal shape summaries.'
    parameters = {
    "type": "object",
    "properties": {
        "image": {
            "type": "string",
            "description": "The input image identifier"
        },
        "param": {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "description": "'external|list'. Model can decide the value."
                },
                "canny_low": {
                    "type": "integer",
                    "description": "<int>"
                },
                "canny_high": {
                    "type": "integer",
                    "description": "<int>"
                },
                "rank": {
                    "type": "integer",
                    "description": "<optional int>"
                },
                "max_contours": {
                    "type": "integer",
                    "description": "<optional int>"
                },
                "epsilon_ratio": {
                    "type": "number",
                    "description": "<float default 0.02>"
                },
                "color": {
                    "type": "array",
                    "items": {
                        "type": "integer"
                    },
                    "description": "BGR color [B, G, R]"
                },
                "thickness": {
                    "type": "integer",
                    "description": "<int>"
                }
            },
            "required": []
        }
    },
    "required": [
        "image","param"
    ]
}

    def call(self, params: Union[str, dict], **kwargs) -> dict:
        params = self._verify_json_format_args(params)
        return generate(params, op_name='approx_poly', **kwargs)

@register_tool('opencv_histogram')
class OpencvHistogram(BaseTool):
    description = 'Applies histogram equalization or CLAHE to enhance image contrast. Supports multiple color modes (grayscale, BGR, HSV) and flexible channel selection to preserve color information or deal with specific channels.'
    parameters = {
    "type": "object",
    "properties": {
        "image": {
            "type": "string",
            "description": "The input image identifier"
        },
        "param": {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "description": "'equalize|clahe'. Model can decide the value."
                },
                "color_mode": {
                    "type": "string",
                    "description": "'grayscale|bgr|hsv' (default: 'grayscale'). Model can decide the value."
                },
                "channels": {
                    "type": "array",
                    "items": {
                        "type": "integer"
                    },
                    "description": "List of channel indices [0, 1, 2]"
                },
                "clip_limit": {
                    "type": "number",
                    "description": "<float> (for CLAHE)"
                },
                "tile_grid_size": {
                    "type": "array",
                    "items": {
                        "type": "integer"
                    },
                    "description": "Tile grid size [width, height], default [8, 8]"
                }
            },
            "required": []
        }
    },
    "required": [
        "image","param"
    ]
}

    def call(self, params: Union[str, dict], **kwargs) -> dict:
        params = self._verify_json_format_args(params)
        return generate(params, op_name='histogram', **kwargs)

@register_tool('opencv_dft')
class OpencvDft(BaseTool):
    description = 'Computes and visualizes the Discrete Fourier Transform magnitude spectrum.'
    parameters = {
    "type": "object",
    "properties": {
        "image": {
            "type": "string",
            "description": "The input image identifier"
        },
        "param": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    "required": [
        "image"
    ]
}

    def call(self, params: Union[str, dict], **kwargs) -> dict:
        params = self._verify_json_format_args(params)
        return generate(params, op_name='dft', **kwargs)

@register_tool('opencv_template_match')
class OpencvTemplateMatch(BaseTool):
    description = 'Matches a template image within the source image. Returns match score and bounding box coordinates.'
    parameters = {
    "type": "object",
    "properties": {
        "image": {
            "type": "string",
            "description": "The input image identifier"
        },
        "param": {
            "type": "object",
            "properties": {
                "template_path": {
                    "type": "string",
                    "description": "'path/to/template'"
                },
                "method": {
                    "type": "string",
                    "description": "'sqdiff|ccorr|ccoeff'. Model can decide the value."
                }
            },
            "required": []
        }
    },
    "required": [
        "image","param"
    ]
}

    def call(self, params: Union[str, dict], **kwargs) -> dict:
        params = self._verify_json_format_args(params)
        return generate(params, op_name='template_match', **kwargs)

@register_tool('opencv_hough_lines')
class OpencvHoughLines(BaseTool):
    description = 'Detects line segments in the image using Hough transform. Returns line coordinates and lengths. By default uses stricter parameters to reduce false positives. You can adjust threshold (higher = stricter, fewer lines) and minLineLength (higher = longer lines only) to control detection sensitivity.'
    parameters = {
    "type": "object",
    "properties": {
        "image": {
            "type": "string",
            "description": "The input image identifier"
        },
        "param": {
            "type": "object",
            "properties": {
                "threshold": {
                    "type": "integer",
                    "description": "<optional integer default 80>"
                },
                "minLineLength": {
                    "type": "integer",
                    "description": "<optional integer default 50>"
                },
                "maxLineGap": {
                    "type": "integer",
                    "description": "<optional integer default 10>"
                },
                "canny_low": {
                    "type": "integer",
                    "description": "<optional integer default 80>"
                },
                "canny_high": {
                    "type": "integer",
                    "description": "<optional integer default 200>"
                },
                "max_lines": {
                    "type": "integer",
                    "description": "<optional integer default 20>"
                }
            },
            "required": []
        }
    },
    "required": [
        "image","param"
    ]
}

    def call(self, params: Union[str, dict], **kwargs) -> dict:
        params = self._verify_json_format_args(params)
        return generate(params, op_name='hough_lines', **kwargs)

@register_tool('opencv_hough_circles')
class OpencvHoughCircles(BaseTool):
    description = 'Detects circles in the image using Hough transform. Returns circle centers and radii. By default uses stricter parameters to reduce false positives. You can adjust param2 (higher = stricter, fewer circles) and minDist (higher = circles must be farther apart) to control detection sensitivity.'
    parameters = {
    "type": "object",
    "properties": {
        "image": {
            "type": "string",
            "description": "The input image identifier"
        },
        "param": {
            "type": "object",
            "properties": {
                "dp": {
                    "type": "number",
                    "description": "<optional float default 1.2>"
                },
                "minDist": {
                    "type": "integer",
                    "description": "<optional integer default 50>"
                },
                "param1": {
                    "type": "integer",
                    "description": "<optional integer default 100>"
                },
                "param2": {
                    "type": "integer",
                    "description": "<optional integer default 50>"
                },
                "minRadius": {
                    "type": "integer",
                    "description": "<optional integer default 10>"
                },
                "maxRadius": {
                    "type": "integer",
                    "description": "<optional integer default 0 = no limit>"
                },
                "max_circles": {
                    "type": "integer",
                    "description": "<optional integer default 20>"
                }
            },
            "required": []
        }
    },
    "required": [
        "image","param"
    ]
}

    def call(self, params: Union[str, dict], **kwargs) -> dict:
        params = self._verify_json_format_args(params)
        return generate(params, op_name='hough_circles', **kwargs)

@register_tool('opencv_watershed')
class OpencvWatershed(BaseTool):
    description = 'Applies watershed segmentation to separate overlapping objects. Returns region count and labels.'
    parameters = {
    "type": "object",
    "properties": {
        "image": {
            "type": "string",
            "description": "The input image identifier"
        },
        "param": {
            "type": "object",
            "properties": {
                "max_regions": {
                    "type": "integer",
                    "description": "<optional integer default 20>"
                }
            },
            "required": []
        }
    },
    "required": [
        "image"
    ]
}

    def call(self, params: Union[str, dict], **kwargs) -> dict:
        params = self._verify_json_format_args(params)
        return generate(params, op_name='watershed', **kwargs)

@register_tool('opencv_grabcut')
class OpencvGrabcut(BaseTool):
    description = 'Performs foreground/background segmentation using GrabCut algorithm with preset rectangle. Returns foreground pixel statistics.'
    parameters = {
    "type": "object",
    "properties": {
        "image": {
            "type": "string",
            "description": "The input image identifier"
        },
        "param": {
            "type": "object",
            "properties": {
                "preset": {
                    "type": "string",
                    "description": "'center|tight|loose'. Model can decide the value."
                }
            },
            "required": []
        }
    },
    "required": [
        "image","param"
    ]
}

    def call(self, params: Union[str, dict], **kwargs) -> dict:
        params = self._verify_json_format_args(params)
        return generate(params, op_name='grabcut', **kwargs)

@register_tool('opencv_features')
class OpencvFeatures(BaseTool):
    description = 'Detects and draws keypoints using various feature detection methods (harris, shi_tomasi, sift, surf, fast, brief, orb). Returns keypoint coordinates.'
    parameters = {
    "type": "object",
    "properties": {
        "image": {
            "type": "string",
            "description": "The input image identifier"
        },
        "param": {
            "type": "object",
            "properties": {
                "method": {
                    "type": "string",
                    "description": "'harris|shi_tomasi|sift|surf|fast|brief|orb'. Model can decide the value."
                },
                "max_points": {
                    "type": "integer",
                    "description": "Maximum number of keypoints to detect"
                }
            },
            "required": []
        }
    },
    "required": [
        "image","param"
    ]
}

    def call(self, params: Union[str, dict], **kwargs) -> dict:
        params = self._verify_json_format_args(params)
        return generate(params, op_name='features', **kwargs)

@register_tool('opencv_denoise')
class OpencvDenoise(BaseTool):
    description = 'Applies fast non-local means denoising. Supports grayscale, color, or per-channel.'
    parameters = {
    "type": "object",
    "properties": {
        "image": {
            "type": "string",
            "description": "The input image identifier"
        },
        "param": {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "description": "'fast_means_gray|fast_means_color|fast_means_bgr_channel' (default: 'fast_means_color'). Model can decide the value."
                },
                "h": {
                    "type": "number",
                    "description": "<float 1-50> (default: 10)"
                },
                "h_color": {
                    "type": "number",
                    "description": "<float 1-50> (default: 10)"
                },
                "template_window": {
                    "type": "integer",
                    "description": "<odd int 3-21> (default: 7)"
                },
                "search_window": {
                    "type": "integer",
                    "description": "<odd int 3-31> (default: 21)"
                },
                "channels": {
                    "type": "array",
                    "items": {
                        "type": "integer"
                    },
                    "description": "List of channel indices [0, 1, 2]"
                }
            },
            "required": []
        }
    },
    "required": [
        "image","param"
    ]
}

    def call(self, params: Union[str, dict], **kwargs) -> dict:
        params = self._verify_json_format_args(params)
        return generate(params, op_name='denoise', **kwargs)

@register_tool('opencv_inpaint')
class OpencvInpaint(BaseTool):
    description = 'Inpaints (fills) regions in the image using automatically generated mask. Supports custom parameters for mask generation and inpainting method.'
    parameters = {
    "type": "object",
    "properties": {
        "image": {
            "type": "string",
            "description": "The input image identifier"
        },
        "param": {
            "type": "object",
            "properties": {
                "preset": {
                    "type": "string",
                    "description": "'canny|threshold' (default: 'canny'). Model can decide the value."
                },
                "canny_low": {
                    "type": "integer",
                    "description": "<int 0-255> (default: 100)"
                },
                "canny_high": {
                    "type": "integer",
                    "description": "<int 0-255> (default: 200)"
                },
                "threshold_value": {
                    "type": "integer",
                    "description": "<int 0-255> (default: 127)"
                },
                "radius": {
                    "type": "integer",
                    "description": "<int 1-10> (default: 3)"
                },
                "method": {
                    "type": "string",
                    "description": "'telea|ns' (default: 'telea'). Model can decide the value."
                }
            },
            "required": []
        }
    },
    "required": [
        "image","param"
    ]
}

    def call(self, params: Union[str, dict], **kwargs) -> dict:
        params = self._verify_json_format_args(params)
        return generate(params, op_name='inpaint', **kwargs)

@register_tool('opencv_inrange_color')
class OpencvInrangeColor(BaseTool):
    description = 'Creates a mask for pixels within a specified color range. Supports HSV or BGR colorspace and flexible output formats.'
    parameters = {
    "type": "object",
    "properties": {
        "image": {
            "type": "string",
            "description": "The input image identifier"
        },
        "param": {
            "type": "object",
            "properties": {
                "colorspace": {
                    "type": "string",
                    "description": "'hsv|bgr' (default: 'hsv'). Model can decide the value."
                },
                "lower": {
                    "type": "array",
                    "items": {
                        "type": "number"
                    },
                    "description": "Lower bound for color range [h/b, s/g, v/r]"
                },
                "upper": {
                    "type": "array",
                    "items": {
                        "type": "number"
                    },
                    "description": "Upper bound for color range [h/b, s/g, v/r]"
                },
                "output_format": {
                    "type": "string",
                    "description": "'mask|masked_image|both' (default: 'both'). Model can decide the value."
                }
            },
            "required": []
        }
    },
    "required": [
        "image","param"
    ]
}

    def call(self, params: Union[str, dict], **kwargs) -> dict:
        params = self._verify_json_format_args(params)
        return generate(params, op_name='inrange_color', **kwargs)

@register_tool('opencv_crop')
class OpencvCrop(BaseTool):
    description = 'Crops a rectangular region from the image. Returns the cropped region with detailed information about the crop area.'
    parameters = {
    "type": "object",
    "properties": {
        "image": {
            "type": "string",
            "description": "The input image identifier"
        },
        "param": {
            "type": "object",
            "properties": {
                "x": {
                    "type": "integer",
                    "description": "<non-negative integer>"
                },
                "y": {
                    "type": "integer",
                    "description": "<non-negative integer>"
                },
                "width": {
                    "type": "integer",
                    "description": "<positive integer>"
                },
                "height": {
                    "type": "integer",
                    "description": "<positive integer>"
                }
            },
            "required": []
        }
    },
    "required": [
        "image","param"
    ]
}

    def call(self, params: Union[str, dict], **kwargs) -> dict:
        params = self._verify_json_format_args(params)
        return generate(params, op_name='crop', **kwargs)

@register_tool('opencv_zoom_in')
class OpencvZoomIn(BaseTool):
    description = 'Zooms into a specific region of the image by cropping and optionally resizing. Useful for focusing on a particular area and enlarging it for better visibility.'
    parameters = {
    "type": "object",
    "properties": {
        "image": {
            "type": "string",
            "description": "The input image identifier"
        },
        "param": {
            "type": "object",
            "properties": {
                "x": {
                    "type": "integer",
                    "description": "<non-negative integer>"
                },
                "y": {
                    "type": "integer",
                    "description": "<non-negative integer>"
                },
                "width": {
                    "type": "integer",
                    "description": "<positive integer>"
                },
                "height": {
                    "type": "integer",
                    "description": "<positive integer>"
                },
                "scale": {
                    "type": "number",
                    "description": "<optional float 0-10>"
                },
                "target_width": {
                    "type": "integer",
                    "description": "<optional positive integer>"
                },
                "target_height": {
                    "type": "integer",
                    "description": "<optional positive integer>"
                }
            },
            "required": []
        }
    },
    "required": [
        "image","param"
    ]
}

    def call(self, params: Union[str, dict], **kwargs) -> dict:
        params = self._verify_json_format_args(params)
        return generate(params, op_name='zoom_in', **kwargs)

@register_tool('opencv_floodfill')
class OpencvFloodfill(BaseTool):
    description = 'Performs flood fill operation starting from a seed point. Fills connected pixels with similar color values. Useful for region segmentation and filling enclosed areas.'
    parameters = {
    "type": "object",
    "properties": {
        "image": {
            "type": "string",
            "description": "The input image identifier"
        },
        "param": {
            "type": "object",
            "properties": {
                "x": {
                    "type": "integer",
                    "description": "<non-negative integer>"
                },
                "y": {
                    "type": "integer",
                    "description": "<non-negative integer>"
                },
                "loDiff": {
                    "type": "array",
                    "items": {
                        "type": "integer"
                    },
                    "description": "Lower difference [B, G, R], default [10, 10, 10]"
                },
                "upDiff": {
                    "type": "array",
                    "items": {
                        "type": "integer"
                    },
                    "description": "Upper difference [B, G, R], default [10, 10, 10]"
                },
                "newVal": {
                    "type": "array",
                    "items": {
                        "type": "integer"
                    },
                    "description": "New fill color [B, G, R], default [0, 255, 0]"
                },
                "flags": {
                    "type": "integer",
                    "description": "<optional integer default 4|FLOODFILL_FIXED_RANGE>"
                }
            },
            "required": []
        }
    },
    "required": [
        "image","param"
    ]
}

    def call(self, params: Union[str, dict], **kwargs) -> dict:
        params = self._verify_json_format_args(params)
        return generate(params, op_name='floodfill', **kwargs)

@register_tool('opencv_connected_components_with_stats')
class OpencvConnectedComponentsWithStats(BaseTool):
    description = 'Finds and analyzes connected components in a binary image. Returns statistics including bounding boxes, areas, and centroids for each component. Useful for object detection and region analysis.'
    parameters = {
    "type": "object",
    "properties": {
        "image": {
            "type": "string",
            "description": "The input image identifier"
        },
        "param": {
            "type": "object",
            "properties": {
                "threshold_mode": {
                    "type": "string",
                    "description": "'otsu|binary|none'. Model can decide the value."
                },
                "threshold_value": {
                    "type": "integer",
                    "description": "<optional integer for binary mode default 127>"
                },
                "connectivity": {
                    "type": "integer",
                    "description": "<4|8 default 8>"
                },
                "max_components": {
                    "type": "integer",
                    "description": "<optional integer default 20>"
                }
            },
            "required": []
        }
    },
    "required": [
        "image","param"
    ]
}

    def call(self, params: Union[str, dict], **kwargs) -> dict:
        params = self._verify_json_format_args(params)
        return generate(params, op_name='connected_components_with_stats', **kwargs)

@register_tool('opencv_convertscaleabs')
class OpencvConvertscaleabs(BaseTool):
    description = 'Converts image to absolute value and scales it to 0-255 range. Useful for visualizing gradient images or other signed data. Applies scaling and offset transformations.'
    parameters = {
    "type": "object",
    "properties": {
        "image": {
            "type": "string",
            "description": "The input image identifier"
        },
        "param": {
            "type": "object",
            "properties": {
                "alpha": {
                    "type": "number",
                    "description": "<optional float default 1.0>"
                },
                "beta": {
                    "type": "number",
                    "description": "<optional float default 0>"
                }
            },
            "required": []
        }
    },
    "required": [
        "image","param"
    ]
}

    def call(self, params: Union[str, dict], **kwargs) -> dict:
        params = self._verify_json_format_args(params)
        return generate(params, op_name='convertscaleabs', **kwargs)

@register_tool('opencv_draw_circle')
class OpencvDrawCircle(BaseTool):
    description = 'Draws a circle on the image at specified center coordinates with given radius. Supports customizable color and thickness. Useful for marking circular regions or objects.'
    parameters = {
    "type": "object",
    "properties": {
        "image": {
            "type": "string",
            "description": "The input image identifier"
        },
        "param": {
            "type": "object",
            "properties": {
                "x": {
                    "type": "integer",
                    "description": "<integer>"
                },
                "y": {
                    "type": "integer",
                    "description": "<integer>"
                },
                "radius": {
                    "type": "integer",
                    "description": "<positive integer>"
                },
                "color": {
                    "type": "array",
                    "items": {
                        "type": "integer"
                    },
                    "description": "Circle color [B, G, R], default [0, 255, 0]"
                },
                "thickness": {
                    "type": "integer",
                    "description": "<optional integer default 2>"
                }
            },
            "required": []
        }
    },
    "required": [
        "image","param"
    ]
}

    def call(self, params: Union[str, dict], **kwargs) -> dict:
        params = self._verify_json_format_args(params)
        return generate(params, op_name='draw_circle', **kwargs)

