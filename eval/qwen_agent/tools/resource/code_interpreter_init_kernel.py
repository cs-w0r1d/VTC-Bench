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

import json  # noqa
import math  # noqa
import os  # noqa
import re  # noqa
import signal
import threading
import time

import matplotlib  # noqa
import matplotlib.pyplot as plt
import numpy as np  # noqa
import pandas as pd  # noqa
import seaborn as sns
from matplotlib.font_manager import FontProperties
from sympy import Eq, solve, symbols  # noqa

try:
    from PIL import Image
except ImportError:
    Image = None

# Import IPython display functions for showing images
try:
    from IPython.display import display, Image as IPythonImage
except ImportError:
    display = None
    IPythonImage = None


def input(*args, **kwargs):  # noqa
    raise NotImplementedError('Python input() function is disabled.')


def _m6_timout_handler(_signum=None, _frame=None):
    raise TimeoutError('M6_CODE_INTERPRETER_TIMEOUT')


try:
    signal.signal(signal.SIGALRM, _m6_timout_handler)
except AttributeError:  # windows
    pass


class _M6CountdownTimer:

    @classmethod
    def start(cls, timeout: int):
        try:
            signal.alarm(timeout)
        except AttributeError:  # windows
            pass  # I haven't found a timeout solution that works with windows + jupyter yet.

    @classmethod
    def cancel(cls):
        try:
            signal.alarm(0)
        except AttributeError:  # windows
            pass


class _AutoCloseImageManager:
    """Manager to automatically close PIL Image windows after showing."""
    _original_show = None
    _auto_close_delay = 0.5  # seconds
    
    @classmethod
    def setup(cls):
        """Setup auto-close for PIL Image.show() method."""
        if Image is None:
            return
        
        if cls._original_show is None:
            cls._original_show = Image.Image.show
            # Create a wrapper function that will be used as the new show method
            def wrapped_show(img_instance, title=None):
                """Wrapped show method that auto-closes the image window."""
                # Call the original show method in a separate thread
                def show_and_close():
                    try:
                        # Call the original show method
                        _AutoCloseImageManager._original_show(img_instance, title=title)
                        # Give the window time to display
                        time.sleep(_AutoCloseImageManager._auto_close_delay)
                    except Exception:
                        pass  # Silently handle any errors
                    finally:
                        try:
                            # Try to close any open image windows
                            import subprocess
                            import sys
                            if sys.platform.startswith('linux'):
                                # On Linux, try to close any display windows
                                subprocess.run(['killall', 'display'], timeout=1, capture_output=True)
                            elif sys.platform == 'darwin':
                                # On macOS, close Preview.app windows
                                subprocess.run(['killall', 'Preview'], timeout=1, capture_output=True)
                        except Exception:
                            pass  # Silently handle any errors
                
                # Run show in background thread so it doesn't block
                thread = threading.Thread(target=show_and_close, daemon=True)
                thread.start()
            
            Image.Image.show = wrapped_show


sns.set_theme()

_m6_font_prop = FontProperties(fname='{{M6_FONT_PATH}}')
plt.rcParams['font.family'] = _m6_font_prop.get_name()

# Setup auto-close for PIL Image windows
_AutoCloseImageManager.setup()
