![Python 3.7](https://img.shields.io/badge/python-3.7-green.svg)

[![PyPI version](https://badge.fury.io/py/ProcessingTools.svg)](https://badge.fury.io/py/ProcessingTools)
[![Downloads](https://pepy.tech/badge/processingtools)](https://pepy.tech/project/processingtools)

# ProcessionTools

- You can install this package using pip. 

```pip install processingtools```

## ProgressBar

```
import processingtools as pt
import time


for i in pt.ProgressBar(range(50)):
    time.sleep(0.1)
```
or
```
import processingtools as pt
import time


for i in pt.ProgressBar(range(50), bar_length=40, start_mark=None, finish_mark='progress done!', total=False):
    time.sleep(0.1)
```
Then, 
```
|████████████████████████████████████████| 100.0% | 50/50 | 0s |  
progress finished!(5311ms)
```

### parameters
**class**　pt.ProgressBar(*in_loop, bar_length: int = 40, start_mark: str = None, finish_mark='progress done!', total: int = None, detail_func: callable = None*)

- **in_loop**: the input loop
- **bar_length**: bar length
- **start_mark**: print string when the progress start
- **finish_mark**: print string what you want when progress finish
- **total**: total value. If you do not fill this, it will calculate automatically, but it may be slow
- **detail_func**: write detail using detail_func
- **remove_last**: If True, remove last progressbar


## EnvRecoder
```
import processingtools as pt
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--save_path', type=str)
args = parser.parse_args()

recoder = pt.EnvReco('/save/path')

args = recoder.arg2abs(args)
recoder.record_arg(args)
recoder.record_code()
recoder.record_os()
recoder.record_gpu()
recoder.put_space()
recoder.print('record logs')
```

Then, record information in the log file
```commandline
Args: 
{
    save_path: None
}

OS Env: 
{
    ALLUSERSPROFILE: ...
    APPDATA: ...
    COMMONPROGRAMFILES: ...
    ⋮
}

GPU Info: 
{
    cuda: True
    num: 1
    names: ['...']
}

[2023-7-3 19:50:9.78]: record logs

```

<details>
<summary>Others</summary>

⚠️ This description was written almost by copilot, with some minor modifications. ⚠️ 

## Others

### MultiProcess
This class provides a set of tools for running functions in parallel using multiple processes. This class is designed to simplify the process of parallel execution, making it easier to utilize multiple CPU cores for improved performance.

#### Methods

1. **`__init__(self, cpu_n: int = mp.cpu_count())`**
    - Initialization function
    - **Parameters**:
        - **cpu_n**: The number of CPUs to use (default: the number of all CPUs)

2. **`duplicate_func(self, func, args_list: typing.Union[tuple, list], progress_args: typing.Union[dict, bool] = True)`**
    - Run the function as a multiprocess
    - **Parameters**:
        - **func**: The function to run as a multiprocess
        - **args_list**: Arguments for the function
        - **progress_args**: Arguments for ProgressBar. If False, it doesn't use ProgressBar; if True, it uses ProgressBar
    - **Returns**: True

3. **`multi_func(self, funcs: typing.Union[tuple, list], args: typing.Union[tuple, list], progress_args: typing.Union[dict, bool] = True)`**
    - Run multiple functions as a multiprocess
    - **Parameters**:
        - **funcs**: The functions to run as a multiprocess
        - **args**: Arguments for the functions
        - **progress_args**: Arguments for ProgressBar. If False, it doesn't use ProgressBar; if True, it uses ProgressBar
    - **Returns**: True

4. **`split_list(self, *args)`**
    - Split a list by the number of `self.cpu_n`
    - **Parameters**:
        - **args**: Input lists
    - **Returns**: Split list

5. **`wrapper(data, *args, **kwargs)`**
    - Static method to wrap a function using `dill`
    - **Parameters**:
        - **data**: Serialized function data
    - **Returns**: Result of the function execution

6. **`adapt_function(function, order=False)`**
    - Adapt a function for multiprocessing
    - **Parameters**:
        - **function**: The function to adapt
        - **order**: If True, maintains order
    - **Returns**: Serialized adapted function


### VideoTools
This provides a set of tools for handling video files, including video capture initialization, frame extraction, video resizing, and video-to-GIF conversion.

#### Methods

1. **`__init__(self, video_path: str)`**
    - Initialization function
    - **Parameters**:
        - **video_path**: The path to the video file

2. **`initial_video_capture(self)`**
    - Initialize video capture and set video properties
    - **Raises**: `FileNotFoundError` if the video cannot be read

3. **`video2images(self, save_path: str, extension: str = 'jpg', start: float = 0, end: float = None, jump: float = 1, option: str = 'frame', size=None) -> True`**
    - Convert video frames to image files
    - **Parameters**:
        - **save_path**: Directory to save the image files
        - **extension**: File extension for the images (default: 'jpg')
        - **start**: Start frame
        - **end**: End frame
        - **jump**: Frame interval to save
        - **option**: 'second' or 'frame' to specify the unit for start, end, and jump
        - **size**: Resize dimensions (height, width) or scale factor
    - **Returns**: True

4. **`video_resize(self, save_path: str, size) -> True`**
    - Resize the video to the specified size
    - **Parameters**:
        - **save_path**: Path to save the resized video
        - **size**: Resize dimensions (height, width) or scale factor
    - **Returns**: True

5. **`second2frame(self, *args)`**
    - Convert seconds to frames
    - **Parameters**:
        - **args**: Time in seconds
    - **Returns**: Corresponding frames

6. **`video2gif(self, save_path: str, speed: float = 1, size=1)`**
    - Convert video to GIF
    - **Parameters**:
        - **save_path**: Path to save the GIF
        - **speed**: Speed factor for the GIF
        - **size**: Resize dimensions (height, width) or scale factor
    - **Raises**: `ModuleNotFoundError` if `moviepy` is not installed


### AutoInputModel
A PyTorch module for automatically processing and normalizing input images. 
This class wraps a given model and provides functionality to read, preprocess, and forward images through the model. 
It supports custom transformers and normalization parameters.

#### Methods

1. **`__init__(self, model, size: typing.Union[tuple, list, None] = None, mean: typing.Union[float, list, torch.Tensor, None] = None, std: typing.Union[float, list, torch.Tensor, None] = None, transformer=None)`**
    - Initialization function
    - **Parameters**:
        - **model**: The model to be used
        - **size**: The size to which images will be resized
        - **mean**: Mean for normalization
        - **std**: Standard deviation for normalization
        - **transformer**: Custom transformer for image preprocessing (Choose one of (size, mean std) or transformer)

2. **`image_read(self, path: str) -> torch.Tensor`**
    - Read and preprocess an image from the given path
    - **Parameters**:
        - **path**: Image file path
    - **Returns**: Normalized image tensor

3. **`forward(self, x: torch.Tensor) -> torch.Tensor`**
    - Forward pass through the model
    - **Parameters**:
        - **x**: Input tensor
    - **Returns**: Output tensor from the model

4. **`to(self, device: str)`**
    - Move the model to the specified device
    - **Parameters**:
        - **device**: Device to move the model to (e.g., 'cpu', 'cuda')


### EnsembleModel
A PyTorch module for performing ensemble predictions using multiple models. This class provides functionality to ensemble model predictions using different methods such as mean and weighted average.

#### Methods

1. **`__init__(self, models)`**
    - Initialization function
    - **Parameters**:
        - **models**: List of models to be ensembled

2. **`forward(self, x: torch.Tensor, option: str = 'mean', weights: typing.Union[list, tuple, None] = None) -> torch.Tensor`**
    - Perform ensemble predictions on the input data
    - **Parameters**:
        - **x**: Input tensor
        - **option**: Ensemble method ('mean' or 'WA' for weighted average)
        - **weights**: Weights for weighted average (must be the same length as models if specified)
    - **Returns**: Ensemble prediction tensor


### s_text

Prints the given text with specified color (RGB) and style.

#### Parameters:
- **text**: The text to be printed.
- **f_rgb**: The RGB color code for the text color.
- **b_rgb**: The RGB color code for the background color.
- **styles**: The styles to be applied to the text. Options are `'bold'`, `'tilt'`, `'underscore'`, and `'cancel'`.


### sprint

Prints the given text with specified color and style.

#### Parameters:
- **text**: The text to be printed.
- **f_rgb**: The RGB color code for the text color.
- **b_rgb**: The RGB color code for the background color.
- **styles**: The styles to be applied to the text. Options are `'bold'`, `'tilt'`, `'underscore'`, and `'cancel'`.
- **sep**: The separator to be used in the print function.
- **end**: The end character to be used in the print function.
- **file**: The file where the output will be written.


### torch_imgs_save

Save images in PNG files.

#### Parameters:
- **imgs**: Torch tensor.
- **save_path**: Save path (default: `'./'`).

#### Returns:
- **True** if normal, otherwise **False**.

</details>
