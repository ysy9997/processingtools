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

## Others
