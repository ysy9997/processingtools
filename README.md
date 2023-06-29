![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)

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


## EnvRecoder
```
import processingtools as pt
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--save_path', type=str)
args = parser.parse_args()

recoder = pt.EnvReco('/save/path')

args = arg2abs(args)
recoder.record_arg()
recoder.record_code()
recoder.record_os()
recoder.record_gpu()
recoder.print('record logs')
```

Then
...