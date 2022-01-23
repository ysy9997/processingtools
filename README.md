![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)

# ProcessionTools

- You can install this package using pip. 

```pip install ProcessingTools```

## usage

```
import ProcessingTools as pt
import time

for i in pt.ProgressBar(range(50)):
    time.sleep(0.1)
```
or
```
import ProcessingTools as pt
import time

for i in pt.ProgressBar(range(50), bar_length=40, start_mark=None, finish_mark='progress done!', max=False):
    time.sleep(0.1)
```
Then, 
```
|████████████████████████████████████████| 100.0% | 50/50 | 0s |  
progress finished!(5311ms)
```

### parameters
**class**　pt.ProgressBar(*in_loop, bar_length: int = 40, start_mark: str = None, finish_mark='progress done!', max=False*)

- **in_loop**: the input loop
- **bar_length**: bar length
- **start_mark**: print string when the progress start
- **finish_mark**: print string what you want when progress finish
- **max**: max value. If you do not fill this, it will calculate automatically, but it may be slow