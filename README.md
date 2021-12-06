# OptimusPrime
Implementation of the Transformer

### Dataset:

EEG of CVPR2021-02785

structure of local files should be:

```
CVPR2021-02785   
│
└───data
│   │   imagenet40-1000-1-00.bdf
│   │   imagenet40-1000-1-01.bdf
│   │   ...
│         
└───design
│   │   run-00.txt
│   │   run-01.txt
│   │   ...
│    
└───stimuli
│   │   n02106662_13.JPEG
│   │   n02106662_25.JPEG
│   │   ...
   

```

### Needs:
torch

python3

mne   # to read .bdf

numpy

glob  # to search local files


### Usage:
Run main.py file.


### issues:

sometimes Error occurs due to non-standard .bdf files. I have already avoided this by skipping those .bdf files.

High time and CPU cost to read .bdf files. To address it, data should be restored as `.pkl` in advance in the future.

Attention on the channel side maybe for useful EEG learning task.

PositionalEncoding isn't powerful enough.

More efficient down-sampling method on raw data should be considered.