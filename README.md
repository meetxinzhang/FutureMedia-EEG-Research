# FutureMedia-EEG-Research

The open-source repository for EEG classification projects.


### Dataset:

EEG of CVPR2021-02785, the structure of local files should be:
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

Our own dataset will be open together with publications.


### Needs:
torch
python3
mne   # to read .bdf

### Usage:
Run main_k_fold.py file or main_parallel.py with multi-gpu.



We are organizing and submitting our current research and continuing to the next stage.


@ Research Institute for Future Media Computing, 
Shenzhen University, 
Shenzhen, China
