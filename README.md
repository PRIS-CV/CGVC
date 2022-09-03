# CGVC
Code release for Cross-Layer Feature based Multi-Granularity Visual Classification (VCIP 2022)


## Changelog
- 2022/09/03 upload the code.


## Requirements

- python 3.6
- PyTorch 1.2.0
- torchvision

## Data
- Download datasets
- Extract them to `data/Birds/`, `data/Airs/` and `data/Cars/`, respectively.
- Split the dataset into train and test folder, the index of each class should follow the Birds.xls, Airs.xls, and Cars.xls

* e.g., CUB-200-2011 dataset
```
  -/birds/train
	         └─── 001.Black_footed_Albatross
	                   └─── Black_Footed_Albatross_0001_796111.jpg
	                   └─── ...
	         └─── 002.Laysan_Albatross
	         └─── 003.Sooty_Albatross
	         └─── ...
   -/birds/test	
             └─── ...         
```


## Training
- `python Birds.py` or `python Airs.py` or `python Cars.py`


## Contact
Thanks for your attention!
If you have any suggestion or question, you can leave a message here or contact us directly:
- chenjunhan@bupt.edu.cn
- mazhanyu@bupt.edu.cn
