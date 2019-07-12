## Training of 2nd stage classifiers

Caution: In order to work, you need to have the patches for each cell classification data set in place.

### Notebooks - WSI alation

In this part, we removed WSI from the training set, as described in the parent directory's readme and the paper.

|Condition|Complete dataset|Training set|Mitotic figures (train)|Notebook used for training|
|---|---|---|---|---|
|3 WSI|11+3 WSI|3 WSI|4432|[CellClassification-ODAEL-3WSI.ipynb](CellClassification-ODAEL-3WSI.ipynb)|
|6 WSI|11+6 WSI|6 WSI|5798|[CellClassification-ODAEL-6WSI.ipynb](CellClassification-ODAEL-6WSI.ipynb)|
|12 WSI|11+12 WSI|12 WSI|12370|[CellClassification-ODAEL-12WSI.ipynb](CellClassification-ODAEL-12WSI.ipynb)|

### Notebooks - HPF ablation

This data set is significantly reduced over the WSI ablation in that the complete 21 WSI training set was used but only a 
small image part (corresponding to 5 HPF, 10 HPF, 50 HPF) around the mitotically most active area (determined by a senior 
pathology expert).

|Condition|Area|Complete dataset|Training set|Mitotic figures (train)|Notebook used for training|
|---|---|---|---|---|---|
|5 HPF|24.885 mm2|11+21 images|21 images|378|[CellClassification-ODAEL-5HPF.ipynb](CellClassification-ODAEL-5HPF.ipynb)|
|10 HPF|124.425 mm2|11+21 images|21 images|745|[CellClassification-ODAEL-10HPF.ipynb](CellClassification-ODAEL-10HPF.ipynb)|
|50 HPF|248.850 mm2|11+21 images|21 images|2467|[CellClassification-ODAEL-50HPF.ipynb](CellClassification-ODAEL-50HPF.ipynb)|

