## Ablation Study

This folder contains all results of the ablation study. We did two kinds of ablation:
- We reduced the area covered by training to be only 5 HPF, 50 HPF or the commonly used 10 HPF
- We reduced the amount of WSI used for training from 23 to 12, 6 and 3. 

### Ablation study: Area covered

We included training on a smaller subset, as this is typically provided by all major data sets. In this regard, the most often used area is 10 High Power Fields, often considered to be 2mm^2 in area. We used the definition of Meuten[1], which is commonly applied in the field of digital veterinary pathology, and used a slightly smaller area of 2.37mm^2. 

[1] D. J. Meuten, F. M. Moore, and J. W. George, “Mitotic Count and theField of View Area,”Veterinary Pathology, vol. 53, pp. 7–9, Jan. 2016.


### Ablation Study: Subsets

The subsets of WSI were chosen from the training set by eliminating from each portion of

|Filename|Mitotic figures|Part of complete Set (23 WSI) | 12 WSI | 6 WSI | 3 WSI |
|---|---|---|---|---|---|
| 2f2591b840e83a4b4358.svs | 3 | x |  - | - | - | 
| ce949341ba99845813ac.svs | 4 | x |  x | x | x | 
| 91a8e57ea1f9cb0aeb63.svs | 6 | x |  x | x | - | 
| 9374efe6ac06388cc877.svs | 7 | x |  - | - | - | 
| 0e56fd11a762be0983f0.svs | 8 | x |  x | x | - | 
| dd6dd0d54b81ebc59c77.svs | 11 | x |  - | - | - | 
| 2e611073cff18d503cea.svs | 18 | x |  x | x | - | 
| 066c94c4c161224077a9.svs | 19 | x |  x | x | - | 
| 285f74bb6be025a676b6.svs | 19 | x |  x | x | - | 
| 2efb541724b5c017c503.svs | 66 | x |  - | - | - | 
| 70ed18cd5f806cf396f0.svs | 85 | x |  x | x | x | 
| 3f2e034c75840cb901e6.svs | 571 | x |  x | x | - | 
| 8bebdd1f04140ed89426.svs | 1000 | x |  - | - | - | 
| 2f17d43b3f9e7dacf24c.svs | 1157 | x |  - | - | - | 
| a0c8b612fe0655eab3ce.svs | 1279 | x |  - | - | - | 
| ac1168b2c893d2acad38.svs | 1329 | x |  x | x | - | 
| fff27b79894fe0157b08.svs | 1744 | x |  - | - | - | 
| 34eb28ce68c1106b2bac.svs | 2279 | x |  x | x | - | 
| 39ecf7f94ed96824405d.svs | 3689 | x |  x | x | - | 
| 20c0753af38303691b27.svs | 4343 | x |  x | x | x | 
| c3eb4b8382b470dd63a9.svs | 4767 | x |  - | - | - |

