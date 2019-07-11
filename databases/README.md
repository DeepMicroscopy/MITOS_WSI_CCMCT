# Database variants

This dataset features three dataset variants:

## MITOS_WSI_CCMCT_MEL: Manually labelled by experts

This is the most straight-forward approach: An expert screened all WSIs twice for mitotic figures and similar cells, additionally, he annotated other cells like granulocytes and normal tumor cells. A second expert gave another opinion on each of these cells and assigned a label accordingly.

|Data format|File|
|---|---|
|SlideRunner (SQlite3 database) | [MITOS_WSI_CCMCT_MEL.sqlite](MITOS_WSI_CCMCT_MEL.sqlite)
|Microsoft COCO | [MITOS_WSI_CCMCT_MEL.json](MITOS_WSI_CCMCT_MEL.json)



## MITOS_WSI_CCMCT_HEAEL: Hard-Example-Augmented Expert Labelled

In this flavour of the data set, a first algorithmic approach was used to differentiate hard negative mitotic figure look-alike cells. All results of the classifier were shown to a veterinary pathology expert, who additionally found some of the "hard negatives" were actually positives.

|Data format|File|
|---|---|
|SlideRunner (SQlite3 database) | [MITOS_WSI_CCMCT_HEAEL.sqlite](MITOS_WSI_CCMCT_HEAEL.sqlite)
|Microsoft COCO | [MITOS_WSI_CCMCT_HEAEL.json](MITOS_WSI_CCMCT_HEAEL.json)



## MITOS_WSI_CCMCT_ODAEL: Object-Detection augmented Expert Labelled

In this variant, we used a two-stage object detection model, much like Li et al. in their DeepMitosis paper, to screen the WSIs for additional mitotic figures that were missed during the first annotation process. The first stage of the model was based on RetinaNet, the second stage was a simple ResNet-18 based classifier. All model detections were filtered for already assigned cells from the database, and new candidates were given to both experts. We believe this data set is the most consistent variant. 

|Data format|File|
|---|---|
|SlideRunner (SQlite3 database) | [MITOS_WSI_CCMCT_ODAEL.sqlite](MITOS_WSI_CCMCT_ODAEL.sqlite)
|Microsoft COCO | [MITOS_WSI_CCMCT_ODAEL.json](MITOS_WSI_CCMCT_ODAEL.json)

