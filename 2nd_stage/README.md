## Training of the second stage detector model

Similar to the works of Li. et al (DeepMitosis -> Deep Detection and Verification), we train a second stage classifier with patches of the cell embedded in its immediate cellular context. The patches are of size 128px X 128px. 

The process is as follows:
1. We extract patches, allowing us to use standard pipelines used for image classification
2. We train a classifier for each data set variant
3. We process the results of the last stage with these classifiers


