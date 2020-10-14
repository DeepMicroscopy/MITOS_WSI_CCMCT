#!/bin/bash

# Convert all TIFF files to pyramidal TIFF so they can be read with openslide without memory limitations
for f in *.tiff; do vips --xres=4000 tiffsave $f pyramid/$f.tif  --compression=jpeg --Q=90 --tile --tile-width=256 --tile-height=256 --pyramid; done

