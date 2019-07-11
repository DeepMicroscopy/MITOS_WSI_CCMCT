import numpy as np 
import SlideRunner.general.dependencies
from SlideRunner.dataAccess.database import Database
from SlideRunner.dataAccess.annotations import ViewingProfile
import os
import openslide
import sqlite3
import cv2

#os.system('cp Slides_final.sqlite Slides_final_cleaned.sqlite')

DB = Database()

vp = ViewingProfile()
vp.majorityClassVote=True

cm=np.zeros((7,7))

threshold = 5

disagreedclass = 0
agreedclass = 0
basepath='../WSI/'
patchSize=128

os.system('mkdir -p DataHEAEL')

dirs = ['Mitosis', 'Mitosislike', 'Tumorcells', 'Granulocytes']
for k in dirs:
    os.system('mkdir -p DataHEAEL/train/%s' % k)
    os.system('mkdir -p DataHEAEL/test/%s' % k)

def listOfSlides(DB):
    DB.execute('SELECT uid,filename from Slides')
    return DB.fetchall()

test_slide_filenames = ['3369_07_B_1_MCT Mitose 2017.svs',
 '3786_09 A MCT Mitose 2017.svs',
 '1659_08_1_MCT Mitose 2017.svs',
 '28_08_A_1_MCT Mitose 2017.svs',
 '3806_09_B_1_MCT Mitose 2017.svs',
 '2253_06_A_1_MCT Mitose 2017.svs',
 '1410_08_A_1_MCT Mitose 2017.svs',
 '1490_08_1_MCT Mitose 2017.svs',
 '2281_14_A_1_MCT Mitose 2017.svs',
 '221_08 MCT Mitose 2017.svs',
 '5187_11 B MCT Mitose 2017.svs']

DB.open('MITOS_WSI_CMCT_HEAEL.sqlite')#Slides_final_cleaned_checked.sqlite')

for slide,filename in listOfSlides(DB):
    DB.loadIntoMemory(slide)
    
    
    slide=openslide.open_slide(basepath+filename)

    for k in DB.annotations.keys():

        anno = DB.annotations[k]

        coord_x = anno.x1
        coord_y = anno.y1

        lu_x = int(coord_x - int(patchSize/2))
        lu_y = int(coord_y - int(patchSize/2))
        img = np.array(slide.read_region(location=(lu_x, lu_y), level=0, size=(patchSize, patchSize)))
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

#        DB.annotations[k].draw(leftUpper=(lu_x,lu_y), image=img, thickness=2, vp=vp, zoomLevel=1.0)

        istest = 'train/' if filename not in test_slide_filenames else 'test/'
        if (anno.agreedClass==2):
            if not cv2.imwrite('DataHEAEL/'+istest+'Mitosis/%d.png' % (k), img):
                  print('Write failed: '+'DataHEAEL/'+istest+'Mitosis/%d.png' % (k))

        if (anno.agreedClass==7):
            cv2.imwrite('DataHEAEL/'+istest+'Mitosislike/%d.png' % k, img)

        if (anno.agreedClass==3):
            cv2.imwrite('DataHEAEL/'+istest+'Tumorcells/%d.png' %k, img)

        if (anno.agreedClass==1):
            cv2.imwrite('DataHEAEL/'+istest+'Granulocytes/%d.png' %k, img) 



