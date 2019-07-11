from SlideRunner.dataAccess.database import Database
import SlideRunner.dataAccess.annotations as annotations
import openslide
import os
import numpy as np 
import sys

if len(sys.argv)<2:
    print('syntax:',sys.argv[0],'<area in WSI>')
else:
    hpf = int(sys.argv[1])

os.system('mkdir -p %dHPF' % hpf)    
DB = Database()
DB.open('../databases/MITOS_WSI_CCMCT_ODAEL.sqlite')

DBRK = Database()
DBRK.open('HighMCAreas.sqlite')

os.system('cp ../databases/MITOS_WSI_CCMCT_ODAEL.sqlite MITOS_WSI_CCMCT_ODAEL_%dHPF.sqlite' % hpf)
DBnew = Database()
DBnew.open('MITOS_WSI_CCMCT_ODAEL_%dHPF.sqlite' % hpf)

DBnew.execute('ATTACH `../databases/MITOS_WSI_CCMCT_ODAEL.sqlite` as orig')

for uid, filename in DB.listOfSlides():
    
    print(uid, filename)

    DB.loadIntoMemory(uid)

    uidRK = DBRK.findSlideWithFilename(filename,'')
    DBRK.loadIntoMemory(uidRK)

    slide = openslide.open_slide('../WSI/'+filename)

    A = 0.237*hpf # mm^2 
    W_hpf_microns = np.sqrt(A*4/3) * 1000 # in microns
    H_hpf_microns = np.sqrt(A*3/4) * 1000 # in microns

    micronsPerPixel = slide.properties[openslide.PROPERTY_NAME_MPP_X]

    W_hpf = int(W_hpf_microns / float(micronsPerPixel))  
    H_hpf = int(H_hpf_microns / float(micronsPerPixel))

    pos=(0,0)
    for anno in DBRK.annotations.keys():
            if ((DBRK.annotations[anno].annotationType == annotations.AnnotationType.AREA) 
                    and (DBRK.annotations[anno].x2-DBRK.annotations[anno].x1>6000)):
                    ds=1
                    center = [int(0.5*(DBRK.annotations[anno].x1+DBRK.annotations[anno].x2)/ds),
                                int(0.5*(DBRK.annotations[anno].y1+DBRK.annotations[anno].y2)/ds)]
                    pos = [DBRK.annotations[anno].x1, DBRK.annotations[anno].y1]

                    size = [DBRK.annotations[anno].x2-DBRK.annotations[anno].x1, 
                            DBRK.annotations[anno].y2-DBRK.annotations[anno].y1]

                    print('Center is: ',center,pos,size,W_hpf,H_hpf)

    img = slide.read_region(location=pos, level=0, size=(W_hpf,H_hpf))
    x2 = pos[0]+W_hpf
    y2 = pos[1]+H_hpf
    x1 = pos[0]
    y1 = pos[1]

    print('Img:',np.array(img).shape)
    newfilename = filename[:-4]+'.tiff'
    img.save('%dHPF/%s'% (hpf, newfilename))

    DBnew.execute('UPDATE Slides set filename="%s" where uid==%d' % (newfilename, uid))    

    DBnew.loadIntoMemory(uid)
    
    annolist = list(DBnew.annotations.keys())
    for annoId in annolist:
        anno = DBnew.annotations[annoId]
        if (anno.x1<x1) or (anno.x1>x2) or (anno.y1<y1) or (anno.y1>y2):
            DBnew.removeAnnotation(anno.uid)
        else:
            DBnew.execute('UPDATE Annotations_coordinates SET coordinateX=%d, coordinateY=%d where annoId==%d' % (anno.x1-x1,anno.y1-y1, anno.uid))

DBnew.commit()
