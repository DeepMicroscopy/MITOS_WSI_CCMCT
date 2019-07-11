from SlideRunner.dataAccess.database import Database

delete_slides = ["13,8,4,7,17,22,28,26,23,24,36,29,15,14,19,32,25,12", # 3 WSI
                 "13,8,4,7,17,22,28,26,23,24,36,29,15,14,19", #6 WSI
                 "13,8,4,7,17,22,28,26,23"]#12 WSI
import os
DB = Database()

WSI_lists = ['3','6','12']

for i,k in enumerate(WSI_lists):
    os.system('cp databases/MITOS_WSI_CMCT_ODAEL.sqlite databases/MITOS_WSI_CMCT_ODAEL_%sWSI.sqlite' % k)

    DB.open('databases/MITOS_WSI_CMCT_ODAEL_%sWSI.sqlite' % k)
    
    cnt = DB.execute('DELETE FROM Annotations_label where annoid in (SELECT uid from Annotations where slide in (%s))' % delete_slides[i])
    cnt = DB.execute('DELETE FROM Annotations_coordinates where slide in (%s)' % delete_slides[i])
    cnt = DB.execute('DELETE FROM Annotations where slide in (%s)' % delete_slides[i])
    cnt = DB.execute('DELETE FROM Slides where uid in (%s)' % delete_slides[i])
    DB.commit()
    
    