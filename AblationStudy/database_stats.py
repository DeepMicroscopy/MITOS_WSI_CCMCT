from SlideRunner.dataAccess.database import Database

DB = Database()

slidelist_test = ['27', '30', '31', '6', '18', '20', '1', '2', '3' ,'9', '11']
clause = ','.join(slidelist_test)

files = ['MITOS_WSI_CCMCT_ODAEL_50HPF.sqlite', 'MITOS_WSI_CCMCT_ODAEL_10HPF.sqlite',     'MITOS_WSI_CCMCT_ODAEL_5HPF.sqlite',
'MITOS_WSI_CCMCT_ODAEL_12WSI.sqlite',   'MITOS_WSI_CCMCT_ODAEL_6WSI.sqlite','MITOS_WSI_CCMCT_ODAEL_3WSI.sqlite','MITOS_WSI_CCMCT_ODAEL.sqlite']

for f in files:
    DB.open('databases/'+f)
    
    cnt = DB.execute('SELECT COUNT(*) FROM Annotations where agreedClass==2 and slide not in (%s)' % clause).fetchone()[0]
    
    print('%40s: %d mitotic figures in training set' % (f, cnt))
    
    

DB.open(files[-1])
slidelist_train = [s[0] for s in DB.execute('SELECT Slides.filename, count(*) as cnt FROM Slides left join Annotations on Annotations.slide == Slides.uid where Slides.uid not in (%s) and Annotations.agreedClass==2 group by slide order by cnt asc' % clause).fetchall()]
                                            
slidelist_cnt = {s[0]:s[1] for s in DB.execute('SELECT Slides.filename, count(*) as cnt FROM Slides left join Annotations on Annotations.slide == Slides.uid where Slides.uid not in (%s) and Annotations.agreedClass==2 group by slide order by cnt asc' % clause).fetchall()}
                                            
partOf = dict()
for f in files[-4:]:
    partOf[f] = {x:'-' for x in slidelist_train}
    DB.open('databases/'+f)
    
    slidelist_train_sub = [s[0] for s in DB.execute('SELECT filename FROM Slides where uid not in (%s)' % clause).fetchall()]   
    for x in slidelist_train_sub:
        partOf[f][x] = 'x'
    
#    cnt = DB.execute('SELECT COUNT(*) FROM Annotations where agreedClass==2 and slide not in (%s)' % clause).fetchone()[0]

for x in slidelist_train:
    print('| %s | %d | x |  %s | %s | %s | ' % (x, slidelist_cnt[x], str(partOf['MITOS_WSI_CCMCT_ODAEL_12WSI.sqlite'][x]),
                                       str(partOf['MITOS_WSI_CCMCT_ODAEL_6WSI.sqlite'][x]),
                                       str(partOf['MITOS_WSI_CCMCT_ODAEL_3WSI.sqlite'][x])))
