
import os
import shutil
import glob
from tqdm import tqdm


# #
# subject ID:
# xxx
# 
# image number:
# xxx
# 
# gender:
# 0 - male
# 1 - famale
# 
# glasses:
# 0 - no
# 1 - yes
# 
# eye state:
# 0 - close
# 1 - open
# 
# reflections:
# 0 - none
# 1 - low
# 2 - high
# 
# lighting conditions/image quality:
# 0 - bad
# 1 - good
# 
# sensor type:
# 01 - RealSense SR300 640x480
# 02 - IDS Imaging, 1280x1024
# 03 - Aptina Imagin 752x480
# 
# example:
# s001_00123_0_0_0_0_0_01.png

Raw_DIR= r'C:\Users\VinayKumar\Desktop\Drowsiness-Alert-System\project\data\mrlEyes_2018_01'
for dirpath, dirname, filenames in os.walk(Raw_DIR):
    for i in tqdm([f for f in filenames if f.endswith('.png')]):
        if i.split('_')[4]=='0':
            shutil.copy(src=dirpath+'/'+i, dst=r'C:\Users\VinayKumar\Desktop\Drowsiness-Alert-System\project\data\Prepared_Data\Closed_Eyes')
        
        elif i.split('_')[4]=='1':
            shutil.copy(src=dirpath+'/'+i, dst=r'C:\Users\VinayKumar\Desktop\Drowsiness-Alert-System\project\data\Prepared_Data\Open_Eyes')


