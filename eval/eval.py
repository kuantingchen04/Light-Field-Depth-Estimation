import os.path
import numpy as np



gt_dir = 'depth_images_v2_400'
trainPath = os.path.join(os.getcwd(),gt_dir)
data = [name for name in os.listdir(trainPath)].sort()

from PIL im
