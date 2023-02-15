import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyts.image import MarkovTransitionField
for i in range(1000):
 data=pd.read_csv("D:\Small Fishing Vessel Recognition\Original Data\Small Fishing\A%i.csv"%(i+1),header=None) #Cyclically read data
 X =data.values.tolist()                          # MTF transformation
 mtf = MarkovTransitionField(image_size=10)
 X_mtf = mtf.fit_transform(X)                     # Show the image for the first time series
 plt.figure(figsize=(10, 10))                     #Define image size
 plt.imshow(X_mtf[0], cmap='rainbow', origin='lower')
 plt.tight_layout()
 plt.axis('off')
 plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
 plt.margins(0, 0)
 plt.savefig("D:\Small Fishing Vessel Recognition\Original Data\Small Fishing\A%i.png"%(i+1))                     #Cycle to save MTF image
 plt.show()
 i=i+1
 
for i in range(1000):
 data=pd.read_csv("D:\Small Fishing Vessel Recognition\Original Data\UCR\A%i.csv"%(i+1),header=None) #Cyclically read data
 X =data.values.tolist()                          # MTF transformation
 mtf = MarkovTransitionField(image_size=10)
 X_mtf = mtf.fit_transform(X)                     # Show the image for the first time series
 plt.figure(figsize=(10, 10))                     #Define image size
 plt.imshow(X_mtf[0], cmap='rainbow', origin='lower')
 plt.tight_layout()
 plt.axis('off')
 plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
 plt.margins(0, 0)
 plt.savefig("D:\Small Fishing Vessel Recognition\Original Data\UCR\A%i.png"%(i+1))                     #Cycle to save MTF image
 plt.show()
 i=i+1
