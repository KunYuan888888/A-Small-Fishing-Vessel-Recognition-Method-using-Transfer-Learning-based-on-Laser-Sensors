# _*_coding:utf-8_*_
# Python imports
import matplotlib.pyplot as plt
import numpy as np
import csv
import os
import pandas as pd
import re


Folder_Path = r'./Original Data'  # The folder to be spliced and its full path
SaveFile_Path = r'./Original Data'  # File path to save after splicing
SaveFile_Name = r'%i.csv'%(i+1)  # File name to save after merging

# Modify the current working directory
os.chdir(Folder_Path)
# Save all file names under this folder into a list
file_list = os.listdir()

# Read the first CSV file and include the header
df = pd.read_csv(Folder_Path + '\\' + file_list[0])  # The encoding defaults to UTF-8, and can be changed automatically if garbled
# Write the first CSV file read into the merged file and save
df.to_csv(SaveFile_Path + '\\' + SaveFile_Name, encoding="utf_8_sig", index=False)

# Cycle through the CSV file names in the list and append them to the merged file
for i in range(1, len(file_list)):
    df = pd.read_csv(Folder_Path + '\\' + file_list[i])
    df.to_csv(SaveFile_Path + '\\' + SaveFile_Name, encoding="utf_8_sig", index=False, header=False, mode='a+')
path = r"./Original Datadata/"

FileNames = os.listdir(path)
for fn in FileNames:
    if re.search(r'\.csv$', fn):
        fullfilename = os.path.join(path, fn)
        df = pd.read_csv(fullfilename,encoding='utf-8',on_bad_lines='skip')
        print(fn)  # file name
        print(df)  # data

for i in range(1000):
    y=pd.read_csv("D:\Small Fishing Vessel Recognition\Original Data\Small Fishing\A%i.csv"%(i+1),header=None)
    reader = csv.reader(open(y, 'rb'))
    l = len(reader)
    x= np.arange(1,l,1)
    z1 = np.polyfit(x, y, 3)
    p1 = np.poly1d(z1)

    # print(p1)
    # yvals = p1(x)
    # plot1 = plt.plot(x, y, 'bo' )
    # plot2 = plt.plot(x, yvals, 'r' )
    # plt.xlabel('$X$')
    # plt.ylabel('$Y$')
    # plt.legend(loc=4)
    # plt.title('polyfitting')
    # plt.show()
    i=i+1