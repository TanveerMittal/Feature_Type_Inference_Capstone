#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
get_ipython().system('{sys.executable} -m pip install googledrivedownloader')


# # Google Drive Download Link
# 
# https://drive.google.com/file/d/1ZPZY2wvDvsmnpQBABLz9ZyZRGvkEmo7B/view?usp=sharing

# In[2]:


from google_drive_downloader import GoogleDriveDownloader as gd
import os


# In[3]:


def get_raw_data():

    data_dir = './RawCSV.zip'
    print('Downloading the raw and preprocessed data into {}.'.format(data_dir))

    if not os.path.exists(data_dir):
        print('Downloading data directory.')
        dir_name = data_dir
        gd.download_file_from_google_drive(file_id='1ZPZY2wvDvsmnpQBABLz9ZyZRGvkEmo7B',
                                           dest_path=dir_name,
                                           unzip=True,
                                           showsize=True)

    print('Data was downloaded.')


# In[4]:


get_raw_data()

