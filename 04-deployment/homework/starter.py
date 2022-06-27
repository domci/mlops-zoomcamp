#!/usr/bin/env python
# coding: utf-8



import pickle
import pandas as pd
import numpy as np
import argparse

# Instantiate the parser
parser = argparse.ArgumentParser()

# Required positional argument
parser.add_argument('--year', type=int)
parser.add_argument('--month', type=int)
args = parser.parse_args()

year = args.year
month = args.month


# In[2]:


with open('model.bin', 'rb') as f_in:
    dv, lr = pickle.load(f_in)


# In[3]:


categorical = ['PUlocationID', 'DOlocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


# In[4]:


df = read_data(f'https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_{year}-{month:02d}.parquet')


# In[5]:


dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = lr.predict(X_val)

print(np.mean(y_pred))

#What's the mean predicted duration for this dataset?
#16.191691679979066


# In[13]:



df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')


# In[16]:


output_file = 'out.parquet'

df.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)

#What's the size of the output file?
#* 39M


# In[ ]:


#jupyter nbconvert mynotebook.ipynb --to python

