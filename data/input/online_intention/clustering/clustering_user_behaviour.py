#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 18:46:02 2021

"""


import umap
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import hdbscan
import seaborn as sns

print(pd.__version__)

"""
Expected Format : userid, time, pagecategory, pageaction

Now the Feature engineering required :
    Timestamp Features : server_timestamp_epoch_ms
        Prepared Features : day of week, hour of day, hour, minutes, seconds, weekend, weekday
    Category Features : product_action, product_skus_hash
        Prepared Features : Dummy Hot encoding - product_skus_hash , Dummy Hot encoding - product_action

"""

# Clustering
# read data
#path = "/Users/yashwanthkumar/Work/upwork/clustering/online_shoppers_intention.csv"
dir_path    = "/Users/yashwanthkumar/Work/upwork/clustering/shopper_intent_prediction"
path        = dir_path+"/shopper_intent_prediction-data.csv"
result_path = "/result_5000rows"
df_full   = pd.read_csv(path)
print(len(df_full))



#########     DATA PRE - PROCESSING      ############

# sampling
df_sample = df_full.sample(n=5000,random_state=1)
print(len(df_sample))
df_sample.head()
df_sample.info()
data = df_sample[df_sample['session_id_hash'].notna()]

# USE THE BELOW CODE TO DROP NA
# check how much uique products skus are there
# print(len(data[["product_skus_hash"]].dropna()))
# d = data[["product_skus_hash"]].dropna().drop_duplicates(["product_skus_hash"])
# print(len(d))


# drop the unused columns
data = data.drop(['hashed_url','event_type'],axis=1) 

# Feature engineering
# time related transformation
"""
Using Duration of a seesion as a feature
Also using:
    1. Hour of Day
    2. Day of week
    3. Weekday or not

"""
data['timestamp']          = pd.to_datetime(data['server_timestamp_epoch_ms'],unit='ms')
data['hour_of_day']        = data['timestamp'].dt.hour.astype("uint8")
#data['minutes']           = data['timestamp'].dt.minute.astype("uint8")
#data['second']            = data['timestamp'].dt.second.astype("uint8")
data['day_of_week']        = data['timestamp'].dt.dayofweek.astype("uint8")
data['weekday']            = data['timestamp'].dt.weekday.astype("uint8").apply(lambda x: 0 if x<5 else 1)
#data                      = data.drop(['server_timestamp_epoch_ms'],axis=1)
data.info()

# category related transformation
data['product_sku_action'] = data['product_action'].astype('str')+"|"+data['product_skus_hash'].astype('str')
# can include is_purchase also as a feature. But since vil be using event action not needed
#data['is_purchase'] = data['product_action'].apply(lambda x: 1 if x == 'purchase' else 0)




# Extracting duration per session
"""
when analysed about whether to take duration in hours or mins or seconds .
For Hours - 98.48 % of sessions was less than 1 hour. So taking hour would be skewed dist. 
For Minutes = 86.72% of sessions was less than 1 minute
For seconds = 85.33% of sessions was less than 1 second
for milliseconds = 85.025 %     ,,           ,,
So took mins as as a measure as representing as seconds/milliseconds offers than 2% exttra coverage

CODE TO CHECK :
s[['maxtime','mintime','duration','duration_mins']].head()
d_under1min = len(s[s['duration_mins']<1])
d = len(s)
print(d_under1min)
print(d)
print("percentage of session less than 1 min")
print(d_under1min/d*100)

"""

df_dt                  = data.groupby('session_id_hash').agg(avgtime =('server_timestamp_epoch_ms','mean'),maxtime =('server_timestamp_epoch_ms','max'),mintime =('server_timestamp_epoch_ms','min'))
df_dt['duration']      = df_dt['maxtime']-df_dt['mintime']
df_dt['duration_mins'] = (df_dt['duration']/(1000*60))



# one hot encoding of categorical variables
"""
For Categorical lets use:
    1. Just the actions alone to know how was his overall activity
    2. His product sku-actions combination to know action at product sku level
    
For Time related:
    1. Hour of Day
    2. Day of week
    3. Weekday or not

"""
colid   = ['session_id_hash']
colcat  = ['product_action','product_skus_hash']
coltime = ['hour_of_day','day_of_week','weekday']


# get cat var into dummies
df_cat_dummies        = pd.get_dummies(data[colid+colcat], columns=colcat).groupby(colid).count()
# get time-cat var into dummies
df_time_dummies       = pd.get_dummies(data[colid+coltime], columns=coltime).groupby(colid).sum()

# concat both df. since they have same index session id. they will be merged properly
model_data            = pd.concat((df_cat_dummies, df_time_dummies), axis=1)
model_data.info()
df_cat_dummies.info()
df_time_dummies.info()

# to check total columns
total_cnt=0
cnt=0
for col in colcat+coltime:
    cnt = len(data[col].unique())
    total_cnt +=cnt
    print(" Total unique values of - {} is {} | Total columns = {}".format(col,cnt,total_cnt))

# the count matches (remmeber for dummy subtract 2 from total cnt)

model_data = model_data.reset_index()
print(len(model_data['session_id_hash']))
# 31928 sessions
print(model_data.duplicated(['session_id_hash']).any())






#########     MODEL TRAINING      ############
fit_data = model_data[model_data.columns[~model_data.columns.isin(['session_id_hash','cluster_id'])]]


# UMAP embedding
sns.set(style='white', rc={'figure.figsize':(10,8)})
standard_embedding = umap.UMAP(random_state=42).fit_transform(fit_data)


# PCA_DBCAN
lowd_data = PCA(n_components=50).fit_transform(fit_data)
#hdbscan_labels = hdbscan.HDBSCAN(min_samples=10, min_cluster_size=100).fit_predict(lowd_data)
model = DBSCAN(eps=3, min_samples=500).fit(lowd_data)
labels = model.labels_
clustered = (labels >= 0)
plt.scatter(standard_embedding[~clustered, 0],
            standard_embedding[~clustered, 1],
            c=(0.5, 0.5, 0.5),
            s=0.1,
            alpha=0.5)
plt.scatter(standard_embedding[clustered, 0],
            standard_embedding[clustered, 1],
            c=labels[clustered],
            s=0.1,
            cmap='Spectral');

plt.title('PCA_DBCAN')
plt.savefig(dir_path+result_path+'/PCA_DBCAN.png')
plt.savefig(dir_path+result_path+'/PCA_DBCAN.svg', format='svg', dpi=1200)
model_data['ClusterID_PCA_DBCAN'] = labels



# PCA_HDBCAN
lowd_data = PCA(n_components=50).fit_transform(fit_data)
labels = hdbscan.HDBSCAN(min_samples=10, min_cluster_size=500).fit_predict(lowd_data)
clustered = (labels >= 0)
plt.scatter(standard_embedding[~clustered, 0],
            standard_embedding[~clustered, 1],
            c=(0.5, 0.5, 0.5),
            s=0.1,
            alpha=0.5)
plt.scatter(standard_embedding[clustered, 0],
            standard_embedding[clustered, 1],
            c=labels[clustered],
            s=0.1,
            cmap='Spectral');

plt.title('PCA_HDBCAN')
plt.savefig(dir_path+result_path+'/PCA_HDBCAN.png')
plt.savefig(dir_path+result_path+'/PCA_HDBCAN.svg', format='svg', dpi=1200)
model_data['ClusterID_PCA_HDBCAN'] = labels 



# UMAP_HDBCAN
clusterable_embedding = umap.UMAP(
    n_neighbors=30,
    min_dist=0.0,
    n_components=2,
    random_state=42,
).fit_transform(fit_data)

labels = hdbscan.HDBSCAN(
    min_samples=10,
    min_cluster_size=500,
).fit_predict(clusterable_embedding)
clustered = (labels >= 0)
plt.scatter(standard_embedding[~clustered, 0],
            standard_embedding[~clustered, 1],
            c=(0.5, 0.5, 0.5),
            s=0.1,
            alpha=0.5)
plt.scatter(standard_embedding[clustered, 0],
            standard_embedding[clustered, 1],
            c=labels[clustered],
            s=0.1,
            cmap='Spectral');

plt.title('UMAP_HDBCAN')
plt.savefig(dir_path+result_path+'/UMAP_HDBCAN.png')
plt.savefig(dir_path+result_path+'/UMAP_HDBCAN.svg', format='svg', dpi=1200)
model_data['ClusterID_UMAP_HDBCAN'] = labels

# DBSCAN
#from sklearn.manifold import TSNE
# Project the data: this step will take several seconds
#tsne = TSNE(n_components=2, init='random', random_state=0)
#standard_embedding = tsne.fit_transform(fit_data)
model = DBSCAN(eps=3, min_samples=500).fit(lowd_data)
labels = model.labels_
clustered = (labels >= 0)
plt.scatter(standard_embedding[~clustered, 0],
            standard_embedding[~clustered, 1],
            c=(0.5, 0.5, 0.5),
            s=0.1,
            alpha=0.5)
plt.scatter(standard_embedding[clustered, 0],
            standard_embedding[clustered, 1],
            c=labels[clustered],
            s=0.1,
            cmap='Spectral');


plt.title('DBSCAN')
plt.savefig(dir_path+result_path+'/DBSCAN.png')
plt.savefig(dir_path+result_path+'/DBSCAN.svg', format='svg', dpi=1200)
model_data['ClusterID_DBCAN'] = labels




##########  SAVING PREDICTION #############
# save predictions
cluster_ids = ['ClusterID_PCA_DBCAN','ClusterID_PCA_HDBCAN','ClusterID_UMAP_HDBCAN']
model_data[['session_id_hash']+cluster_ids].to_csv(dir_path+result_path+'/prediction_output_sid_cid.csv')
model_data.to_csv(dir_path+result_path+'/feature_outputs.csv')





"""
Tried K-Means . Not working well

Code:
# clustering
# finding the right K using elbow method
Sum_of_squared_distances = []
K = range(1,20)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(model_data.loc[:, model_data.columns != 'session_id_hash'])
    # evaluating clusters
    Sum_of_squared_distances.append(km.inertia_)

plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()
# from the plot choose K at the elbow point


# predict clusters
#k = 4
#km = KMeans(n_clusters=k)
#model = km.fit(model_data.loc[:, model_data.columns != 'session_id_hash'])
fit_data = model_data[model_data.columns[~model_data.columns.isin(['session_id_hash','cluster_id'])]]


"""