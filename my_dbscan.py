# -*- coding: utf-8 -*-
"""
Created on Sat May 11 15:58:20 2019

@author: reuve
"""

import numpy as np
from sklearn.datasets import load_iris
from numpy import linalg as LA
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

#build point_dist: distance between all point pairs
def init(data):        
    #building length matrics
    for i in range(len(data)):
        for j in range(len(data)):
            points_dist[i][j]=LA.norm(data[i]-data[j]) 
   
# build an array of all the legal neighbors     of a point   
def regionQuery(point):
    neighbors=[]
    for i in range(len(points_dist)):
        if ((points_dist[i][point]<=epsilon) and (i!=point)):
            neighbors.append(i)
    return neighbors


# begins in a core point and expands the cluster till it can’t be expanded          
def expandcluster(seed):
    cluster=regionQuery(seed) #list of unclustered neighbors
    dbscan_class[seed]=cluster_idx
    while len(cluster):
        next_gen=[]
        for i in cluster:
            sub_cluster=regionQuery(i)
            dbscan_class[i]=cluster_idx #gave a dbscan_class to all neighbors
                
            #if a neighbor is a cluster point, add all his neighbors to the cluster
            if len(sub_cluster)>=min_points: 
                for j in sub_cluster: 
                    if dbscan_class[j]==0: #if not clustered yet
                        next_gen.append(j)             
        cluster=next_gen[:] #proceed to next generation

# iterates over all points in the db and if they are core points expands them 
def my_DBSCAN(): #main function running on all data points 
    global cluster_idx      
    for i in range(len(data)):
        if (len(regionQuery(i))>=min_points) and (dbscan_class[i]==0): #new cluster point
            expandcluster(i)
            cluster_idx+=1#increase number for next cluster


def show_clusters(x,y): #input - axis column x and y
    X=data[:,x]
    Y=data[:,y]
    fig, ax = plt.subplots()
    for i in range(cluster_idx):
        if i==0:
            plt.scatter(X[dbscan_class==i],Y[dbscan_class==i],label="Noise", alpha=0.3, edgecolors='none')
        else:
            plt.scatter(X[dbscan_class==i],Y[dbscan_class==i],label=("cluster# "+str(i)))
    ax.legend()
    plt.title('my DBSCAN')
    plt.show()


#plot the next iteration
def print_clusters(data,labels):
#    plt.title('Iteretion {}'.format(iter))
#    plt.scatter(data[:, 0], data[:, 1], c=labels, marker ="*", s=50);
#    plt.scatter(data[:, 2], data[:, 1], c=labels, marker ="*", s=50);
    plt.scatter(data[:, 1], data[:, 3], c=labels, marker ="*", s=50);
    plt.show()
    
#init
iris = load_iris()
data = iris.data
epsilon=0.4
min_points=5
points_dist=np.zeros((len(data),len(data))) 
cluster_idx=1 #running cluster number
dbscan_class=np.zeros(len(data))
init(data)

# Compute DBSCAN using Iris dataset
db = DBSCAN(eps=0.4, min_samples=5).fit(data)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = 'k'
    class_member_mask = (labels == k)
    data = iris['data']
    xy = data[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=6)

    xy = data[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=6)

plt.title('sklearn DBSCAN, Estimated N: %d' % n_clusters_)
plt.show()

##do my  algorithm
my_DBSCAN()
show_clusters(0,1)#input - axis column x and y
##print_clusters(data,dbscan_class)

    