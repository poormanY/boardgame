#-*- coding: utf-8 -*-
import sys
from collections import Counter
from sklearn.decomposition import PCA
from scipy.spatial import distance
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
# 한글 폰트 사용을 위해서 세팅
from matplotlib import font_manager, rc
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--n_clusters', type=int, default=4)
args = parser.parse_args()
N_CLUSTERS = args.n_clusters

plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

features = {}
vecs = []
names = []
scores = []
player = {}
data = {}
col = []
col_n = 0

for i, line in enumerate(sys.stdin):
    words = line.split(',', col_n-1)
    if i == 0:
        col = words
        col_n = len(col)
        questions = words[5:-1]
        continue
    
    name = words[1] 
    email = words[4]
    answers = words[5:-1]

    player[email] = {
        'name': name,
        'answers': {} 
    }

    for x, y in zip(questions, answers):
        if not x in data:
            data[x] = []
        data[x].append(y)
        player[email]['answers'][x] = y

min_data = {}
for k, v in data.items():
    cnts = Counter(v).most_common()
    min_data[k] = cnts[-1][0]

for k, v in player.items():
    features[k] = {
        'name': v['name'],
        'vec': []
    }

    score = 0
    for q, a in v['answers'].items():
        if a == min_data[q]:
            score += 1
            features[k]['vec'].append(-1)
        else:
            features[k]['vec'].append(1)
    #print('{}\t{}\t{}'.format(v['name'], score, features[k]['vec']))
    names.append(features[k]['name'])
    vecs.append(features[k]['vec'])
    scores.append(score)


### plot kmean cluster
pca = PCA(2)
 
#Transform the data
df = pca.fit_transform(vecs)

#Initialize the class object
kmeans = KMeans(n_clusters= N_CLUSTERS)
 
#predict the labels of clusters.
label = kmeans.fit_predict(df)
 
#Getting the Centroids
centroids = kmeans.cluster_centers_
labels = kmeans.labels_
print(centroids)



unbalance_groups = True
while(unbalance_groups):
    for cent_idx, (cent_x, cent_y) in enumerate(centroids):
        min_idx = None
        min_cent_idx = None
        min_distance = None
        cnt = np.count_nonzero(labels == cent_idx)

        for idx, value in enumerate(df):
            _name = names[idx]
            _label = labels[idx]
            _score = scores[idx]
            x, y = value[0], value[1]
            if cnt >= len(labels)//N_CLUSTERS:
                continue
            if cent_idx == labels[idx]:
                continue
            d = distance.cdist(np.array([[cent_x, cent_y],]), np.array([[x, y],]))
            if (min_idx == None) or (d < min_distance):
                min_distance = d
                min_idx = idx
                min_cent_idx = cent_idx
        if not min_idx == None:
            labels[min_idx] = min_cent_idx

    unbalance_groups = False
    for cent_idx, (cent_x, cent_y) in enumerate(centroids):
        cnt = np.count_nonzero(labels == cent_idx)
        if cnt < len(labels)//N_CLUSTERS:
            unbalance_groups = True
            break


#Getting unique labels
u_labels = np.unique(label)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

print('#######################################')
print('idx', 'name', 'label', 'score', 'x, y')
for idx, (x,y) in enumerate(df):
    print(idx, names[idx], labels[idx], scores[idx], (round(x,4), round(y,4)))
    plt.scatter(x, y, s=8, c=[colors[labels[idx]]]) 
    plt.text(x+0.05, y-0.02, idx)

for idx, (x, y) in enumerate(centroids):
    plt.scatter(x, y, s=10, c=[colors[idx]]) 
    plt.text(x-0.05, y+0.05, 'G{}'.format(idx))
plt.legend()
plt.show()
