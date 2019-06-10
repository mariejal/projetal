import matplotlib.pyplot as plt
import random
from matplotlib import style
style.use('ggplot')
import numpy as np

X = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8 ],
              [8, 8],
              [1, 0.6],
              [9,11]])

##plt.scatter(X[:,0], X[:,1], s=150)
##plt.show()

colors = 10*["g","r","c","b","k"]


class K_Means:
    def __init__(self, k=2, tol=0.001, max_iter=300, seeds=None):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        self.seeds = seeds

    def fit(self, data):

        self.centroids = {}

        if self.seeds==None: 
            #selectionne les centroids de départ (les deux premiers points de la liste de donnees melangee)
            for i in range (self.k):
                random.shuffle(data)
                self.centroids[i] = data[i] 
        else: 
            self.centroids[i] = seeds[i]


        for i in range (self.max_iter):
            self.classifications = {}

            for i in range (self.k):
                self.classifications[i] = []

            for featureset in data:
                #disatance du point avec les centroides
                distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids] 
                classification = distances.index(min(distances)) #renvoie l'indice de la classe/du controide le plus proche
                self.classifications[classification].append(featureset) #on ajoute l'exemple  au dico de classification

            prev_centroids = dict(self.centroids) #on stocke les centroids précédents

            for classification in self.classifications:
                #on calcule les nouveaux centroides
                self.centroids[classification] = np.average(self.classifications[classification],axis=0)


            #on teste si nos centroides sont optimaux   
            optimized = True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                #calcule la somme des écarts pour chaque feature et compare la valeur à la tolérance
                if np.sum((current_centroid-original_centroid)/original_centroid*100.0) > self.tol:  
                    print(np.sum((current_centroid-original_centroid)/original_centroid*100.0))
                    optimized = False
            
            if optimized:
                break


    def predict(self, data):
        distances = [np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids] 
        classification = distances.index(min(distances))
        return classification

clf=K_Means()
clf.fit(X)


for centroid in clf.centroids:
    for centroid in clf.centroids:
        plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1],
                    marker="o", color="k", s=150, linewidths=5)

for classification in clf.classifications:
    color = colors[classification]
    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0], featureset[1], marker="x", color=color, s=150, linewidths=5)
        
plt.show()
