#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from collections import defaultdict
import argparse
import pprint as pp
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import re
import matplotlib.pyplot as plt
import random
from matplotlib import style
style.use('ggplot')
import numpy as np
import random
import glob
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy import sparse
import scipy.spatial.distance as distance
from sklearn.decomposition import PCA
#import plotly
#import plotly.plotly as py
#import plotly.graph_objs as go

parser = argparse.ArgumentParser()
parser.add_argument("vb_choisi", help = "verbe à clusteriser", default = None)
parser.add_argument("pourcentage_graine", help="nb d'exemple pour constituer une graine", default = 3)
args = parser.parse_args()

liste_prep = [] #la liste des prep qui peuvent suivre le verbe

class Desamb:


	"""
	Classe qui implémente un désambiguisateur lexical grâce à un Kmeans semi-supervisé.


	Attributs:

	vb_choisi: str
		lemme à désambiguiser

	taille_fenetre: int
		fenêtre utilisée pour extraire le contexte du lemme

	golds, tok_ids: list
		contiennent respectivement les classes golds et les ids des tokens cibles

	emb: defaultdict(float)
		contient les embeddings de chaque mot présent dans le fichier d'embeddings -> emb[mot] = son embedding

	liste_conll: list
		contient toutes les phrases sous leur forme conll

	kmeans: K_Means
		algorithme de clustering utilisé par le désambiguisateur

	X, Y: list
		X contient les vecteurs et Y les classes golds. Y[i] = classe gold de X[i]

	gold2vec: dict
		contient tous les vecteurs pour une classe -> gold2vec[gold] = [ vecteurs associés à cette classe ]

	pourcentage_graine: int
		nombre d'exemples utilisés pour créer une graine

	seeds: list
		graines passées au KMeans

	liste_prep: list
		contient toutes les prépositions présentes dans le corpus


	Méthodes:

		extraction_conll(self, phr, sentence_id, taille_fenetre)

		emb2dic(self, fichier)

		average_emb(self, dic_emb, phr, sentence_id, taille_fenetre)

		create_vector(self, phr, sentence_id, taille_fenetre, taille_emb, dic_emb)

		createX_Y(self)

		create_seeds(self, nb_clusters)
	"""

	def __init__(self, vb_choisi, pourcentage_graine):

		self.vb_choisi = vb_choisi
		self.taille_fenetre = 10
		
		conll_stream = open(glob.glob("data/" + vb_choisi + "/" + vb_choisi + "*.conll")[0], encoding="utf8")
		gold_stream = open(glob.glob("data/" + vb_choisi + "/" + vb_choisi + "*.gold")[0], encoding="utf8")
		tok_ids_stream = open(glob.glob("data/" + vb_choisi + "/" + vb_choisi + "*.tok_ids")[0], encoding="utf8")

		self.golds = gold_stream.readlines()
		self.tok_ids = tok_ids_stream.readlines()
		
		self.emb = defaultdict(float)

		self.liste_conll = []
		for sentence in (conll_stream.read()).split("\n\n"):
			self.liste_conll += [sentence.split("\n")]

		self.kmeans = None 

		self.X, self.Y = [], []
		self.gold2vec = {}
		self.pourcentage_graine = 1 #int(pourcentage_graine)
		self.seeds = []
		self.g = []

		self.liste_prep = []


			
	def extraction_conll(self, phr, sentence_id, taille_fenetre):
		
		obs = defaultdict(lambda: defaultdict(int))

		gold = int(self.golds[sentence_id])
		obs["gold"][""] = gold
		verb_id = int(self.tok_ids[sentence_id])


		#id du gouverneur du verbe (initialisé à lui-même)
		vb_gouv = verb_id

		#on regarde les infos du verbe_id
		info_verbid = phr[verb_id-1].split("\t")

		#si le lemme dépend d'un vb racine, il faut regarder les arguments de ce dernier
		if ("dm=ind" not in info_verbid[5].split("|")):

			if not ( "dm=inf" in info_verbid[5].split("|") and "obj.p" or "obj" in re.split('||:', info_verbid[7]) ):
				vb_gouv = int(info_verbid[6].split("|")[0]) #c'est mtn son gouverneur qui gouverne notre recherche d'infos

		for i in range(len(phr)):

			mot = phr[i].split("\t")

			#--nbr d'arguments + prep
			#premiere condition = pour les cas ou le gouverneur est qqch comme "4|3"
			#deuxieme condition = cas normaux 
			
			gouv = [int(n) for n in mot[6].split("|")]

			if (vb_gouv in gouv):
				not_wanted = ["P","PONCT","P+D","V","C","V"]

				if (mot[3] not in not_wanted) and ("mod" not in mot[7].split("|")):
					obs["nb_arg"][""]+=1
				
				if mot[3]=="P":
					obs["prep"][mot[2]]+=1
					#--m.a.j de la liste des prep suivant le verbe
					#print(mot[2])
					#print(liste_prep)
					#print(type(liste_prep))
					if mot[2] not in self.liste_prep:
						self.liste_prep.append(mot[2])

			#--fenetre de mots
			#verbe_id-1-i est la position du mot courant par rapport au verbe
			pos_mot = verb_id-1-i # on fait -1 pcq en python on compte à partir de 0
			if pos_mot > 0 and pos_mot <= taille_fenetre and mot[3] != "P":

				obs["mots_fenetre"]["avant  " + mot[2]] += 1

			if pos_mot < 0 and pos_mot >= -taille_fenetre and mot[3] != "P":
				
				obs["mots_fenetre"]["apres  " + mot[2]] += 1
		#pp.pprint(obs)	
		#input()
		return obs


	#--retourne un dictionnaire key=mot, value=embedding du mot, pour chaque mot du fichier
	def emb2dic(self, fichier):
		dic_emb = defaultdict(np.array)
		with open(fichier,encoding="utf8") as stream :
			line = stream.readline()
			while line : 
				line = line.strip('\n').split(" ")
				if line[0] not in dic_emb and line[0].isalpha():
					word_emb = np.array([float(comp) for comp in line[1:] if comp != ''])
					taille_emb = len(word_emb)
					dic_emb[line[0]] = word_emb
				line = stream.readline()
		stream.close()
		return dic_emb,taille_emb

	
	#--renvoie un dictionnaire key=mot, value= la moyenne des embeddings des mots de la fenetre
	def average_emb(self, dic_emb, phr, sentence_id, taille_fenetre):
		tmp = 0
		dic_av_emb = defaultdict(float)
		for word_emb in dic_emb:
			tmp=0
			if word_emb not in dic_av_emb:
				for comp in dic_emb[word_emb]:
					tmp+=comp
					if len(dic_emb[word_emb]) !=0:
						dic_av_emb[word_emb] = tmp/len(dic_emb[word_emb])
					else : 
						print(word_emb)
		return dic_av_emb


	#--trouve la liste complète des prep eventuelles a mettre dans le vecteur
	#def get_liste_prep():
	#	for i in range(len(d.liste_conll)-1):
	#		#v = create_vector(d.liste_conll[i],i,3)
	#		obs = d.extraction_conll(d.liste_conll[i],i,3)
	#		if obs["nb_arg"][""]>=4: return(liste_prep)


	#-- crée le vecteur associé à chaque phrase
	def create_vector(self, phr, sentence_id, taille_fenetre, taille_emb, dic_emb):
		
		obs = d.extraction_conll(phr, sentence_id, taille_fenetre)
		taille_embeg = taille_emb
		sum_emb = np.zeros(taille_embeg)
		nb_arg =obs["nb_arg"][""]


		for elt in obs["mots_fenetre"]:
			elt = elt.split(" ")
			if elt[2] in dic_emb : 
				#print(dic_emb[elt[2]])
				sum_emb = sum_emb + dic_emb[elt[2]]
		
		average_emb = sum_emb/taille_embeg		

		##vector = np.zeros(3+1+taille_embeg)
		vector = np.zeros(taille_embeg)
		#vect[pos]=1 si i+1 est le nombre d'arguments du verbe dans la phrase 
		##vector[nb_arg-1]=1
		#print(vector)
		#input()
		#vect[pos]= 1 si la preposition à l'indice pos dans list_prep suit le verbe dans la phr
		list_of_prep = obs["prep"].keys()
		#for i in range(len(liste_prep)):
		#	if liste_prep[i] in list_of_prep:
		##if len(obs["prep"])>0:

		##	vector[3]=1

		#--les composantes restantes du vecteur correspondent aux composantes du word embedding	
		##for i in range(3+1,len(vector)):
		for i in range(len(vector)):
			##vector[i] = average_emb[i-(3+1)]
			vector[i] = average_emb[i]
		return(vector, obs["gold"][""]) #utile pour créer la liste des golds dans la suite


	def createX_Y(self):

		datas = [] #la liste des vecteurs
		golds = [] #la liste des golds 
		### golds[i] est l'idx de la classe associée au vecteur datas[i]

		#stocke les vecteurs dans datas(list), et leur classe gold associée dans golds(list)
		#liste_prep=get_liste_prep()

		dic_emb, taille_emb = self.emb2dic("w2v_final")
		for i in range (len(self.liste_conll)-1):
			vector, gold = self.create_vector(self.liste_conll[i], i, self.taille_fenetre, taille_emb, dic_emb)
			datas.append(vector)
			golds.append(gold)


		#création d'un dico avec key=classe_gold, value=liste des vecteurs associés
		gold2vec = defaultdict(list)
		for i in range(len(datas)):
			gold2vec[golds[i]].append(datas[i])

		self.gold2vec = gold2vec

		for classe in gold2vec:
			self.X += gold2vec[classe][:40]
			self.Y +=[classe for i in range(40)]

		gold2vecb = defaultdict(list)

		for i in range(len(self.X)):
			gold2vecb[self.Y[i]].append(self.X[i])

		self.gold2vec = gold2vecb
		print("gold2vec", [(elt,len(gold2vecb[elt])) for elt in gold2vecb]) #juste un test 

		"""
		for gold in self.gold2vec:

			print("gold", gold)
			for i in range(len(self.gold2vec[gold])-1):
				#print(1 - distance.cosine(self.gold2vec[gold][i], self.gold2vec[gold][i+1]))
				c = cosine_similarity([self.gold2vec[gold][i]], [self.gold2vec[gold][i+1]])
				#dists = euclidean_distances([self.gold2vec[gold][i]], [self.gold2vec[gold][i+1]])
				print(c)
				#if dists > 0.6:
				#	print("dists > 0.6")
				#	print(self.gold2vec[gold][i])
				#	print(self.gold2vec[gold][i+1])
				#	print()

			input()"""

		#on peut instancier le Kmeans et créer les seeds mtn qu'on a gold2vec
		self.kmeans = K_Means(k=len(self.gold2vec), g=self.g)
		self.seeds, self.g = self.create_seeds(len(self.gold2vec)) #a décommenter pour le supervisé


	def create_seeds(self, nb_clusters):

		#pb à p-ê rajouter dans rapport: galère avec les types et les opérations entre matrices?

		#on initialise une matrice (nb_cluster, taille d'un vecteur)
		seeds = np.empty( (nb_clusters, len(self.X[0])), dtype=list ) 
		
		#position actuelle dans seeds
		n = 0
	
		g = []

		for gold in self.gold2vec:

			exs = []

			#nomrbe d'exemples utilisés pour cette classe
			taille_graine = len(self.gold2vec[gold])*(self.pourcentage_graine/100)
			#print("ex gardés pour", gold, "=", taille_graine)
			if taille_graine<1: taille_graine=1
			#input()
			
			for i in range(self.pourcentage_graine):#round(taille_graine)): #/!\ mettre qqpart des garde fous sur pourcentage_graine
												#certaines gold n'ont que 3 ex

				#on sélectione un objet au hasard dans la classe voulue
				#puis on le supprime - du dico gold2vec
				#					 - du dataset
				r = random.randint(0, len(self.gold2vec[gold]))-1
				obj = self.gold2vec[gold][r]
				del self.gold2vec[gold][r]

				#galere pour retrouver obj dans le dataset
				#-> méthode trouvée est trop lourde: à modifier si ya le temps
				#   pcq il faut mettre X en liste de liste pour utiliser .index()
				#	et X est une liste d'array
				tmp = [list(ex) for ex in self.X]
				id_a_supp = tmp.index(list(obj))
				del self.X[id_a_supp]
				del self.Y[id_a_supp]

				#on ajoute l'ex tiré au sort à la liste d'exemples
				exs += [obj]


			#on fait la moyenne des x exemples tirés au sort
			#et on ajoute le résultat à la liste de graines
			moy = np.mean(exs, axis=0)
			seeds[n] = moy
			n+=1
			g+= [gold]

		return (seeds, g)	


class K_Means:

	"""
	Classe qui implémente un algorithme de clustering K-Means.


	Attributs:

	k: int
		nombre de clusters

	tol: int

	max_iter: int
		nombre d'itération maximum

	seeds: list
		éventuelles graines passées à l'algo


	Méthodes:

		fit(self, data)

		predict(self, data)

		evaluation
	"""

	def __init__(self, k, g, tol=0.001, max_iter=30, seeds=None):
		self.k = k
		self.tol = tol
		self.max_iter = max_iter
		self.seeds = seeds
		self.centroids = {}
		self.classifications = {}
		self.g = g

	def fit(self, data, golds):

		#self.centroids = {}
		if len(self.seeds)==0: 
			np.random.shuffle(data)
			#selectionne les centroids de départ (les deux premiers points de la liste de donnees melangee)
			for i in range (self.k):
				self.centroids[i] = data[i] 
		else:
			i=0
			for classe in self.g:
				self.centroids[classe] = self.seeds[i]
				i+=1

		#print("centroids: ", self.centroids)

		for i in range (self.max_iter):

			self.classifications = {}

			for classe in self.g:
				self.classifications[classe] = []

			bonnes_rep = 0
			for k in range(len(data)):
				
				#distance du point avec les centroides
				#print([self.centroids[centroid] for centroid in self.centroids])

				#distances = [ euclidean_distances( [data[k]] , [self.centroids[centroid]] ) for centroid in self.centroids ]
				distances = [np.linalg.norm(data[k]-self.centroids[centroid]) for centroid in self.centroids] 
				classification = distances.index(min(distances)) #renvoie l'indice de la classe/du controide le plus proche
				classification = self.g[classification]




				self.classifications[classification].append(data[k]) #on ajoute l'exemple  au dico de classification

				if classification == golds[k]: bonnes_rep+=1

			print("evaluation epoch n°%s :" % str(i+1), "bonnes_rep", round((bonnes_rep/len(data) *100), 1), "%")
			pp.pprint(self.eval(data, golds))
			input()

			prev_centroids = dict(self.centroids) #on stocke les centroids précédents

			for classification in self.classifications:
				
				#méthode A: moyenne de tous les vecteurs du centroid

				#on calcule les nouveaux centroides
				self.centroids[classification] = np.average(self.classifications[classification], axis=0)


				#méthode B: moyenne des vecteurs ayant le même sens que le centroid

				"""
				4_average = []
				#on prend que les vecteurs ayant le meme sens que le centroid

				for vecteur in self.centroids[classification]:

					vector_list = vector.tolist()
					idxTrueClass = listeVec.index(vector_list) #l'index du vecteur dans listeVec
					#print("vecteur: ", vector, "   ", "index : ", idxTrueClass)
					etiquette = listeEtique[idxTrueClass] #l'etiquette dans la listeEtique
					if etiquette != classifition: 
						pass
					else: 
						4_average += [vecteur]

				self.centroids[classification] = np.average(np.array(4_average), axis=0)
				"""


			#on teste si nos centroides sont optimaux   
			#optimized = True

			for c in self.centroids:
				original_centroid = prev_centroids[c]
				#print(original_centroid)
				current_centroid = self.centroids[c]
				#print(current_centroid)
				#calcule la somme des écarts pour chaque feature et compare la valeur à la tolérance
				#print("sum: ", np.sum((current_centroid-original_centroid)/original_centroid*100.0))
				#input()
				#if np.sum((current_centroid-original_centroid)/original_centroid*100.0) > self.tol:  
				#	print(np.sum((current_centroid-original_centroid)/original_centroid*100.0))
				#	optimized = False

			#if optimized:
			#	break

			#print("fit: ", self.classifications)

		"""print("distances centroids")
		for i in range(len(self.centroids)):
			if i+1<len(self.centroids):
				print(euclidean_distances([self.centroids[i]], [self.centroids[i+1]]))
			else:
				print(euclidean_distances([self.centroids[i]], [self.centroids[0]]))"""



	def predict(self, data):
		distances = [np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids] 
		classification = distances.index(min(distances))
		return classification


	def eval(self, listeVec, listeEtique):

		#print("clasification: ", self.classifications)
		#listeVec = listeVec.tolist() #passe de np.array a liste simple
		#print("listeVec: ", listeVec)
		#print(listeVec)
		tmp = [list(ex) for ex in listeVec]
		listeVec = tmp
		self.evaluation = {} #dico de la forme evaluation = {0 (centroide): 1(classe): 50%, 2: 30%...}
		#print(listeEtique)
		#input()
		for centroid in self.classifications:
			
			cluster_size = len(self.classifications[centroid])
			cluster = self.classifications[centroid] #la liste des vecteurs associés au cluster
			self.evaluation[centroid] = defaultdict(int)

			for vector in cluster:
				
				vector = vector.tolist()
				idxTrueClass = listeVec.index(vector) #l'index du vecteur dans listeVec
				#print("vecteur: ", vector, "   ", "index : ", idxTrueClass)
				etiquette = listeEtique[idxTrueClass] #l'etiquette dans la listeEtique
				self.evaluation[centroid][etiquette] += (1/cluster_size) * 100 #on incrémente en pourcentage

		return self.evaluation



####################################################### MAIN

d = Desamb(args.vb_choisi, args.pourcentage_graine)
d.createX_Y()


# non supervisé
#d.kmeans.fit(d.X, d.Y)
#print("\nfinal eval:")
#print(pp.pprint(d.kmeans.eval(d.X, d.Y)))
#input()
#print("centroids", d.kmeans.centroids)

# supervisé
#print(d.seeds)
d.kmeans = K_Means(k=len(d.gold2vec), seeds=d.seeds, g=d.g) #on réinstancie le kmeans avec nos seeds
#print(d.kmeans.seeds)
d.kmeans.fit(d.X,d.Y)


#--------------graphiques / comparaisons 


def matplotlib_to_plotly(cmap, pl_entries):
    h = 1.0/(pl_entries-1)
    pl_colorscale = []
    
    for k in range(pl_entries):
        C = map(np.uint8, np.array(cmap(k*h)[:3])*255)
        for i in range(len(C)):
        	pl_colorscale.append([k*h, 'rgb'+str((C[0], C[1], C[2]))])
        
    return pl_colorscale



pca = PCA(n_components=2)
X_2d = pca.fit_transform(d.X)
centroids = np.array([item for item in d.kmeans.centroids.values()])
c_2d = pca.fit_transform(centroids)
h = .02


x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Y = d.Y

"""
#pour afficher les limites des clusters mais foireux pour l'instant
back = go.Heatmap(x=xx[0][:len(Y)],
                  y=xx[0][:len(Y)],
                  z=Y,
                  showscale=False,
                  colorscale=[[0, 'green'], [0.5, 'red'], [1.0, 'rgb(0, 0, 255)']])


#afficher les points
markers = go.Scatter(x=X_2d[:, 0], 
                     y=X_2d[:, 1],
                     showlegend=True,
                     mode='markers', 
                     marker=dict(
                             size=3, color='black'))
#afficher les centres
center = go.Scatter(x=c_2d[:, 0],
                    y=c_2d[:, 1],
                    showlegend=True,
                    mode='markers', 
                    marker=dict(
                            size=10, color='red'))
data=[markers, center]

layout = go.Layout(title ='K-means clustering on the digits dataset (PCA-reduced data)<br>'
                           'Centroids are marked with red',
                   xaxis=dict(ticks='', showticklabels=True,
                              zeroline=False),
                   yaxis=dict(ticks='', showticklabels=True,
                              zeroline=False))
fig = go.Figure(data=data, layout=layout)

plotly.offline.plot(fig)

#plt.savefig("graph_%s_pourcentage_%s" % (d.vb_choisi, int(d.pourcentage_graine)))
     
plt.savefig("graph_%s_pourcentage_%s" % (d.vb_choisi, int(d.pourcentage_graine)))
"""