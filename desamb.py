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
from sklearn.cluster import KMeans
import plotly
import plotly.plotly as py
import plotly.graph_objs as go

parser = argparse.ArgumentParser()
parser.add_argument("vb_choisi", help = "verbe à clusteriser")
parser.add_argument("pourcentage_graine", help="pourcentage d'exemple utilisés pour constituer une graine")
parser.add_argument("version", help="version non/faible/fortement supervisé\nEntrer non_superv ou faible_superv ou forte_superv")
args = parser.parse_args()
liste_prep = []

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
		graines passées au K-Means

	liste_golds: list
		contient la liste des classes

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


	def __init__(self, vb_choisi, pourcentage_graine, version):
		self.vb_choisi = vb_choisi
		self.taille_fenetre = 5
		self.version = version
		
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

		self.pourcentage_graine = int(pourcentage_graine)

		self.seeds = []
		self.liste_golds = []

		self.liste_prep = []

			
	def extraction_conll(self, phr, sentence_id, taille_fenetre):

		"""

		Extrait d'un fichier conll les informations relavatives à une phrase passée en argument

		Entrée: 

			phr => list
				une phrase sous sa représentation conll

			sentence_id => int
				indice de la phrase

			taille_fenetre => int
				taille de la fenetre

		Sortie: dict de dict, par exemple les mots du contexte sont présents dans obs["mots dans la fenetre lexicale"] = dict{ 

																				["mot avant"]: "murailles"
																				["mot apres"]: "Marc"
																				}
																			

		"""
		
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
					if mot[2] not in self.liste_prep:
						self.liste_prep.append(mot[2])

			#--fenetre de mots
			#verbe_id-1-i est la position du mot courant par rapport au verbe
			# on fait -1 car on compte à partir de 0
			pos_mot = verb_id-1-i
			if pos_mot > 0 and pos_mot <= taille_fenetre and mot[3] != "P":

				obs["mots_fenetre"]["avant  " + mot[2]] += 1

			if pos_mot < 0 and pos_mot >= -taille_fenetre and mot[3] != "P":
				
				obs["mots_fenetre"]["apres  " + mot[2]] += 1

		return obs


	def emb2dic(self, fichier):
		
		"""
		Construit un dico qui stocke le word embedding associé à chaque mot 

		Entrée: 
			string => nom du fichier contenant les embeddings

		Sortie:
			dict => key=mot, value=embedding du mot, pour chaque mot du fichier
		"""

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

	
	def average_emb(self, dic_emb, phr, sentence_id, taille_fenetre):

		"""
		Construit un dico qui stocke le word embedding moyen représentant les mots en contexte dans la fenêtre du mot à clusteriser
			
		Entrée:

			dic_emb => dict
				dictionnaire des word embeddings

			phr => list
				une phrase sous sa représentation conll

			sentence_id => int
				indice de la phrase

			taille_fenetre => int
				taille de la fenetre

		Sortie:
			dict =>  key=mot, value= la moyenne des embeddings des mots de la fenetre

		"""

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


	def create_vector(self, phr, sentence_id, taille_fenetre, taille_emb, dic_emb):

		"""
		Crée le vecteur associé à chaque phrase
		Sont en commentaire ##-- les instructions qui étaient utilisées
		pour construire noc vecteurs initiaux 
		(avec les infos sur le nombre d'arguments et les prépositions en plus des word embeddings )

		Entrée:
			phr => list
				une phrase sous sa représentation conll
			
			sentence_id => int
				indice de la phrase

			taille_fenetre => int
				taille de la fenetre 

			taille_emb => int 
				taille d'un vecteur de word embeddings

			dic_emb => dict
				dictionnaire des word embeddings
		
		Sortie:
			tuple => (le vecteur crée, sa classe gold)
		"""

		obs = d.extraction_conll(phr, sentence_id, taille_fenetre)
		taille_embeg = taille_emb
		sum_emb = np.zeros(taille_embeg)
		nb_arg =obs["nb_arg"][""]


		for elt in obs["mots_fenetre"]:
			elt = elt.split(" ")
			if elt[2] in dic_emb : 
				sum_emb = sum_emb + dic_emb[elt[2]]
		
		average_emb = sum_emb/taille_embeg		

		vector = np.zeros(3+1+taille_embeg)
		#vector = np.zeros(taille_embeg)
		
		#vect[pos]=1 si i+1 est le nombre d'arguments du verbe dans la phrase 
		vector[nb_arg-1]=1

		#si la preposition à l'indice pos dans list_prep suit le verbe dans la phr
		#--vect[pos]= 1 
		
		if len(obs["prep"])>0:

			vector[3]=1

		#les composantes restantes du vecteur correspondent aux composantes du word embedding	
		for i in range(3+1,len(vector)):
		#for i in range(len(vector)):
			vector[i] = average_emb[i-(3+1)]
			#vector[i] = average_emb[i]

		return(vector, obs["gold"][""]) #utile pour créer la liste des golds dans la suite


	def createX_Y(self, use_seeds):

		"""

		Crée 
		- les objets X (contient les vecteurs) et Y (les sens associés)
			Y[i] est l'idx de la classe associée au vecteur X[i] 
		- gold2vec
		- et potentiellement les graines

		Entrée:

			use_seeds: bool
				si == True, on instanciera aussi des graines pour la partie semi-supervisée

		Sortie:
			aucune, les instances remplies sont affectées en place aux attributs correspondant avec self
		"""

		#la liste des vecteurs
		datas = []
		#la liste des golds
		golds = []


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

		self.X = datas
		self.Y = golds


		#on peut instancier le Kmeans et créer les seeds maintenant qu'on a gold2vec
		self.kmeans = K_Means(k=len(self.gold2vec), liste_golds=self.liste_golds)
		
		if use_seeds:
			self.seeds, self.liste_golds = self.create_seeds(len(self.gold2vec))


	def create_seeds(self, nb_clusters):

		"""

		Crée les graines à donner au K-means semi-supervisé

		Entrée: 

			nb_clusters: int
				k / nombre de sens associés à un verbe

		Sortie:
			tuple: (liste de listes/vecteurs pour chaque sens, liste des sens cibles)
		"""

		#on initialise une matrice (nb_cluster, taille d'un vecteur)
		seeds = np.empty( (nb_clusters, len(self.X[0])), dtype=list ) 
		
		#position actuelle dans seeds
		n = 0
		
		#contient la liste des classes
		g = []

		for gold in self.gold2vec:

			exs = []
				
			taille_graine = len(self.gold2vec[gold])*(self.pourcentage_graine/100)
			if taille_graine<1: taille_graine=1 #on utilise au minimum 1 graine

			for i in range(round(taille_graine)):

				#on sélectione un objet au hasard dans la classe voulue
				#puis on le supprime - du dico gold2vec
				#					 - du dataset
				r = random.randint(0, len(self.gold2vec[gold]))-1
				obj = self.gold2vec[gold][r]
				del self.gold2vec[gold][r]

				#il faut mettre X en liste de liste pour utiliser .index()
				#et X est une liste d'array
				#on stocke tout X en changeant les types de ses objets
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


	max_iter: int
		nombre d'itération maximum

	seeds: list
		éventuelles graines passées à l'algo

	centroids: dict
		contient un vecteur pour chaque centroid/ sens -> centroids[sens] = vecteur moyenné

	classifications: dict
		contient les vecteurs associé pour chaque centroid -> classification[centroid] = list(de np array/vecteurs contenus dans ce centroid)

	liste_golds: list
		contient une liste des labels (= [1, 2, 5, 3] pour abattre par exemple)

	resultats: list
		contient les résultats pour chaque itération du K-means


	Méthodes:

		fit(self, data)

		predict(self, data)

		evaluation
	"""

	def __init__(self, k, liste_golds, tol=0.001, max_iter=10, seeds=[]):

		self.k = k
		self.max_iter = max_iter
		self.seeds = seeds
		self.centroids = {}
		self.classifications = {}
		self.liste_golds = liste_golds
		self.resultats = [] #sera aussi long que le nbr d'itération

	def fit(self, data, golds, use_seeds, maj_controlees): #maj_controlees = boolean mis a True si on fait des mises a jour controlees

		"""
			Initialise les centroides et les met à jour (self.centroids)
			A chaque itération, un dictionnaire de classification spécifiant à quel centroid est assigné chaque point 
			est mis a jour (self.classifications)
			Génère un dictionnaire retenant la composition des clusters finale pour chaque test 

			Entrée:

				data => liste de np arrays
				golds => liste des étiquettes associées à chaque vecteur

			Sortie: aucune 
		"""

		#on détermine la clé (le centroid) par un sens si on est en supervisé, par des clés non représentatives sinon
		if not(use_seeds): 

			np.random.shuffle(data)
			#selectionne les centroids de départ (les deux premiers points de la liste de donnees melangee)

			for i in range (self.k):
				self.centroids[i] = data[i] 
		else:
			i=0
			for classe in self.liste_golds:
				self.centroids[classe] = self.seeds[i]
				i+=1

		

		for i in range (self.max_iter):

			#on vide classifications à chaque itération
			self.classifications = {}

			#on détermine la clé (le centroid) par un sens si on est en supervisé, par des clés non représentatives sinon
			if use_seeds:
				for classe in self.liste_golds:
					self.classifications[classe] = []
			else:
				for j in range(self.k):
					self.classifications[j] = []

			bonnes_rep = 0
			for m in range(len(data)):
				
				#distance du point avec les centroides
				distances = [np.linalg.norm(data[m]-self.centroids[centroid]) for centroid in self.centroids] 

				#renvoie l'indice de la classe/du controide le plus proche
				classification = distances.index(min(distances))
				
				#on remplace la clé par le sens associé au cluster le plus proche
				if use_seeds:
					classification = self.liste_golds[classification]

				#on ajoute l'exemple  au dico de classification
				self.classifications[classification].append(data[m])

				if classification == golds[m]: bonnes_rep+=1

			#print("evaluation epoch n°%s :" % str(i+1), "bonnes_rep", round((bonnes_rep/len(data) *100), 1), "%")
			#pp.pprint(self.eval(data, golds))
			self.resultats += [round((bonnes_rep/len(data) *100), 1)]
			#input()

			#on stocke les centroids précédents
			prev_centroids = dict(self.centroids)

			tmp = [list(ex) for ex in data]
			listeVec = tmp

			for classification in self.classifications:
				
				#méthode A: moyenne de tous les vecteurs du centroid
				if not(maj_controlees):

					#on calcule les nouveaux centroides avec tous les vecteurs qui y sont affecté
					self.centroids[classification] = np.average(self.classifications[classification], axis=0)

				else:
					
					#méthode B: moyenne des vecteurs ayant le même sens que le centroid
					for_average = []

					#on prend que les vecteurs ayant le meme sens que le centroid
					for vecteur in self.classifications[classification]:

						vector_list = vecteur.tolist()

						#l'index du vecteur dans listeVec
						idxTrueClass = listeVec.index(vector_list)
						
						#l'etiquette dans la listeEtique
						etiquette = golds[idxTrueClass]

						if etiquette != classification: 
							pass
						else: 
							for_average += [vecteur]

					self.centroids[classification] = np.average(np.array(for_average), axis=0)

			#on remplace les anciens centroids par les nouveaux
			for c in self.centroids:
				original_centroid = prev_centroids[c]
				current_centroid = self.centroids[c]


	def eval(self, listeVec, listeEtique):


		"""
		Fait le bilan sur la composition d'un centroide, à savoir, quelle classe est majoritaire, s'il y en a une 
		
		Entrée:

			listeVec => la liste de vecteurs utilisées
			listeEtique => la liste des étiquettes correspondantes
			listeEtique[0] est l'étiquette de listeVec[0]

		Sortie: 
		
			un dico evaluation de forme {0(le numéro de centroide/cluster) : 1(classe): 50%, 2(classe): 25%...}
		"""


		tmp = [list(ex) for ex in listeVec]
		listeVec = tmp
		self.evaluation = {}

		for centroid in self.classifications:
			
			cluster_size = len(self.classifications[centroid])

			#la liste des vecteurs associés au cluster
			cluster = self.classifications[centroid]
			self.evaluation[centroid] = defaultdict(int)

			for vector in cluster:
				
				vector = vector.tolist()
				
				#l'index du vecteur dans listeVec
				idxTrueClass = listeVec.index(vector)
				
				#l'etiquette dans la listeEtique
				etiquette = listeEtique[idxTrueClass]

				#on incrémente en pourcentage
				self.evaluation[centroid][etiquette] += (1/cluster_size) * 100

		return self.evaluation



####################################################### MAIN

#on instancie notre Desambiguisateur
d = Desamb(args.vb_choisi, args.pourcentage_graine, args.version)





#PARTIE NORMALE

#on choisit 1 version parmi les 3 possibles

# ---------------------------------- 1 lancé de l'algo normal sans création de graphique ou quoi

###version "non superv", version "faible superv", version "forte superv"

###version 1: non supervisé
def non_superv():
	d.createX_Y(False) # false pcq on veut pas creer des seeds
	d.kmeans.fit(d.X, d.Y, False, False)

###supervisé
#version 2: si tu veux juste utiliser les graines

def faible_superv():
	d.createX_Y(True) #true pcq on veut creer des seeds
	d.kmeans = K_Means(k=len(d.gold2vec), seeds=d.seeds, liste_golds=d.liste_golds) #on réinstancie le kmeans avec nos seeds
	d.kmeans.fit(d.X, d.Y, True, False)

#version 3: si tu veux utiliser les graines + maj contrôlées
def forte_superv():
	d.createX_Y(True) #true pcq on veut creer des seeds
	d.kmeans = K_Means(k=len(d.gold2vec), seeds=d.seeds, liste_golds=d.liste_golds) #on réinstancie le kmeans avec nos seeds
	d.kmeans.fit(d.X, d.Y, True, True)


if d.version=="non_superv":
	non_superv()
elif d.version=="faible_superv":
	faible_superv()
elif d.version=="forte_superv":
	forte_superv()
else  :
	print("VERSION ERROR, please try again")

#print("\nfinal eval:")
#print(pp.pprint(d.kmeans.eval(d.X, d.Y)))
#print("résultats:", d.kmeans.resultats)
#input()






# PARTIE GRAPHIQUE 


"""
# ---------------------------------- non supervise VS semi supervise -- VS semi supervisé ++		COMPARAISON RÉSULATS FINAUX

#ATTENTION c'est long à tourner

lances_Kmeans = 10

#contient les résultats du non supervisé
res_nSup = []
#contient les résulats du semi supervisé où on instancie seulement les graines
res_SupFaible = []

#contient les résultats du semi supervisé avec graines + màj contrôlées
res_SupFort = []


# on lance le Kmeans x fois, on recupère le meilleur résultat
# -> res_nsup[0] = meilleur résultat au premier lancé, pour le non supervisé
# on plottera les trois listes sur un graphique 

for i in range(lances_Kmeans):

	#on lance les trois types de supervisé


# 1) non supervisé
	if d.version == "non_superv":
		non_superv()
		print("résultat de ce %s ieme lancé - non supervisé: " % str(i+1), d.kmeans.resultats)	
		res_nSup += [max(d.kmeans.resultats)]

# 2) supervisé faible
	if d.version == "faible_superv":
		faible_superv()
		print("résultat de ce %s ieme lancé - supervisé faible: " % str(i+1), d.kmeans.resultats)
		res_SupFaible += [max(d.kmeans.resultats)]

# 3) supervisé fort
	if d.version == "forte_superv":
		forte_superv()
		print("résultat de ce %s ieme lancé - supervisé fort: " % str(i+1), d.kmeans.resultats)
		res_SupFort += [max(d.kmeans.resultats)]

print("res_nSup", res_nSup)
print("res_SupFaible", res_SupFaible)
print("res_SupFort", res_SupFort)

y = [i+1 for i in range(lances_Kmeans)]

fig = plt.figure()
plt.plot(y, res_nSup, label="non supervisé")
plt.plot(y, res_SupFaible, label="supervisé faible")
plt.plot(y, res_SupFort, label="supervisé fort")
plt.ylabel("taux de bonnes réponses")
plt.xlabel("i-eme lancé du Kmeans")
plt.title("Comparaison de performance Kmeans selon le degré supervisé\n%s" % d.vb_choisi)
plt.tight_layout()
plt.legend()
plt.savefig("comparaison_supervise_%s" % d.vb_choisi)

"""
"""
# ---------------------------------- évolution non supervise VS semi supervise -- VS semi supervisé ++		COMPARAISON EVOLUTION SELON ITÉRATION
#																											POUR UN SEUL LANCÉ

# pas long à tourner ca va 

#contient l'évolution du non supervisé
evolution_nSup = []

#contient l'évolutinon du semi supervisé où on instancie seulement les graines
evolution_SupFaible = []

#contient l'évolution du semi supervisé avec graines + màj contrôlées
evolution_SupFort = []


# 1) non supervisé
non_superv()
print("évolution des résultats - non supervisé: ", d.kmeans.resultats)	
evolution_nSup = d.kmeans.resultats

# 2) supervisé faible
faible_superv()
print("évolution des résultats - supervisé faible: ", d.kmeans.resultats)
evolution_SupFaible = d.kmeans.resultats

# 3) supervisé fort
forte_superv()
print("évolution résultat de ce  - supervisé fort: ", d.kmeans.resultats)
evolution_SupFort = d.kmeans.resultats


y = [i+1 for i in range(d.kmeans.max_iter)]

fig = plt.figure()
plt.plot(y, evolution_nSup, label="non supervisé")
plt.plot(y, evolution_SupFaible, label="supervisé faible")
plt.plot(y, evolution_SupFort, label="supervisé fort")
plt.ylabel("taux de bonnes réponses")
plt.xlabel("i-eme itération du Kmeans")
plt.title("Evolution des performance Kmeans pour un lancé\nSelon le degré supervisé\n%s" % d.vb_choisi)
plt.tight_layout()
plt.legend()
plt.savefig("evolution_selon_supervise_%s" % d.vb_choisi)
"""


# ---------------------------------- lancer plusieurs fois le kmeans 										COMPARAISONS RÉSULATS SUR X LANCÉS


res_lances = []

for i in range(100):
	
	# 1) non supervisé
	if d.version == "non_superv":
		non_superv()
		print("résultat de ce %s ieme lancé - non supervisé: " % str(i+1), d.kmeans.resultats)	
		res_lances += [max(d.kmeans.resultats)]

	# 2) supervisé faible
	if d.version == "faible_superv":
		faible_superv()
		print("résultat de ce %s ieme lancé - supervisé faible: " % str(i+1), d.kmeans.resultats)
		res_lances += [max(d.kmeans.resultats)]

	# 3) supervisé fort
	if d.version == "forte_superv":
		forte_superv()
		print("résultat de ce %s ieme lancé - supervisé fort: " % str(i+1), d.kmeans.resultats)
		res_lances += [max(d.kmeans.resultats)]

fig = plt.figure()
y = [i+1 for i in range(100)]
x = [i+1 for i in range(1, 60, 10)]
plt.plot(y, res_lances)
#plt.yticks(x)
plt.xticks([i+1 for i in range(1, 100, 20)])
plt.ylabel("taux de bonnes réponses")
plt.xlabel("i-eme lancé du Kmeans")
plt.title("Performance Kmeans sur %s lances\n%s" % (str(len(y)), d.vb_choisi))
fig.canvas.draw()
plt.tight_layout()
plt.savefig("2résultats_sur_%s_lances_\n%s" % (str(len(y)), d.vb_choisi))


"""
# ----------------------------------- plotter l'evolution selon création des graines 				COMPARAISON RÉSULATS SELON POURCENTAGE GRAINES					
#liste_pourcentages = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
#res_pourcentages = []

for pourcent in liste_pourcentages:
	
	# on écrase l'ancien pourcentage à chaque fois
	d.pourcentage_graine = pourcent

	# 1) non supervisé
	if d.version == "non_superv":
		non_superv()
		print("résultat de ce %s ieme lancé - non supervisé: " % str(i+1), d.kmeans.resultats)	
		res_lances += [max(d.kmeans.resultats)]

	# 2) supervisé faible
	if d.version == "faible_superv":
		faible_superv()
		print("résultat de ce %s ieme lancé - supervisé faible: " % str(i+1), d.kmeans.resultats)
		res_lances += [max(d.kmeans.resultats)]

	# 3) supervisé fort
	if d.version == "forte_superv":
		forte_superv()
		print("résultat de ce %s ieme lancé - supervisé fort: " % str(i+1), d.kmeans.resultats)
		res_laces += [max(d.kmeans.resultats)]

#for i in range(len(res_pourcentages)):
print(res_pourcentages)
fig = plt.figure()
plt.plot(liste_pourcentages, res_pourcentages)
plt.yticks(res_pourcentages)
plt.xticks(liste_pourcentages)
plt.ylabel("taux de bonnes réponses")
plt.xlabel("x % d'exemples utilisés pour instancier les graines")
plt.title("Performance Kmeans selon pourcentage d'ex utilisés pr graines\n%s" % (d.vb_choisi))
fig.canvas.draw()
plt.tight_layout()
plt.savefig("résultats_selon_pourcent_graines%s" % (d.vb_choisi))
"""


"""
#----------------------------------- plotter le clustering											UTILISATION PCA POUR VISUALISER CLUSTERING
#avec sklean: on rentre nos données et nos seeds et les mêmes paramètres 
#mais n'a pas les màj contrôlées
# solution A: ne peut plotter que le supervisé faible où les graines sont instanciées

# solutions B: inon, on on plotte pas les limites des clusters 

def matplotlib_to_plotly(cmap, pl_entries):
	h = 1.0/(pl_entries-1)
	pl_colorscale = []

	for k in range(pl_entries):
		C = list(map(np.uint8, np.array(cmap(k*h)[:3])*255))
		pl_colorscale.append([k*h, 'rgb'+str((C[0], C[1], C[2]))])

	return pl_colorscale


pca = PCA(n_components=2)
d.createX_Y(True)
X_2d = pca.fit_transform(d.X)
h = .02

#solution A: plotter les vecteurs, les clusters et leurs délimitations avec un Kmeans de Sklearn parametré comme le notre
#sklearn_Kmeans = KMeans(init=pca.fit_transform(d.seeds), n_clusters=d.kmeans.k, n_init=1, max_iter=d.kmeans.max_iter,
#						precompute_distances=False, verbose=0, random_state=None)

#sklearn_Kmeans.fit(X_2d)
#c_2d = sklearn_Kmeans.cluster_centers_


#solution B: plotter seulement les vecteurs et les clusters
d.kmeans = K_Means(k=len(d.gold2vec), seeds=d.seeds, g=d.liste_golds)
d.kmeans.fit(d.X, d.Y, True, True)
centroids = np.array([item for item in d.kmeans.centroids.values()])
c_2d = pca.fit_transform(centroids) # /!\ erreur ici si un des centroids est vide


x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))


##----à décommenter si on plot la solution B
#Y = sklearn_Kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
#Y = Y.reshape(xx.shape)
#pour afficher les limites des clusters mais foireux pour l'instant
#back = go.Heatmap(x=xx[0][:len(Y)],
#                  y=xx[0][:len(Y)],
#                  z=Y,
#                  showscale=False,
#                  colorscale=matplotlib_to_plotly(plt.cm.Paired, len(Y)))
#-----

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

#solution A: on affiche tout (limites, vecteurs et clusters)
#data=[back, markers, center]

#solution B: on affiche que les vecteurs et clusters
data=[markers, center]

layout = go.Layout(title ='Visualisation de la répartition des données en k clusters<br>'
                           'Les centroids sont notés en rouge<br>'
                           '\n%s' %d.vb_choisi,
                   xaxis=dict(ticks='', showticklabels=True,
                              zeroline=False),
                   yaxis=dict(ticks='', showticklabels=True,
                              zeroline=False))
fig = go.Figure(data=data, layout=layout)

plotly.offline.plot(fig)"""