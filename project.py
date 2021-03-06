import os
from collections import defaultdict
import argparse
import pprint as pp
import numpy as np
import re
import matplotlib.pyplot as plt
import random
from matplotlib import style
style.use('ggplot')
import numpy as np
import random

parser = argparse.ArgumentParser()
parser.add_argument("vb_choisi", help = "verbe à clusteriser", default = None)
parser.add_argument("taille_graine", help="nb d'exemple pour constituer une graine", default = 3)
args = parser.parse_args()

liste_prep = [] #la liste des prep qui peuvent suivre le verbe

class Desamb:

	def __init__(self, vb_choisi, taille_graine):

		self.vb_choisi = vb_choisi
		self.liste_fichier = [f for f in os.listdir("data/" + vb_choisi) if f[0] != "."] #a revoir: verifier que la liste est ds le bon ordre
		self.conll_stream = open("data/" + vb_choisi + "/" + self.liste_fichier[0],encoding="utf8")
		self.gold_stream = open("data/" + vb_choisi + "/" + self.liste_fichier[1],encoding="utf8")
		self.tok_ids_stream = open("data/" + vb_choisi + "/" + self.liste_fichier[2],encoding="utf8")

		self.golds = self.gold_stream.readlines() #liste
		self.tok_ids = self.tok_ids_stream.readlines() #liste
		self.emb = defaultdict(float)
		self.liste_conll = [] #contient toutes les phrases sous leur forme conll
		for sentence in (self.conll_stream.read()).split("\n\n"):
			self.liste_conll += [sentence.split("\n")]

		self.kmeans = K_Means()

		self.X = []
		self.Y = []

		self.liste_prep = []
		self.gold2vec = {}

		self.taille_graine = taille_graine

		"""#bloc qui existe ici pour tester
		for i in range(len(liste_conll)): 
			self.extraction_conll(liste_conll[i], i, 3)
		"""

			
	def extraction_conll(self,phr,sentence_id,taille_fenetre):
		obs = defaultdict(lambda: defaultdict(int)) #pas sure d'utiliser des dico de dico

		gold = int(self.golds[sentence_id])
		obs["gold"][""] = gold
		verb_id = int(self.tok_ids[sentence_id])
		#print("---------------------- " + str(verb_id))
		#print()

		vb_gouv = verb_id #id du verbe gouvernant le lemme(init a lui mm)

		#on regarde les infos du verbe_id
		info_verbid = phr[verb_id-1].split("\t")

		#si le lemme dépend d'un vb racine, il faut regarder les arguments de ce dernier
		if ("dm=ind" not in info_verbid[5].split("|")):

			if not ( "dm=inf" in info_verbid[5].split("|") and "obj.p" or "obj" in re.split('||:', info_verbid[7]) ):
				vb_gouv = int(info_verbid[6].split("|")[0]) #c'est mtn son gouverneur qui gouverne notre recherche d'infos

		for i in range(len(phr)):

			mot = phr[i].split("\t")
			#print(mot)

			#--nbr d'arguments + prep
			#premiere condition = pour les cas ou le gouverneur est qqch comme "4|3"
			#deuxieme condition = cas normaux 
			
			gouv = [int(n) for n in mot[6].split("|")]

			if (vb_gouv in gouv):
				not_wanted = ["P","PONCT","P+D","V","C","V"]

				if (mot[3] not in not_wanted) and ("mod" not in mot[7].split("|")):
					#print("mot[3]:",mot[3])
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
	def average_emb(self, dic_emb,phr,sentence_id,taille_fenetre):
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
		#print(self.liste_prep,len(self.liste_prep))
		#pp.pprint(obs)
		#input()

		taille_embeg = taille_emb
		sum_emb = np.zeros(taille_embeg)
		nb_arg =obs["nb_arg"][""]
		#print("nb arg =", nb_arg)
		for elt in obs["mots_fenetre"]:
			elt = elt.split(" ")
			if elt[2] in dic_emb : 
				#print(dic_emb[elt[2]])
				sum_emb = sum_emb + dic_emb[elt[2]]
		
		average_emb = sum_emb/taille_embeg
		#print(average_emb) 		

		vector = np.zeros(3+1+taille_embeg)
		#print(vector)
		#print(len(vector))
		#input()
		#vect[pos]=1 si i+1 est le nombre d'arguments du verbe dans la phrase 
		vector[nb_arg-1]=1
		#print(vector)
		#input()
		#vect[pos]= 1 si la preposition à l'indice pos dans list_prep suit le verbe dans la phr
		list_of_prep = obs["prep"].keys()
		#for i in range(len(liste_prep)):
		#	if liste_prep[i] in list_of_prep:
		if len(obs["prep"])>0:
			#print(obs["prep"])		
			vector[3]=1
			#print(vector)
			#input()

		#--les composantes restantes du vecteur correspondent aux composantes du word embedding	
		for i in range(3+1,len(vector)):
			vector[i] = average_emb[i-(3+1)]
		#print(list_of_prep)
		#print(vector)
		#input()

		return(vector, obs["gold"][""]) #utile pour créer la liste des golds dans la suite


	def createX_Y(self):
		datas = [] #la liste des vecteurs
		golds = [] #la liste des golds 
		### golds[i] est l'idx de la classe associée au vecteur datas[i]

		#stocke les vecteurs dans datas(list), et leur classe gold associée dans golds(list)
		#liste_prep=get_liste_prep()

		dic_emb, taille_emb = self.emb2dic("w2v_final")
		for i in range (len(self.liste_conll)-1):
			vector, gold = self.create_vector(self.liste_conll[i], i, 3, taille_emb, dic_emb)
			datas.append(vector)
			golds.append(gold)

		self.X = datas
		self.Y = golds

		#création d'un dico avec key=classe_gold, value=liste des vecteurs associés
		gold2vec = defaultdict(list)
		#print(gold2vec)
		for i in range(len(datas)):
			gold2vec[golds[i]].append(datas[i])
		self.gold2vec = gold2vec
		#print([(elt,len(gold2vec[elt])) for elt in gold2vec]) #juste un test 


	def create_seeds(self, nb_clusters): #a prederminer pour chq vb
		seeds = []

		for gold in self.gold2vec:

			ex = np.array([])

			print("gold2vec avant", self.gold2vec[gold])

			for i in range(len(self.taille_graine)):

				r = random.randint(0,len(self.gold2vec[gold]))
				obj = self.gold2vec[gold][r]
				print("r", r)
				#j= np.where(np.isin(self.gold2vec[gold], r))
				#print(j)
				print(np.array_equal(r, self.gold2vec[gold][r]))
				np.delete(self.gold2vec[gold], r)
				input()
				#self.gold2vec[gold].remove(r)
				print("gold2vec apres", self.gold2vec)
				np.append(ex, r, axis=0)
				print("ex apres", ex)
				

				#on supprime du dataset l'ex tiré au sort
				id_to_del = self.datas.index(r)
				del self.X[id_to_del]
				del self.Y[id_to_del]
				input()

				#faire la moyenne des exemples
				#moy = np.mean(ex)
				#seeds.append(moy)
			


class K_Means:
    def __init__(self, k=4, tol=0.001, max_iter=300, seeds=None):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        self.seeds = seeds

    def fit(self, data):

        self.centroids = {}

        if self.seeds==None: 
            np.random.shuffle(data)
            print(data)
            #selectionne les centroids de départ (les deux premiers points de la liste de donnees melangee)
            for i in range (self.k):
                print(i)
                self.centroids[i] = data[i] 
        else: 
            self.centroids[i] = seeds[i]

        print("centroids: ", self.centroids)

        


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
                #print(original_centroid)
                current_centroid = self.centroids[c]
                #print(current_centroid)
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



#######################################################MAIN



#clf=K_Means()
#clf.fit(X)


d = Desamb(args.vb_choisi, args.taille_graine)
d.createX_Y()


# non supervisé
#d.kmeans.fit(d.X)


# supervisé
d.create_seeds(4)
d.kmeans = Kmeans(seeds=d.seeds) #on réinstancie le kmeans avec nos seeds
d.kmeans.fit(d.X)


# plot centroids mais foireux 
centroids = np.array([item for item in d.kmeans.centroids.values()])
print(centroids)
plt.scatter(centroids[:,0], centroids[:,1], marker="x", color='r')
colors = 10*["g","r","c","b","k"]


for centroid in d.kmeans.centroids:
    for centroid in d.kmeans.centroids:
        plt.scatter(d.kmeans.centroids[centroid][0], d.kmeans.centroids[centroid][1],
                    marker="o", color="k", s=150, linewidths=5)

for classification in d.kmeans.classifications:
    color = colors[classification]
    for featureset in d.kmeans.classifications[classification]:
        plt.scatter(featureset[0], featureset[1], marker="x", color=color, s=150, linewidths=5)
        
plt.show()