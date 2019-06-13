import os
from collections import defaultdict
import argparse
import pprint as pp
import numpy as np
import re

parser = argparse.ArgumentParser()
parser.add_argument("vb_choisi", help = "verbe à clusteriser", default = None)
args = parser.parse_args()

liste_prep = [] #la liste des prep qui peuvent suivre le verbe

class Desamb:

	def __init__(self, vb_choisi):

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

		"""#bloc qui existe ici pour tester
		for i in range(len(liste_conll)): 
			self.extraction_conll(liste_conll[i], i, 3)
"""

			
	def extraction_conll(self,phr,sentence_id,taille_fenetre):
		obs = defaultdict(lambda: defaultdict(int)) #pas sure d'utiliser des dico de dico

		gold = int(self.golds[sentence_id])
		obs["gold"][""] = gold
		verb_id = int(self.tok_ids[sentence_id])
		print("---------------------- " + str(verb_id))
		print()

		vb_gouv = verb_id #id du verbe gouvernant le lemme(init a lui mm)

		#on regarde les infos du verbe_id
		info_verbid = phr[verb_id-1].split("\t")

		if not ( "dm=inf" in info_verbid[5].split("|") and "obj.p" or "obj" in re.split('||:', info_verbid[7]) ):
			vb_gouv = int(info_verbid[6].split("|")[0]) #c'est mtn son gouverneur qui gouverne notre recherche d'infos
###############
		for i in range(len(phr)):

			mot = phr[i].split("\t")
			#print(mot)

			#--nbr d'arguments + prep
			#premiere condition = pour les cas ou le gouverneur est qqch comme "4|3"
			#deuxieme condition = cas normaux 
			
			gouv = [int(n) for n in mot[6].split("|")]

			if (vb_gouv in gouv):
				not_wanted = ["P","PONCT","P+D","V","C","V"]

				if (mot[3] not in not_wanted) and ("mod" not in mot[7].split("|") and obs["nb_arg"][""]<3):
					print("mot[3]:",mot[3])
					obs["nb_arg"][""]+=1
				
				if mot[3]=="P":
					obs["prep"][mot[2]]+=1
					#--m.a.j de la liste des prep suivant le verbe
					if mot[2] not in liste_prep:
						liste_prep.append(mot[2])

			#--fenetre de mots
			#verbe_id-1-i est la position du mot courant par rapport au verbe
			pos_mot = verb_id-1-i # on fait -1 pcq en python on compte à partir de 0
			if pos_mot > 0 and pos_mot <= taille_fenetre and mot[3] != "P":

				obs["mots_fenetre"]["avant  " + mot[2]] += 1

			if pos_mot < 0 and pos_mot >= -taille_fenetre and mot[3] != "P":
				
				obs["mots_fenetre"]["apres  " + mot[2]] += 1
		pp.pprint(obs)	
		input()
		return obs

d = Desamb(args.vb_choisi)

#--retourne un dictionnaire key=mot, value=embedding du mot, pour chaque mot du fichier
def emb2dic(fichier):
	dic_emb = defaultdict(np.array)
	with open(fichier,encoding="utf8") as stream :
		line = stream.readline()
		while line : 
			line = line.strip('\n').split(" ")
			if line[0] not in dic_emb and line[0].isalpha():
				word_emb = np.array([float(comp) for comp in line[1:]])
				taille_emb = len(word_emb)
				dic_emb[line[0]] = word_emb
			line = stream.readline()
	stream.close()
	return dic_emb,taille_emb

dic_emb,taille_emb = emb2dic("w2v_final")

#--renvoie un dictionnaire key=mot, value= la moyenne des embeddings des mots de la fenetre
def average_emb(dic_emb,phr,sentence_id,taille_fenetre):
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
def get_liste_prep():
	for i in range(len(d.liste_conll)-1):
		#v = create_vector(d.liste_conll[i],i,3)
		obs = d.extraction_conll(d.liste_conll[i],i,3)
		if obs["nb_arg"][""]>=4:
	return(liste_prep)
#-- crée le vecteur associé à chaque phrase
def create_vector(phr,sentence_id,taille_fenetre):
	obs = d.extraction_conll(phr,sentence_id,taille_fenetre)
	print(obs)
	taille_embeg = taille_emb
	sum_emb = np.zeros(taille_embeg)
	nb_arg =obs["nb_arg"][""]
	print("nb arg =", nb_arg)
	for elt in obs["mots_fenetre"]:
		elt = elt.split(" ")
		if elt[2] in dic_emb : 
			#print(dic_emb[elt[2]])
			sum_emb = sum_emb + dic_emb[elt[2]]
	
	average_emb = sum_emb/taille_embeg
	#print(average_emb) 		

	vector = np.zeros(3+len(liste_prep)+taille_embeg)

	#vect[pos]=1 si i+1 est le nombre d'arguments du verbe dans la phrase 
	vector[nb_arg-1]=1

	#vect[pos]= 1 si la preposition à l'indice pos dans list_prep suit le verbe dans la phr
	list_of_prep = obs["prep"].keys()
	for i in range(len(liste_prep)):
		if liste_prep[i] in list_of_prep:
			vector[3+i]=1

	#--les composantes restantes du vecteur correspondent aux composantes du word embedding		
	for i in range(3+len(liste_prep),len(vector)):
		vector[i] = average_emb[i-(3+len(liste_prep))]
	#print(liste_prep,len(liste_prep))
	#print(list_of_prep)
	#print(vector)

	return(vector,obs["gold"][""]) #utile pour créer la liste des golds dans la suite


datas = [] #la liste des vecteurs
golds = [] #la liste des golds 
### golds[i] est l'idx de la classe associée au vecteur datas[i]

#stocke les vecteurs dans datas(list), et leur classe gold associée dans golds(list)
liste_prep=get_liste_prep()
for i in range (len(d.liste_conll)-1):
	vector,gold = create_vector(d.liste_conll[i],i,3)
	datas.append(vector)
	golds.append(gold)

#création d'un dico avec key=classe_gold, value=liste des vecteurs associés
gold2vec = defaultdict(list)
for i in range(len(datas)):
	gold2vec[golds[i]].append(datas[i])

print([(elt,len(gold2vec[elt])) for elt in gold2vec]) #juste un test 
