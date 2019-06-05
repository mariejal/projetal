import os
from collections import defaultdict
import argparse
import pprint as pp
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("vb_choisi", help = "verbe à clusteriser", default = None)
args = parser.parse_args()

class Desamb:

	def __init__(self, vb_choisi):

		self.vb_choisi = vb_choisi
		self.liste_fichier = [f for f in os.listdir("data/" + vb_choisi) if f[0] != "."] #a revoir: verifier que la liste est ds le bon ordre
		
		self.conll_stream = open("data/" + vb_choisi + "/" + self.liste_fichier[0])
		self.gold_stream = open("data/" + vb_choisi + "/" + self.liste_fichier[1])
		self.tok_ids_stream = open("data/" + vb_choisi + "/" + self.liste_fichier[2])

		self.golds = self.gold_stream.readlines() #liste
		self.tok_ids = self.tok_ids_stream.readlines() #liste

		liste_conll = [] #contient toutes les phrases sous leur forme conll
		for sentence in (self.conll_stream.read()).split("\n\n"):
			liste_conll += [sentence.split("\n")]

		#bloc qui existe ici pour tester
		for i in range(len(liste_conll)): 
			self.extraction_conll(liste_conll[i], i, 3)
		

	def extraction_conll(self, phr, sentence_id, taille_fenetre):

		obs = defaultdict(lambda: defaultdict(int)) #pas sure d'utiliser des dico de dico

		gold = int(self.golds[sentence_id])
		obs["gold"][""] = gold
		verb_id = int(self.tok_ids[sentence_id])
		print("---------------------- " + str(verb_id))
		print()


		for i in range(len(phr)):

			mot = phr[i].split("\t")
			print(mot)

			#--nbr d'arguments + prep
			#premiere condition = pour les cas ou le gouverneur est qqch comme "4|3"
			#deuxieme condition = cas normaux 
			if (len(mot[6])>0 and verb_id in [int(n) for n in mot[6].split("|")]) or (len(mot[6])==1 and int(mot[6]) == verb_id):

				if mot[3] != "P" and mot[3] != "V":

					obs["nb_arg"][""] += 1

				if mot[3] == "P":

					obs["prep"][mot[2]] += 1

			#--fenetre de mots
			#verbe_id-1-i est la position du mot courant par rapport au verbe
			pos_mot = verb_id-1-i # on fait -1 pcq en python on compte à partir de 0
			if pos_mot > 0 and pos_mot <= taille_fenetre and mot[3] != "P":

				obs["mots_fenetre"]["avant  " + mot[2]] += 1

			if pos_mot < 0 and pos_mot >= -taille_fenetre and mot[3] != "P":
				
				obs["mots_fenetre"]["apres  " + mot[2]] += 1
			
		pp.pprint(obs)	
		input()


d = Desamb(args.vb_choisi)


#sum_emb/taille_embeg


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



dic_emb = emb2dic("w2v_final")[0]

taille_emb = emb2dic("w2v_final")[1]



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

def create_vector(phr,sentence_id,taille_fenetre):

	obs = d.extraction_conll(phr,sentence_id,taille_fenetre)

	print(obs)

	taille_embeg = taille_emb

	sum_emb = np.zeros(taille_embeg)

	nb_arg =obs["nb_arg"][""]

	for elt in obs["mots_fenetre"]:

		elt = elt.split(" ")

		if elt[2] in dic_emb : 

			print(dic_emb[elt[2]])

			sum_emb = sum_emb + dic_emb[elt[2]]

	

	average_emb = sum_emb/taille_embeg

	#print(average_emb) 		

	vector = np.zeros(9)

	vector[nb_arg-1]=1

	# fonctionne pas car vector[8] est un np array alors que vector[i!=8] sont des reels

	# vector[8] = average_emb #la moyenne des embedding des mots de la fenetre

	

	#reste a ajouter les prepositions

	

for i in range (len(d.liste_conll)-1):

	create_vector(d.liste_conll[i],i,3)	