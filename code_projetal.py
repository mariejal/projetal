import os
from collections import defaultdict
import argparse
import pprint as pp

parser = argparse.ArgumentParser()
parser.add_argument("vb_choisi", help = "verbe à clusteriser", default = None)
args = parser.parse_args()

class Desamb:

	def __init__(self, vb_choisi):

		self.vb_choisi = vb_choisi
		self.liste_fichier = [f for f in os.listdir("data/" + vb_choisi) if f[0] != "."] #verifier que la liste est ds le bon ordre
		
		self.conll_stream = open("data/" + vb_choisi + "/" + self.liste_fichier[0])
		self.gold_stream = open("data/" + vb_choisi + "/" + self.liste_fichier[1])
		self.tok_ids_stream = open("data/" + vb_choisi + "/" + self.liste_fichier[2])

		self.golds = self.gold_stream.readlines() #liste
		self.tok_ids = self.tok_ids_stream.readlines() #liste

		#print(self.conll_stream)

		liste_conll = []
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

			b = False
			mot = phr[i].split("\t")
			print(mot)

			#nbr d'arguments + 
			#premiere condition = pour les cas ou le gouverneur est qqch comme "4|3"
			#deuxieme condition = cas normaux 
			if (len(mot[6])>0 and verb_id in [int(n) for n in mot[6].split("|")]) or (len(mot[6])==1 and int(mot[6]) == verb_id):

				if mot[3] != "P" and mot[3] != "V":

					obs["nb_arg"][""] += 1

				if mot[3] == "P":

					obs["prep"][mot[2]] += 1

			#fenetre de mots
			#verbe_id-1-i est la position du mot courant par rapport au verbe
			pos_mot = verb_id-1-i # on fait -1 pcq en python on compte à partir de 0
			if pos_mot > 0 and pos_mot <= taille_fenetre and mot[3] != "P":

				obs["mots_fenetre"]["avant  " + mot[2]] += 1

			if pos_mot < 0 and pos_mot >= -taille_fenetre and mot[3] != "P":
				
				obs["mots_fenetre"]["apres  " + mot[2]] += 1
			
		pp.pprint(obs)	
		input()



d = Desamb(args.vb_choisi)
