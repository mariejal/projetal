#le dossier data doit être dans le même fichier que le code
import argparse
import os
from collections import defaultdict
import pprint as pp


parser = argparse.ArgumentParser()
parser.add_argument("verb_chosen", help = "verbe utilisé pour faire le clustering", default = None)
args = parser.parse_args()


class Example:

	def __init__(self):

		self.gold = None
		self.nb_ex = None
		self.vecteur = []

class Desamb:

	def __init__(self, verb_chosen):

		self.verb_chosen = verb_chosen
		verb_rep = os.listdir("data/" + self.verb_chosen)

		self.conll_stream = open("data/" + self.verb_chosen + "/" + verb_rep[1])
		self.gold_stream = open("data/" + self.verb_chosen + "/" + verb_rep[2])
		self.tok_ids_stream = open("data/" + self.verb_chosen + "/" + verb_rep[3])

		self.golds = self.gold_stream.readlines()
		self.tok_ids = self.tok_ids_stream.readlines()
		
		self.liste_phr = [] #liste qui contient toutes les phrases du conll
		for sentence in (self.conll_stream.read()).split("\n\n"):
			self.liste_phr += [sentence.split("\n")]

		print(self.liste_phr)
		for i in range(len(self.liste_phr)):
			print(self.liste_phr[i])
			input()
			self.extraction_conll(self.liste_phr[i], 3, i)


	def extraction_conll(self, phr, len_fenetre, sentence_id):

		"""line_conll = conll_stream.readline()
		line_gold = gold_stream.readline()
		line_tok_ids = tok_ids_stream.readline()

		corpus = []
		
		indice = 0

		while line_conll != "\n":
			
			phr = 
			print(line_conll)
			
			line_conll = line_conll.split("\t")
			
			corpus["gold"] = line_gold[0]
			
			if line_conll[2] == self.verb_chosen:
				print("trouve")

			input()

			line_conll = conll_stream.readline()

		print("exemple fini")
		indice +=1
		line_tok_ids = tok_ids_stream.readline()
		corpus += [[phr]]"""

		print(phr)
		print("sentence_id: " + str(sentence_id))
		#print("len_gold: " + str(self.gold_stream.readlines()[2]))

		print(len(self.golds))
		gold = int(self.golds[sentence_id])
		verb_id = int(self.tok_ids[sentence_id])
		print("verb_id: " + str(verb_id))

		obs = defaultdict(lambda: defaultdict(int))

		#classe gold

		obs["gold"][gold] = 1 

		#nb d'arguments + prep

		for conll_word in phr:
			conll_word = conll_word.split("\t")
			#print(len(conll_word[6]))
			# ----------------------------ATTENTION: a gerer quand il y a 4|3 comme gouverneur
			if (len(conll_word[6])== 1 and int(conll_word[6]) == verb_id) or (len(conll_word[6])>0 and verb_id in conll_word[6]):
				if conll_word[3] != 'V' and conll_word[3] != 'PONCT' and conll_word[3] != "P":
					print(conll_word)
					obs["nb_arg"][""] +=1

				if conll_word[3] == "P":
					obs["prep"][conll_word[2]] +=1

			#input()

		

		#mots de la fenetre
		for i in range(verb_id-1-(len_fenetre), len(phr)):

			#print("i: " + str(i))
			if i != verb_id-1:
				if (i<0):
					obs["mots_fenetre"]["w+" + str(i+len_fenetre+1) + " = " + phr[i].split("\t")[2]] = 1

				if (i>0):
					obs["mots_fenetre"]["w" + str(i-len_fenetre-1) + " = " + phr[i].split("\t")[2]] = 1



		pp.pprint(obs)

		input()






d = Desamb(args.verb_chosen)

#d.extraction_conll(d.conll_stream, d.tok_ids_stream, 4)