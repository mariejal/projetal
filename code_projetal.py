import os
from collections import defaultdict
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("verb_chosen", help = "verbe à clusteriser", default = None)
args = parser.args()

class Desamb:

	def __init__(self, verb_chosen):

		self.verb_chosen = verb_chosen
		self.


