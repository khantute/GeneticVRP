import math
import sys
import numpy
import random
import time
import matplotlib.pyplot as plt

matrix = [[0, 32, 44, 46, 33, 25, 59, 50, 71, 35, 58, 57, 30, 58, 64, 26, 46, 95, 31, 14], [32, 0, 34, 53, 29, 58, 88, 9, 32, 60, 46, 70, 66, 11, 42, 14, 19, 63, 44, 82], [44, 34, 0, 45, 33, 64, 50, 29, 38, 37, 66, 53, 58, 33, 53, 20, 37, 31, 33, 46], [46, 53, 45, 0, 48, 44, 34, 38, 53, 44, 44, 63, 83, 19, 50, 25, 74, 75, 68, 54], [33, 29, 33, 48, 0, 71, 67, 17, 50, 83, 31, 64, 37, 18, 94, 49, 85, 22, 7, 79], [25, 58, 64, 44, 71, 0, 73, 36, 56, 53, 45, 79, 55, 78, 63, 17, 69, 68, 25, 68], [59, 88, 50, 34, 67, 73, 0, 29, 14, 12, 48, 49, 43, 39, 71, 75, 82, 50, 59, 63], [50, 9, 29, 38, 17, 36, 29, 0, 59, 34, 34, 59, 43, 34, 70, 28, 50, 25, 30, 38], [71, 32, 38, 53, 50, 56, 14, 59, 0, 73, 43, 54, 41, 33, 57, 73, 30, 39, 47, 87], [35, 60, 37, 44, 83, 53, 12, 34, 73, 0, 74, 38, 53, 41, 20, 26, 17, 57, 13, 36], [58, 46, 66, 44, 31, 45, 48, 34, 43, 74, 0, 33, 72, 92, 72, 69, 66, 31, 45, 45], [57, 70, 53, 63, 64, 79, 49, 59, 54, 38, 33, 0, 72, 49, 30, 62, 88, 74, 28, 27], [30, 66, 58, 83, 37, 55, 43, 43, 41, 53, 72, 72, 0, 47, 83, 51, 32, 57, 36, 44], [58, 11, 33, 19, 18, 78, 39, 34, 33, 41, 92, 49, 47, 0, 91, 52, 69, 35, 80, 53], [64, 42, 53, 50, 94, 63, 71, 70, 57, 20, 72, 30, 83, 91, 0, 59, 15, 20, 47, 20], [26, 14, 20, 25, 49, 17, 75, 28, 73, 26, 69, 62, 51, 52, 59, 0, 72, 74, 28, 38], [46, 19, 37, 74, 85, 69, 82, 50, 30, 17, 66, 88, 32, 69, 15, 72, 0, 29, 83, 47], [95, 63, 31, 75, 22, 68, 50, 25, 39, 57, 31, 74, 57, 35, 20, 74, 29, 0, 52, 56], [31, 44, 33, 68, 7, 25, 59, 30, 47, 13, 45, 28, 36, 80, 47, 28, 83, 52, 0, 61], [14, 82, 46, 54, 79, 68, 63, 38, 87, 36, 45, 27, 44, 53, 20, 38, 47, 56, 61, 0]]
matrix_cst = [[0, 65, 31, 14, 21, 42, 57, 39, 60, 36, 61, 50, 69, 17, 63, 38, 62, 40, 57, 9], [65, 0, 63, 44, 30, 48, 40, 50, 41, 33, 48, 60, 44, 59, 58, 31, 55, 52, 11, 58], [31, 63, 0, 19, 66, 40, 62, 58, 52, 81, 45, 33, 52, 52, 27, 70, 59, 63, 47, 58], [14, 44, 19, 0, 68, 67, 57, 42, 76, 73, 80, 37, 20, 47, 11, 64, 59, 55, 64, 17], [21, 30, 66, 68, 0, 44, 60, 27, 52, 57, 45, 56, 71, 18, 80, 54, 17, 70, 25, 58], [42, 48, 40, 67, 44, 0, 56, 32, 88, 27, 13, 60, 41, 11, 46, 36, 50, 47, 64, 82], [57, 40, 62, 57, 60, 56, 0, 34, 59, 81, 29, 38, 55, 42, 91, 82, 62, 59, 34, 63], [39, 50, 58, 42, 27, 32, 34, 0, 45, 22, 38, 35, 43, 17, 13, 10, 23, 51, 55, 74], [60, 41, 52, 76, 52, 88, 59, 45, 0, 38, 49, 61, 55, 44, 66, 64, 56, 44, 65, 93], [36, 33, 81, 73, 57, 27, 81, 22, 38, 0, 49, 38, 31, 22, 33, 85, 84, 43, 90, 19], [61, 48, 45, 80, 45, 13, 29, 38, 49, 49, 0, 50, 54, 47, 22, 70, 66, 66, 41, 62], [50, 60, 33, 37, 56, 60, 38, 35, 61, 38, 50, 0, 84, 40, 63, 69, 83, 41, 74, 39], [69, 44, 52, 20, 71, 41, 55, 43, 55, 31, 54, 84, 0, 30, 41, 36, 37, 31, 29, 73], [17, 59, 52, 47, 18, 11, 42, 17, 44, 22, 47, 40, 30, 0, 68, 49, 93, 26, 59, 28], [63, 58, 27, 11, 80, 46, 91, 13, 66, 33, 22, 63, 41, 68, 0, 37, 23, 39, 44, 41], [38, 31, 70, 64, 54, 36, 82, 10, 64, 85, 70, 69, 36, 49, 37, 0, 40, 58, 66, 56], [62, 55, 59, 59, 17, 50, 62, 23, 56, 84, 66, 83, 37, 93, 23, 40, 0, 16, 43, 42], [40, 52, 63, 55, 70, 47, 59, 51, 44, 43, 66, 41, 31, 26, 39, 58, 16, 0, 51, 53], [57, 11, 47, 64, 25, 64, 34, 55, 65, 90, 41, 74, 29, 59, 44, 66, 43, 51, 0, 52], [9, 58, 58, 17, 58, 82, 63, 74, 93, 19, 62, 39, 73, 28, 41, 56, 42, 53, 52, 0]]
upper_bound = 7000
max_points = 200

class VRPGenome:

	# __slots__ = ['nb_pts', 'predecessors', 'genome', 'fitness']

	def __init__(self, genome: list[int] = None) -> None:
		self.nb_pts = len(matrix)
		self.predecessors = [-1]*self.nb_pts
		self.genome = genome if genome is not None else self.generate_random()
		self.fitness = self.split()

	def generate_random(self) -> list[int]:
		rand_genome = [x for x in range(1, self.nb_pts)]
		random.shuffle(rand_genome)
		return rand_genome

	def split(self) -> int:
		values = [sys.maxsize]*self.nb_pts
		values[0] = 0
		for i in range(1, self.nb_pts):
			constraint: int = 0
			cost = 0
			j = i
			while (j < self.nb_pts) and (constraint <= upper_bound) and (j-i+1 <= max_points):
				if (i == j):
					constraint = matrix_cst[0][self.genome[j-1]] + matrix_cst[self.genome[j-1]][0]
					cost = matrix[0][self.genome[j-1]] + matrix[self.genome[j-1]][0]
				else:
					cost += - matrix[self.genome[j-2]][0] + matrix[self.genome[j-2]][self.genome[j-1]] + matrix[self.genome[j-1]][0]
					constraint += - matrix_cst[self.genome[j-2]][0] + matrix_cst[self.genome[j-2]][self.genome[j-1]] + matrix_cst[self.genome[j-1]][0]

				if (constraint <= upper_bound) and (j-i+1 <= max_points):
					if (values[i-1] + cost < values[j]):
						values[j] = values[i-1] + cost
						self.predecessors[j] = i-1
					j += 1
				else:
					break
		return values[self.nb_pts-1]

	def get_path(self) -> list:
		tournees = []
		# traiter les non servis
		j = self.nb_pts-1
		i = -1
		while i != 0:
			tour = []
			i = self.predecessors[j]
			for k in range(i+1, j+1):
				tour.append(self.genome[k-1])
			tournees.append(tour)
			j = i
		return tournees

#     __slots__  = []


# definir __gt__ et eq etc pour pouvoir faire le sort sur la pop

'''
 Cet algo permet de trouver une solution au problème de VRP (Vehicle
 Routing Problem) qui consiste à trouver les meilleures tournées de 
 véhicule afin de servir un certain nombre de clients. On doit minimiser 
 la distance totale des tournées, et respecter les contraintes (distance,
 nombre de points maximums par tournées, etc...).
 Cet algorithme est grandement inspiré de : 
 https://doi.org/10.1016/S0305-0548(03)00158-8
 '''
class GeneticAlgorithmVRP:

	def __init__(self) -> None:
		self.population: list[VRPGenome] = []
		self.prodIter = 0
		self.unprodIter = 0
		self.mutationRate = 0.1
		self.generationSize = 30
		self.maxProductiveIter = 10000
		self.maxNoProductiveIter = 3000
		self.spacement = 1
		self.replacement = 8
		self.genomeSize = len(matrix) - 1

		# Solution
		self.tempsTotal = 0
		self.distanceTotale = 0
		self.tournees = []
		self.nonServis = []
  
	'''/**
	 * Prétraitement et initilisation.
	 * On regarde si tous les points peuvent être servis, si ce n'est pas le cas
	 * à cause du maxDistance alors on les met dans nonServis et on modifie les 
	 * temps et distances pour les mettre à la même valeur que s'ils étaient 
	 * confondus avec le dépôt. A la fin on les enlevera de la solution.
	 */'''
	def init(self) -> bool:
		for i in range(1, len(matrix_cst)):
			if (matrix_cst[0][i] + matrix_cst[i][0] > upper_bound):
				self.nonServis.append(i)
		
		for non in self.nonServis:
			for idx in matrix[non]:
				matrix[non][idx] = matrix[0][idx]
				matrix[idx][non] = matrix[idx][0]
			for idx in matrix_cst[non]:
				matrix_cst[non][idx] = matrix_cst[0][idx]
				matrix_cst[idx][non] = matrix_cst[idx][0]
	
		return len(self.nonServis) != self.genomeSize
  
	'''
	 * Crée une population initiale aléatoire. On essaie d'utiliser des heuristiques
	 * pour avoir des solutions correctes dès le début.
	'''
	def generateRandomPop(self) -> None:
		CW = self.clarkAndWright()
		print("| Fitness CW :", CW.fitness)
		self.population.append(CW)

		'''Il faudrait rajouter 2 autres heuristiques pour avoir une
		bonne population de départ.'''

		g = self.greedy()
		self.population.append(g)
		print("| Fitness Greedy :", g.fitness)

		while (len(self.population) < self.generationSize):
			indiv = VRPGenome()
			if (self.spaced(self.population, indiv)):
				self.population.append(indiv)

		self.population.sort(key=lambda x: x.fitness)
  
	'''    /**
	 * Utile pour Clarke & Wwright, permet de retrouver la tournée à laquelle
	 * appartient i.
	 */'''
	def findRoute(self, tournees, i: int) -> int:
		route = -1
		for tour in range(len(tournees)):
			if i in tournees[tour]['points']:
				route = tour
		return route
  
	'''/**
	 * Heuristique de Clarke & Wright permettant d'avoir une bonne population 
	 * de départ. Elle se base sur l'idée du "saving" : savings contient un 
	 * tableau de trip: un point A, un point B et une valeur : le "saving". 
	 * Cette valeur représente la différente entre (A, dépôt)->(dépôt, B) et (A,B). 
	 * Plus cette valeur est grande plus il sera intéressant de passer par A puis 
	 * directement par B plutôt que de repasser par le dépôt entre deux.
	 * On essaie donc itérativement, en partant de tournées ne contenant qu'un 
	 * point, de leur ajouter les chemins de plus grand saving.
	 */'''
	def clarkAndWright(self) -> VRPGenome:
		tournees = [] # list of dict
		savings = []

		for i in range(len(matrix_cst) -1):
			for j in range(i+1, len(matrix_cst)):
				if i != 0 and j != 0:
					savings.append((i,j,
						matrix_cst[i][0] +
						matrix_cst[0][j] -
						matrix_cst[i][j]))
			tournees.append({
				'distance': matrix_cst[0][i+1] + matrix_cst[i+1][0], 
				'points': [i+1]})

		savings.sort(key=lambda x:x[2], reverse=True)

		while (len(savings) > 0):
			routeI = self.findRoute(tournees, savings[0][0])
			routeJ = self.findRoute(tournees, savings[0][1])
			if (routeI != routeJ and 
				  (tournees[routeI]['distance'] + tournees[routeJ]['distance'] 
				  - savings[0][2] <= upper_bound) and 
				  (len(tournees[routeI]['points']) + len(tournees[routeJ]['points']) + 1 <= max_points)):
				tournees[routeI]['points'].extend(tournees[routeJ]['points'])
				tournees[routeI]['distance'] += tournees[routeJ]['distance']
				tournees[routeI]['distance'] -= savings[0][2]
				tournees.pop(routeJ)
			savings.pop(0)
		
		res = [k for x in tournees for k in x['points']]
		return VRPGenome(res)
  
	# /**
	#  * Algorithme glouton qui retourne un chromosome. Récursivement on choisit un 
	#  * point, puis le plus proche de celui-ci etc
	#  */
	def greedy(self) -> VRPGenome:
		gene: list[int] = []
		min = sys.maxsize
		argmin = -1
		for i in range(len(matrix)):
			if i != 0:
				if matrix[0][i] < min:
					min = matrix[0][i]
					argmin = i
		gene.append(argmin)
	
		for j in range(len(matrix) -2):
			min = sys.maxsize
			for i in range(len(matrix)):
				if (i != 0) and (i != j) and (i not in gene):
					if (matrix[j][i] < min):
						min = matrix[j][i]
						argmin = i
			gene.append(argmin)
		return VRPGenome(gene)
  
	'''/**
	 * Permet d'extraire n éléments aléatoires d'une liste.
	 */'''
	def getRandElements(self, genomes: list[VRPGenome], n: int) -> list[VRPGenome]:
		rand_genome = [x for x in range(len(genomes))]
		random.shuffle(rand_genome)
		return [genomes[k] for k in rand_genome[:n]]
  
	# /**
	#  * Meme principe que pour crossoverOneCut() mais avec deux points : i et j
	#  * C1[i ... j] = P1[i ... j] puis complété par les valeurs de P2 circulairement à partir de j+1
	#  */
	def crossoverVrpTwoCuts(self, parents: list[VRPGenome]) -> list[VRPGenome]:
		break1 = random.randint(0, self.genomeSize)
		break2 = break1 + random.randint(0, self.genomeSize - break1)
		
		P1 = parents[0].genome[:] # First parent
		P2 = parents[1].genome[:] # Second
		C1 = [None]*self.genomeSize # On ne gardera que le 'milieu' (entre break1 et 2)
		C2 = [None]*self.genomeSize # Idem
		C1[break1:break2] = P1[break1:break2]
		C2[break1:break2] = P2[break1:break2]

		iter1 = break2 # Premier élément nul de C1 après break2
		iter2 = break2
		for i in range(len(P1)):
			if P2[(break2 + i) % len(P1)] in C1:
				continue
			C1[iter1 % len(P1)] = P2[(break2 + i) % len(P1)]
			iter1 += 1
		for i in range(len(P1)):
			if P1[(break2 + i) % len(P1)] in C2:
				continue
			C2[iter2 % len(P1)] = P1[(break2 + i) % len(P1)]
			iter2 += 1

		return [VRPGenome(C1), VRPGenome(C2)]
	
	# /**
	#  * Méthode de recherche locale. Etant donné un chemin entre tous les points, le but de
	#  * cet algo est de retourner la meilleure permutation possible entre 2 arêtes. Pour ça 
	#  * on prend une partie du chemin et on la "retourne" (c'est équivalent).
	#  */
	def twoOpt(self, individu: VRPGenome) -> VRPGenome:
		best = individu
		for i in range(self.genomeSize - 1):
			for j in range(i+1, self.genomeSize):
				gen = individu.genome[:]
				gen[i], gen[j] = gen[j], gen[i]
				k = 1
				while k < (j-i)/2:
					gen[i+k], gen[j-k] = gen[j-k], gen[i+k]
					k += 1
				current = VRPGenome(gen)
				if (current.fitness < best.fitness):
					best = current
		return best
	
  
	# /**
	#  * Cette méthode renvoit 2 individus qui vont servir pour le crossover. 
	#  * Chaque individu est choisi par la méthode dite "du tournoi" : on 
	#  * choisit alétoirement 2 individus parmi toute la population et on 
	#  * retourne le meilleur. Idem pour le deuxième.
	#  */
	def selectParentsByTournament(self) -> list[VRPGenome]:
		first = self.getRandElements(self.population, 2)
		second = self.getRandElements(self.population, 2)

		return [first[max(enumerate(first), key=lambda x: x[1].fitness)[0]], second[max(enumerate(second), key=lambda x: x[1].fitness)[0]]]
  
	# /**
	#  * Retourne un enfant aléatoire provenant du croisement des parents
	#  */
	def getRandomChild(self, parents: list[VRPGenome]) -> VRPGenome:
		children = self.crossoverVrpTwoCuts(parents)
		return children[random.randint(0,1)]
	
  
	# /**
	#  * Retourne le meilleur enfant provenant du croisement des parents
	#  */
	def getBestChild(self, parents: list[VRPGenome]) -> VRPGenome:
		children = self.crossoverVrpTwoCuts(parents)
		if children[0].fitness < children[1].fitness:
			return children[0]
		return children[1]
	
  
	# /**
	#  * On tente d'échanger l'enfant avec un mauvais individu (i.e. qui était 
	#  * dans la deuxième moitié de la population). Si l'enfant est meilleur on
	#  * le remplace et on a réussi une itération. On en profite pour faire la
	#  * mutation ici.
	#  */
	def tradingChildWithBadElement(self, child: VRPGenome) -> None:
		rank = random.randint(len(self.population)/2, len(self.population) - 1)
		mutant = child
	
		if (random.random() < self.mutationRate):
			m = self.twoOpt(child)
			if (self.spaced(self.population, m, rank)):
				mutant = m
			
		if (self.spaced(self.population, mutant, rank)):
			self.prodIter += 1
			if (self.population[0].fitness > mutant.fitness):
				self.unprodIter = 0
			else:
				self.unprodIter += 1
			
			self.replaceAndSort(rank, mutant)

  
	# /**
	#  * Cette méthode renvoie vrai si element qu'on cherche à faire rentrer
	#  * dans la population est assez espacé des autres solutions. Cela permet 
	#  * une hétérogénéité de la population pour éviter de converger trop rapidement
	#  * dans un minimum local.
	#  */
	def spaced(self, population: list[VRPGenome], individu: VRPGenome, skipped = None) -> bool:
		for pop in range(len(population)):
			if (pop != skipped) and (math.floor(population[pop].fitness/self.spacement) == math.floor(individu.fitness/self.spacement)):
				return False
		return True
  
	# /**
	#  * Pour aider l'algo on remplace quelques mauvais individus par de nouveaux
	#  */
	def partialReplacement(self) -> None:
		newPop: list[VRPGenome] = []
		# On genere une population bien espacée
		while (len(newPop) < self.replacement):
			indiv = VRPGenome()
			if (self.spaced(self.population, indiv)):
				newPop.append(indiv)
  
		for new_indiv in newPop:
	    # si l'individu est meilleur que le pire de la population on le remplace
	    # sinon on le croise avec tous les individus et on garde le meilleur enfant
			if (new_indiv.fitness< self.population[-1].fitness):
				self.replaceAndSort(-1, new_indiv)
				self.unprodIter = 0
			else:
				child = VRPGenome()
				for pop in (self.population + newPop):
					newChild = self.getBestChild([new_indiv, pop])
					if (newChild.fitness < child.fitness):
						child = newChild
				if (child.fitness < self.population[-1].fitness):
					self.replaceAndSort(-1, child)
					self.unprodIter = 0

	# /**
	#  * Permet de retirer un élément de la liste et de le remplacer
	#  * par un autre mais au bon endroit. Petite optimisation permettant
	#  * d'avoir toujours une liste triée sans la retriée entièrement.
	#  */
	def replaceAndSort(self, rank: int, val: VRPGenome) -> None:
		self.population.pop(rank)
		i = 0
		while (i < len(self.population) and self.population[i].fitness < val.fitness):
			i += 1
		self.population.insert(i, val)
	
  
	def optimizeVRP(self) -> None:
		if not self.init():
			print('Problème init()')
			exit()
		
		if (self.genomeSize - len(self.nonServis) <= 6):
			print(self.twoOpt(self.clarkAndWright()).get_path())
			exit()
		
		values = []
  
		self.generateRandomPop()
		print("Start with : ", self.population[0].fitness)
		values.append(self.population[0].fitness)
	  
		replacementDone = 0
		while (self.prodIter < self.maxProductiveIter and self.unprodIter < self.maxNoProductiveIter):
			parents = self.selectParentsByTournament()
			child = self.getRandomChild(parents)
			self.tradingChildWithBadElement(child)
	
			if (replacementDone < 5) and (self.unprodIter == math.floor(self.maxNoProductiveIter / 4)):
				self.partialReplacement()
				replacementDone += 1

			values.append(self.population[0].fitness)

		print("Prod iter : ", self.prodIter, " / Unprod iter : ", self.unprodIter)
		print("Best : ", self.population[0].fitness)
		sol = self.population[0].get_path()
		print(len(values))
		plt.plot(values)
		plt.ylabel('Meilleur genome')
		plt.xlabel('Iteration')
		plt.show()
		print(sol)
		print("------")

if __name__ == '__main__':
	# size = 20
	# matrix = numpy.random.randint(0, 50, (size,size))
	# matrix_cst = numpy.random.randint(0, 50, (size,size))
	# matrix = matrix + matrix.T
	# matrix_cst = matrix_cst + matrix_cst.T
	# for i in range(size):
	# 	matrix[i][i] = 0
	# 	matrix_cst[i][i] = 0
	# matrix_cst = matrix_cst.tolist()
	# matrix = matrix.tolist()

	# print(matrix)
	# print(matrix_cst)

	start = time.time()
	vrai = GeneticAlgorithmVRP()
	vrai.optimizeVRP()
	print(f"Temps total : {round((time.time() - start)*100)/100} secondes")
	print()