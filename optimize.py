import numpy
import random

matrix = []
matrix_cst = []
upper_bound = 3000
max_points = 200

class VRPGenome:
	def __init__(self, genome: list[int] = None) -> None:
		self.nb_pts = len(matrix)
		self.predecessors = [-1]*len(matrix)
		self.genome = genome if genome is not None else self.generate_random()
		self.fitness = self.split()

	def generate_random(self) -> list[int]:
		rand_genome = [x for x in range(1, self.nb_pts)]
		random.shuffle(rand_genome)
		return rand_genome

	def split(self) -> int:
		values = [99999999999999999]*self.nb_pts
		values[0] = 0
		for i in range(1, self.nb_pts):
			constraint = 0
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
		self.population = []
		self.prodIter = 0
		self.unprodIter = 0
		self.mutationRate = 0.1
		self.generationSize = 30
		self.maxProductiveIter = 30000
		self.maxNoProductiveIter = 10000
		self.spacement = 5
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
		for i in range(1, len(matrix_cst.length)):
			if (matrix_cst[0][i] + matrix_cst[i][0] > upper_bound
					or self.data.points[i].close < self.data.costMatrixTime[0][i] + self.data.points[i].treatment
					+ self.data.heureDepart 
					or self.data.points[i].close - self.data.points[i].open < self.data.points[i].treatment):
				self.nonServis.push(i)
		
		for non in self.nonServis:
			for idx in matrix[non]:
				matrix[non][idx] = matrix[0][idx]
				matrix[idx][non] = matrix[idx][0]
			for idx in matrix_cst[non]:
				matrix_cst[non][idx] = matrix_cst[0][idx]
				matrix_cst[idx][non] = matrix_cst[idx][0]
	
		return self.nonServis.length != self.genomeSize
  
	'''
	 * Crée une population initiale aléatoire. On essaie d'utiliser des heuristiques
	 * pour avoir des solutions correctes dès le début.
	'''
	def generateRandomPop(self) -> None:
		CW = self.clarkAndWright()
		print("| Fitness CW :", CW.getFitness())
		self.population.push(CW)

		'''Il faudrait rajouter 2 autres heuristiques pour avoir une
		bonne population de départ.'''

		g = self.greedy()
		self.population.push(g)
		print("| Fitness Greedy :", g.getFitness())

		while (len(self.population) < self.generationSize):
			indiv = VRPGenome(self.data)
			if (self.spaced(self.population, indiv)):
				self.population.push(indiv) 

		self.population.sort()
  
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
	# def greedy(): VRPGenome:
	#   gene: number[] = []
	#   min = Number.POSITIVE_INFINITY
	#   argmin = -1
	#   for (i = 0  i < matrix.length  i++):
	#     if (i !== 0):
	#       if (matrix[0][i] < min):
	#         min = matrix[0][i]
	#         argmin = i
	#       }
	#     }
	#   }
	#   gene.push(argmin)
  
	#   for (j = 0  j < matrix.length -2 j++):
	#     min = Number.POSITIVE_INFINITY
	#     for (i = 0  i < matrix.length  i++):
	#       if (i !== 0 and i !== j and gene.indexOf(i) == -1):
	#         if (matrix[j][i] < min):
	#           min = matrix[j][i]
	#           argmin = i
	#         }
	#       }
	#     }
	#     gene.push(argmin)
	#   }
	#   return VRPGenome(self.data, gene)
	# }
  
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
	# def getRandomChild(parents: VRPGenome[]): VRPGenome:
	#   children = self.crossoverVrpTwoCuts(parents)
	#   rand =  Math.floor(Math.random() + 0.5) // rand = 0 ou 1 ici
	#   return children[rand]
	# }
  
	# /**
	#  * Retourne le meilleur enfant provenant du croisement des parents
	#  */
	#  def getBestChild(parents: VRPGenome[]): VRPGenome:
	#   children = self.crossoverVrpTwoCuts(parents)
	#   return children[0].getFitness() < children[1].getFitness() ? children[0] : children[1]
	# }
  
	# /**
	#  * On tente d'échanger l'enfant avec un mauvais individu (i.e. qui était 
	#  * dans la deuxième moitié de la population). Si l'enfant est meilleur on
	#  * le remplace et on a réussi une itération. On en profite pour faire la
	#  * mutation ici.
	#  */
	# def tradingChildWithBadElement(child: VRPGenome): void:
	#   rank = Math.floor(self.genomeSize - (Math.random() * self.genomeSize / 2)) - 1
	#   mutant = child
  
	#   if (Math.random() < self.mutationRate):
	#     m = self.twoOpt(child)
	#     if (self.spaced(self.population, m, rank)):
	#       mutant = m
	#     }
	#   }
	#   if (self.spaced(self.population, mutant, rank)):
	#     self.prodIter++
	#     if (self.population[0].getFitness() > mutant.getFitness()):
	#       self.unprodIter = 0
	#     } else:
	#       self.unprodIter++
	#     }
	#     self.replaceAndSort(rank, mutant)
	#   }
	# }
  
	# /**
	#  * Cette méthode renvoie vrai si element qu'on cherche à faire rentrer
	#  * dans la population est assez espacé des autres solutions. Cela permet 
	#  * une hétérogénéité de la population pour éviter de converger trop rapidement
	#  * dans un minimum local.
	#  */
	# def spaced(pop: VRPGenome[], individu: VRPGenome, skipped?: number): boolean:
	#   res = true
	#   pop.forEach((elt, idx) =>:
	#     if (idx !== skipped and 
	#         Math.floor(elt.getFitness()/self.spacement) == Math.floor(individu.getFitness()/self.spacement))
	#       res = false
	#   })
	#   return res
	# }
  
	# /**
	#  * Pour aider l'algo on remplace quelques mauvais individus par de nouveaux
	#  */
	# def partialReplacement(): void:
	#   newPop: VRPGenome[] = []
	#   // On genere une population bien espacée
	#   while (newPop.length < self.replacement):
	#     indiv = VRPGenome(self.data)
	#     if (self.spaced(newPop, indiv)):
	#       newPop.push(indiv)
	#     }
	#   }
  
	#   newPop.forEach(element =>:
	#     // si l'individu est meilleur que le pire de la population on le remplace
	#     // sinon on le croise avec tous les individus et on garde le meilleur enfant
	#     rank = self.generationSize-1
	#     if (element.getFitness() < self.population[rank].getFitness()):
	#       self.replaceAndSort(rank, element)
	#       self.unprodIter = 0
	#     } else:
	#       child: VRPGenome = VRPGenome(self.data)
  
	#       self.population.forEach(popIndiv =>:
	#         newChild = self.getBestChild([element, popIndiv])
	#         if (newChild.getFitness() < child.getFitness()):
	#           child = newChild
	#         }
	#       })
	#       newPop.forEach(popIndiv =>:
	#         newChild = self.getBestChild([element, popIndiv])
	#         if (newChild.getFitness() < child.getFitness()):
	#           child = newChild
	#         }
	#       })
	#       if (child.getFitness() < self.population[rank].getFitness()):
	#         self.replaceAndSort(rank, child)
	#         self.unprodIter = 0
	#       }
	#     }
	#   })
	# }
  
	# /**
	#  * Permet de retirer un élément de la liste et de le remplacer
	#  * par un autre mais au bon endroit. Petite optimisation permettant
	#  * d'avoir toujours une liste triée sans la retriée entièrement.
	#  */
	# def replaceAndSort(rank: number, val: VRPGenome): void:
	#   self.population.splice(rank, 1)
	#   i = 0
	#   while (i < self.population.length and self.population[i].getFitness() < val.getFitness()):
	#     i++
	#   }
	#   self.population.splice(i, 0, val)
	# }
  
	# def optimizeVRP(): ISolutionVrp:
	#   if (!self.init()):
	#     return self.sol
	#   }
	  
	#   if (self.genomeSize - self.sol.nonServis.length <= 6):
	#     return self.twoOpt(self.clarkAndWright()).recoverPath(self.sol)
	#   }
  
	#   self.generateRandomPop()
	#   print("Start with : ", self.population[0].getFitness().toFixed(2))
	  
	#   replacementDone = 0
	#   while (self.prodIter < self.maxProductiveIter and self.unprodIter < self.maxNoProductiveIter):
	#     parents = self.selectParentsByTournament()
	#     child = self.getRandomChild(parents)
	#     self.tradingChildWithBadElement(child)
  
	#     if (replacementDone < 5 and self.unprodIter == Math.floor(self.maxNoProductiveIter / 4)):
	#       self.partialReplacement()
	#       replacementDone++
	#     }
	#   }
	#   print("Prod iter : ", self.prodIter, " / Unprod iter : ", self.unprodIter)
	#   print("Best : ", self.population[0].getFitness().toFixed(2))
	#   print("------")
	#   sol = self.population[0].recoverPath(self.sol)
	#   sol.data = self.data
	#   // Au début de l'algo on a échangé le 0 avec le dépot, on le remet ici
	#   if (sol.nonServis.indexOf(sol.data.depot) !== -1):
	#     sol.nonServis.splice(sol.nonServis.indexOf(sol.data.depot))
	#     sol.nonServis.push(0)
	#   }
	#   return sol
	# }

if __name__ == '__main__':
	matrix = numpy.random.randint(0, 50, (10,10))
	matrix_cst = numpy.random.randint(0, 50, (10,10))
	matrix = matrix + matrix.T
	matrix_cst = matrix_cst + matrix_cst.T
	for i in range(len(matrix)):
		matrix[i][i] = 0
		matrix_cst[i][i] = 0

	
	v = GeneticAlgorithmVRP()
	vrp = VRPGenome()
	vrp2 = VRPGenome()
	print(vrp.fitness, vrp2.fitness)
	print([v.twoOpt(x).fitness for x in [vrp, vrp2]])

	v.population.extend([VRPGenome(), VRPGenome(), VRPGenome(), VRPGenome()])
	print([c.fitness for c in v.population])
	print([c.fitness for c in (v.selectParentsByTournament())])