import math
import sys
import numpy
import random
import time
import matplotlib.pyplot as plt

matrix = []
matrix_cst = []
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


class GeneticAlgorithmVRP:
    """Cet algo permet de trouver une solution au problème de VRP (Vehicle
    Routing Problem) qui consiste à trouver les meilleures tournées de
    véhicule afin de servir un certain nombre de clients. On doit minimiser
    la distance totale des tournées, et respecter les contraintes (distance,
    nombre de points maximums par tournées, etc...).
    Cet algorithme est grandement inspiré de :
    https://doi.org/10.1016/S0305-0548(03)00158-8"""

    def __init__(self) -> None:
        self.population: list[VRPGenome] = []
        self.productive_iter = 0
        self.unprod_iter = 0
        self.mutation_rate = 0.1
        self.pop_size = 30
        self.max_prod_iter = 10000
        self.max_unprod_iter = 3000
        self.spacement = 1
        self.replacement = 8
        self.genome_size = len(matrix) - 1

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
    
        return len(self.nonServis) != self.genome_size
  
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

        while (len(self.population) < self.pop_size):
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
        break1 = random.randint(0, self.genome_size)
        break2 = break1 + random.randint(0, self.genome_size - break1)
        
        P1 = parents[0].genome[:] # First parent
        P2 = parents[1].genome[:] # Second
        C1 = [None]*self.genome_size # On ne gardera que le 'milieu' (entre break1 et 2)
        C2 = [None]*self.genome_size # Idem
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
        for i in range(self.genome_size - 1):
            for j in range(i+1, self.genome_size):
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
    
        if (random.random() < self.mutation_rate):
            m = self.twoOpt(child)
            if (self.spaced(self.population, m, rank)):
                mutant = m
            
        if (self.spaced(self.population, mutant, rank)):
            self.productive_iter += 1
            if (self.population[0].fitness > mutant.fitness):
                self.unprod_iter = 0
            else:
                self.unprod_iter += 1
            
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
                self.unprod_iter = 0
            else:
                child = VRPGenome()
                for pop in (self.population + newPop):
                    newChild = self.getBestChild([new_indiv, pop])
                    if (newChild.fitness < child.fitness):
                        child = newChild
                if (child.fitness < self.population[-1].fitness):
                    self.replaceAndSort(-1, child)
                    self.unprod_iter = 0

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
        
        if (self.genome_size - len(self.nonServis) <= 6):
            print(self.twoOpt(self.clarkAndWright()).get_path())
            exit()
        
        values = []
  
        self.generateRandomPop()
        print("Start with : ", self.population[0].fitness)
        values.append(self.population[0].fitness)
      
        replacementDone = 0
        while (self.productive_iter < self.max_prod_iter and self.unprod_iter < self.max_unprod_iter):
            parents = self.selectParentsByTournament()
            child = self.getRandomChild(parents)
            self.tradingChildWithBadElement(child)
    
            if (replacementDone < 5) and (self.unprod_iter == math.floor(self.max_unprod_iter / 4)):
                self.partialReplacement()
                replacementDone += 1

            values.append(self.population[0].fitness)

        print("Prod iter : ", self.productive_iter, " / Unprod iter : ", self.unprod_iter)
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
    size = 30
    matrix = numpy.random.randint(0, 50, (size,size))
    matrix_cst = numpy.random.randint(0, 50, (size,size))
    matrix = matrix + matrix.T
    matrix_cst = matrix_cst + matrix_cst.T
    for i in range(size):
    	matrix[i][i] = 0
    	matrix_cst[i][i] = 0
    matrix_cst = matrix_cst.tolist()
    matrix = matrix.tolist()

    # print(matrix)
    # print(matrix_cst)

    start = time.time()
    vrai = GeneticAlgorithmVRP()
    vrai.optimizeVRP()
    print(f"Temps total : {round((time.time() - start)*100)/100} secondes")
    print()