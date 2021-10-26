import math
import sys
import numpy
import random
import time
import matplotlib.pyplot as plt
import cProfile
from pstats import Stats, SortKey

matrix = []
matrix_cst = []
upper_bound = 5000
max_points = 6


class VRPGenome:

    __slots__ = ['nb_pts', 'predecessors', 'genome', 'fitness']

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
            while (j < self.nb_pts) and (j-i+1 <= max_points) and (constraint <= upper_bound):
                if (i == j):
                    constraint = matrix_cst[0][self.genome[j-1]] + matrix_cst[self.genome[j-1]][0]
                    cost = matrix[0][self.genome[j-1]] + matrix[self.genome[j-1]][0]
                else:
                    cost += - matrix[self.genome[j-2]][0] + matrix[self.genome[j-2]][self.genome[j-1]] + matrix[self.genome[j-1]][0]
                    constraint += - matrix_cst[self.genome[j-2]][0] + matrix_cst[self.genome[j-2]][self.genome[j-1]] + matrix_cst[self.genome[j-1]][0]

                if (j-i+1 <= max_points) and (constraint <= upper_bound):
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
            tournees.insert(0, tour)
            j = i
        return tournees

class GeneticAlgorithmVRP:
    """Cet algo permet de trouver une solution au problème de VRP (Vehicle
    Routing Problem) qui consiste à trouver les meilleures tournées de
    véhicule afin de servir un certain nombre de clients. On doit minimiser
    la distance totale des tournées, et respecter les contraintes (distance,
    nombre de points maximums par tournées, etc...).
    Cet algorithme est grandement inspiré de :
    https://doi.org/10.1016/S0305-0548(03)00158-8"""

    def __init__(self) -> None:
        self.start = time.time()
        self.population: list[VRPGenome] = []
        self.productive_iter = 0
        self.unprod_iter = 0
        self.mutation_rate = 0.05
        self.pop_size = 30
        self.max_prod_iter = 2000
        self.max_unprod_iter = 500
        self.spacement = .5
        self.replaced = 8
        self.replacement = 3
        self.genome_size = len(matrix) - 1
        self.stats = [0]*8
        self.nb_cross = 0

        # Solution
        self.tempsTotal = 0
        self.distanceTotale = 0
        self.tournees = []
        self.nonServis = []

    def init(self) -> bool:
        """Prétraitement et initilisation.
        On regarde si tous les points peuvent être servis, si ce n'est pas le cas
        à cause du maxDistance alors on les met dans nonServis et on modifie les 
        temps et distances pour les mettre à la même valeur que s'ils étaient 
        confondus avec le dépôt. A la fin on les enlevera de la solution."""

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

    def generateRandomPop(self) -> None:
        """Crée une population initiale aléatoire. On essaie d'utiliser des heuristiques
        pour avoir des solutions correctes dès le début."""

        clarke_wright = self.localSearch(self.clarkAndWright())
        print("| Fitness CW :", clarke_wright.fitness)
        self.population.append(clarke_wright)

        # Il faudrait rajouter 2 autres heuristiques pour avoir une bonne population de départ.

        g = self.localSearch(self.greedy())
        self.population.append(g)
        print("| Fitness Greedy :", g.fitness)

        self.population.append(self.localSearch(VRPGenome()))
        self.population.append(self.localSearch(VRPGenome()))

        while (len(self.population) < self.pop_size):
            indiv = VRPGenome()
            if (self.spaced(self.population, indiv)):
                self.population.append(indiv)

        self.population.sort(key=lambda x: x.fitness)

    def findRoute(self, tournees, i: int) -> int:
        """Utile pour Clarke & Wright, permet de retrouver la tournée à laquelle appartient i."""

        route = -1
        for tour in range(len(tournees)):
            if i in tournees[tour]:
                route = tour
        return route

    def clarkAndWright(self) -> VRPGenome:
        """Heuristique de Clarke & Wright permettant d'avoir une bonne population
        de départ. Elle se base sur l'idée du "saving" : savings contient un
        tableau de trip: un point A, un point B et une valeur : le "saving".
        Cette valeur représente la différente entre (A, dépôt)->(dépôt, B) et (A,B).
        Plus cette valeur est grande plus il sera intéressant de passer par A puis
        directement par B plutôt que de repasser par le dépôt entre deux.
        On essaie donc itérativement, en partant de tournées ne contenant qu'un
        point, de leur ajouter les chemins de plus grand saving."""

        tournees = [] # list of dict
        savings = []

        for i in range(len(matrix) -1):
            for j in range(i+1, len(matrix)):
                if i != 0 and j != 0:
                    savings.append((i,j,
                        matrix[i][0] +
                        matrix[0][j] -
                        matrix[i][j]))
            tournees.append([i+1])

        savings.sort(key=lambda x:x[2], reverse=True)

        while (len(savings) > 0):
            routeI = self.findRoute(tournees, savings[0][0])
            routeJ = self.findRoute(tournees, savings[0][1])
            if (routeI != routeJ and (len(tournees[routeI]) + len(tournees[routeJ]) + 1 <= max_points)):
                tournees[routeI].extend(tournees[routeJ])
                tournees.pop(routeJ)
                #savings = [(i,j,k) for (i,j,k) in savings if (i != savings[0][0]) and (j != savings[0][1])]
            if len(savings) > 0:
                savings.pop(0)
        
        res = [k for x in tournees for k in x]
        return VRPGenome(res)

    def clarkAndWrightBis(self) -> VRPGenome:
        """Heuristique de Clarke & Wright avec modif"""

        tournees = [] # list of dict
        savings = []

        for i in range(len(matrix) -1):
            for j in range(i+1, len(matrix)):
                if i != 0 and j != 0:
                    savings.append((i,j,
                        matrix[i][0] +
                        matrix[0][j] -
                        matrix[i][j]))
            tournees.append({
                'distance': matrix[0][i+1] + matrix[i+1][0], 
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
                savings = [(i,j,k) for (i,j,k) in savings if (i != savings[0][0]) and (j != savings[0][1])]
            if len(savings) > 0:
                savings.pop(0)
        
        res = [k for x in tournees for k in x['points']]
        return VRPGenome(res)

    def greedy(self) -> VRPGenome:
        """Algorithme glouton qui retourne un chromosome. Récursivement on choisit un 
        point, puis le plus proche de celui-ci etc"""
        gene: list[int] = []
        mini = sys.maxsize
        argmin = -1
        for i in range(len(matrix)):
            if i != 0:
                if matrix[0][i] < mini:
                    mini = matrix[0][i]
                    argmin = i
        gene.append(argmin)
    
        for j in range(len(matrix) -2):
            mini = sys.maxsize
            for i in range(len(matrix)):
                if (i != 0) and (i != j) and (i not in gene):
                    if (matrix[j][i] < mini):
                        mini = matrix[j][i]
                        argmin = i
            gene.append(argmin)
        return VRPGenome(gene)

    def getRandElements(self, genomes: list[VRPGenome], n: int) -> list[VRPGenome]:
        """Permet d'extraire n éléments aléatoires d'une liste."""
        rand_genome = [x for x in range(len(genomes))]
        random.shuffle(rand_genome)
        return [genomes[k] for k in rand_genome[:n]]

    def crossoverVrpTwoCuts(self, parents: list[VRPGenome]) -> list[VRPGenome]:
        """Meme principe que pour crossoverOneCut() mais avec deux points : i et j
        C1[i ... j] = P1[i ... j] puis complété par les valeurs de P2 circulairement à partir de j+1"""
        self.nb_cross += 1
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

    def twoOpt(self, individu: VRPGenome) -> VRPGenome:
        """Méthode de recherche locale. Etant donné un chemin entre tous les points, le but de
        cet algo est de retourner la meilleure permutation possible entre 2 arêtes. Pour ça 
        on prend une partie du chemin et on la "retourne" (c'est équivalent)."""
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

    def localSearch(self, individu: VRPGenome) -> VRPGenome:
        """Défini la mutation en 9 étapes simples : soient x et y les successeurs de u et v dans leur
        tournées respectives ([[...], [..., u, x, ...], ... [..., v, y, ...]]). Ces points peuvent
        appartenir à la même tournée ou à des différentes et un point peut être le dépôt."""

        best_fit = individu.fitness
        best_gen = individu.genome[:]
        for u in range(len(individu.genome) -2):
            x = u+1
            for v in range(u+1, len(individu.genome) -1):
                y = v+1
                while True:
                    # M1 : If u is a client node, remove u then insert it after v,
                    gen = best_gen[:]
                    a = gen.pop(u)
                    gen.insert(v+1, a)
                    current_fit = VRPGenome(gen).fitness
                    if best_fit > current_fit:
                        best_fit = current_fit
                        best_gen = gen[:]
                        self.stats[0] += 1
                        #print("M1 :", VRPGenome(best_gen).get_path(), best_fit)
                        continue

                    # M2 : If u and x are clients, remove them then insert (u;x) after v
                    gen = best_gen[:]
                    b = gen.pop(x)
                    a = gen.pop(u)
                    gen.insert(v+1, a)
                    gen.insert(v+2, b)
                    current_fit = VRPGenome(gen).fitness
                    if best_fit > current_fit:
                        best_fit = current_fit
                        best_gen = gen[:]
                        self.stats[1] += 1
                        #print("M2 :", VRPGenome(best_gen).get_path(), best_fit)

                    # M3 : If u and x are clients, remove them then insert (x;u) after v
                    gen = best_gen[:]
                    b = gen.pop(x)
                    a = gen.pop(u)
                    gen.insert(v+1, b)
                    gen.insert(v+2, a)
                    current_fit = VRPGenome(gen).fitness
                    if best_fit > current_fit:
                        best_fit = current_fit
                        best_gen = gen[:]
                        self.stats[2] += 1
                        #print("M3 :", VRPGenome(best_gen).get_path(), best_fit)
                        continue

                    # M4 : If u and v are clients, swap u and v
                    gen = best_gen[:]
                    gen[u], gen[v] = gen[v], gen[u]
                    current_fit = VRPGenome(gen).fitness
                    if best_fit > current_fit:
                        best_fit = current_fit
                        best_gen = gen[:]
                        self.stats[3] += 1
                        #print("M4 :", VRPGenome(best_gen).get_path(), best_fit)
                        continue

                    # M5 : If u, x and v are clients, swap (u,x) and v
                    gen = best_gen[:]
                    gen[u], gen[v] = gen[v], gen[u]
                    a = gen.pop(x)
                    gen.insert(v+1, a)
                    current_fit = VRPGenome(gen).fitness
                    if best_fit > current_fit:
                        best_fit = current_fit
                        best_gen = gen[:]
                        self.stats[4] += 1
                        # print("M5 :", VRPGenome(best_gen).get_path(), best_fit)
                        continue

                    # M6 : If (u;x) and (v;y) are clients, swap (u;x) and (v;y)
                    if x != v and u != y:
                        gen = best_gen[:]
                        gen[u], gen[x], gen[v], gen[y] = gen[v], gen[y], gen[u], gen[x]
                        current_fit = VRPGenome(gen).fitness
                        if best_fit > current_fit:
                            best_fit = current_fit
                            best_gen = gen[:]
                            self.stats[5] += 1
                            #print("M6 :", VRPGenome(best_gen).get_path(), best_fit)
                            continue
                    
                    # M7 : If T(u) == T(v), replace (u;x) and (v;y) by (u;v) and (x;y)
                    # M8 : If T(u) != T(v), replace (u;x) and (v;y) by (u;v) and (x;y)
                    gen = best_gen[:]
                    gen[x], gen[v] = gen[v], gen[x]
                    current_fit = VRPGenome(gen).fitness
                    if best_fit > current_fit:
                        best_fit = current_fit
                        best_gen = gen[:]
                        self.stats[6] += 1
                        #print("M7 :", VRPGenome(best_gen).get_path(), best_fit)
                        continue
                        
                    # M9 : If T(u) != T(v), replace (u;x) and (v;y) by (u;y) and (x;v)
                    if x != v and u != y:
                        path = [x[:] for x in VRPGenome(best_gen).get_path()]
                        Tu = -1
                        Tv = -1
                        for a in range(len(path)):
                            if u in path[a]:
                                Tu = a
                            if v in path[a]:
                                Tv = a
                        if Tu != Tv:
                            gen[x], gen[v], gen[y] = gen[y], gen[x], gen[v]
                            current_fit = VRPGenome(gen).fitness
                            if best_fit > current_fit:
                                best_fit = current_fit
                                best_gen = gen[:]
                                self.stats[7] += 1
                                #print("M9 :", VRPGenome(best_gen).get_path(), best_fit)
                                continue
                    break
        return VRPGenome(best_gen)

    def selectParentsByTournament(self) -> list[VRPGenome]:
        """Cette méthode renvoit 2 individus qui vont servir pour le crossover. 
        Chaque individu est choisi par la méthode dite "du tournoi" : on 
        choisit alétoirement 2 individus parmi toute la population et on 
        retourne le meilleur. Idem pour le deuxième."""
        first = self.getRandElements(self.population, 2)
        second = self.getRandElements(self.population, 2)

        return [first[max(enumerate(first), key=lambda x: x[1].fitness)[0]], second[max(enumerate(second), key=lambda x: x[1].fitness)[0]]]

    def getRandomChild(self, parents: list[VRPGenome]) -> VRPGenome:
        """Retourne un enfant aléatoire provenant du croisement des parents"""
        children = self.crossoverVrpTwoCuts(parents)
        return children[random.randint(0,1)]

    def getBestChild(self, parents: list[VRPGenome]) -> VRPGenome:
        """Retourne le meilleur enfant provenant du croisement des parents"""
        children = self.crossoverVrpTwoCuts(parents)
        if children[0].fitness < children[1].fitness:
            return children[0]
        return children[1]

    def tradingChildWithBadElement(self, child: VRPGenome) -> None:
        """On tente d'échanger l'enfant avec un mauvais individu (i.e. qui était 
        dans la deuxième moitié de la population). Si l'enfant est meilleur on
        le remplace et on a réussi une itération. On en profite pour faire la
        mutation ici."""
        rank = random.randint(len(self.population)/2, len(self.population) - 1)
        mutant = child
    
        if (random.random() < self.mutation_rate):
            m = self.localSearch(child)
            if (self.spaced(self.population, m, rank)):
                mutant = m
            
        if (self.spaced(self.population, mutant, rank)):
            self.productive_iter += 1
            if (self.population[0].fitness > mutant.fitness):
                self.unprod_iter = 0
            else:
                self.unprod_iter += 1
            
            self.replaceAndSort(rank, mutant)

    def spaced(self, population: list[VRPGenome], individu: VRPGenome, skipped = None) -> bool:
        """Cette méthode renvoie vrai si element qu'on cherche à faire rentrer
        dans la population est assez espacé des autres solutions. Cela permet 
        une hétérogénéité de la population pour éviter de converger trop rapidement
        dans un minimum local."""
        for pop in range(len(population)):
            if (pop != skipped) and (math.floor(population[pop].fitness/self.spacement) == math.floor(individu.fitness/self.spacement)):
                return False
        return True

    def partialReplacement(self) -> None:
        """Pour aider l'algo on remplace quelques mauvais individus par de nouveaux"""
        newPop: list[VRPGenome] = []
        # On genere une population bien espacée
        while (len(newPop) < self.replaced):
            indiv = VRPGenome()
            if (self.spaced(self.population, indiv)):
                newPop.append(indiv)
  
        for new_indiv in newPop:
        # si l'individu est meilleur que le pire de la population on le remplace
        # sinon on le croise avec tous les individus et on garde le meilleur enfant
            if (new_indiv.fitness < self.population[-1].fitness):
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

    def replaceAndSort(self, rank: int, val: VRPGenome) -> None:
        """Permet de retirer un élément de la liste et de le remplacer
        par un autre mais au bon endroit. Petite optimisation permettant
        d'avoir toujours une liste triée sans la retriée entièrement.
        """
        self.population.pop(rank)
        i = 0
        while (i < len(self.population) and self.population[i].fitness < val.fitness):
            i += 1
        self.population.insert(i, val)

    def optimizeVRP(self) -> VRPGenome:
        if not self.init():
            print('Problème init()')
            exit()
        
        if (self.genome_size - len(self.nonServis) <= 6):
            print(self.localSearch(self.clarkAndWright()).get_path())
            exit()
        
        values = []
        restarts = []
  
        self.generateRandomPop()
        print("All starting values : ", [round(x.fitness) for x in self.population])
        print("Best one : ", self.population[0].fitness)
        values.append(self.population[0].fitness)
      
        replacement_done = 0
        while (self.productive_iter < self.max_prod_iter and self.unprod_iter < self.max_unprod_iter):
            parents = self.selectParentsByTournament()
            child = self.getRandomChild(parents)
            self.tradingChildWithBadElement(child)
    
            if (replacement_done < self.replacement) and (self.unprod_iter == math.floor(self.max_unprod_iter / 4)):
                restarts.append(len(values))
                self.partialReplacement()
                replacement_done += 1

            values.append(self.population[0].fitness)
            update_progress((self.productive_iter + self.unprod_iter)/(self.max_prod_iter + self.max_unprod_iter), start_time=self.start)

        update_progress(2)
        self.population[0] = self.localSearch(self.population[0])
        values.append(self.population[0].fitness)
        print(f'Itérations :         {len(values)} ({self.productive_iter}/{self.unprod_iter})')
        print(f'Solution finale :    {round(self.population[0].fitness, 2)} mètres')
        print(f'Temps total :        {round((time.time() - self.start)*100)/100} secondes')
        print(f'Local search stats : {self.stats}')
        print(f'Nb OX :              {self.nb_cross}')
        print("------")

        plt.figure()
        plt.plot(values)
        plt.vlines(restarts, ymin=[values[x] - 25 for x in restarts], ymax=[values[x] + 25 for x in restarts], linestyles='dashed', colors='k')
        plt.ylabel('Meilleur genome')
        plt.xlabel('Iteration')
        return self.population[0]

def update_progress(progress, texte='Iterations', start_time=None):
    barLength = 20 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    elif start_time is not None:
        status = f'{round(time.time() - start_time)}s'
    block = int(round(barLength*progress))
    text = "{0} : [{1}] {2}% {3}\r".format(texte, "#"*block + "-"*(barLength-block), round(progress*100), status)
    sys.stdout.write(text)
    sys.stdout.flush()

if __name__ == '__main__':
    size = 50

    # xs = [0] + [random.randint(-250, 250) for x in range(size)]
    # ys = [0] + [random.randint(-250, 250) for x in range(size)]
    # print(xs)
    # print(ys)

    # Pour 30 :
    xs = [0, 102, 37, 25, -124, 217, 229, 155, -227, 129, -99, -62, 164, 239, 196, -167, 195, -4, 41, 99, -217, 246, 100, 96, 174, -89, -29, 136, -194, -91, -111]
    ys = [0, -144, 246, 21, -122, -22, -178, -144, 199, -66, 83, 104, 38, -35, -211, 1, 70, 123, -96, -108, 115, -179, -169, 221, -72, 66, -124, 37, 235, 169, -185]
    
    # Pour 40
    # xs = [0, -164, -74, 1, -211, -180, -102, -132, -174, -23, 161, 169, 199, 97, -93, -150, -14, 103, -63, 231, 249, -111, 83, 106, 46, -19, -241, 250, 214, -170, 147, 80, 23, 120, -189, -35, 5, 164, -162, -164, -121]
    # ys = [0, 160, -142, -86, -133, 82, 137, -191, 211, -16, -143, -18, -163, -150, 157, 49, -133, -15, -50, 31, 228, -212, -148, -72, -152, -5, 250, 161, 1, -184, 121, -41, -138, -19, 51, 143, -81, 250, 59, -157, -79]
    
    # Pour 50 
    # xs = [0, 49, -167, -144, -175, -115, -71, 109, 30, 0, -211, 39, -171, -13, -95, -85, -36, 235, -184, -210, 155, 170, -200, -229, -90, 126, -86, -36, -99, 0, 47, -105, -50, -232, 204, -100, 92, 125, -209, -56, 121, 120, -228, -76, 112, 172, 33, 175, -136, 152, 92]
    # ys = [0, 164, 99, -231, -216, 100, -65, -123, -38, -74, 199, 65, -187, 200, -153, 180, -5, 31, 217, 108, -245, 19, -241, -206, 140, -97, 212, -161, -228, 135, -211, 153, -236, 218, -36, 145, 169, -230, -46, -250, -74, -163, -162, -242, -246, 63, -163, -63, 176, -237, -22]

    matrix = [[math.sqrt((xs[x] - xs[y])**2 + (ys[x] - ys[y])**2) for x in range(len(xs))] for y in range(len(ys))]
    matrix_cst = [[0 for x in xs] for y in ys]

    vrai = GeneticAlgorithmVRP()
    opt = vrai.optimizeVRP()

    fig, ax = plt.subplots()
    ax.scatter(x=xs, y=ys)
    for i in range(len(xs)):
        ax.annotate(i, (xs[i], ys[i]))

    plt.plot(0, 0, 'ro')
    plt.ylabel('Y')
    plt.xlabel('X')

    for a in opt.get_path():
        color = (random.random(), random.random(), random.random())
        plt.plot([0, xs[a[0]]], [0, ys[a[0]]], c=color)
        for b in range(len(a) -1):
            plt.plot([xs[a[b]], xs[a[b+1]]], [ys[a[b]], ys[a[b+1]]], c=color)
        plt.plot([0, xs[a[-1]]], [0, ys[a[-1]]], c=color)

    plt.title(f'{len(opt.get_path())} (val = {round(opt.fitness, 2)})')
    plt.show()

    # with cProfile.Profile() as pr:
    #     start = time.time()
    #     vrai = GeneticAlgorithmVRP()
    #     vrai.optimizeVRP()
    #     print(f"Temps total : {round((time.time() - start)*100)/100} secondes")

    # with open('profiling_stats.txt', 'w') as stream:
    #     stats = Stats(pr, stream=stream)
    #     stats.strip_dirs()
    #     stats.sort_stats('time')
    #     stats.dump_stats('.prof_stats')
    #     stats.print_stats()
