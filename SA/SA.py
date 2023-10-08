import random
import math
import matplotlib.pyplot as plt

class SimAnneal():

    def __init__(self, T=20, stopping_T=1e-8, alpha=0.9998, stopping_iter=20000, interval = (-512,512), neighbour_delta=1024):
        self.T = T
        self.stopping_T = stopping_T
        self.alpha = alpha
        self.stopping_iter = stopping_iter
        self.interval = interval
        self.neighbour_delta = neighbour_delta

        self.og_T = T
        self.best_solution = None
        self.best_fitness = float("Inf")
        self.fitness_list = []
        self.iteration = 1

    def gen_solution(self, xinterval, yinterval):
        return (random.uniform(*xinterval), random.uniform(*yinterval))
    
    def gen_intervals(self, value):
        # min = value-self.neighbour_delta if value-self.neighbour_delta >= self.interval[0] else self.interval[0]
        # max = value+self.neighbour_delta if value+self.neighbour_delta <= self.interval[1] else self.interval[1]
        # return (min,max)
        return (-512,512)


    #gen a random solution with random values (x,y)
    def inital_solution(self):
        x,y = self.gen_solution(self.interval, self.interval)
        cur_fit = self.fitness(x,y)
        self.fitness_list.append(cur_fit)
        if cur_fit > self.best_fitness:
            self.best_solution = (x,y)
            self.best_fitness = cur_fit

        return (x,y),cur_fit

    #evaluate eggholder function
    def fitness(self,x,y):
        return -(y+47)*math.sin(math.sqrt(abs(x/2+y+47)))-x*math.sin(math.sqrt(abs(x-(y+47))))
    
    def p_accept(self, candidate_fitness):
        return math.exp(-abs(candidate_fitness - self.cur_fitness) / self.T)
    
    def accept(self, candidate):
        candidate_fitness = self.fitness(*candidate)
        if candidate_fitness < self.cur_fitness:
            self.cur_fitness = candidate_fitness
            self.cur_solution = candidate
            if candidate_fitness < self.best_fitness:
                self.best_fitness, self.best_solution = candidate_fitness, candidate
       
        else:
            if random.random() < self.p_accept(candidate_fitness):
                self.cur_fitness, self.cur_solution = candidate_fitness, candidate

    def anneal(self):
        self.cur_solution, self.cur_fitness = self.inital_solution()
        sol_iter = 0
        print("     ->Starting annealing.")

        while self.T >= self.stopping_T and self.iteration < self.stopping_iter:
            # candidate = self.gen_solution()

            (x,y) = self.cur_solution
            candidate = self.gen_solution(self.gen_intervals(x),self.gen_intervals(y))

            best = self.best_fitness
            self.accept(candidate)
            
            if(self.best_fitness < best):
              sol_iter = self.iteration
              
            self.T *= self.alpha
            self.iteration += 1
            self.neighbour_delta *= self.alpha
            self.fitness_list.append(self.cur_fitness)

        print("     ->Best solutions obtained: ", self.best_solution)
        print("     ->Best fitness obtained: ", self.best_fitness)
        print('     ->itered %d times' % self.iteration)

        return self.best_solution ,self.best_fitness, sol_iter

    def plot_learning(self):
        """
        Plot the fitness through iterations.
        """
        plt.plot([i for i in range(len(self.fitness_list))], self.fitness_list)
        plt.ylabel("Fitness")
        plt.xlabel("Iteration")
        plt.show()
