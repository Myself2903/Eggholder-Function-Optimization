from geneticalgorithm2 import geneticalgorithm2 as ga
from geneticalgorithm2 import Population_initializer
import numpy as np
from ipywidgets import interactive, fixed
import matplotlib.pyplot as plt
from time import time

def f(X):
    return (-(X[1] + 47.0) * np.sin(np.sqrt(np.abs(X[0]/2.0 + (X[1] + 47.0)))) - X[0] * np.sin(np.sqrt(np.abs(X[0] - (X[1] + 47.0)))))

def plotter(E,A, solution):
    try:
        x, y, z = solution
        fig=plt.figure(figsize=[12,8])
        ax=plt.axes(projection='3d')
        ax.plot_surface(X1,X2,f((X1,X2)),color='red',alpha=0.7)
        ax.plot_wireframe(X1,X2,f((X1,X2)),ccount=2,rcount=2, color='orange',alpha=0.8)   
        ax.scatter(x, y, z, color='blue', s=100)
        ax.view_init(elev=E,azim=A)
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('f(x1,x2)')
        plt.show()
    except Exception as e:
        print(e)

if __name__ == '__main__':
    experiment_iter = 30
    execution_times = []
    best_solution = None
    best_fitness = float("Inf")
    fitnesses = []

    var_bound = np.array([[-512,512]]*2)
    params = {'max_num_iteration': 5000,\
                   'population_size':100,\
                   'mutation_probability':0.1,\
                   'elit_ratio': 0.01,\
                   'parents_portion': 0.3,\
                   'crossover_type':'uniform',\
                   'selection_type': 'roulette',
                   'max_iteration_without_improv':None}
    
    for i in range(experiment_iter):
        model = ga(function=f, dimension=2, variable_type='real', variable_boundaries=var_bound, algorithm_parameters=params)
        
        init_time = time()
        print('-'*100)
        print(f'starting iter {i}:')
        result = model.run(
            no_plot = True if i != experiment_iter-1 else False,
            progress_bar_stream = 'stdout',
            disable_printing= True,
            population_initializer = Population_initializer(select_best_of = 1, local_optimization_step = 'never', local_optimizer = None)
        )

        final_time = time()
        ex_time = final_time-init_time

        execution_times.append(ex_time)
        fitnesses.append(result.score)

        if result.score < best_fitness:
            best_solution = result.variable
            best_fitness = result.score
        
        print(f'\nSolution: {result.variable}')
        print(f'Fitness: {result.score}')
        print(f'Execution time: {ex_time:.4f}')
    
    print('-'*100)
    print(f'Best Solution: {best_solution}')
    print(f'Best Fitness: {best_fitness}')
    print(f'Average Fitness: {np.mean(fitnesses):.4f}')
    print(f'Fitness Variance: {np.var(fitnesses):.4f}')
    print(f'Average Execution Time: {np.mean(execution_times)}')


    x1=np.linspace(-512,512,100)
    x2=np.linspace(-512,512,100)
    X1,X2=np.meshgrid(x1,x2)
    
    sol = (*best_solution, best_fitness)
    iplot=interactive(plotter,E=(-90,90,5),A=(-90,90,5), solution=fixed(sol))
    iplot
    