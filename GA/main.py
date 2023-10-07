from geneticalgorithm import geneticalgorithm as ga
import numpy as np
from ipywidgets import interactive, fixed
import matplotlib.pyplot as plt
from time import time

def f(X):
    return (-(X[1] + 47.0) * np.sin(np.sqrt(np.abs(X[0]/2.0 + (X[1] + 47.0)))) - X[0] * np.sin(np.sqrt(np.abs(X[0] - (X[1] + 47.0)))))

def plotter(E,A, solution):
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

if __name__ == '__main__':
    var_bound = np.array([[-512,512]]*2)
    params = {'max_num_iteration': 5000,\
                   'population_size':100,\
                   'mutation_probability':0.1,\
                   'elit_ratio': 0.01,\
                   'crossover_probability': 0.5,\
                   'parents_portion': 0.3,\
                   'crossover_type':'uniform',\
                   'max_iteration_without_improv':None}
    model = ga(function=f, dimension=2, variable_type='real', variable_boundaries=var_bound, algorithm_parameters=params)
    init_time = time()
    model.run()
    final_time = time()
    ex_time = final_time-init_time
    convergence=model.report
    solution=model.output_dict
    
    print(f'execution time: {ex_time:.4f}')
    x1=np.linspace(-512,512,100)
    x2=np.linspace(-512,512,100)
    X1,X2=np.meshgrid(x1,x2)
    (points,z) = solution.values()
    
    sol = (*points,z)
    iplot=interactive(plotter,E=(-90,90,5),A=(-90,90,5), solution=fixed(sol))
    iplot
    