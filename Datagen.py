# I used the terminal to run this code and generate the data, only for the trigonometric basis.
# I input "fenicsproject run" then "fenicsproject create ppde" to create a session
# Then "fenicsproject run ppde" to run the session and finally "python3 Datagen.py" to execute the code.
# It results in a folder containing 4 files: G.npy contains the square root of the gram matrix.
# X1.npy contains the input vectors and Y1.npy contains the output vectors.

from ast import literal_eval
from fenics import *
from itertools import combinations_with_replacement
from math import floor, ceil
import multiprocessing as mp
import numpy as np
import os
import pickle
import sys


def update_params(argv, params):
    if len(argv) > 1:
        params.update(literal_eval(argv[1]))

    return params


def matrix_sqrt(mat):
    w, V = np.linalg.eigh(mat)

    if np.min(w) < 0:
        print('Smallest eigenvalue: ', np.min(w))
        print('Warning negative eigenvalues set to Zero')
        diag_mat = np.diag(np.sqrt(np.maximum(0, w)))
    else:
        diag_mat = np.diag(np.sqrt(w))

    mat_sqrt = np.linalg.multi_dot([V, diag_mat, V.T])

    return mat_sqrt


def sqrt_gram_matrix(V):
    u = TrialFunction(V)
    v = TestFunction(V)
    G1 = assemble(u * v * dx).array()
    G2 = assemble(dot(grad(u), grad(v)) * dx).array()

    G = G1 + G2

    return matrix_sqrt(G.astype(np.float32))


def get_polynomial(num_var, max_deg):
    var = ['c']  # add dummy variable
    for i in range(num_var):
        var.append('x[' + str(i) + ']')

    # compute all combinations of variables
    terms = []
    for x in combinations_with_replacement(var, max_deg):
        terms.append(list(x))

    for el in terms:
        while "c" in el:
            el.remove("c")

    monomial_list = ['*'.join(x) for x in terms]
    monomial_list[0] = '1'  # fix constant term

    return monomial_list


def trigonometric_basis(n, p, mu=1, sigma=-1):
    num_terms = p

    coeff = np.random.uniform(0, 1., (n, num_terms))
    expr_list = []

    template_a = "+(1+{}*{}*(1+{}))"
    template_sin = "sin({}*3.14159*x[0])*sin({}*3.14159*x[1])"

    for i in range(n):
        temp = str(mu)
        for j in range(1, num_terms + 1):
            sin = template_sin.format(floor((j + 2) / 2), ceil((j + 2) / 2))
            a = template_a.format(j ** (sigma), str(coeff[i, j - 1]), sin)
            temp += a
        expr_list.append(temp)

    return expr_list, coeff


def boundary(x, on_boundary):
    return on_boundary


def solver(expr):
    '''Solver for parametric diffusion equation'''

    u = Function(V)
    v = TestFunction(V)

    a = Expression(expr, degree=params['inter_deg'])
    F = dot(a * grad(u), grad(v)) * dx - f * v * dx

    solve(F == 0, u, bc)

    return np.array(u.vector().get_local())


# All parameters
default_parameters = {
    'num_samples': 5000,
    'basis_fct': 'trigonometric',
    'mu': 1,  # shift hyper-parameter
    'mode': 'x',  # only required for cookies
    'sigma': 0,  # only required for trigonometric polnomials
    'num_basis': 2,  # actual number of basis functions may vary
    'finite_element': 'P',  # see http://femtable.org/
    'inter_deg': 1,  # degree of interpolation
    'mesh_size': 10,  # number of edges in the mesh (number of nodes: edges + 1)
    'right_hand_side': '20+10*x[0]-5*x[1]',
    'folder_name': 'some_dataset',  # folder where data is saved
    'part': 1,
    'part_max': 5
}

if __name__ == '__main__':

    # update default parameters with system arguments
    params = update_params(sys.argv, default_parameters)
    data_path = './datasets/' + params['folder_name'] + '/'
    params['data_path'] = data_path
    part = str(params['part'])
    print(params)

    if not os.path.exists(params['data_path']):
        os.makedirs(params['data_path'])

    # save parameter dictionary only for first part
    if part == '1':
        with open(params['data_path'] + 'params', 'wb') as pickle_out:
            pickle.dump(params, pickle_out)

    # create mesh and function space
    mesh = UnitSquareMesh(params['mesh_size'], params['mesh_size'])
    f = Expression(params['right_hand_side'], degree=params['inter_deg'])
    V = FunctionSpace(mesh, params['finite_element'], params['inter_deg'])

    # define homogeneous Diriclet boundary
    u_D = Expression('0', degree=params['inter_deg'])
    bc = DirichletBC(V, u_D, boundary)

    # generate coefficients
    num_samples = params['num_samples'] // params['part_max']
    if params['basis_fct'] == 'polynomial':
        expr_list, coeff = polynomial_basis(num_samples, params['num_basis'], params['mu'])
    elif params['basis_fct'] == 'trigonometric':
        expr_list, coeff = trigonometric_basis(num_samples, params['num_basis'], params['mu'], params['sigma'])
    elif params['basis_fct'] == 'squares':
        expr_list, coeff = squares_basis(num_samples, params['num_basis'], params['mu'])
    elif params['basis_fct'] == 'cookies':
        expr_list, coeff = cookies_basis(num_samples, params['num_basis'], params['mu'], params['mode'])
    else:
        print('Basis function not found')

    # save sqrt of Gram matrix only for first part
    if part == '1':
        G_sqrt = sqrt_gram_matrix(V)
        np.save(data_path + 'G', G_sqrt)

    np.save(data_path + 'X' + part, coeff)

    # multithreading for quicker data generation, upping processes
    num_processes = 2
    pool = mp.Pool(processes=num_processes)
    results = pool.map(solver, expr_list)
    data = np.stack(results, 0)

    np.save(data_path + 'Y' + part, data)

    # check if dimensions are as expected
    print(data.shape)
    print(coeff.shape)
