import numpy as np
import itertools
from scipy.optimize import fsolve
import functools as ft 
import random


def significance_level(episilon,M,delta1,N,sigma):
    return  ((M)*(((episilon**2)-4*delta1)**2))/(16*N*(episilon**2)) + np.log(sigma)

def solving_epsilon(M,delta1,N,sigma):
    return lambda x: significance_level(x,M,delta1,N,sigma)

def get_epsilon(M,delta1,N,sigma):
    initial_guess = 0.5
    func = solving_epsilon(M,delta1,N,sigma)
    epsilon_sol = fsolve(func,initial_guess)
    return epsilon_sol[0]

def Probability_guessing(Answ, epsilon,H):
    epsilon_tilde = np.sqrt(epsilon**2 + epsilon**4)
    if Answ : 
        return 1/H + epsilon_tilde
    if not Answ :
        return (1 - epsilon_tilde )/H

def Privacy(epsilon,eta,N):
    return ((1-eta)**N)*epsilon*np.sqrt(1+epsilon**2) + (1 - (1-eta)**N)

def Election_accepted(epsilon,N,sigma,gamma):
    S = (1-2**(-gamma))**sigma
    
    return ((1-epsilon*(1-S))**N)

import numpy as np

def number_of_sample(epsilon, delta, N, sigma):
    # Vérification des conditions d'entrée
    if epsilon**2 <= 4 * delta:
        raise ValueError("L'expression (epsilon^2 - 4*delta) doit être positive pour éviter une division par zéro.")

    # Calcul des constantes
    factor = (epsilon**2 - 4 * delta)**2
    denominator = 16 * N * epsilon**2
    constant = denominator / factor

    # Borne inférieure
    result = -np.log(sigma) * constant  # Borne supérieure (même formule dans ce cas)

    return result


def getting_xxxx_outcome() : 
    outcome_list = [
                    (-1.0, -1.0, -1.0, -1.0), (-1.0, -1.0, -1.0, 1.0),
                    (-1.0, -1.0, 1.0, -1.0), (-1.0, -1.0, 1.0, 1.0),
                    (-1.0, 1.0, -1.0, -1.0), (-1.0, 1.0, -1.0, 1.0),
                    (-1.0, 1.0, 1.0, -1.0), (-1.0, 1.0, 1.0, 1.0),
                    (1.0, -1.0, -1.0, -1.0), (1.0, -1.0, -1.0, 1.0),
                    (1.0, -1.0, 1.0, -1.0), (1.0, -1.0, 1.0, 1.0),
                    (1.0, 1.0, -1.0, -1.0), (1.0, 1.0, -1.0, 1.0),
                    (1.0, 1.0, 1.0, -1.0), (1.0, 1.0, 1.0, 1.0)]
        
        # Replace -1.0 with 0 in each tuple
    outcome_list = [list(0 if x == -1.0 else x for x in tpl) for tpl in outcome_list]
    
    return outcome_list

import numpy as np
import scipy.optimize as sp
import time
import math
import itertools
from scipy.stats import norm
from copy import deepcopy
from pathlib import Path
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import glob
import pandas as pd
import random
import counts_statistics as cs

def unitary(angle, u_compensation, angle_rz):

    a = angle[0]
    b = angle[1]
    y = angle[2]

    rz = [[np.exp(-1j*angle_rz/2),0],
          [0,np.exp(1j*angle_rz/2)]]

    u = rz@u_compensation
    
    f = (1/2)*(-np.cos(2*(a - b))-np.cos(2*(b-y))) - np.real(u[0][0])
    g = (1/2)*(np.sin(2*(a - b)) + np.sin(2*(b - y))) - np.real(u[0][1])
    h = (1/2)*(-np.sin(2*(a - b)) - np.sin(2*(b - y))) - np.real(u[1][0])
    v = (1/2)*(-np.cos(2*(a - b))- np.cos(2*(b - y))) - np.real(u[1][1])

    k = (1/2)*(-np.cos(2*b) + np.cos(2*(a - b + y))) - np.imag(u[0][0])
    m = (1/2)*(-np.sin(2*b) + np.sin(2*(a - b + y))) - np.imag(u[0][1])
    z = (1/2)*(-np.sin(2*b) + np.sin(2*(a - b + y))) - np.imag(u[1][0])
    e = (1/2)*(np.cos(2*b) - np.cos(2*(a - b + y))) - np.imag(u[1][1])

    return  (f, g, h, v, k, m, z, e)

def solving(angle, u, angle_rz):
    result = sp.least_squares(unitary, angle, method='trf', args=[u, angle_rz], max_nfev=1000000000)
    QWP1 = result.x[0]
    HWP1 = result.x[1]
    QWP2 = result.x[2]
    return(QWP2*180/np.pi, HWP1*180/np.pi, QWP1*180/np.pi)

def toss_coins(gamma):
    """Toss gamma coins; return 1 if all are heads, else return 0."""
    return int(all(random.choice([0, 1]) == 1 for _ in range(gamma)))

def generate_rk(n, pk):
    """Generate an N-bit string `rk` with parity matching `pk`."""
    rk = [random.choice([0, 1]) for _ in range(n - 1)]
    # Calculate the parity bit to make ⊕ rk = pk
    parity_bit = pk ^ (sum(rk) % 2)
    rk.append(parity_bit)
    return rk

def generate_orderings(n):
    """Generate N orderings where each voter is last exactly once."""
    orderings = []
    for i in range(n):
        ordering = list(range(n))  # Start with all voters
        ordering.remove(i)  # Remove voter i from early positions
        random.shuffle(ordering)  # Shuffle remaining voters
        ordering.append(i)  # Place voter i at the end
        orderings.append(ordering)
    random.shuffle(orderings)
    return orderings

def LOGICALOR(n, sigma, gamma, vote, parity = True):
    """Run the logical OR protocol on the input votes."""
    # Step 1: Generate orderings
    orderings = generate_orderings(n)
    y = 0  # Initialize output y
    wk_values_sum = 0
    wk_order_list =[]
    for ordering in orderings:
        vote_order = []
        for i in ordering:
            vote_order.append(vote[i])
        for _ in range(sigma):  # Repeat sigma times for each ordering
            pk_values = []
            rk_values = []

            # Step 2-3: Determine pk for each voter
            for k in range(n):
                if vote_order[k] == 0:
                    pk = 0
                else:
                    pk = toss_coins(gamma)
                pk_values.append(pk)

            # Step 4: Generate rk with parity pk for each voter
            rk_values = [generate_rk(n, pk) for pk in pk_values]

            # Step 5: Simulate sending rk^i_k to voter i
            received_bits = [[rk_values[k][i] for k in range(n)] for i in range(n)]

            # Step 6: Each voter computes the parity zi of received bits
            zi_values = [sum(received_bits[i]) % 2 for i in range(n)]

            # Step 7: Compute the parity of the original bits
            y_current = sum(zi_values) % 2       
                     
            wk_order = [0]*n
            wk_values_sum = []
                
            for k in range(n):
                wk = []
                for x in range(n):
                    wk.append((zi_values[x] + rk_values[k][x])%2)
                wk_values_sum.append(sum(wk)%2)
                
            for k in range(0,len(wk_values_sum)):
                wk_order[ordering[k]] = wk_values_sum[k]
                
            wk_order_list.append(wk_order)
            
            y |= y_current  # OR the result into y (if any repetition results in 1, y becomes 1)
            
    col_sums = [sum(col) for col in zip(*wk_order_list)]
    
    if parity == True :
        return y,col_sums # Output final result
    if parity == False : 
        return y
   
def RandomBit(agent_index, n, sigma, gamma):
    """Generate a random bit for a single agent's vote."""
    vote = [0] * n
    vote[agent_index] = random.choice([0, 1])
    return LOGICALOR(n, sigma, gamma, vote, parity=False)

def RandomAgent(agent_index, n, sigma, gamma):
    """Generate a random identifier for an agent."""
    bits = [RandomBit(agent_index, n, sigma, gamma) for _ in range(int(np.log2(n)))]
    return int("".join(map(str, bits)), 2)

def UniqueIndex(n, sigma, gamma):
    indices_assigned = [None] * n  # Liste pour stocker les indices uniques attribués
    round_number = 0  # Initialiser R = 1
    
    while round_number < 4:  # Tant que tous les indices ne sont pas attribués
        indices_assigned_this_round = [None] * n 
        # Étape 2: Chaque agent choisit x_k
        x_values = []
        for k in range(n):
            if indices_assigned[k] is not None:  # Si l'agent a déjà un index, x_k = 0
                x_values.append(0)
            else:  # Sinon, choisir x_k = 1 avec probabilité 1 / (N - R)
                prob = 1 / (n - round_number)
                x_values.append(1 if random.random() < prob else 0)
        # Effectuer LogicalOr pour obtenir y
        y,wk = LOGICALOR(n, sigma, gamma, x_values)

        # Étape 3: Vérifier si y = 0, sinon on répète
        if y == 0:
            continue
        # Étape 4: Identifier les agents avec x_k = 1 et w_k = 0
        collision = False
        
        for k in range(n):
            if x_values[k] == 1 and wk[k] == 0:
                indices_assigned_this_round[k] = round_number  # Assigner l'index ω_k = R
                
        if indices_assigned_this_round.count(round_number) > 1: 
            collision = True
        
        if collision == True:
            continue            
        
        for k in range(n):
            if indices_assigned_this_round[k] !=None : 
                indices_assigned[k] = indices_assigned_this_round[k]
                
        # Étape 5: Notification - chaque agent effectue un LogicalOr avec 0, sauf ceux ayant un index
        notification_input = [1 if indices_assigned[k] == round_number else 0 for k in range(n)]
        notification_output = LOGICALOR(n, sigma, gamma, notification_input,parity=False)
        # Étape 6: Vérification de la notification
        
        if notification_output == 0:  # Pas d'index assigné, on répète
            continue

        # Étape 7: Si notification == 1, on augmente R
        round_number += 1


    return indices_assigned  # Retourner la liste des indices uniques assignés

def getting_xxxx_outcome() : 
    outcome_list = [
                    (-1.0, -1.0, -1.0, -1.0), (-1.0, -1.0, -1.0, 1.0),
                    (-1.0, -1.0, 1.0, -1.0), (-1.0, -1.0, 1.0, 1.0),
                    (-1.0, 1.0, -1.0, -1.0), (-1.0, 1.0, -1.0, 1.0),
                    (-1.0, 1.0, 1.0, -1.0), (-1.0, 1.0, 1.0, 1.0),
                    (1.0, -1.0, -1.0, -1.0), (1.0, -1.0, -1.0, 1.0),
                    (1.0, -1.0, 1.0, -1.0), (1.0, -1.0, 1.0, 1.0),
                    (1.0, 1.0, -1.0, -1.0), (1.0, 1.0, -1.0, 1.0),
                    (1.0, 1.0, 1.0, -1.0), (1.0, 1.0, 1.0, 1.0)]
        
        # Replace -1.0 with 0 in each tuple
    outcome_list = [list(0 if x == -1.0 else x for x in tpl) for tpl in outcome_list]
    
    return outcome_list

def Verification_Analysis(data_dirs,file,file2):
    
    os.chdir(data_dirs[0])
    filenames = [i for i in glob.glob(f"{file}*")]
    measurement_basis = []
    
    for file in filenames:
        os.chdir(f"{data_dirs[0]}/{file}/{file2}")
        file_aux=[i for i in glob.glob("ABCD*")]    
        measurement_basis  = file_aux
        # Defining (the sign associated with each stabilizer)
        stabilizers = ["xxxx",'vvvv']
        stab_sign = [1, -1]
        STABILIZER_TO_INDEX = {stab: i for i, stab in enumerate(stabilizers)} 
        working_dir_data =  data_dirs[0]
        # # We re-calculate the statistics for the randomly selected data
        stats = cs.Operartors_Statistics(4, stabilizers, stab_sign, STABILIZER_TO_INDEX, measurement_basis, data_dirs[0] + "//" + filenames[0] + f"/{file2}")
        writting_dir = working_dir_data + "//test"
        os.makedirs(f"{writting_dir}", exist_ok=True)
        stats.shuffle_data()
        samples_evolution = np.arange(2, stats.n_loops+1, 1)
        p_rejection_list, n_samples_list = stats.get_fail_probability_evolution(samples_evolution, writting_dir)
        Mean_rejection = sum(p_rejection_list)/len(p_rejection_list)
    
    return Mean_rejection

def Voting_Analysis(Vote,data_dirs):
    
    os.chdir(data_dirs[0])
    filenames = [i for i in glob.glob("TEST")]
    for file in filenames :
        os.chdir(f"{data_dirs[0]}/{file}")
        file_aux = [i for i in glob.glob("ABCD*")]
    counts = []
    Measurement =[]
    Agent_index =[]
    for i in file_aux :
        Agent_index.append(((i.split("=")[1]).split("_")[2]).split(".")[0])

    while ',' in Agent_index :
        Agent_index.remove(',')

    for i,j in enumerate(Agent_index):
        Agent_index[i] = int(j)

    for file2 in file_aux : 
        with open(file2,mode = 'r') as file3 :
            for line in file3 :
                counts.append(line.split())
                
    counts = np.array(counts,dtype = int)
    result_outcome = getting_xxxx_outcome()
    for i in range(4) :
        
        Measurement.append([result_outcome[z] for z in range(len(counts[i])) if counts[i][z] == 1])
                
    B = np.zeros((4,4))
    for i in range(4):
        for j in range(4):
            B[i][j] = Measurement[i][0][j]

    for i,j in enumerate(Agent_index):
        B[i][j] = (int(Vote[i]) ^ int(B[i][j]))
        
    E = [sum(B[:][i])%2 for i in range(4)]

    T = [0]*2

    for i in E : 
        if i == 0 :
            T[0] = T[0] + 1
        if i == 1 :
            T[1] = T[1] + 1
    
    return(B,E,T)

def generate_angles_in_degrees():
    # Randomly generate three angles between 0 and 180 degrees
    angles = np.random.uniform(0, 180, 3)
    
    # Calculate the fourth angle to make the sum a multiple of 180
    total = sum(angles)
    remainder = total % 180
    fourth_angle = (180 - remainder) if remainder != 0 else 0

    # Ensure the fourth angle is within the range [0, 180]
    if fourth_angle > 180:
        fourth_angle -= 180

    angles = list(angles) + [fourth_angle]
    return angles
