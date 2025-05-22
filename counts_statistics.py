import numpy as np
import matplotlib.pyplot as plt
import os
import random
import functools as ft
import itertools
import operator
from Pauli import *
            
"""
The following function "eigenvals" will define the sign associated with each eigenvalue
(for eg. needed to calculate the expectation value of an operator)
This method should work for measurement resulting from the combination of Paulis

In principle, we would only need a list of eigenvalues in this script but here we also calculate the
eigenvectors so we can check the ordering of the eigenvalues list is matching the order of our data
collection (i.e., to make sure we associate the correct eigenvalue to the correct column in the 
get_expectation_values function below)

Note: Getting the eigenvectors and eigenvalues from the operator matrix using directly a
python package works but then, it is hard to know to which column (in our data file) the eigenvalues
correspond to
"""
def eigenvals(n_qubits, operators):
    eigenvalues = []
    eigenvectors = []

    for stab_basis in operators:
        """
        For each operator we define 2 lists:
        - pauli_outcomes_list - list with the eigenvectors of the basis measured in each qubit "q" ("q in stab_basis" goes letter by letter in stab_basis)
        - pauli_eigvals_list - list with the eigenvalues of the basis measured in each qubit "q"
        """
        pauli_outcomes_list = []
        pauli_eigvals_list =[]

        for q in stab_basis:
            evals, evecs = np.linalg.eigh(PAULI[q])
            sorting = np.argsort(evals)[::-1]
            pauli_eigvals_list.append(evals[sorting])
            pauli_outcomes_list.append(evecs[sorting])
    

        """
        Then, calculate
        - eigenvectors (outcomes) of the operator by doing the kronecker product between the outcomes of each qubit
        - eigenvalues (eigenvals) of the operator by doing the multiplication between the eigenvalues of each qubit
        """
        outcomes = list(ft.reduce(np.kron, (pauli_outcomes_list[0][k],
                                            pauli_outcomes_list[1][l],
                                            pauli_outcomes_list[2][m],
                                            pauli_outcomes_list[3][n])) for k, l, m, n in itertools.product([0, 1], repeat=4))
        
        eigenvals = list(ft.reduce(operator.mul, (pauli_eigvals_list[0][k],
                                                pauli_eigvals_list[1][l],
                                                pauli_eigvals_list[2][m],
                                                pauli_eigvals_list[3][n])) for k, l, m, n in itertools.product([0, 1], repeat=4))

        eigenvalues.append(eigenvals)
        eigenvectors.append(outcomes)

    return np.array(eigenvalues, dtype=float), np.array(eigenvectors, dtype=complex)



"""
The class Operators_statistics is meant to read data files (filenames) in working_dir_data
(each data file corresponding to a measurement basis that can be extracted from the filename)
The columns in the files should be associated with a specific eigenvector
The lines in the file should be associated with different measurement rounds
Each elements should correspond to the number of detected events

This data can then be treated to calculate expectation values or passing probabilities
(further used to calculate failure rates and violations of inequalities)
associated with each stabilizer in stabilizers. Each stabilizer has a stab_sign associated,
which determines the outcome winning loosing outcomes (stab_sign=+1 means that lambda=+1 is a winning outcome).

set_loops is set to None if the same operators is measured several times (over several rounds)
"""
class Operartors_Statistics:

    def __init__(self, n_qubits, stabilizers, stab_sign, STABILIZER_TO_INDEX, filenames, working_dir_data, set_loops=None):

        os.chdir(working_dir_data)

        self.n_qubits = n_qubits
        self.n_outcomes = 2**self.n_qubits
        self.n_loops = len(filenames)

        self.stabilizers = stabilizers
        self.STABILIZER_TO_INDEX=STABILIZER_TO_INDEX
        self.n_stabilizers = len(self.stabilizers)
        self.stab_sign = stab_sign

        self.eigenvalues, self.eigenvectors = eigenvals(self.n_qubits, self.stabilizers)

        self.labels = []
        self.loops = []
        # Counts array is a self.n_loops by self.n_outcomes array
        # Each loop corresponds to a different randomly chosen stabilizer
        self.counts = []
        self.counts_array = np.zeros((self.n_loops, self.n_outcomes))
        self.measurement = []
        self.N_pass =  np.zeros((self.n_stabilizers))
        self.N_fail = np.zeros((self.n_stabilizers))
        self.N_old_pass =   np.zeros((self.n_stabilizers))
        self.N_old_fail =   np.zeros((self.n_stabilizers))


        self.N_pass_error = np.zeros((self.n_stabilizers))
        self.N_fail_error = np.zeros((self.n_stabilizers))

        for i, file in enumerate(filenames):
            file_label = str(((file.split("=")[1]).split(".")[0]).split("_")[0])
            ### We only gather the counts that resulted from a stabilizer measurement
            if any(file_label == c for c in self.stabilizers):
                self.labels.append(file_label)
                if set_loops is None:
                    self.loops.append(int((file.split("_"))[3].split(".")[0]))
                else:
                    self.loops.append(i)

                with open(file) as file:
                    for line in file:
                        self.counts.append(line.split())
                        self.counts_array[i] = np.array(self.counts[-1][:16], dtype=float)
            else:
                self.labels.append(0)
                self.loops.append(0)
        self.counts_array_error = np.sqrt(self.counts_array)


    def shuffle_data(self):
        np.random.seed(1234)
        order = np.arange(self.n_loops)
        np.random.shuffle(order)

        self.labels = np.array(self.labels)[order]
        self.loops = np.array(self.loops)[order]
        self.counts_array = self.counts_array[order]
        self.counts_array_error = self.counts_array_error[order]
        assert len(self.labels) == len(self.loops) == len(self.counts_array) == len(self.counts_array_error) == self.n_loops


    def remove_random_loop(self):
        # Select random loop and set it to zero in the self.counts_array
        loop_rand = random.randrange(self.n_loops)
        self.counts_array[loop_rand] = np.zeros(16)
        print(f"Certified sample: loop {loop_rand}")


    def set_counts_per_outcome(self, start=None, end=None, aux_init=True):
        # If counts_aux was not initialized in get_pass_probability, aux_init should be True
        # If we want to plot the evolution of p_pass and want to avoid re-writting counts_aux we should set it to False
        if aux_init is True:
            self.counts_aux = np.zeros((self.n_stabilizers, self.n_loops+1, self.n_outcomes))
            self.counts_aux_error = np.zeros((self.n_stabilizers, self.n_loops+1, self.n_outcomes))

        idx=np.arange(self.n_loops)
        if end is None:
            labels, loops = self.labels, self.loops
        else:
            labels, loops, idx = self.labels[start:end], self.loops[start:end], idx[start:end]
        
        for label, loop, i in zip(labels, loops, idx):
            if any(label == c for c in self.stabilizers):
                self.counts_aux[self.STABILIZER_TO_INDEX[label]][loop] = self.counts_array[i]
                self.counts_aux_error[self.STABILIZER_TO_INDEX[label]][loop] = self.counts_array_error[i]
                

        self.counts_per_outcome = np.sum(self.counts_aux, axis=1)
        self.counts_per_outcome_error = np.sqrt(np.sum(self.counts_aux_error**2, axis=1))



    def get_pass_probability(self, start=0, end=-1, aux_init=True):
        # start and end limits the number of loops considered in the counts_per_outcome
        # aux_init=True unless we are calling this function from inside get_pass_probability_evolution() function
        self.set_counts_per_outcome(start, end, aux_init)
        
        self.N_pass =  np.zeros((self.n_stabilizers))
        self.N_fail = np.zeros((self.n_stabilizers))
        self.N_pass_error = np.zeros((self.n_stabilizers))
        self.N_fail_error = np.zeros((self.n_stabilizers))

        self.positive_eigen = (self.eigenvalues+[1]*self.n_outcomes)/2
        self.negative_eigen = -(self.eigenvalues-[1]*self.n_outcomes)/2

        for i,_ in enumerate(self.stabilizers):
            
            self.N_positive = self.positive_eigen*self.counts_per_outcome[i]
            self.N_negative = self.negative_eigen*self.counts_per_outcome[i]

            N_positive_error = self.positive_eigen*self.counts_per_outcome_error[i]
            N_negative_error = self.negative_eigen*self.counts_per_outcome_error[i]

            if self.stab_sign[i]==1:
                self.N_pass[i] = np.sum(self.N_positive)
                self.N_fail[i] = np.sum(self.N_negative)
                self.N_pass_error[i] = np.sqrt(np.sum(N_positive_error**2))
                self.N_fail_error[i] = np.sqrt(np.sum(N_negative_error**2))

            elif self.stab_sign[i]==-1:
                self.N_pass[i] = np.sum(self.N_negative)
                self.N_fail[i] = np.sum(self.N_positive)
                self.N_pass_error[i] = np.sqrt(np.sum(N_negative_error**2))
                self.N_fail_error[i] = np.sqrt(np.sum(N_positive_error**2))

        N_pass_all = np.sum(self.N_pass)
        N_pass_all_error = np.sqrt(np.sum(self.N_pass_error**2))
        N_fail_all = np.sum(self.N_fail)
        N_fail_all_error = np.sqrt(np.sum(self.N_fail_error**2))

        self.p_pass = N_pass_all/(N_pass_all+N_fail_all)
        self.p_pass_error = np.sqrt((N_pass_all*N_fail_all_error/(N_pass_all+N_fail_all)**2)**2+(N_fail_all*N_pass_all_error/(N_pass_all+N_fail_all)**2)**2)

        self.N_total = N_pass_all + N_fail_all
        self.N_total_error = np.sqrt(N_pass_all_error**2+N_fail_all_error**2)
            
        return self.p_pass, self.p_pass_error


    def get_pass_probability_evolution(self, lim, writting_dir=None):
        # We should add the expectation values evolution or maybe not because we never use that

        self.p_pass_list = []
        self.p_pass_error_list = []
        self.n_samples_list = []
        self.n_samples_error_list = []

        k_last=0

        # Initiazize self.counts_aux once and set aux_init=False such that it doesn't do it again inside the k in lim loop
        # self.n_loops+1 because when we remove 1 file from filenames, which is the certified file,
        # but the dimensions still need to match
        self.counts_aux = np.zeros((self.n_stabilizers, self.n_loops+1, self.n_outcomes))
        self.counts_aux_error = np.zeros((self.n_stabilizers, self.n_loops+1, self.n_outcomes))

        for k in lim:
            p_pass, p_pass_error = self.get_pass_probability(k_last, k, aux_init=False)

            self.p_pass_list.append(p_pass)
            self.p_pass_error_list.append(p_pass_error)

            self.n_samples_list.append(self.N_total)
            self.n_samples_error_list.append(self.N_total_error)
            k_last=k

        if writting_dir is not None:
            f = open(f"{writting_dir}\\n_samples.txt", "w")
            f.write(str(self.n_samples_list)) 
            f.close()

            f = open(f"{writting_dir}\\n_samples_error.txt", "w")
            f.write(str(self.n_samples_error_list))
            f.close()

            f = open(f"{writting_dir}\\p_pass.txt", "w")
            f.write(str(self.p_pass_list))
            f.close()

            f = open(f"{writting_dir}\\p_pass_error.txt", "w")
            f.write(str(self.p_pass_error_list))
            f.close()
                
        return self.p_pass_list, self.n_samples_list


    def plot_pass_probability(self, save_dir=False):
        measurment_basis = list(map(lambda x: x.upper(), self.stabilizers))
        self.get_expectation_values()

        p = 100*self.N_pass/(self.N_pass+self.N_fail)
        error_values = 100*np.sqrt((self.N_pass*self.N_fail_error/(self.N_pass+self.N_fail)**2)**2+(self.N_fail*self.N_pass_error/(self.N_pass+self.N_fail)**2)**2)

        # Choose a blue-toned colormap
        cmap = plt.get_cmap('Blues')
        plt.figure(figsize=(10, 6))
        bars = plt.bar(measurment_basis, p, color=cmap(p))
        # Adding error bars and values
        plt.errorbar(measurment_basis, p, yerr=error_values, fmt="o", color='black', capsize=5)
        for bar, prob, err in zip(bars, p, error_values):
            plt.text(bar.get_x() + bar.get_width() / 2, prob - 7, f'{prob:.1f}', ha='center', color='white', fontsize=19)

        plt.xlabel('Measurement Basis', fontsize=20)
        plt.ylabel('Success Probability (%)', fontsize=20)

        plt.ylim(0, 100)  # Set y-axis limits from 0 to 1
        plt.xticks(rotation=45, fontsize=18)  # Rotate x-axis labels for better readability
        plt.yticks(fontsize=18) 
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        if save_dir is not False:
            plt.savefig(save_dir)
        plt.show()
        
    def get_both_probability(self, start=0, end=-1, aux_init=True):
        # start and end limits the number of loops considered in the counts_per_outcome
        # aux_init=True unless we are calling this function from inside get_pass_probability_evolution() function
        self.set_counts_per_outcome(start, end, aux_init)
        positive_eigen = (self.eigenvalues+[1]*self.n_outcomes)/2
        negative_eigen = -(self.eigenvalues-[1]*self.n_outcomes)/2
        
        for i,_ in enumerate(self.stabilizers):
            N_positive = positive_eigen[i]*self.counts_per_outcome[i]
            N_negative = negative_eigen[i]*self.counts_per_outcome[i]
            N_positive_error = positive_eigen[i]*self.counts_per_outcome_error[i]
            N_negative_error = negative_eigen[i]*self.counts_per_outcome_error[i]
            
            if self.stab_sign[i]==1:    
                self.N_old_pass[i] =  self.N_pass[i]
                self.N_old_fail[i] = self.N_fail[i]
                self.N_pass[i] = np.sum(N_positive)
                self.N_fail[i] = np.sum(N_negative)
                self.N_pass_error[i] = np.sqrt(np.sum(N_positive_error**2))
                self.N_fail_error[i] = np.sqrt(np.sum(N_negative_error**2))

            elif self.stab_sign[i]==-1:
                self.N_old_pass[i] =  self.N_pass[i]
                self.N_old_fail[i] = self.N_fail[i]
                self.N_pass[i] = np.sum(N_negative)
                self.N_fail[i] = np.sum(N_positive)
                self.N_pass_error[i] = np.sqrt(np.sum(N_negative_error**2))
                self.N_fail_error[i] = np.sqrt(np.sum(N_positive_error**2))

        if sum(self.N_pass) > sum(self.N_old_pass):
            self.measurement.append(0)
        if  sum(self.N_fail) > sum(self.N_old_fail):
            self.measurement.append(1)
        
        N_pass_all = np.sum(self.N_pass)
        N_pass_all_error = np.sqrt(np.sum(self.N_pass_error**2))
        
        N_fail_all = np.sum(self.N_fail)
        N_fail_all_error = np.sqrt(np.sum(self.N_fail_error**2))

        self.p_fail = N_fail_all/(N_pass_all+N_fail_all)
        self.p_fail_error = np.sqrt((N_fail_all*N_pass_all_error/(N_pass_all+N_fail_all)**2)**2+(N_pass_all*N_fail_all_error/(N_pass_all+N_fail_all)**2)**2)

        self.p_pass = N_pass_all/(N_pass_all+N_fail_all)
        self.p_pass_error = np.sqrt((N_pass_all*N_fail_all_error/(N_pass_all+N_fail_all)**2)**2+(N_fail_all*N_pass_all_error/(N_pass_all+N_fail_all)**2)**2)
        
        self.N_total = N_pass_all + N_fail_all
        self.N_total_error = np.sqrt(N_pass_all_error**2+N_fail_all_error**2)
        
        return self.p_pass,self.p_fail, self.p_pass_error,self.p_fail_error
    
    def get_both_probability_evolution(self, lim, writting_dir=None):
        # We should add the expectation values evolution or maybe not because we never use that
        self.p_pass_list = []
        self.p_pass_error_list = []

        self.p_fail_list = []
        self.p_fail_error_list = []

        self.n_samples_list = []
        self.n_samples_error_list = []

        k_last=0

        # Initiazize self.counts_aux once and set aux_init=False such that it doesn't do it again inside the k in lim loop
        # self.n_loops+1 because when we remove 1 file from filenames, which is the certified file,
        # but the dimensions still need to match
        self.counts_aux = np.zeros((self.n_stabilizers, self.n_loops+1, self.n_outcomes))
        self.counts_aux_error = np.zeros((self.n_stabilizers, self.n_loops+1, self.n_outcomes))

        for k in lim:
            p_pass,p_fail,p_pass_error,p_fail_error = self.get_both_probability(k_last, k, aux_init=False)

            self.p_pass_list.append(p_pass)
            self.p_pass_error_list.append(p_pass_error)
            self.p_fail_list.append(p_fail)
            self.p_fail_error_list.append(p_fail_error)

            self.n_samples_list.append(self.N_total)
            self.n_samples_error_list.append(self.N_total_error)
            k_last=k

        if writting_dir is not None:
            f = open(f"{writting_dir}\\n_samples.txt", "w")
            f.write(str(self.n_samples_list)) 
            f.close()

            f = open(f"{writting_dir}\\n_samples_error.txt", "w")
            f.write(str(self.n_samples_error_list))
            f.close()

            f = open(f"{writting_dir}\\p_fail.txt", "w")
            f.write(str(self.p_fail_list))
            f.close()
            
            f = open(f"{writting_dir}\\p_pass.txt", "w")
            f.write(str(self.p_pass_list))
            f.close()

            f = open(f"{writting_dir}\\p_pass_error.txt", "w")
            f.write(str(self.p_pass_error_list))
            f.close()

            f = open(f"{writting_dir}\\p_fail_error.txt", "w")
            f.write(str(self.p_pass_error_list))
            f.close()
                
        return self.p_pass_list,self.p_pass_error_list,self.p_fail_list,self.p_fail_error_list, self.n_samples_list
    
    def plot_fail_probability(self, save_dir=False):
        measurment_basis = list(map(lambda x: x.upper(), self.stabilizers))
        self.get_expectation_values()

        p = self.N_fail/(self.N_pass+self.N_fail)
        error_values = np.sqrt((self.N_fail*self.N_pass_error/(self.N_pass+self.N_fail)**2)**2+(self.N_pass*self.N_fail_error/(self.N_pass+self.N_fail)**2)**2)

        # Choose a blue-toned colormap
        cmap = plt.get_cmap('Blues')
        plt.figure(figsize=(10, 6))
        bars = plt.bar(measurment_basis, p, color=cmap(p))
        # Adding error bars and values
        plt.errorbar(measurment_basis, p, yerr=error_values, fmt="o", color='black', capsize=5)
        for bar, prob, err in zip(bars, p, error_values):
            plt.text(bar.get_x() + bar.get_width() / 2, prob - 7, f'{prob:.1f}', ha='center', color='white', fontsize=19)

        plt.xlabel('Measurement Basis', fontsize=20)
        plt.ylabel('Rejection ration', fontsize=20)

        plt.ylim(0,0.2)  # Set y-axis limits from 0 to 1
        plt.xticks(rotation=45, fontsize=18)  # Rotate x-axis labels for better readability
        plt.yticks(fontsize=18) 
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        if save_dir is not False:
            plt.savefig(save_dir)
        plt.show()


    def get_expectation_values(self):
        self.expectation_values = np.zeros((self.n_stabilizers))
        self.expectation_values_error = np.zeros((self.n_stabilizers))

        self.set_counts_per_outcome()

        for i,_ in enumerate(self.stabilizers):

            # x and y are just for an easier p_outcome_error calculation
            x = self.counts_per_outcome[i]
            x_error = self.counts_per_outcome_error[i]
            y = np.sum(self.counts_per_outcome[i])-x
            y_error = np.sqrt(np.sum(self.counts_per_outcome_error[i]**2)+x_error**2)


            denom = y + x
            p_outcome = np.zeros_like(x)
            p_outcome_error = np.zeros_like(p_outcome)
            non_zero = denom > 0 
            # p_outcome is the N_outcome/N_all_outcomes
            p_outcome[non_zero] = x[non_zero] / denom[non_zero]
            p_outcome_error[non_zero] = np.sqrt((y[non_zero]*x_error[non_zero])**2+(x[non_zero]*y_error[non_zero])**2)/(denom[non_zero]**2)

            self.expectation_values[i] = np.sum(self.eigenvalues[i]*p_outcome)
            self.expectation_values_error[i] = np.sum(p_outcome_error**2)

        # print(self.expectation_values)

        return self.expectation_values, self.expectation_values_error

