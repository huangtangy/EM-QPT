
'''
Useful functions for EM-QPT
'''



from QPT import QPT
from QPT import *
from numExp_qiskit import NumExp
from joblib import Parallel, delayed
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import itertools, random
from qutip import Qobj
from qutip.random_objects import rand_super_bcsz, rand_kraus_map, rand_unitary
from qiskit.quantum_info import random_unitary, Operator, average_gate_fidelity, Kraus, Chi, Choi
from scipy.linalg import sqrtm
from os import listdir
import pandas as pd

def get_average(chi1,chi2,N):
    return (get_chiF(chi1, chi2)*(2**N)+1)/(2**N+1)  # /np.sqrt(np.trace(chi1.conj().T@chi1)*np.trace(chi2.conj().T@chi2))

def get_chiF(chi1, chi2):
    """Calculate the fidelity between two quantum processes represented by χ matrices."""
    sqrt_chi1 = sqrtm(chi1)
    return np.real(np.trace(sqrtm(sqrt_chi1 @ chi2 @ sqrt_chi1))**2)
def from_chiF_to_gateF(chiF,N):
    return (chiF*(2**N)+1)/(2**N+1)

def EM_QPT(N,idea_channel,measure_data_gate,measure_data_id,chi_digital=[],chi_fid=True):
    '''
    N: qubit num
    idea_channel: the target quantum channel
    measure_data_gate: the measurement data for quantum channel
    measure_data_id: the measuremnt data for identity channel
    chi_digital: the digital of identity channel

    '''
    chi_idea = get_idea_chi_matrix(idea_channel, N)
    if len(chi_digital)==False:
        # (2) the identity QPT
        qptid = QPT(N, measure_data_id, [np.identity(2 ** N)])
        chi_id_pred = qptid.get_chi_LS_X(qptid.rho_in_idea, qptid.observables)
        # (3) the standard QPT 
        qpt0 = QPT(N, measure_data_gate, idea_channel)
        chi_pred = qpt0.get_chi_LS_X(qpt0.rho_in_idea, qpt0.observables) 
        # (4) the error-mitigated  QPT 
        proj_noisy = qpt0.get_noisy_proj_1(chi_id_pred)
        rho_in_noisy = qpt0.get_noisy_state(chi_id_pred)
        # chi_EM_a = qpt0.get_chi_LS_X(rho_in_noisy, qpt0.observables)
        # chi_EM_b = qpt0.get_chi_LS_X(qpt0.rho_in_idea, proj_noisy)
        FF_EM_a = 0#get_average(chi_EM_a, chi_idea,N)
        #print(f"Fidelity (with EMa): {FF_EM_a}")
        FF_EM_b = 0#get_average(chi_EM_b, chi_idea,N)
        #print(f"Fidelity (with EMb): {FF_EM_b}")
        chi_EM_c = qpt0.get_chi_LS_X(rho_in_noisy, proj_noisy)
        FF_noEM = get_chiF(chi_pred, chi_idea)
        FF_EM_c = get_chiF(chi_EM_c, chi_idea)
        #print(f"Fidelity (with EMc): {FF_EM_c}")
    else:
        chi_id_pred =chi_digital
        qpt0 = QPT(N, measure_data_gate, idea_channel)
        # (5) ML-QPT 
        proj_noisy = qpt0.get_noisy_proj_1(chi_id_pred)
        rho_in_noisy = qpt0.get_noisy_state(chi_id_pred)
        chi_EM_c = qpt0.get_chi_LS_X(rho_in_noisy, proj_noisy)
        if chi_fid==True:
            FF_noEM,FF_EM_a,FF_EM_b,FF_EM_c = 0,0,0,get_chiF(chi_EM_c, chi_idea)
        else:
            FF_noEM,FF_EM_a,FF_EM_b,FF_EM_c = 0,0,0,get_average(chi_EM_c, chi_idea,N)
    return FF_noEM,FF_EM_a,FF_EM_b,FF_EM_c

# def EM_QPT(N,idea_channel,measure_data_gate,measure_data_id):

#     random_channel = [idea_channel]
#     # (2) the identity QPT
#     qptid = QPT(N, measure_data_id, [np.identity(2 ** N)])
#     chi_id_pred = qptid.get_chi_LS_X(qptid.rho_in_idea, qptid.observables)

#     # (3) the standard QPT 
#     qpt0 = QPT(N, measure_data_gate, random_channel)
#     chi_pred = qpt0.get_chi_LS_X(qpt0.rho_in_idea, qpt0.observables) 
#     chi_idea = get_idea_chi_matrix(random_channel, N)
#     #print(Qobj(chi_pred),Qobj(chi_idea))
#     # (4) the error-mitigated  QPT 
#     proj_noisy = qpt0.get_noisy_proj_1(chi_id_pred)
#     rho_in_noisy = qpt0.get_noisy_state(chi_id_pred)

#     #rho_out_idea = [Qobj(np.sum([K @ rho.full() @ K.conj().T for K in random_channel], axis=0),dims=ddim,) for rho in qpt0.rho_in_idea]

#     # chi_EM_a = qpt0.get_chi_LS_X(rho_in_noisy, qpt0.observables)
#     # chi_EM_b = qpt0.get_chi_LS_X(qpt0.rho_in_idea, proj_noisy)
#     chi_EM_c = qpt0.get_chi_LS_X(rho_in_noisy, proj_noisy)

#     FF_noEM = get_average(chi_pred, chi_idea,N)
#     #print(f"Fidelity (no EM): {FF_noEM}")

#     FF_EM_a = 0#get_average(chi_EM_a, chi_idea,N)
#     #print(f"Fidelity (with EMa): {FF_EM_a}")

#     FF_EM_b = 0#get_average(chi_EM_b, chi_idea,N)
#     #print(f"Fidelity (with EMb): {FF_EM_b}")

#     FF_EM_c = get_average(chi_EM_c, chi_idea,N)
#     #print(f"Fidelity (with EMc): {FF_EM_c}")
#     return FF_noEM,FF_EM_a,FF_EM_b,FF_EM_c,chi_id_pred

def generate_valid_cptp_kraus_operators(n_qubits, num_kraus):


    """
    Generate a set of Kraus operators that satisfy the CPTP condition, ensuring ∑ K_i† K_i = I.

    Parameters:
    - n_qubits (int): Number of qubits
    - num_kraus (int): Number of Kraus operators to generate

    Returns:
    - list of numpy.ndarray: List of Kraus operators
    """
    # Define Pauli matrices for single qubits
    pauli = [qeye(2), sigmax(), sigmay(), sigmaz()]
    
    # Generate the tensor product of Pauli matrices for all possible combinations of qubits
    Elist = [tensor(*op).full() for op in itertools.product(pauli, repeat=n_qubits)]
    
    # Create Kraus operators by summing weighted Pauli matrices
    kraus_ops = [
        np.sum([(np.random.randn() + 1j * np.random.randn()) * E for E in Elist], axis=0)
        for _ in range(num_kraus)
    ]
    
    # Ensure the CPTP condition: ∑ K_i† K_i = I
    sum_kdag_k = sum(K.conj().T @ K for K in kraus_ops)
    sqrt_inv = np.linalg.inv(sqrtm(sum_kdag_k))
    
    # Normalize the Kraus operators to satisfy CPTP
    kraus_ops = [K @ sqrt_inv for K in kraus_ops]
    
    # Verify the CPTP condition again
    sum_kraus = sum(K.conj().T @ K for K in kraus_ops)
    
    if not np.allclose(sum_kraus, np.eye(2**n_qubits), rtol=1e-5):
        raise ValueError("The generated Kraus operators do not satisfy the CPTP condition!")
    
    # If only one Kraus operator is requested, generate a random unitary operator
    if num_kraus == 1:
        kraus_ops = [rand_unitary(2 ** n_qubits).full()]
    
    return kraus_ops

def get_idea_chi_matrix(random_channel, N):
    pauli = [qeye(2), sigmax(), sigmay(), sigmaz()]
    Elist = [tensor(*op) for op in product(pauli, repeat=N)]
    Elist = [E.full() for E in Elist]
    chi_exact = np.zeros((2**(2*N), 2**(2*N)), dtype=complex)

    for a in range(2**(2*N)):
        for b in range(2**(2*N)):
            Ea, Eb = Elist[a], Elist[b]
            chi_exact[a, b] = np.sum([np.trace(Ea @ gate) * np.trace(Eb.conj().T @ gate.conj().T) / 2**(2*N)
                                        for gate in random_channel])
    return chi_exact

def EM_id_chi(N,measure_data_id):
    # (2) the identity QPT
    qptid = QPT(N, measure_data_id, [np.identity(2 ** N)])
    chi_id_pred = qptid.get_chi_LS_X(qptid.rho_in_idea, qptid.observables)
    return chi_id_pred

# def get_chiF(chi1, chi2):
#     """Calculate the fidelity between two quantum processes represented by χ matrices."""
#     sqrt_chi1 = sqrtm(chi1)
#     return np.real(np.trace(sqrtm(sqrt_chi1 @ chi2 @ sqrt_chi1))**2)


# def get_exp_plot(measure_data):
    '''
    Visualizing the measurement data for a N-qubit gate
    '''
    rotation = ["I", "ry(90)", 'rx(90)']
    initial_state =['|0>','|+>','|->','|1>']
    observables =['|0><0|','|1><1|']
    cir_lab = list(product(initial_state,rotation))
    
    z = measure_data  
    th = 0
    ylabel = []
    plt.figure()
    fig, ax = plt.subplots(figsize=(4,8))
    ax.imshow(z)
    ax.set_xticks(np.arange(len(observables)), labels=observables)
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    for j in range(len(z)):
        for k in range(len(z[0])):
            text = ax.text(k,th, round(z[th,k], 3), ha="center", va="center", color="w")
        th+=1
        ylabel.append(cir_lab[j][0]+','+cir_lab[j][1])
    ax.set_yticks(np.arange(len(ylabel)), labels=ylabel)
    plt.show()