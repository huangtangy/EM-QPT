
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

def EM_QPT(N,idea_channel,measure_data_gate,measure_data_id,chi_digital=[],chi_fid=True,notes=None):
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
        qptid = QPT(N, measure_data_id, [np.identity(2 ** N)],notes)
        chi_id_pred = qptid.get_chi_LS_X(qptid.rho_in_idea, qptid.observables)
        # (3) the standard QPT 
        qpt0 = QPT(N, measure_data_gate, idea_channel,notes)
        chi_pred = qpt0.get_chi_LS_X(qpt0.rho_in_idea, qpt0.observables) 
        # (4) the error-mitigated  QPT 
        proj_noisy = qpt0.get_noisy_proj_1(chi_id_pred)
        rho_in_noisy = qpt0.get_noisy_state(chi_id_pred)
        chi_EM_c = qpt0.get_chi_LS_X(rho_in_noisy, proj_noisy)
        FF_noEM = get_chiF(chi_pred, chi_idea)
        FF_EM_c = get_chiF(chi_EM_c, chi_idea)
        #print(f"Fidelity (with EMc): {FF_EM_c}")
    else:
        chi_id_pred =chi_digital
        qpt0 = QPT(N, measure_data_gate, idea_channel,notes)
        # (5) ML-QPT 
        proj_noisy = qpt0.get_noisy_proj_1(chi_id_pred)
        rho_in_noisy = qpt0.get_noisy_state(chi_id_pred)
        chi_EM_c = qpt0.get_chi_LS_X(rho_in_noisy, proj_noisy)
        if chi_fid==True:
            FF_noEM,FF_EM_c = 0,get_chiF(chi_EM_c, chi_idea)
        else:
            FF_noEM,FF_EM_c = 0,get_average(chi_EM_c, chi_idea,N)
    return FF_noEM,FF_EM_c
 

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

def EM_id_chi(N,measure_data_id,notes=None):
    # (2) the identity QPT
    qptid = QPT(N, measure_data_id, [np.identity(2 ** N)],notes)
    chi_id_pred = qptid.get_chi_LS_X(qptid.rho_in_idea, qptid.observables)
    return chi_id_pred

##====================== useful function in ML training============
import torch
import torch.nn.functional as F
def is_CPTP_chi_np(chi_matrix, N, tol=1e-5):
    """
    判断 NumPy Chi 矩阵是否是 CPTP
    """
    dim2 = chi_matrix.shape[0]
    dim = int(dim2 ** 0.5)

    pauli = [qeye(2), sigmax(), sigmay(), sigmaz()]
    pauli_basis = [tensor(*op) for op in product(pauli, repeat=N)]
    pauli_basis = [p.full() for p in pauli_basis]


    if not np.allclose(chi_matrix, chi_matrix.T.conj(), atol=tol):
        return False

    # 2. CP
    eigenvalues = np.linalg.eigvalsh(chi_matrix)
    # print('eigenvalues',eigenvalues)
    if np.any(eigenvalues < -tol):
        return False

    # 3. TP
    identity = np.eye(dim, dtype=np.complex64)
    trace_test = sum(chi_matrix[m, n] * pauli_basis[n].conj().T @ pauli_basis[m]
                     for m in range(dim2) for n in range(dim2))

    if not np.allclose(trace_test, identity, atol=tol * 100):
        print('trace perseving',np.allclose(trace_test, identity, atol=tol),trace_test)
        return False

    return True

def clean_cholesky(img):
    """
    清洗输入矩阵，得到用于 Cholesky 分解的矩阵 T

    Args:
        img (torch.Tensor): 形状为 (batch_size, hilbert_size, hilbert_size, 2)
                            的张量，表示神经网络的随机输出。
                            最后一个维度用于分离实部和虚部。

    Returns:
        T (torch.Tensor): 形状为 (N, hilbert_size, hilbert_size) 的张量，
                          表示 N 个用于 Cholesky 分解的矩阵（复数张量）。
    """
    # 分离实部和虚部
    real = img[:,0,:, :]
    imag = img[:,1,:, :]

    # 取虚部的对角线元素
    diag_all = torch.diagonal(imag, dim1=1, dim2=2)  # shape: (batch_size, hilbert_size)
    # 构造对角矩阵
    diags = torch.diag_embed(diag_all)

    # 将虚部对角线元素置零
    imag = imag - diags

    # 取下三角部分
    real = torch.tril(real)
    imag = torch.tril(imag)

    # 构造复数矩阵
    T = torch.complex(real, imag)
    return T

def density_matrix_from_T(tmatrix):
    """
    从 T 矩阵得到密度矩阵，并进行归一化

    Args:
        tmatrix (torch.Tensor): 形状为 (N, hilbert_size, hilbert_size)
                                 的张量，表示 N 个有效的 T 矩阵（复数张量）。

    Returns:
        rho (torch.Tensor): 形状为 (N, hilbert_size, hilbert_size) 的张量，
                            表示 N 个密度矩阵。
    """
    T = tmatrix
    # 计算共轭转置 T†
    T_dagger = T.transpose(-1, -2).conj()

    # 计算 T_dagger @ T
    proper_dm = torch.matmul(T_dagger, T)

    # 计算每个矩阵的迹
    traces = torch.einsum('bii->b', proper_dm)
    # 计算归一化因子 1/trace，并调整形状便于广播
    inv_traces = 1.0 / traces
    inv_traces = inv_traces.view(-1, 1, 1)

    # 归一化密度矩阵
    rho = proper_dm * inv_traces

    # rho = rho.view(rho.shape[0], rho.shape[1]*rho.shape[2])
    return rho

def reshape_expdata(data_I):
    exp_data_I =np.zeros(shape=(12,2)) 
    th = 0
    for i in range(4):
        for j in range(3):
            exp_data_I[th,:2]=data_I[:2,i,j]/np.sum(data_I[:2,i,j])
            th+=1
    return exp_data_I
