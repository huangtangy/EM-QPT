#=============================================================================
#==============This is the main class for the EM-QPT==========================
#=============================================================================
#=====python packages are recommended in 'requirements.txt'.

import numpy as np
import cvxpy as cp
import qutip as qt
from itertools import product
from qutip.qip.operations.gates import rx, ry
from qutip import Qobj
import itertools,h5py,scipy,jax,pickle
from qutip import basis, qeye, sigmax, sigmay, sigmaz, tensor, ket2dm, Qobj, fidelity, rand_ket
#import qutip.qip as qugate
import qutip.qip.operations.gates as qugate
from itertools import product
from qutip.random_objects import rand_super_bcsz, rand_kraus_map, rand_unitary
import jax
from jax import numpy as jnp
import cvxpy as cp

class QPT():
    def __init__(self, N, measure_data,random_channel=None,notes=None):  # ,Amatrix,psi_in_idea,obervables
        """
        :param N: Number of qubits
        :param measure_data: Measurement data (outcomes)
        :param random_channel: Random channel for generating Kraus operators (optional)
        """
        # introducing the dimension of N-qubit system while using the Qutip class (Qobj)
        

        self.N = N
        self.measure_data = measure_data

        # Setup dimensions based on N
        self.dim = [[2] * N] * N if N > 1 else [[2], [2]]

        # Pauli basis and tensor products
        pauli = [qeye(2), sigmax(), sigmay(), sigmaz()]
        self.pauli_sys = [tensor(*op) for op in product(pauli, repeat=N)]
        zero, one = basis(2, 0), basis(2, 1)

        #=========================================================================================================
        # ==========#(single-qubit gate and numeric) Prepare state rotations and input states=====================
        #=========================================================================================================
        if notes==None:
            stat_rot = [qeye(2), qugate.rx(-np.pi / 2), qugate.ry(-np.pi / 2), qugate.ry(np.pi)]
            inputst = [ro * zero for ro in stat_rot]
            psi_in_idea = [tensor(*psi) for psi in product(inputst, repeat=N)]
            # Rotation matrices for measurement
            rotation = [qeye(2), qugate.rx(np.pi / 2), qugate.ry(np.pi / 2)]
            U_rotation = [tensor(*rot) for rot in product(rotation, repeat=N)]
        else:
        #=========================================================================================================
        # ================ the CZ setup ==========================================================================  
        #=========================================================================================================
            # plus, mins = (zero + one).unit(), (zero + 1j * one).unit()  # |+x> state # |-y> state Rxy(90,0);Rxy(90,90)
            # inputst = [zero, one, plus, mins]
            # psi_in_idea = [tensor(*psi) for psi in product(inputst, repeat=N)]
            # rotation = [qeye(2), qugate.ry(np.pi / 2 * (-1)), qugate.rx(np.pi / 2)]  
            # U_rotation = [tensor(*rot) for rot in product(rotation, repeat=N)]


            #=========================for Liangyu's data============================
            # 16 idea rho_in (Liangyu's data)
            zero, one = basis(2, 0), basis(2, 1)  # |0>  |1>state
            plus, mins = (zero -one).unit(),(zero - 1j* one).unit()   # |+x> state # |-y> state Rxy(90,0);Rxy(90,90)
            #plus, mins = (zero - 1j*one).unit() ,(zero + one).unit() 
            inputst = [zero, one, plus, mins]
            psi_in_idea = [tensor(*psi) for psi in product(inputst, repeat=N)]
            rotation = [qeye(2),  qugate.ry(-np.pi / 2), qugate.rx(np.pi / 2 * (1))]

            # zero, one = basis(2, 0), basis(2, 1)  # |0>  |1>state
            # plus, mins = (zero - 1j*one).unit() ,(zero + one).unit()   # |+x> state # |-y> state Rxy(90,0);Rxy(90,90)
            # inputst = [zero, one, plus, mins]
            # psi_in_idea = [tensor(*psi) for psi in product(inputst, repeat=N)]
            # rotation = [qeye(2), qugate.rx(np.pi / 2 * (1)), qugate.ry(np.pi / 2)]
            U_rotation = [tensor(*rot) for rot in product(rotation, repeat=self.N)] # 9 rotations
            

        # Measurement projectors
        e_sys = [tensor(*state) for state in product([basis(2, i) for i in range(2)], repeat=N)]
        proj_qst = [ket2dm(e_sys[i]) for i in range(len(e_sys))]
        self.observables=  [U_rotation[m].dag() * proj_qst[n] * U_rotation[m] for m in range(len(U_rotation)) for n in range(len(proj_qst))]

        # Ideal states and corresponding outputs
        self.rho_in_idea = [ket2dm(psi_in_idea[i]) for i in range(len(psi_in_idea))]
        self.rho_out_idea = [
            Qobj(np.sum([K @ rho.full() @ K.conj().T for K in random_channel], axis=0), dims=self.dim)
            for rho in self.rho_in_idea
        ]

    def get_chi_LS_X(self,rho_in_list,proj_list):
        '''
        :input: the input states and projs which are revised or perfect
        :return: the chi-matrix estimated by given SPAM and measurement results
        '''
       
        dim_chi = 2 ** (2 * self.N)  # the dim of chi
        E, N = self.pauli_sys, self.N
        rho_in_nos, proj_list = rho_in_list,proj_list#rho_in_list, self.obervables  # the input noisy states

        Ob = self.measure_data.reshape(len(proj_list), len(rho_in_nos))
        Ob = np.array(Ob).reshape(len(rho_in_nos) * len(proj_list))
        coeff_with_spam = np.empty(shape=(len(rho_in_nos) * len(proj_list), (2 ** (2 * N)) ** 2), dtype=complex)  # the corresponding experimental measurement

        # the A matrix
        row = 0
        for i in range(len(rho_in_nos)):
            for j in range(len(proj_list)):
                col = 0
                for m in range(dim_chi):
                    for n in range(dim_chi):
                        coeff_with_spam[row, col] = (E[m] * rho_in_nos[i] * E[n].dag() * proj_list[j]).tr()
                        col += 1
                row += 1

        X1 = cp.Variable((4 ** N, 4 ** N), hermitian=True)
        chi1 = cp.reshape(X1.T, ((4 ** N) ** 2,),order='F')# default order in different version of cp, the numpy default is order='C'

        obj1 = cp.Minimize(cp.norm(coeff_with_spam @ chi1 - Ob, 2))
        #constraints1 = [X1 >> 0, cp.trace(X1) == 1]
        d = rho_in_nos[0].shape[0]  
        I_d = np.eye(d, dtype=complex)
        
        tp_expr = 0
        for m in range(dim_chi):
            for n in range(dim_chi):
                # E[n].dag() * E[m] 
                tp_expr += X1[m, n] * (E[n].dag() * E[m]).full()  
        
        constraints1 = [
            X1 >> 0,
            tp_expr == I_d   
        ]

        prob1 = cp.Problem(obj1, constraints1)
        prob1.solve(solver=cp.SCS,eps=1e-8,warm_start= True);  # , verbose=True#cp.SCScp.OSQP#,eps=1e-10 ,eps=1e-10 ,alpha=2.0 

        return X1.value
 
    def get_noisy_state(self, chi_matrix):
        """
        Estimates the chi-matrix using least squares optimization.
        :param rho_in_list: List of input states
        :param proj_list: List of measurement projectors
        :return: Optimized chi matrix
        """
        N = self.N
        dim_chi = 2 ** (2 * N)
        rho_in_idea = self.rho_in_idea
        E = self.pauli_sys  # 16 pauli basis
        observables = self.observables
        idea_measure_proj = self.measure_data.reshape(len(self.rho_in_idea),len(observables))  # 原始数据是这样的

        # with SPAM A using 16 states
        rho_out_list = []
        for i in range(len(rho_in_idea)):
            row = 0
            coeff_with_spam = np.empty((len(observables), dim_chi ** 2), dtype=complex)
            for j in range(len(observables)):
                col = 0
                for m in range(dim_chi):
                    for n in range(dim_chi):
                        # print(i,j,rho_in_idea[i], obervables[j])
                        coeff_with_spam[row, col] = (E[m] * rho_in_idea[i] * E[n].dag() * observables[j]).tr()
                        col += 1
                row += 1

            Ob = idea_measure_proj[i]

            # chi matrix including spam using 16 stats
            X1 = cp.Variable((2 ** N, 2 ** N), hermitian=True)  # the density matrix
            X1_vec = cp.reshape(X1, ((2 ** N) ** 2,),order='F')  # vecterize rho
            chi_matrix = np.reshape(chi_matrix, (dim_chi ** 2,))  # the chi matrix
            #print(np.shape(observables),(len(observables), dim_chi))
            # notes: if some bug on the reshape, try this :
            #observables1 = np.array([ob.full() for ob in observables])
            #obervables_vec = observables1.reshape( (len(observables), dim_chi))
            obervables_vec = np.reshape(observables, (len(observables), dim_chi))
            #print('obervables_vec',obervables_vec)
            obj1 = cp.Minimize(cp.norm(coeff_with_spam @ chi_matrix - X1_vec @ obervables_vec.T,2))  # acting the chi matrix on the idea states
            #obj1 = cp.Minimize(cp.norm(Ob - X1_vec @ obervables_vec.T, 2)) # using thr raw measurement data instead of chi (really small difference)
            constraints1 = [X1 >> 0, cp.trace(X1) == 1]

            prob1 = cp.Problem(obj1, constraints1)
            prob1.solve(solver=cp.SCS,eps=1e-6,warm_start= True);  # , verbose=True max_iters=5000,

            rho_estimate = Qobj(X1.value, dims=self.dim)
            rho_out_list.append(rho_estimate)
        return rho_out_list
 
    def get_noisy_proj_1(self,  chi_matrix):
        '''
        # chi_matrix: for a given identity-chi matrix, return the revised POVM projectors while we assume the projector is perfect
        '''

        N = self.N
        dim_chi = 2 ** (2 * N)
        proj = self.observables
        #EE = pauli_basis
        rho_in_idea = self.rho_in_idea

        # ==================== get noisy projector =================#

        proj_out_list = []
        for j in range(3**N): #3^N groups of POVMs
            coeff_with_spam = np.empty((2 ** N, len(rho_in_idea), dim_chi ** 2), dtype=complex)
            h = 0
            for jj in range(j * 2 ** N, (j + 1) * 2 ** N):
                row = 0
                #print('h', h)
                for i in range(len(rho_in_idea)):
                    col = 0
                    for m in range(dim_chi):
                        for n in range(dim_chi):
                            coeff_with_spam[h, row, col] = ( self.pauli_sys [m] * rho_in_idea[i] *  self.pauli_sys [n].dag() * proj[jj]).tr()
                            col += 1
                    row += 1
                h += 1
            # define the POVMs
            d,n = 2 ** N,2**N  #  the dimension and number of POVM in one group
            E = [cp.Variable((d, d), complex=True) for _ in range(n)] # the variable is a set of POVMs
            # the condition of POVMs
            constraints = []
            for i in range(n):
                #constraints.append(E[i] == E[i].H)  # Hermitian 
                constraints.append(E[i] >> 0)  # 1. (E_i ⪰ 0) the positive
            constraints.append(sum(E) == np.eye(d))# 2. the completeness

            # 定义 Loss function
            chi_matrix = np.reshape(chi_matrix, (dim_chi ** 2,))  # the chi matrix
            rho_list = [rho.full().T for rho in rho_in_idea]  # 特别注意：这里是后面等效计算trace的前提就是 tr(AB)= vec(A)@vec(B.T)
            rho_vec = np.reshape(rho_list, (len(rho_in_idea), dim_chi))

            loss = sum(cp.norm(coeff_with_spam[i, :, :]@chi_matrix - cp.reshape(E[i], ((2 ** N) ** 2,), order='F')@rho_vec.T, 2) for i in range(n))
            prob1 = cp.Problem(cp.Minimize(loss), constraints)  #
            prob1.solve(solver=cp.SCS,eps=1e-6,warm_start= True);  # , verbose=True max_iters=50000,

            optimized_matrices = [Ei.value for Ei in E]
            for M in optimized_matrices:
                proj_out_list.append(Qobj(M.conj(), dims=self.dim))

        return proj_out_list
