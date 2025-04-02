import numpy as np
import qiskit
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_state_city, plot_histogram
from qiskit.quantum_info import Kraus
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_aer.noise import NoiseModel, QuantumError, ReadoutError,pauli_error, depolarizing_error, thermal_relaxation_error,amplitude_damping_error,coherent_unitary_error,phase_damping_error,mixed_unitary_error
from itertools import product
from qutip import qeye, sigmax, sigmay, sigmaz, tensor, ket2dm

class NumExp:
    def get_circuit(self, N, circ, circuit_gate,p_unitay=0, random_channel=None):
        if p_unitay==0:
            factor = 1
        else:
            factor =  1+(np.random.uniform()-0.5)*p_unitay#np.random.normal(loc=(1-p_unitay), scale=0.01, size=1)[0]
        #print('factor:',factor)
        theta = factor*(np.pi/2)

        for n in range(N):
            gate = circuit_gate[0][N-n-1]
            
            if gate == 'I':
                circ.id(n)
            elif gate == 'X':
                circ.x(n)
            elif gate == 'RY':
                circ.ry(theta, n)
            elif gate == '-RY':
                circ.ry(-theta, n)
            elif gate == '-RX':
                circ.rx(-theta, n)
            elif gate == 'RX':
                circ.rx(theta, n)

        if random_channel:
            kraus_op = Kraus(random_channel)
            circ.append(kraus_op, range(N))

        for n in range(N):
            gate_ro = circuit_gate[1][N-n-1]
            if gate_ro == 'I':
                circ.id(n)
            elif gate_ro == '-RX':
                circ.rx(-theta, n)
            elif gate_ro == 'RX':
                circ.rx(theta, n)
            elif gate_ro == 'RY':
                circ.ry(theta, n)
            elif gate_ro == '-RY':
                circ.ry(-theta, n)

        return circ

    def get_results_qiskit(self, N, circuit_gate, p_reset, p_meas,p_unitay, random_channel=None):
        circ = QuantumCircuit(N)
        circ.reset(range(N))
        circ = self.get_circuit(N, circ, circuit_gate, p_unitay,random_channel)
        circ.measure_all()

        # Noise model setup
        noise_model = NoiseModel()
        # if p_reset>0 and p_reset>0:
        #     error_reset =pauli_error([('X', p_reset), ('I', 1 - p_reset)])#depolarizing_error(p_reset, 1)
        #     error_meas = pauli_error([('X', p_meas), ('I', 1 - p_meas)])#depolarizing_error(p_meas, 1)
        # else:
        #     p_reset,p_meas = np.abs(p_reset),np.abs(p_meas)
        #     error_reset =pauli_error([('I', p_reset), ('X', 1 - p_reset)])#depolarizing_error(p_reset, 1)
        #     error_meas = pauli_error([('I', p_meas), ('X', 1 - p_meas)])#depolarizing_error(p_meas, 1)
        error_reset =depolarizing_error(p_reset, 1)#pauli_error([('X', p_reset), ('I', 1 - p_reset)])#
        error_meas = depolarizing_error(p_meas, 1)#pauli_error([('X', p_meas), ('I', 1 - p_meas)])#

        noise_model.add_all_qubit_quantum_error(error_reset, "reset")
        noise_model.add_all_qubit_quantum_error(error_meas, "measure")

        # error_reset1 = amplitude_damping_error(p_reset)
        # error_meas1 = amplitude_damping_error(p_meas)

        # error_reset2 = phase_damping_error(p_reset)
        # error_meas2 = phase_damping_error(p_reset)

        # mixed_error_reset =  error_reset2.compose( error_reset1)
        # mixed_error_mea =  error_meas2.compose( error_meas1)

        # mixed_error_reset =  mixed_error_reset.compose( error_reset)
        # mixed_error_mea =  mixed_error_mea.compose( error_meas)

        # noise_model.add_all_qubit_quantum_error(mixed_error_reset, "reset")
        # noise_model.add_all_qubit_quantum_error(mixed_error_mea, "measure")

        # mixed_error_reset =  error_reset1.compose( error_reset)
        # mixed_error_mea =  error_meas1.compose( error_meas)

        # mixed_error_reset = error_reset2.compose(mixed_error_reset)
        # mixed_error_mea =error_meas2.compose( mixed_error_mea)
  

        # Noisy simulator
        sim_noise = AerSimulator(noise_model=noise_model)
        circ_tnoise = transpile(circ, sim_noise)
        result_noise = sim_noise.run(circ_tnoise, shots=4096).result()
        return result_noise.get_counts()

    def get_measurement(self, N, p_reset, p_meas,p_unitay, random_channel):
        gate_pre = ['I', '-RX', '-RY', 'X']
        gate_rot = ['I', 'RX', 'RY']
        mea_basi0 = list(product(['0', '1'], repeat=N))

        psi_in_str = list(product(gate_pre, repeat=N))
        rotation_str = list(product(gate_rot, repeat=N))
        str_list = list(product(psi_in_str, rotation_str))

        nois_measure = []
        for circuit in str_list:
            noisy_result = self.get_results_qiskit(N, circuit, p_reset, p_meas,p_unitay, random_channel)
            nos_res = [noisy_result.get(mea, 0) for mea in [''.join(mea) for mea in mea_basi0]]
            nois_measure.append(np.array(nos_res) / sum(nos_res))
        
        return np.array(nois_measure)

    def get_idea_chi_matrix(self, random_channel, N):
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
