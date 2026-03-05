import numpy as np
import qiskit
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_state_city, plot_histogram
from qiskit.quantum_info import Kraus,Operator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_aer.noise import NoiseModel, QuantumError, ReadoutError,pauli_error, depolarizing_error, thermal_relaxation_error,amplitude_damping_error,coherent_unitary_error,phase_damping_error,mixed_unitary_error
from itertools import product
from qutip import qeye, sigmax, sigmay, sigmaz, tensor, ket2dm

import numpy as np
from itertools import product


class NumExp:
    def get_circuit(self, N, circ, circuit_gate,p_unitay, random_channel=None):
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

        # if p_unitay==0:
        #     factor = 1
        # else:
        #     factor =  1+(np.random.uniform()-0.5)*p_unitay #np.random.normal(loc=(1-p_unitay), scale=0.01, size=1)[0]
        # #print('factor:',factor)
        # theta = factor*(np.pi/2)

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
        result_noise = sim_noise.run(circ_tnoise, shots=10000).result()
        return result_noise.get_counts()

    def get_measurement(self, N, p_reset, p_meas,p_unitay, random_channel):
        gate_pre = ['I', '-RX', '-RY', 'X']
        gate_rot = ['I', 'RX', 'RY']

        # gate_pre = ['I', 'X', '-RY', 'RX']#qugate.ry(-np.pi / 2),qugate.rx(np.pi / 2)
        # gate_rot = ['I', '-RY','RX']

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

class QPTDataFromUnitaryQiskit:
    """
    Generate QPT "measure_data" compatible with your QPT class, from an arbitrary N-qubit unitary U.

    Design matches your QPT(notes!=None) branch:
      - input states (per qubit): |0>, |1>, |+x>, |+y>
      - measurement rotations (per qubit): I, Ry(-pi/2), Rx(+pi/2)
      - output layout: (4^N * 3^N, 2^N), input-major then rotation-major

    Bit-order:
      - Qiskit count/prob keys are shown as q_{N-1}...q0 (MSB on left).
      - Your QuTiP tensor convention is q0...q_{N-1} (q0 as MSB).
      - We reorder probabilities accordingly (bit-reversal on the index).
    """

    def __init__(
        self,
        N: int,
        outcome_perm=None,
        simulator_method: str = "density_matrix",
    ):
        if N < 1:
            raise ValueError("N must be >= 1")
        self.N = N
        self.d = 2**N
        self.outcome_perm = tuple(range(self.d)) if outcome_perm is None else tuple(outcome_perm)
        if len(self.outcome_perm) != self.d:
            raise ValueError(f"outcome_perm must have length 2^N={self.d}")

        self._sim = AerSimulator(method=simulator_method)

        # QPT design (notes!=None)
        self.input_1q = ["0", "1", "+x", "+y"]
        self.rots_1q = ["I", "Ry(-pi/2)", "Rx(pi/2)"]

        self.inputs = list(product(self.input_1q, repeat=N))      # 4^N
        self.rotations = list(product(self.rots_1q, repeat=N))    # 3^N

        # Permutation: Qiskit displayed index (q_{N-1}...q0) -> QuTiP tensor index (q0...q_{N-1})
        self._qiskit_index_for_qutip_index = np.array(
            [self._bit_reverse(k, N) for k in range(self.d)], dtype=int
        )

    # ---------------- public API ----------------
    def simulate_measure_data(self, U, shots=None):
        """
        U: can be
           - np.ndarray (2^N x 2^N) unitary
           - qiskit.quantum_info.Operator
           - qiskit.QuantumCircuit (must be N qubits and unitary)

        shots: None -> exact probabilities (density_matrix simulator)
               int  -> sampling to match finite-shot experiment

        returns: measure_data with shape (4^N * 3^N, 2^N)
        """
        Uop = self._to_operator(U)
        if Uop.dim[0] != self.d:
            raise ValueError(f"Unitary dimension mismatch: expected {self.d}, got {Uop.dim[0]}")

        rows = []
        for inp in self.inputs:
            for rot in self.rotations:
                p = self._one_setting_probs(Uop, inp, rot, shots=shots)
                rows.append(p)

        return np.vstack(rows)

    # ---------------- internals: circuit + probs ----------------
    def _one_setting_probs(self, Uop: Operator, inp_labels, rot_labels, shots=None):
        qc = QuantumCircuit(self.N, self.N)

        # prepare input
        for q, lab in enumerate(inp_labels):
            self._prep_1q(qc, q, lab)

        # apply unitary
        qc.append(Uop, list(range(self.N)))

        # apply measurement rotations (before Z measurement)
        for q, rot in enumerate(rot_labels):
            self._meas_rot_1q(qc, q, rot)

        # measure in Z
        qc.measure(list(range(self.N)), list(range(self.N)))

        if shots is None:
            qc2 = qc.copy()
            qc2.save_probabilities_dict()
            result = self._sim.run(qc2).result()
            pd = result.data(0)["probabilities_dict"]

            # p_qiskit indexed by displayed bitstrings q_{N-1}...q0: '0...0'..'1...1'
            p_qiskit = np.array(
                [pd.get(format(i, f"0{self.N}b"), 0.0) for i in range(self.d)],
                dtype=float,
            )
        else:
            result = self._sim.run(qc, shots=shots).result()
            counts = result.get_counts(0)
            p_qiskit = np.array(
                [counts.get(format(i, f"0{self.N}b"), 0) / shots for i in range(self.d)],
                dtype=float,
            )

        # reorder to QuTiP tensor order q0...q_{N-1}
        p_qutip = p_qiskit[self._qiskit_index_for_qutip_index]

        # apply outcome permutation if your pipeline expects it
        p_qutip = p_qutip[list(self.outcome_perm)]
        return p_qutip

    # ---------------- helpers ----------------
    @staticmethod
    def _prep_1q(qc, qubit, label):
        if label == "0":
            return
        if label == "1":
            qc.x(qubit)
            return
        if label == "+x":
            qc.h(qubit)
            return
        if label == "+y":
            qc.h(qubit)
            qc.s(qubit)  # H then S: |0> -> |+y>
            return
        raise ValueError(f"Unknown input label: {label}")

    @staticmethod
    def _meas_rot_1q(qc, qubit, rot):
        if rot == "I":
            return
        if rot == "Ry(-pi/2)":
            qc.ry(-np.pi / 2, qubit)
            return
        if rot == "Rx(pi/2)":
            qc.rx(np.pi / 2, qubit)
            return
        raise ValueError(f"Unknown rotation: {rot}")

    def _to_operator(self, U):
        if isinstance(U, Operator):
            return U
        if isinstance(U, QuantumCircuit):
            if U.num_qubits != self.N:
                raise ValueError(f"Circuit qubits mismatch: expected N={self.N}, got {U.num_qubits}")
            return Operator(U)
        # assume numpy array
        U = np.asarray(U, dtype=complex)
        if U.shape != (self.d, self.d):
            raise ValueError(f"U must have shape ({self.d},{self.d}), got {U.shape}")
        return Operator(U)

    @staticmethod
    def _bit_reverse(x: int, nbits: int) -> int:
        y = 0
        for _ in range(nbits):
            y = (y << 1) | (x & 1)
            x >>= 1
        return y
