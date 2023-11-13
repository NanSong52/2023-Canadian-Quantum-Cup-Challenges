import json
import pennylane as qml
import pennylane.numpy as np

edges = [(0, 1), (1, 2), (2, 0), (2, 3)]
num_wires = 4

# We define the Hamiltonian for you!

ops = [qml.PauliZ(0), qml.PauliZ(1),qml.PauliZ(2), qml.PauliZ(3), qml.PauliZ(0)@qml.PauliZ(1), qml.PauliZ(0)@qml.PauliZ(2),qml.PauliZ(1)@qml.PauliZ(2),qml.PauliZ(2)@qml.PauliZ(3)]
coeffs = [0.5, 0.5, 1.25, -0.25, 0.75, 0.75, 0.75, 0.75]

cost_hamiltonian = qml.Hamiltonian(coeffs, ops)

# Write any helper functions you need here
optimal_value = np.min(np.linalg.eigvalsh(cost_hamiltonian.sparse_matrix().todense()))


dev = qml.device('default.mixed', wires = num_wires)

@qml.qnode(dev) 
def qaoa_circuit(params, noise_param):

    """
    Define the noisy QAOA circuit with only CNOT and rotation gates, with Depolarizing noise
    in the target qubit of each CNOT gate.

    Args:
        params(list(list(float))): A list with length equal to the QAOA depth. Each element is a list that contains 
        the two QAOA parameters of each layer.
        noise_param (float): The noise parameter associated with the depolarization gate

    Returns: 
        (np.tensor): A numpy tensor of 1 element corresponding to the expectation value of the cost Hamiltonian
    
    """


    # Put your code here #
    # Put your code here #
    for wire in range(num_wires):
        qml.RZ(np.pi/2, wires=wire)
        qml.RX(np.pi/2, wires=wire)
        qml.RZ(np.pi/2, wires=wire)
    # Apply the QAOA layers
    for gamma, beta in params:
        # Apply the cost layer for single-qubit Z terms with their respective coefficients
        qml.RZ(2 * gamma * 0.5, wires=0)
        qml.RZ(2 * gamma * 0.5, wires=1)
        qml.RZ(2 * gamma * 1.25, wires=2)
        qml.RZ(2 * gamma * -0.25, wires=3)

        # Apply the cost layer for two-qubit ZZ terms with their respective coefficients
        # Note that the evolution under ZZ is simulated using CNOT-RZ-CNOT sequences
        for edge, coeff in [([0, 1], 0.75), ([0, 2], 0.75), ([1, 2], 0.75), ([2, 3], 0.75)]:
            qml.CNOT(wires=[edge[0], edge[1]])
            qml.RZ(2 * gamma * coeff, wires=edge[1])
            qml.CNOT(wires=[edge[0], edge[1]])
            # Add noise on the target qubit
            qml.DepolarizingChannel(noise_param, wires=edge[1])
        # Apply the mixer layer with noisy CNOTs
        for i, (control_qubit, target_qubit) in enumerate(edges):
            qml.RX(2 * beta, wires=target_qubit)
            qml.RY(2 * 0.0001, wires=target_qubit)
            qml.RY(-2 * 0.0001, wires=target_qubit)
            # qml.CNOT(wires=[control_qubit, target_qubit])
            qml.DepolarizingChannel(noise_param, wires=target_qubit)

    # Return the expectation value of the cost Hamiltonian
    return qml.expval(cost_hamiltonian)


def approximation_ratio(qaoa_depth, noise_param):
    """
    Returns the approximation ratio of the QAOA algorithm for the Minimum Vertex Cover of the given graph
    with depolarizing gates after each native CNOT gate

    Args:
        qaoa_depth (float): The number of cost/mixer layer in the QAOA algorithm used
        noise_param (float): The noise parameter associated with the depolarization gate
    
    Returns: 
        (float): The approximation ratio for the noisy QAOA
    """
    # Put your code here #
    params = np.random.uniform(0, 2*np.pi, (qaoa_depth, 2))

    # Define the optimization step
    opt = qml.AdamOptimizer(stepsize=0.1)

    # Set the number of optimization steps
    steps = 500
    for i in range(steps):
        params, prev_val = opt.step_and_cost(lambda p: qaoa_circuit(p, noise_param), params)
        if i % 20 == 0:
            print(f"Step {i}: cost = {prev_val}")

    # Get the expectation value after optimization
    exp_val = qaoa_circuit(params, noise_param)

    # Compute the approximation ratio
    ratio = exp_val / optimal_value
    return ratio

    # Put your code here #


# These functions are responsible for testing the solution.
random_params = np.array([np.random.rand(2)])

ops_2 = [qml.PauliX(0), qml.PauliX(1), qml.PauliX(2), qml.PauliX(3)]
coeffs_2 = [1,1,1,1]

mixer_hamiltonian = qml.Hamiltonian(coeffs_2, ops_2)

@qml.qnode(dev)
def noiseless_qaoa(params):

    for wire in range(num_wires):

        qml.Hadamard(wires = wire)

    for elem in params:

        qml.ApproxTimeEvolution(cost_hamiltonian, elem[0], 1)
        qml.ApproxTimeEvolution(mixer_hamiltonian, elem[1],1)

    return qml.expval(cost_hamiltonian)

random_params = np.array([np.random.rand(2)])

circuit_check = (np.isclose(noiseless_qaoa(random_params) - qaoa_circuit(random_params,0),0)).numpy()

def run(test_case_input: str) -> str:
    input = json.loads(test_case_input)
    output = approximation_ratio(*input)

    return str(output)

def check(solution_output: str, expected_output: str) -> None:
    solution_output = json.loads(solution_output)
    expected_output = json.loads(expected_output)
    
    tape = qaoa_circuit.qtape
    names = [op.name for op in tape.operations]
    random_params = np.array([np.random.rand(2)])

    assert circuit_check, "qaoa_circuit is not doing what it's expected to."

    assert names.count('ApproxTimeEvolution') == 0, "Your circuit must not use the built-in PennyLane Trotterization."
     
    assert set(names) == {'DepolarizingChannel', 'RX', 'RY', 'RZ', 'CNOT'}, "Your circuit must use qml.RX, qml.RY, qml.RZ, qml.CNOT, and qml.DepolarizingChannel."

    assert solution_output > expected_output - 0.02


# These are the public test cases
test_cases = [
    ('[2,0.005]', '0.4875'),
    ('[1, 0.003]', '0.1307')
]

# This will run the public test cases locally
for i, (input_, expected_output) in enumerate(test_cases):
    print(f"Running test case {i} with input '{input_}'...")

    try:
        output = run(input_)

    except Exception as exc:
        print(f"Runtime Error. {exc}")

    else:
        if message := check(output, expected_output):
            print(f"Wrong Answer. Have: '{output}'. Want: '{expected_output}'.")

        else:
            print("Correct!")