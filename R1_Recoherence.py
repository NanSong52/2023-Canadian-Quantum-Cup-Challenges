import json
import pennylane as qml
import pennylane.numpy as np

dev = qml.device("default.qubit", wires=5)


@qml.qnode(dev)
def evolve_state(coeffs, time):
    """
    Args:
        coeffs (list(float)): A list of the coupling constants g_1, g_2, g_3, and g_4
        time (float): The evolution time of th system under the given Hamiltonian

    Returns:
        (numpy.tensor): The density matrix for the evolved state of the central spin.
    """

    # We build the Hamiltonian for you

    operators = [
        qml.PauliZ(0) @ qml.PauliZ(1),
        qml.PauliZ(0) @ qml.PauliZ(2),
        qml.PauliZ(0) @ qml.PauliZ(3),
        qml.PauliZ(0) @ qml.PauliZ(4),
    ]
    hamiltonian = qml.dot(coeffs, operators)


    # Put your code here #

    # Return the required density matrix.

    
    # Return the required density matrix.
    qml.RY(0.5 * np.pi, wires=0)
    qml.RY(0.4 , wires=1)
    qml.RY(1.2 , wires=2)
    qml.RY(1.8 , wires=3)
    qml.RY(0.6 , wires=4)
    hamiltonian_prime = qml.Hamiltonian(coeffs, operators)
    qml.CommutingEvolution(hamiltonian_prime, time)

    return qml.density_matrix(wires=0)

def purity(rho):
    """
    Args:
        rho (array(array(complex))): An array-like object representing a density matrix

    Returns:
        (float): The purity of the density matrix rho

    """


    # Put your code here

    # Return the purity

    rho2 = np.dot(rho, rho)
    return np.trace(rho2).real

def recoherence_time(coeffs):
    """
    Args:
        coeffs (list(float)): A list of the coupling constants g_1, g_2, g_3, and g_4.

    Returns:
        (float): The recoherence time of the central spin.

    """


    max_iterations = 10000  # Define a maximum number of iterations
    t = 0.1

    iteration = 0
    while iteration < max_iterations:
        rho = evolve_state(coeffs, t)  # Make sure evolve_state is defined
        if np.isclose(purity(rho), 1, rtol=1e-2):  # Make sure purity is defined
            break
        t += 0.01  # Smaller increment
        iteration += 1
    return t

# These functions are responsible for testing the solution.
def run(test_case_input: str) -> str:
    params = json.loads(test_case_input)
    output = recoherence_time(params)

    return str(output)


def check(solution_output: str, expected_output: str) -> None:
    solution_output = json.loads(solution_output)
    expected_output = json.loads(expected_output)

    assert np.isclose(solution_output, expected_output, rtol=5e-2)


# These are the public test cases
test_cases = [
    ('[5,5,5,5]', '0.314'),
    ('[1.1,1.3,1,2.3]', '15.71')
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