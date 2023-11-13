import json
import pennylane as qml
import pennylane.numpy as np

def U(n_wires, label):

    """
    This function will define the gate U. It should not return anything,
    just include the operators that define the gate.

    Args:
        n_wires (int): Number of wires
        label (int): Number of mountains that we believe the function has

    """
    # Put your code here #

    # qml.AmplitudeEmbedding(phi, wires = range(n_wires-1))
    qml.QFT(wires=range(n_wires-1))
    # return qml.state()
    label_binary = bin(label)[2:]
    # fill more zeros infront such that the length matches n_wires-1
    label_binary = '0'*(n_wires-1-len(label_binary)) + label_binary
    qml.MultiControlledX(wires = range(n_wires), control_values=label_binary)
    # also collect the anti-binary
    label_binary = bin(2**(n_wires-1)-label)[2:]
    # fill more zeros infront such that the length matches n_wires-1
    label_binary = '0'*(n_wires-1-len(label_binary)) + label_binary
    qml.MultiControlledX(wires = range(n_wires), control_values=label_binary)


# These functions are responsible for testing the solution.
def run(test_case_input: str) -> str:

    inputs = json.loads(test_case_input)
    n_wires = int(inputs[0])
    phi = np.array(inputs[1])
    label = int(inputs[2])

    dev = qml.device("default.qubit", wires = n_wires)
    @qml.qnode(dev)
    def circuit():
        qml.AmplitudeEmbedding(phi, wires = range(n_wires-1))
        U(n_wires, label)
        return qml.probs(wires = n_wires-1)

    return str(circuit()[0].numpy())


def check(solution_output: str, expected_output: str) -> None:
    
    solution_output = json.loads(solution_output)

    assert(np.isclose(solution_output, 0.) or np.isclose(solution_output, 1.)), "Make sure that with one shot you always get the same output"

    if np.isclose(solution_output, 0.):
        assert expected_output == "Well labeled", "The function did not predict the result correctly"

    if np.isclose(solution_output, 1.):
        assert expected_output == "Mislabeled", "The function did not predict the result correctly"


# These are the public test cases
test_cases = [
    ('[0, 0.7853981633974483, 0.25, 0.75]', '0.8952847075210476'),
    ('[1.83259571459, 1.88495559215, 0.5, 0.5]', '0.52616798')
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