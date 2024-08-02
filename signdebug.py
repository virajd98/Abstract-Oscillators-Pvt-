
from typing import cast
from classiq import *
from classiq.execution import ExecutionPreferences
import numpy as np
import matplotlib.pyplot as plt
from classiq.execution import ClassiqBackendPreferences


# @qfunc
# def init_state_phase(state: QNum):
#     state_in_qubit = QArray("state_in_qubit")
#     msb = QArray("msb", QBit)
#     size = np.log2(init_state.size)
#     allocate(size, msb)
#     bind(state, state_in_qubit)
#     repeat(state_in_qubit.len, lambda i: CX(state_in_qubit[i], msb[i]))
#     # control(msb[size-1], lambda: PHASE(-np.pi/2, state_in_qubit[size-1]))
#     bind(state_in_qubit, state)


@qfunc
def main(state: Output[QNum]):
    prepare_amplitudes(amplitudes=[0j, 0j, 0j, 0j, -0j, (1-0j), -0j, -0j], out=state, bound=0.01)
    # init_state_phase(state)


qmod = create_model(main)
backend_preferences = ClassiqBackendPreferences(backend_name="simulator_statevector")
model_pref = set_execution_preferences(qmod, ExecutionPreferences(num_shots=4000, backend_preferences=backend_preferences))
qprog = synthesize(model_pref)

job = execute(qprog)
parsed_state_vector= job.result()[0].value.parsed_state_vector  # to get the phases
print(parsed_state_vector)in