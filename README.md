**Simulating Classical Coupled Harmonic Oscillators Using [Classiq](https://www.classiq.io/)**

In recent years, Quantum Computing has made remarkable strides, with increasingly powerful quantum computers being developed annually. Today, machines with hundreds of qubits 
can execute quantum algorithms with circuit depths reaching into the thousands, all while preserving a significant signal. However, a key challenge within the quantum computing 
ecosystem is the development of efficient and innovative quantum algorithms that are not only practical but also offer exponential advantages over classical methods.

One of the most promising areas of research in quantum algorithms is Hamiltonian simulation, particularly in simulating classical coupled harmonic oscillators. Our project aims to
bridge the gap between theory and practice by translating the theoretical framework presented in the **2023** study by R. Babbush, D. W. Berry, R. Kothari, R. D. Somma, and N. Wiebe,
*Exponential Quantum Speedup in Simulating Coupled Classical Oscillators*, published in Phys. Rev. X 13, 041041 (2023), into a practical implementation compatible with current quantum 
hardware and software capabilities.

The objective of our [Womanium Quantum+AI 2024](https://womanium.org/Quantum/AI) Final Project is to implement, optimize, and evaluate the simulation of classical coupled harmonic 
oscillators as detailed in the study. You can access the original paper [here](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.13.041041).

## Table of Contents

1. **Implementation of toy problem** (the simplest possible cases) that covers:
- Encoding of the problem.
- The key algorithmic building blocks
- The readout and post-processing.
- All theoretical details 
<img src="https://github.com/user-attachments/assets/0e59441e-b1e3-4f41-8410-d5be388b8bde" alt="Description" width="175">

Refer [here](https://github.com/virajd98/Abstract-Oscillators-Pvt-/blob/main/ToySuzuki.ipynb) and [here](https://github.com/virajd98/Abstract-Oscillators-Pvt-/blob/main/ToyExponentiation.ipynb)


The implementation should be scalable, such that it is clear how to extend it for a more complicated scenario, and it should be checked and tested using
a simulator.

2) **Enlarging the problem for a more complicated scenario
In this step, the actual problem from the paper has been implemented. We attempt to solve Problem 2 of estimating kinetic energy of system of Coupled Oscillators. Our codes are general, and for demonstration we simulate for 8 masses. Refer [here](https://github.com/virajd98/Abstract-Oscillators-Pvt-/blob/main/KineticEnergyEstimationProblem2.ipynb)

Resources estimation in terms of circuit depth, circuit width and number of 2-qubit gates should be made and compared across several hardwares has been done. 

3) **Optimize the solution for the most adequate hardware** that was found in the second step. 
 
Deliverables:
● Slides that summarize the work
● **The .qmod and .qprog files for each step.** This can be found in the folder named [Figures](https://github.com/virajd98/Abstract-Oscillators-Pvt-/tree/main/Figures)
● The Python Jupyter notebooks of each step

## Installation and Usage

Please fork the repository and clone the fork to your local device. Certain files use other files which has helper functions. 

The project uses the following packages: classiq (0.43.3), scipy, numpy, matplotlib, tqdm, qutip (optional).

For installation run the following on codecell of jupyter notebook

`!pip install <package-name>` 


## Contributing

If you wish to contribute to this repository, please fork the repository, make your contributions in your fork, and submit a pull request


## License

This project is licensed under the MIT License - see the [LICENSE](MIT-LICENSE.txt) file for details.

## Bibliography:

1. R. Babbush, D. W. Berry, R. Kothari, R. D. Somma, and N. Wiebe, *Exponential Quantum Speedup in Simulating Coupled Classical Oscillators*, published in Phys. Rev. X 13, 041041 (2023), https://doi.org/10.1103/PhysRevX.13.041041, [link to paper](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.13.041041)
2. G.H. Low, I.L. Chuang, *Hamiltonian simulation by Qubitization*, published in Quantum, 3 (2019), p. 163, 	https://doi.org/10.22331/q-2019-07-12-163, [link to paper](https://quantum-journal.org/papers/q-2019-07-12-163/)
3. András Gilyén, Yuan Su, Guang Hao Low, and Nathan Wiebe. 2019. Quantum singular value transformation and beyond: exponential improvements for quantum matrix arithmetics. In Proceedings of the 51st Annual ACM SIGACT Symposium on Theory of Computing (STOC 2019). Association for Computing Machinery, New York, NY, USA, 193–204. https://doi.org/10.1145/3313276.3316366,[link to paper](https://dl.acm.org/doi/10.1145/3313276.3316366)
4. [Classiq Github Hamiltonian Qubitization](https://github.com/Classiq/classiq-library/tree/9c43f05f3d498c8c72be7dcb3ecdaba85d9abd6e/tutorials/hamiltonian_simulation/hamiltonian_simulation_with_block_encoding)
5.  [Classiq Github Glued Trees](https://github.com/Classiq/classiq-library/blob/9c43f05f3d498c8c72be7dcb3ecdaba85d9abd6e/algorithms/glued_trees/glued_trees.ipynb#L4)
6. [Classiq documentation](https://docs.classiq.io/latest/)
 

## Acknowledgments

- A heartfelt gratitude to [Womanium Team](https://womanium.org/Quantum/AI) for designing & organizing this program, and offering scholarships. 
- Special thanks to [Eden Shirman](https://www.linkedin.com/in/eden-schirman-71bb7a1b9/?originalSubdomain=il), [Tomer Goldfriend](https://www.linkedin.com/in/tomer-goldfriend-3422341b2/), and [everyone at Classiq](https://app.slack.com/client/T04KVKJKKFY/search).
- This project uses [Classiq Github](https://github.com/Classiq/classiq-library/tree/main) by [Classiq](https://www.classiq.io/).
![image](https://github.com/user-attachments/assets/71d911d2-f9f3-4ff8-88d0-be8e894334c3)

