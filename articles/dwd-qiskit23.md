---
title: "Qiskit で遊んでみる (23) — 最新の Qiskit で量子化学計算"
emoji: "🪐"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["Qiskit", "Python"]
published: false
---

# 目的

[IBM Quantum Challenge Fall 2022](https://www.ibm.com/blogs/think/jp-ja/quantum-challenge-fall-2022/) というイベントが 2022 年の冬に開催され、その中で

> Lab 4：量子化学

というのがあった。水素原子 H についての問題があるが、これはざっくりとはハミルトニアン

$$
\begin{align*}
\mathcal{H} = - \frac{\hbar^2}{2 \mu} \Delta - \frac{Z}{|x|}
\end{align*}
$$

に対して $\mathcal{H} \psi = E \psi$ を解きたいというものである。特に基底状態の固有値 $E_1$ に興味がある場合、

- $\mathcal{H}$ を第二量子化して、
- Jordan-Wigner 変換とかを通してスピン演算子形式に変更した後に（これも $\mathcal{H}$ と書く）、
- 適当な ansatz（今回は UCCSD ansatz を用いる）$\ket{\psi (\theta)} = U(\theta) \ket{0}^{\otimes n}$ を使って、

$$
\begin{align*}
f(\theta) = \braket{\psi (\theta) | \mathcal{H} | \psi (\theta)} \geq E_1
\end{align*}
$$

を最小化すると、$\inf \braket{\psi (\theta) | \mathcal{H} | \psi (\theta)} = E_1$ が近似的に成立することが期待される。

— という、わりと浪漫のありそうな内容だが、実は最新の Qiskit でかつてのコードでは動かなくなっているので、何とか動かしたいというのが目的である。

# かつてのコード

問題製作者側の解答が入ったノートブックが [solutions-by-authors/lab-4](https://github.com/qiskit-community/ibm-quantum-challenge-fall-22/tree/main/solutions-by-authors/lab-4) にあるので有難く使わせてもらう。Apache-2.0 license らしい。

```sh
%%bash

pip install "qiskit==1.0.2" "qiskit[visualization]==1.0.2"
pip install "qiskit_aer==0.13.3" "qiskit_algorithms==0.3.0" "qiskit_nature==0.7.2"
pip install "pyscf==2.5.0"
pip list | grep qiskit
```

> qiskit                           1.0.2
> qiskit-aer                       0.13.3
> qiskit-algorithms                0.3.0
> qiskit-nature                    0.7.2

の環境下では、幾つかの API やモジュールが deprecation してしまって動かない。

[(最小) EigensolverFactory マイグレーション・ガイド](https://qiskit-community.github.io/qiskit-nature/locale/ja_JP/migration/0.6_b_mes_factory.html) といった移行ガイドや [問題の変換](https://qiskit-community.github.io/qiskit-nature/locale/ja_JP/tutorials/05_problem_transformers.html) を参考に書き換えた。

ほとんどの変更箇所は `construct_problem` の中にある。

# アップデート後のコード

とりあえず以下で動くようになった。

まずは以下くらいを import すれば良いらしい。色々なものが `qiskit_nature.second_q` の下にまとめられたようなのでそれに従う。

```python
# Import necessary libraries and packages
import math
import matplotlib.pyplot as plt
import numpy as np

import warnings
warnings.filterwarnings('ignore')

from qiskit import QuantumCircuit
from qiskit.primitives import Estimator
from qiskit_aer import StatevectorSimulator

# Import Qiskit libraries for VQE
from qiskit_algorithms import MinimumEigensolverResult, VQE
from qiskit_algorithms.optimizers import SLSQP, SPSA
from qiskit_algorithms.utils import algorithm_globals
algorithm_globals.random_seed = 1024

# Import Qiskit Nature libraries
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_nature.second_q.algorithms.initial_points import HFInitialPoint
from qiskit_nature.second_q.circuit.library import UCC, UCCSD, HartreeFock
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import (
    BravyiKitaevMapper, JordanWignerMapper, ParityMapper
)
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer

from qiskit_nature.settings import settings

settings.dict_aux_operators = True
```

## 1. Helper Function for constructing problem

新しい流儀では `PySCFDriver` を直接使うようなので、移行ガイドや当時のコードでのパラメータの渡し方を参考に書き直す。元々は `ParityMapper` が使われていたが、ためしに `JordanWignerMapper` を使ってみた。

```python
def construct_problem(
    geometry, charge, multiplicity, basis, num_electrons, num_molecular_orbitals
):

    # 'H 0.0 0.0 0.0; H 0.0 0.0 0.735' のような形式に変換
    atom = "; ".join(
        [" ".join([atm[0]] + [str(v) for v in atm[1]])
            for atm in geometry]
    )
    # https://github.com/qiskit-community/qiskit-nature/blob/0.5.2/qiskit_nature/drivers/second_quantization/pyscfd/pyscfdriver.py#L335
    spin = multiplicity - 1
    driver = PySCFDriver(atom=atom, charge=charge, spin=spin, basis=basis)

    # Run the preliminary quantum chemistry calculation
    problem = driver.run()

    # Set the active space
    active_space_trafo = ActiveSpaceTransformer(
        num_electrons=num_electrons,
        num_spatial_orbitals=num_molecular_orbitals
    )

    # Now you can get the reduced electronic structure problem
    problem_reduced = active_space_trafo.transform(problem)

    # The second quantized Hamiltonian of the reduce problem
    second_q_ops_reduced = problem_reduced.hamiltonian.second_q_op()

    # Set the mapper to qubits
    # JordanWignerMapper を使ってみた
    mapper = JordanWignerMapper()

    # Compute the Hamitonian in qubit form
    qubit_op = mapper.map(second_q_ops_reduced)
    aux_ops = {}
    aux_ops.update(
        mapper.map(problem_reduced.properties.particle_number.second_q_ops())
    )
    aux_ops.update(
        mapper.map(problem_reduced.properties.angular_momentum.second_q_ops())
    )

    ansatz = UCCSD(
        problem_reduced.num_spatial_orbitals,
        problem_reduced.num_particles,
        mapper,
        initial_state=HartreeFock(
            problem_reduced.num_spatial_orbitals,
            problem_reduced.num_particles,
            mapper,
        ),
    )

    initial_point = HFInitialPoint()
    initial_point.ansatz = ansatz
    initial_point.problem = problem_reduced

    solver = VQE(Estimator(), ansatz, SLSQP())
    solver.initial_point = initial_point.to_numpy_array()
    result = solver.compute_minimum_eigenvalue(qubit_op, aux_ops)
    real_solution = result.optimal_value

    return ansatz, qubit_op, real_solution, problem_reduced
```

## 2. Helper function for Running VQE

ここはオーソドックスな VQE の最適化の計算部分で、特に変更する必要がなかったのでそのまま使った。最近の流儀である `Estimator` を使って期待値計算をして `SPSA` でパラメータの最適化を行っている。クラウド環境でやる時の話かもしれないが、[EstimatorV2](https://docs.quantum.ibm.com/api/qiskit-ibm-runtime/qiskit_ibm_runtime.EstimatorV2) というのが導入されるらしいので、また何か影響を受けるかもしれない。

```python
def custom_vqe(
    estimator, ansatz, ops, problem_reduced, optimizer = None,
    initial_point=None,
):

    # Define convergence list
    convergence = []

    # Keep track of jobs (Do-not-modify)
    job_list = []

    # Define evaluate_expectation function
    def evaluate_expectation(x):
        x = list(x)

        # Define estimator run parameters
        job = estimator.run(
            circuits=[ansatz], observables=[ops], parameter_values=[x]
        ).result()
        results = job.values[0]
        job_list.append(job)

        # Pass results back to callback function
        return np.real(results)

    # Call back function
    def callback(x,fx,ax,tx,nx):
        # Callback function to get a view on internal states and statistics of the optimizer for visualization
        convergence.append(evaluate_expectation(fx))

    np.random.seed(10)

    # Define initial point. We shall define a random point here based on the number of parameters in our ansatz
    if initial_point is None:
        initial_point = np.random.random(ansatz.num_parameters)

    # Define optimizer and pass callback function
    if optimizer == None:
        optimizer = SPSA(maxiter=100, callback=callback)

    # Define minimize function
    result =  optimizer.minimize(evaluate_expectation, x0=initial_point)

    vqe_interpret = []
    for i in range(len(convergence)):
        sol = MinimumEigensolverResult()
        sol.eigenvalue = convergence[i]
        sol = problem_reduced.interpret(sol).total_energies[0]
        vqe_interpret.append(sol)

    return vqe_interpret, job_list, result
```

## 3. Custom Function to Plot Graphs

最適化の履歴を可視化する関数であるが、これもそのまま使った。

```python
import matplotlib.pyplot as plt

def plot_graph(energy, real_solution, molecule, color="tab:blue"):

    plt.rcParams["font.size"] = 14

    # plot loss and reference value
    plt.figure(figsize=(12, 6), facecolor='white')
    plt.plot(energy, label="Estimator VQE {}".format(molecule),color = color)
    plt.axhline(y=real_solution.real, color="tab:red", ls="--", label="Target")

    plt.legend(loc="best")
    plt.xlabel("Iteration")
    plt.ylabel("Energy [H]")
    plt.title("VQE energy")
    plt.show()
```

## 1. Compute the Ground State Energy of Each Ingredient

**Define Geometry**

3 次元空間の原点に陽子を配置するよというだけ。これもそのまま使った。

```python
# Constructing H
hydrogen_a = [["H", [0.0 ,0.0, 0.0]]]
```

## Construct Problem

**H - VQE Run**

VQE による計算。ここも特に変えていないのだが、固有値などが `numpy.complex128` になっているので、見てくれだけの話だが実部をとるようにした。

```python
algorithm_globals.random_seed = 1024
# For H
ansatz_a, ops_a, real_solution_a, problem_reduced_a = \
    construct_problem(geometry=hydrogen_a, charge=0, multiplicity=2,
                      basis="ccpvdz", num_electrons=(1,0),
                      num_molecular_orbitals=2)

# Estimator VQE for H
Energy_H_a,_,jobs = custom_vqe(estimator=Estimator(), ansatz=ansatz_a,
                               ops=ops_a, problem_reduced=problem_reduced_a)


Energy_H_a = [v.real for v in Energy_H_a]
real_solution_a = real_solution_a.real

# Plot Graph H
plot_graph(Energy_H_a, real_solution_a, "H",color = "tab:purple")
```

![](/images/dwd-qiskit23/001.png)

## 結果を見る

VQE で求まった基底状態のエネルギーはハートリーエネルギーという単位で求まる。[CODATA Value: Hartree energy in eV](https://physics.nist.gov/cgi-bin/cuu/Value?hrev) によると換算については

> 27.211 386 245 988(53) eV

を使えば良いそうなので、


```python
# Energy of the ground state of the hydrogen atom.: -13.6 [eV]
E = -13.6

# 1 hartree-energy = 27.211386245988 eV
print(np.min(Energy_H_a) * 27.211386245988)
```

> -13.586057479730476

となる。精度の良し悪しは分からないが、概ね一致しているのではないだろうか。

# まとめ

他のケースで動作確認をしていないので、ちゃんとコードの移行ができているのかよく分からないが、とりあえず水素原子のケースは動作するようになったと思う。

動かなかった部分というのは、ハミルトニアンをスピン演算子形式に変換するまでの部分で、かつて qiskit-terra と呼ばれていた部分の変更および自身の変更によって qiskit-nature の API が変わったことに起因するのだが、この部分はまったく詳しくないので移行ガイドを読みながらチマチマ置き換えていくしかなくて大変だった。

普通に遊ぶ分には、当時動いていたバージョンと当時の書き方でやるほうが楽かもしれない。
