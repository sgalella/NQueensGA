# Genetic N-Queens
Genetic algorithm to solve the n queens problem. The problem arises from a generalization of the eight puzzle:

> _"The eight queen puzzle is the problem of placing eight chess queens on an 8x8 board chessboard so that no two queens threaten each other."_ — From [Wikipedia](https://en.wikipedia.org/wiki/Eight_queens_puzzle)

<p align="center">
    <img width="480" height="360" src="images/board.jpg">
</p>



## Images

<p align="center">
    <img width="400" height="300" src="images/convergence.jpg">
  <img width="400" height="300" src="images/diversity.jpg">
</p>



## Installation

To install the dependencies, run the following command:

```bash
pip install -r requirements.txt
```

If using Conda, you can also create an environment with the requirements:

```bash
conda env create -f environment.yml
```

By default the environment name is `genetic-nqueens`. To activate it run:

```bash
conda activate genetic-nqueens
````



## Usage

Then use the following command to run the genetic algorithm:

```bash
python -m genetic_nqueens 
```

The parameters of the algorithm can be changed in in `genetic_nqueens/__main__.py`.

