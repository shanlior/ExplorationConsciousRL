This is a partial baselines package, containing an added implementation of the Expected and Surrogate Sigma-DDPG Algorithms, and the Expected and Surrogate alpha-DQN algorithms, presented in the paper:

"Exploration Conscious Reinforcement Learning Revisited", Shani, L., Efroni, Y., & Mannor, S. (2018).
https://arxiv.org/abs/1812.05551

# Running the Algorithms

## alpha-DQN
Running the alpha-DQN for Atari is made by running the following line:

`python baselines/deepq/experiment/run_atari.py`

- The *alpha* flag turns on the alpha-DQN.
- The *surrogate* flag turns on the surrogate version.
- The *expected* flag turns on the expected version.

## sigma-DDPG

Running the sigma-DDPG for Mujoco is made by running the following line:

`python baselines/ddpg/main.py`

- The *sigma* flag turns on sigma-DDPG.
- The *surrogate* flag turns on the surrogate version.
- The *expected* flag turns on the expected version.

    For the expected version:
    - The *sigma_num_samples* and *grad_num_samples* flags determine the number of samples used to approximate the expected target and the expected gradient, respectively.


# Installation

For installation, first follow the normal baselines installation.

Notice that this is an old baselines implementation, tested with tensorflow-gpu-1.6.0 and CUDA 9.0.

<img src="data/logo.jpg" width=25% align="right" />

# Baselines

OpenAI Baselines is a set of high-quality implementations of reinforcement learning algorithms.

These algorithms will make it easier for the research community to replicate, refine, and identify new ideas, and will create good baselines to build research on top of. Our DQN implementation and its variants are roughly on par with the scores in published papers. We expect they will be used as a base around which new ideas can be added, and as a tool for comparing a new approach against existing ones. 

You can install it by typing:

```bash
git clone https://github.com/openai/baselines.git
cd baselines
pip install -e .
```

- [DDPG](baselines/ddpg)
- [DQN](baselines/deepq)

To cite this repository in publications:

    @misc{baselines,
      author = {Dhariwal, Prafulla and Hesse, Christopher and Klimov, Oleg and Nichol, Alex and Plappert, Matthias and Radford, Alec and Schulman, John and Sidor, Szymon and Wu, Yuhuai},
      title = {OpenAI Baselines},
      year = {2017},
      publisher = {GitHub},
      journal = {GitHub repository},
      howpublished = {\url{https://github.com/openai/baselines}},
    }
