# Multi-Source Tranfser Learning for Deep Model-Based Reinforcement Learning [[Paper](https://openreview.net/pdf?id=1nhTDzxxMA)]

## Multi-Task Learning
For training a multi task agent: 
```
python dreamer-multi-task.py --task1 HopperBulletEnv-v0 --task2 AntBulletEnv-v0 --task3 HalfCheetahBulletEnv-v0 --batch_length 50 --envs 3 --steps 2e6 --transfer False --logdir './logdir/'
```

## Modular and Fractional Transfer Learning
For modular and fractional transfer learning, first place the variables of the source (multi-task) agent in the folder for the agent you are about to train. Say we transfer to a HalfCheetah agent, we create a folder ```'./logdir/frac-cheetah/'```, place the variables.pkl of the multi-task agent in that folder, and then run:
```
python dreamer-FTL.py --task1 HalfCheetahBulletEnv-v0 --batch_length 50 --envs 1 --steps 1e6 --transfer True --transfer_factor 0.2 --logdir './logdir/frac-cheetah/'
```
<img src="https://github.com/remosasso/multi-source-TL-for-deep-MBRL/blob/main/images/ftlgit.png" width=20% height=20%>



For meta-model transfer learning, first locally make use of functions ```agent.load('./logdir/variables.pkl')``` and ```agent.save_single(agent._encode, "./encoder.pkl")``` to load the variables of a multi-task agent, and then to save the corresponding autoencoder in ```'./logdir/'```. Then you can train single agents with the frozen autoencoder using 'dreamer-MMTL.py'. You can then save the reward models similarly to saving the autoencoder with ```agent.save_single(agent._reward, "./stored_meta/cheetah.pkl")```. Say we trained a HalfCheetah and Ant agent with the UFS frozen autoencoder, we place the aforementioned reward parameters in ```'./stored_meta/'```. Then, when wanting to perform MMTL, run e.g.:
```
python dreamer-MMTL.py --task1 HopperBulletEnv-v0 --n_meta 2 --meta1 HalfCheetahBulletEnv-v0 --meta2 AntBulletEnv-v0 --batch_length 50 --envs 1 --steps 1e6 --logdir './logdir/mmtl-hopper/'
```

<img src="https://github.com/remosasso/multi-source-TL-for-deep-MBRL/blob/main/images/mmtlfinal.png" width=40% height=40%>
See below for general Dreamer requirements and instructions.

____________________________________________________________________________________________________________________________________________________________


# Dream to Control

Fast and simple implementation of the Dreamer agent in TensorFlow 2.

<img width="100%" src="https://imgur.com/x4NUHXl.gif">

If you find this code useful, please reference in your paper:

```
@article{hafner2019dreamer,
  title={Dream to Control: Learning Behaviors by Latent Imagination},
  author={Hafner, Danijar and Lillicrap, Timothy and Ba, Jimmy and Norouzi, Mohammad},
  journal={arXiv preprint arXiv:1912.01603},
  year={2019}
}
```

## Method

![Dreamer](https://imgur.com/JrXC4rh.png)

Dreamer learns a world model that predicts ahead in a compact feature space.
From imagined feature sequences, it learns a policy and state-value function.
The value gradients are backpropagated through the multi-step predictions to
efficiently learn a long-horizon policy.

- [Project website][website]
- [Research paper][paper]
- [Official implementation][code] (TensorFlow 1)

[website]: https://danijar.com/dreamer
[paper]: https://arxiv.org/pdf/1912.01603.pdf
[code]: https://github.com/google-research/dreamer

## Instructions

Get dependencies:

```
pip3 install --user tensorflow-gpu==2.1.0
pip3 install --user tensorflow_probability
pip3 install --user git+git://github.com/deepmind/dm_control.git
pip3 install --user pandas
pip3 install --user matplotlib
```

Train the agent:

```
python3 dreamer.py --logdir ./logdir/dmc_walker_walk/dreamer/1 --task dmc_walker_walk
```

Generate plots:

```
python3 plotting.py --indir ./logdir --outdir ./plots --xaxis step --yaxis test/return --bins 3e4
```

Graphs and GIFs:

```
tensorboard --logdir ./logdir
```
