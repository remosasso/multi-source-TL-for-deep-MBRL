# Multi-Source Tranfser Learning for Deep Model-Based Reinforcement Learning
Code for the [paper](https://arxiv.org/abs/2205.14410?context=cs.AI) Multi-Source Tranfser Learning for Deep Model-Based Reinforcement Learning. If this code was useful to your research, please acknowledge it:
```
@article{
      sasso2023multisource,
      title={Multi-Source Transfer Learning for Deep Model-Based Reinforcement Learning},
      author={Remo Sasso and Matthia Sabatelli and Marco A. Wiering},
      journal={Transactions on Machine Learning Research},
      issn={2835-8856},
      year={2023},
      url={https://openreview.net/forum?id=1nhTDzxxMA}
}
```

## Instructions
Please see the [Dreamer repository](https://github.com/google-research/dreamer) for general dependency requirements and installation instructions.
### Multi-Task Learning
For training a multi task agent with, say, the Hopper, Ant, and Cheetah task for 2M environment steps: 
```
python dreamer-multi-task.py --task1 HopperBulletEnv-v0 --task2 AntBulletEnv-v0 --task3 HalfCheetahBulletEnv-v0 --batch_length 50 --envs 3 --steps 2e6 --logdir './logdir/multi-hopper-ant-cheetah'
```

### Modular and Fractional Transfer Learning
For modular and fractional transfer learning, first place the variables of the source (multi-task) agent in the folder for the agent you are about to train. Say we transfer to a HalfCheetah agent, we create a folder ```'./logdir/frac-cheetah/'```, place the variables.pkl of the multi-task agent in that folder, and then run:
```
python dreamer-FTL.py --task1 HalfCheetahBulletEnv-v0 --batch_length 50 --envs 1 --steps 1e6 --transfer True --transfer_factor 0.2 --logdir './logdir/frac-cheetah/'
```
<img src="https://github.com/remosasso/multi-source-TL-for-deep-MBRL/blob/main/images/multi.png" width=40% height=40%>     <img src="https://github.com/remosasso/multi-source-TL-for-deep-MBRL/blob/main/images/ftl.PNG" width=44.5% height=44.5%>

### Meta-Model Transfer Learning

For meta-model transfer learning, first locally make use of functions ```agent.load('./logdir/variables.pkl')``` and ```agent.save_single(agent._encode, "./encoder.pkl")``` to load the variables of a multi-task agent, and then to save the corresponding autoencoder in ```'./logdir/'```. Then you can train single agents with the frozen autoencoder using 'dreamer-MMTL.py'. You can then save the reward models similarly to saving the autoencoder with ```agent.save_single(agent._reward, "./stored_meta/cheetah.pkl")```. Say we trained a HalfCheetah and Ant agent with the UFS frozen autoencoder, we place the aforementioned reward parameters in ```'./stored_meta/'```. Then, when wanting to perform MMTL, run e.g.:
```
python dreamer-MMTL.py --task1 HopperBulletEnv-v0 --n_meta 2 --meta1 HalfCheetahBulletEnv-v0 --meta2 AntBulletEnv-v0 --batch_length 50 --envs 1 --steps 1e6 --logdir './logdir/mmtl-hopper/'
```

<img src="https://github.com/remosasso/multi-source-TL-for-deep-MBRL/blob/main/images/mmtlfinal.png" width=30% height=30%>

