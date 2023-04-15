# PyBulletPerAct

The following repository contains files that implement the Perceivor-Actor (PerAct) of the [paper](https://arxiv.org/abs/2209.05451) for the
[panda_gym](https://github.com/qgallouedec/panda-gym) environment in Pybullet (PyBulletPerAct) and for setups suitable for training PyBulletPerAct
on pouring "liquid". 


## Panda_gym

We've modified the original panda_gym environment 
(see [documentation](https://panda-gym.readthedocs.io/en/latest/) and [paper](https://arxiv.org/abs/2106.13687)) 
substantially to accommodate its observations and environment with the visual/voxel approach of PerAct.
Some of the major key points are:

* We essentially completely change the pick-and-place environment (feel free to rename it, just be careful to change the registration in the 
[__init__ file](https://github.com/fluderm/PyBulletPerAct/blob/main/panda_gym/__init__.py)).
The other environments are left unchanged from panda_gym 2.0.0.

* We add two glasses (red and blue) to the environment, where the blue glass is the 'object' and the red one is the 'target'. Furthermore, 
we add several (feel free to change the amount/size/mass) small spheres (initialized to be in the blue cup). Both the blue and red cup 
are at arbitrary locations upon initialization on the table and the task is to pick up the blue cup and pour a certain amount of spheres into 
the red cup. The idea is to simulate the pouring of powder as asked by the client.

* If you want to apply a reinforcement learning pipeline you should modify the reward functions in the 
[file](https://github.com/fluderm/PyBulletPerAct/blob/main/panda_gym/envs/tasks/pick_and_place.py). 
However, we would like to point out that reinforcement learning is very time and computing power consuming, and it's worthwhile investigating alternative
approaches (e.g. PyBulletPerAct).

* In [Pybullet.py](https://github.com/fluderm/PyBulletPerAct/blob/main/panda_gym/pybullet.py), we 
substantially and carefully changed the rendering such that it is compatible with 
[Create_and_save_demo.ipynb](https://github.com/fluderm/PyBulletPerAct/blob/main/Create_and_save_demo.ipynb).
Furthermore, we add functions that allow for changing the environment such that we can simulate pouring small spheres.

Some additional remarks about panda_gym and simulations in general:

* We chose panda_gym and PyBullet, as it ran without issues on Colab. We tried various other simulation environments and a lot of them require 
more computing power than we had access to. In particular, Gazebo seemed quite troublesome. A decent alternatively (on linux) would be 
[RLBench](https://sites.google.com/view/rlbench), which is based on 
[PyRep](https://github.com/stepjam/PyRep) and [CoppeliaSim](https://www.coppeliarobotics.com/). 
Unfortunately, PyRep only works on linux at the time of this writing.

* Our starting point is panda_gym version 2.0.0 (rather than the updated versions). The reason for this is that it is compatible with 
[stable-baseline3](https://stable-baselines3.readthedocs.io/en/master/) which we used in earlier reinforcement learning approaches. 
The colab files should be future-proof now, but even within the short time of the fellowship some dependencies broke.


## Create_and_save_demo Colab

The colab file 
[Create_and_save_demo.ipynb](https://github.com/fluderm/PyBulletPerAct/blob/main/Create_and_save_demo.ipynb)
implements a pipeline that easily lets you create and save demos in a way that the saved demos of the modified panda_gym and PyBullet environment
are compatible with PerAct (and more broadly with RLBench).

Some Remarks:

* I would suggest you implement a way to save text prompts with your demos. There are various comments to that regard and it should be trivial 
(and worthwhile) to change a couple of lines of code.
* We have some pouring actions implemented but it's probably worth trying some more sophisticated approaches.
* I would further suggest using more sophisticated motion planning (see, e.g. [here](https://github.com/yijiangh/pybullet_planning/issues/7) 
for an example). In particular it is crucial for real-life implementations that there's a way to avoid collisions with other objects in the environment (the current, naive inverse kinematics implementation needs some modification to account for that).
Currently, the approach of the gripper is to simply take a straight line from current location to target. 
There are quite a lot of fancy implementations of such motion-planners and it's probably worthwhile investigating it a bit more.
* It is quite straightforward to parallelize the demo creation and saving. So it's probably worthwhile if you want to quickly create large datasets.
Notice however, that one of the benefits of PerAct is that it's not very data intensive.


## PyBulletPerAct Colab

Finally, the [PyBulletPerAct.ipynb](https://github.com/fluderm/PyBulletPerAct/blob/main/PyBulletPerAct.ipynb) contains the 
part of the pipeline which trains PyBulletPerAct on the data generated in Create_and_save_demo. We've carefully code from various sources and made sure
all conventions are compatible. In particular we create a voxelized environment of our panda_gym which serves as the action and observation space of PerAct.

Some remarks:

* We've decided to focus on the PerAct approach for a variety of reasons. In particular, Voxelizing the environment makes the PerceiverIO part of the
pipeline effectively independent of camera positioning, etc. (for instance RT-1 by Google is dependent on a fixed camera and camera angle). Furthermore,
discretizing the environment improves training speed and requires less data and you can get decent results within ~ an hour of training on Colab pro.
There are many other novel approaches to robot training, but a lot of them -- while really cool -- require expensive computing resources we did not have.
In general, I would stay away from anything dubbed "reinforcement learning", as it seems miles behind transformer-type approaches.
* It is worthwhile going through the code and understanding what it's doing.
* In particular, I would suggest playing around with the voxelization. In particular, if you eventually want PerAct to pour a certain amount of spheres,
the voxel size better be able to discern between spheres (at least approximately). In particular, my feeling is to limit the voxel bounds to a smaller
volume.
* If you're just starting up, I've shared data (for "pick and place") generated by myself as well as the weights of a pretrained model for "pick and place".
This should give you a good starting point.

## To do:

A lot of our time was spent trying to figure out the goal for our group. In order to help you with that and give you explicit direction, let me provide you with some (to me) reasonable next steps:

* First, I would suggest trying to understand the code we provide in this repo. Personally, I think this is a very good approach and seems suitable to
making progress given limited resources.

* Then, I would suggest to change the modified panda_gym environment so that the blue cup with the spheres in it is in perfect position to do pouring 
(i.e. tilting the 6th joint of the panda robot to a certain angle). Test out how well it works for pouring (if they go into the red cup and jump out 
maybe consider changing the sphere masses or making the red cup larger). Once pouring seems reasonable, start generating lots of data where the shifted
angle is picked arbitrarily and you record how many spheres were poured successfully (there's already a function for that). Once you've generated
some data (here it's important to store the prompts, e.g. smth like "pour 3 spheres from the blue to the red cup" -- 
this requires some simple modifications of our code), run the PyBulletPerAct file and train the model on this data. In order to see whether it works,
you won't require fancy motion-planning just twist the 6th joint until the orientation predicted by PerAct is reached.

* Once this is running and reliably pours an explicit amount, I would play around with adding more spheres, etc. But it's then also useful to look into
motion planners and see whether you can combine the pouring with the picking and placing by e.g. changing the prompt to "pick up the blue cup and place
it near the red one and pour x spheres from the blue into the red one". I think this should work; if not, you can try to train it all in one.

* Once these things work reliably you can increase the difficulty by adding obstacles to the environment, more cups with different amount of spheres,
etc. and have fun exploring the limitations.

If you run into any trouble or want some help, feel free to reach out to [me](mailto:fluderm@gmail.com)





