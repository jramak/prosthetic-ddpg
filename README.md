Contributors: Mark Paine, Jagdish Ramakrishnan

Code we used for the crowdAI, NIPS 2018 AI for Prosthetics Challenge. We trained the agent using OpenAI’s Deep Deterministic Policy Gradient (DDPG) baselines implementation. Our best performing submission (~1260 reward) used a SELU activation function and feature embellishment (e.g., torso and legs lean, knee flexion).

## Installation

Install Miniconda or Anaconda and set up your conda environment by following these intructions: http://osim-rl.stanford.edu/docs/quickstart/

## Using Tensorboard

    $ export OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard'

    $ python -m baselines.ddpg.main ...
    Logging to /var/folders/<very long path>

    $ tensorboard --logdir /var/<very long path>

You can also choose the logs directory by setting OPENAI_LOGDIR environment variable.

## Running jobs in parallel using MPI

    $ mpiexec -n <NUM_CPUS> python -m baselines.ddpg.main --nb-epochs 3 --model 3D --difficulty 0 --frameskip 4 --activation leaky_relu --actor-layer-sizes '[256,256]' --critic-layer-sizes '[256,256]'

## Saving and Restoring Models

Models are always saved after each epoch. If left unspecified, the base filename is generated randomly. Each saved model file is appended with epoch number. To specify a base name for the saved files, use the --saved-model-basename argument:

    $ python -m baselines.ddpg.main --nb-epochs 50 --model 2D --difficulty 0 --evaluation --frameskip 1 --eval-frameskip=1 --saved-model-basename skip1-shapeNone

To restore from a saved model of a particular epoch (note the "-1" suffix to designate the model that was saved after epoch 1):

    $ python -m baselines.ddpg.main --nb-epochs 50 --model 2D --difficulty 0 --evaluation --frameskip 1 --eval-frameskip=1 --restore-model-name skip1-shapeNone-1

It's even possible to restore from an existing model and save new models to a different file basename:

	$ python -m baselines.ddpg.main --nb-epochs 50 --model 2D --difficulty 0 --evaluation --frameskip 1 --eval-frameskip=1 --restore-model-name skip1-shapeNone-1 --save-model-basename my-new-model-from-skip1-shapeNone-1

## Using feature embellishment and reward shaping

Feature embellishment adds derived features to the
observation space. For example, torso lean, femur lean,
and knee flexion are features of potential interest.

Reward shaping implies feature embellishment and takes
it a step further by applying negative rewards for
undesired behaviors. For example, attempting to run
with excessive rearward torso lean or with both
knees hyperextended is undesired.

    $ python -m baselines.ddpg.main --nb-epochs 50 --model 2D --difficulty 0 --evaluation --feature-embellishment

    $ python -m baselines.ddpg.main --nb-epochs 50 --model 2D --difficulty 0 --evaluation --reward-shaping

Since reward shaping relies on feature embellishment,
the following invocation doesn't make sense (but will
result in reward shaping with feature embellishment
turned on, overriding the --no-feature-embellishment
setting):

    $ python -m baselines.ddpg.main --nb-epochs 50 --model 2D --difficulty 0 --evaluation --no-feature-embellishment --reward-shaping

## Setting up on GCloud

Spin up a VM (with enough CPU / memory). Run these commands to install git and conda:

    $ sudo apt-get update
    $ sudo apt-get install bzip2 git libxml2-dev
    $ wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
    $ bash Miniconda3-latest-Linux-x86_64.sh
    $ rm Miniconda3-latest-Linux-x86_64.sh
    $ source .bashrc
    $ conda update -n base conda

Set up your ssh keys:

    $ ssh-keygen -t rsa -b 4096 -C "user@gmail.com"

Copy your public key to github and clone repos:

    $ git config --global user.email "user@gmail.com"
    $ git config --global user.name "Your Name"
    $ git clone git@bitbucket.org:mobylick/prosthetic.git

Follow instructions here to set up environment: http://osim-rl.stanford.edu/docs/quickstart/ .

Finally, install remaining packages:

    $ pip install tensorflow
    $ pip install scipy
    $ conda install mpi4py

## Running a job on GCloud

The run_job_gcloud.sh script can run the job and shutdown the vm instance if needed. If the first argument is the command, the script will simply evaluate the command. In order to shutdown the vm after execution of command, provide "stop-vm" as first argument and command as second argument.

    $ source activate opensim-rl
    $ ./run_job_gcloud.sh "python -m baselines.ddpg.main --nb-epochs 1 --model 3D --difficulty 0 --frameskip 4 --activation selu --saved-model-basename skip4-shapexz-eval-relxzpos-rsx2-ou-selu --noise-type ou_0.2 --reward-shaping --reward-shaping-x=2" > logs/test.log  # vm will not be stopped after execution
    $ ./run_job_gcloud.sh stop-vm "python -m baselines.ddpg.main --nb-epochs 1 --model 3D --difficulty 0 --frameskip 4 --activation selu --saved-model-basename skip4-shapexz-eval-relxzpos-rsx2-ou-selu --noise-type ou_0.2 --reward-shaping --reward-shaping-x=2" > logs/test.log  # vm will be stopped after execution

You may use the "screen" command in the google cloud browser window before running the above commands.
