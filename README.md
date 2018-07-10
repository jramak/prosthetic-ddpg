## Using Tensorboard

    $ export OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard'

    $ python -m baselines.ddpg.main ...
    Logging to /var/folders/<very long path>

    $ tensorboard --logdir /var/<very long path>

## Saving and Restoring Models

To save models after each epoch

    $ python -m baselines.ddpg.main --nb-epochs 50 --model 2D --difficulty 0 --evaluation --frameskip 1 --eval-frameskip=1 --saved-model-name skip1-shapeNone

To restore from a saved model of a particular epoch (note the "-1" suffix to designate the model that was saved after epoch 1):

    $ python -m baselines.ddpg.main --nb-epochs 50 --model 2D --difficulty 0 --evaluation --frameskip 1 --eval-frameskip=1 --saved-model-name skip1-shapeNone-1 --restore-saved-model

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
