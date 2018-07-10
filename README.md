## Using Tensorboard

    $ export OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard'

    $ python -m baselines.ddpg.main ...
    Logging to /var/folders/<very long path>

    $ tensorboard --logdir /var/<very long path>

## Saving and Restoring Models

To save models after each epoch

    $ python -m baselines.ddpg.main --nb-epochs 50 --model 2D --difficulty 0 --evaluation --frameskip 1 --eval-frameskip=1 --saved-model-name skip1-shapeNone

To restore from a saved model of a particular epoch (note the "-1" suffix to designate the model that was saved after epoch 1):

python -m baselines.ddpg.main --nb-epochs 50 --model 2D --difficulty 0 --evaluation --frameskip 1 --eval-frameskip=1 --saved-model-name skip1-shapeNone-1 --restore-saved-model
