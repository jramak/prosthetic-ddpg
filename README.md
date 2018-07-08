## Using Tensorboard

    $ export OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard'

    $ python -m baselines.ddpg.main ...
    Logging to /var/folders/<very long path>

    $ tensorboard --logdir /var/<very long path>
