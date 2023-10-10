from stable_baselines3.common.callbacks import BaseCallback


class TensorboardCallback(BaseCallback):
    def __init__(self):
        super(TensorboardCallback, self).__init__()

    def _on_step(self) -> bool:
        self.logger.record('reward/reward', self.training_env.get_attr('reward')[0])
        self.logger.record('reward/maximum environment score', self.training_env.get_attr('max_score')[0])
        # self.logger.record('reward/mean reward', self.training_env.get_attr('mean_reward')[0])
        self.logger.record('reward/max reward', self.training_env.get_attr('max_reward')[0])
        # self.logger.record('reward/backward step', self.training_env.get_attr('startingID')[0])
        return True
