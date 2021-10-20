import tensorboard
from collections import defaultdict


class ResultsLogger():

  def __init__(metric_list, log_to_tensorboard=True):
    self.metrics = defaultdict(dict)
  def dump(self, state):
    pass

  def record_summary(self, data):
    with train_summary_writer.as_default():
      tf.summary.scalar('loss', train_loss.result(), step=epoch)
      tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)

  def log_results(inputs):
    pass
