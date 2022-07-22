from tensorboard import summary as summary_lib
import tensorflow as tf

labels = tf.constant([False, True, True, False, True], dtype=tf.bool)
predictions = tf.random.uniform(labels.get_shape(), maxval=1.0)
print(predictions)
print(predictions.get_shape())
print(labels)
print(labels.get_shape())
summary_lib.pr_curve(name='foo',
                     predictions=predictions,
                     labels=labels,
                     num_thresholds=11)
merged_summary = tf.compat.v1.summary.merge_all()

with tf.compat.v1.Session() as sess:
  writer = tf.compat.v1.summary.FileWriter('/tmp/logdir', sess.graph)
  for step in range(43):
    writer.add_summary(sess.run(merged_summary), global_step=step)
