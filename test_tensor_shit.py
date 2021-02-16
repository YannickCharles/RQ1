import tensorflow as tf


zero_action_nodes = tf.zeros(shape=(3 - 1, 4), dtype=tf.dtypes.float32)
action_on_nodes = tf.concat([zero_action_nodes, tf.constant([[0.,0.,0.,1.]])], axis=0)

with tf.Session() as sess:
    test = sess.run(action_on_nodes)

print('end')