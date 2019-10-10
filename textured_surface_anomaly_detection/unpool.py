import tensorflow as tf

def unpool():

    value = tf.constant([[[[1],[2]],[[3],[4]]],[[[1],[2]],[[3],[4]]],[[[1],[2]],[[3],[4]]]])

    sh = value.get_shape().as_list()
    dim = len(sh[1:-1])
    out = (tf.reshape(value, [-1] + sh[-dim:]))

    for i in range(dim, 0, -1):
        out = tf.concat([out, out], i)
    out_size = [-1] + [s * 2 for s in sh[1:-1]] + [sh[-1]]
    out = tf.reshape(out, out_size)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    print(sess.run(value))
    print(sess.run(out))
    return out
# unpool()