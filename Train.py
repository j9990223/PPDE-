import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


tf.compat.v1.reset_default_graph()
precision = tf.float32

num_train_pts = 1000
num_test_pts = 5000
hidden_dim = 300
depth = 11
input_dim = 2
output_dim = 121
batch_size = 250
delta = 0.1
epochs = 10000

x_test = np.load("X_test.npy")
x_train = np.load("X_train.npy")
y_train = np.load("Y_train.npy")
y_test = np.load("Y_test.npy")
Gram = np.load("G.npy")

x_test=np.transpose(x_test).astype(np.float32)
x_train=np.transpose(x_train).astype(np.float32)
y_train=np.transpose(y_train).astype(np.float32)
y_test=np.transpose(y_test).astype(np.float32)

rho = tf.nn.leaky_relu


def default_block(x, layer, dim1, dim2, weight_bias_initializer, rho, precision=tf.float32):
    W = tf.compat.v1.get_variable(name='l' + str(layer) + '_W', shape=[dim1, dim2],
                                  initializer=weight_bias_initializer, dtype=precision)

    b = tf.compat.v1.get_variable(name='l' + str(layer) + '_b', shape=[dim2, 1],
                                  initializer=weight_bias_initializer, dtype=precision)

    return rho(tf.matmul(W, x) + b)


def funcApprox(x, layers=11, input_dim=2, output_dim=121, hidden_dim=300, precision=tf.float32):
    print('Constructing the tensorflow nn graph')

    weight_bias_initializer = tf.random_normal_initializer(stddev=delta)

    with tf.compat.v1.variable_scope('UniversalApproximator'):
        # input layer description
        in_W = tf.compat.v1.get_variable(name='in_W', shape=[hidden_dim, input_dim],
                                         initializer=weight_bias_initializer, dtype=precision)

        in_b = tf.compat.v1.get_variable(name='in_b', shape=[hidden_dim, 1],
                                         initializer=weight_bias_initializer, dtype=precision)

        z = tf.matmul(in_W, x) + in_b

        x = rho(z)



        for i in range(layers):
            choice = 0
            x = default_block(x, i, hidden_dim, hidden_dim, weight_bias_initializer, precision=precision,
                              rho=rho)
            choice = 1

        out_v = tf.compat.v1.get_variable(name='out_v', shape=[output_dim, hidden_dim],
                                          initializer=weight_bias_initializer, dtype=precision)

        out_b = tf.compat.v1.get_variable(name='out_b', shape=[output_dim, 1],
                                          initializer=weight_bias_initializer, dtype=precision)

        z = tf.math.add(tf.linalg.matmul(out_v, x, name='output_vx'), out_b, name='output')
        return z

def get_batch(X_in, Y_in, batch_size):
    X_cols = X_in.shape[0]
    Y_cols = Y_in.shape[0]

    for i in range(X_in.shape[1]//batch_size):
        idx = i*batch_size + np.random.randint(0,10,(1))[0]

        yield X_in.take(range(idx,idx+batch_size), axis = 1, mode = 'wrap').reshape(X_cols,batch_size), \
              Y_in.take(range(idx,idx+batch_size), axis = 1, mode = 'wrap').reshape(Y_cols,batch_size)


tf.compat.v1.reset_default_graph()
tf.compat.v1.disable_eager_execution()


with tf.compat.v1.variable_scope('Graph') as scope:
    # inputs to the NN
    x = tf.compat.v1.placeholder(precision, shape=[2, None], name='input')
    y_true = tf.compat.v1.placeholder(precision, shape=[121, None], name='y_true')

    y = funcApprox(x, layers=11, input_dim=2,output_dim=121, hidden_dim=300,precision=tf.float32)

    print(y)
    with tf.compat.v1.variable_scope('Loss'):
        # Mean squared error function
        loss = ((tf.compat.v1.losses.mean_squared_error(tf.linalg.matmul(Gram,y),tf.linalg.matmul(Gram,y_true)))/tf.linalg.norm(tf.linalg.matmul(Gram,y_true)))
    init_rate = 0.0002
    lrn_rate = init_rate
    opt = tf.compat.v1.train.AdamOptimizer(learning_rate=lrn_rate)
    train_op = opt.minimize(loss)

    print(tf.compat.v1.trainable_variables())

    losses = []

    print(np.shape(x_train))
    with tf.compat.v1.Session() as sess:
        # init variables
        sess.run(tf.compat.v1.global_variables_initializer())

        for i in range(epochs):


          count = 0
          for x_in_train_batch, y_true_train_batch in get_batch(x_train, y_train, batch_size):
            count = count + 1
            current_loss, _ = sess.run([loss, train_op],
                            feed_dict={x: x_train_batch, \
                                        y_true: y_true_train_batch})
            losses.append(current_loss)

        y_res = sess.run([y], feed_dict = {x: x_test.reshape(2,num_test_pts)})

print('done')
x=range(20000)
print(losses)
print("y_res0",y_res[0])
plt.semilogy(x,losses)
