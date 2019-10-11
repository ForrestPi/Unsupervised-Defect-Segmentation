# python -m pdb aae_wgan.py --load 0 --comment "aae wgan" --model_name "None" --step "0"
# python -m pdb aae_wgan.py --load 1 --comment "2018-02-22 12:40:34.603793_35_Adversarial_Autoencoder_WGAN retrain" --model_name "2018-02-22 12:40:34.603793_35_Adversarial_Autoencoder_WGAN" --step "-4933"


import os
os.environ['CUDA_VISIBLE_DEVICES'] = 'GPU_NAME'
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from model import encoder, decoder, discriminator

import import_dataset as datasets
from funcs.preproc import *

# Parameters
BATCH_SIZE =64
EPOCHS = 300
LR = 2e-5
WEIGHT=0.5
results_path = './Results/Adversarial_Autoencoder'

def form_results():
    """
    Forms folders for each run to store the tensorboard files, saved models and the log files.
    :return: three string pointing to tensorboard, saved models and log paths respectively.
    """
    results_path = './Results/Adversarial_Autoencoder'
    folder_name = "/cAdversarial_Autoencoder_WGAN"
    tensorboard_path = results_path + folder_name + '/Tensorboard'
    saved_model_path = results_path + folder_name + '/Saved_models/'
    log_path = results_path + folder_name + '/log'
    if not os.path.exists(results_path + folder_name):
        os.mkdir(results_path + folder_name)
        os.mkdir(tensorboard_path)
        os.mkdir(saved_model_path)
        os.mkdir(log_path)
    return tensorboard_path, saved_model_path, log_path, folder_name

def train(z_dim=None, model_name=None):
    """
    Used to train the autoencoder by passing in the necessary inputs.
    :param train_model: True -> Train the model, False -> Load the latest trained model and show the image grid.
    :return: does not return anything
    """
    X_train, y_train = datasets.create_datasets(retrain=0, task="aae_wgan_" + str(z_dim),
                                                num_aug=0)

    batch_size = BATCH_SIZE
    input_dim = X_train.shape[-1]

    with tf.device("/gpu:0"):
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        x_input = tf.placeholder(dtype=tf.float32, shape=[batch_size, input_dim, input_dim, 1],
                                 name='Input')
        x_target = tf.placeholder(dtype=tf.float32, shape=[batch_size, input_dim, input_dim, 1],
                                  name='Target')
        real_distribution = tf.placeholder(dtype=tf.float32, shape=[batch_size, z_dim],
                                           name='Real_distribution')
        decoder_input = tf.placeholder(dtype=tf.float32, shape=[1, z_dim],
                                       name='Decoder_input')

        encoder_output = encoder(x_input, reuse=False, is_train=True)
        encoder_output_test = encoder(x_input, reuse=True, is_train=False)
        d_fake, d_fake_logits = discriminator(encoder_output, reuse=False)
        d_real, d_real_logits = discriminator(real_distribution, reuse=True)

        d_fake_test, d_fake_logits_test = discriminator(encoder_output, reuse=True)
        d_real_test, d_real_logits_test = discriminator(real_distribution, reuse=True)

        decoder_output, std = decoder(encoder_output, reuse=False, is_train=True)
        encoder_output_z = encoder(decoder_output, reuse=True, is_train=False)
        decoder_output_test, std_ = decoder(encoder_output, reuse=True, is_train=False)
        encoder_output_z_test = encoder(decoder_output_test, reuse=True, is_train=False)

        #decoder_image = decoder(decoder_input, reuse=True, is_train=False)

        # Autoencoder loss
        # summed = tf.reduce_mean(tf.square(decoder_output-x_target),[1,2,3])
        summed = tf.reduce_sum(tf.square(decoder_output - x_target), [1, 2, 3])
        # sqrt_summed = summed
        sqrt_summed = tf.sqrt(summed + 1e-8)
        autoencoder_loss = tf.reduce_mean(sqrt_summed)

        summed_test = tf.reduce_sum(tf.square(decoder_output_test - x_target), [1, 2, 3])
        # sqrt_summed_test = summed_test
        sqrt_summed_test = tf.sqrt(summed_test + 1e-8)
        autoencoder_loss_test = tf.reduce_mean(sqrt_summed_test)

        # l2 loss of z
        enc = tf.reduce_sum(tf.square(encoder_output - encoder_output_z), [1])
        encoder_l2loss = tf.reduce_mean(enc)
        enc_test = tf.reduce_sum(tf.square(encoder_output_test - encoder_output_z_test), [1])
        encoder_l2loss_test = tf.reduce_mean(enc_test)

        dc_loss = tf.reduce_mean(d_real_logits - d_fake_logits)
        dc_loss_test = tf.reduce_mean(d_real_logits_test - d_fake_logits_test)

        with tf.name_scope("Gradient_penalty"):
            eta = tf.placeholder(tf.float32, shape=[batch_size, 1], name="Eta")
            interp = eta * real_distribution + (1 - eta) * encoder_output
            _, c_interp = discriminator(interp, reuse=True)

            # taking the zeroth and only element because tf.gradients returns a list
            c_grads = tf.gradients(c_interp, interp)[0]

            # L2 norm, reshaping to [batch_size]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(c_grads), axis=[1]))
            tf.summary.histogram("Critic gradient L2 norm", slopes)

            grad_penalty = tf.reduce_mean((slopes - 1) ** 2)
            lambd = 10.0
            dc_loss += lambd * grad_penalty

        # Generator loss
        # generator_loss = tf.reduce_mean(
        #    tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_fake), logits=d_fake_logits))
        generator_loss = tf.reduce_mean(d_fake_logits)
        generator_loss_test = tf.reduce_mean(d_fake_logits_test)

        all_variables = tf.trainable_variables()
        dc_var = tl.layers.get_variables_with_name('Discriminator', True, True)
        en_var = tl.layers.get_variables_with_name('Encoder', True, True)
        #print en_var
        # dc_var = [var for var in all_variables if 'dc' in var.name]
        # en_var = [var for var in all_variables if 'encoder' in var.name]
        var_grad_autoencoder = tf.gradients(autoencoder_loss, all_variables)[0]
        var_grad_discriminator = tf.gradients(dc_loss, dc_var)[0]
        var_grad_generator = tf.gradients(generator_loss, en_var)[0]

        # Optimizers
        with tf.device("/gpu:0"):
            autoencoderl2_optimizer = tf.train.AdamOptimizer(learning_rate=LR,
                                                             beta1=0.5, beta2=0.9).minimize(
                autoencoder_loss + 0.5 *encoder_l2loss)
            autoencoder_optimizer = tf.train.AdamOptimizer(learning_rate=LR,
                                                           beta1=0.5, beta2=0.9).minimize(autoencoder_loss)
            discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=LR,
                                                             beta1=0.5, beta2=0.9).minimize(dc_loss, var_list=dc_var)
            generator_optimizer = tf.train.AdamOptimizer(learning_rate=LR,
                                                         beta1=0.5, beta2=0.9).minimize(generator_loss, var_list=en_var)

            tl.layers.initialize_global_variables(sess)
        # Reshape immages to display them
        input_images = tf.reshape(x_input, [-1, input_dim, input_dim, 1])
        generated_images = tf.reshape(decoder_output, [-1, input_dim, input_dim, 1])
        # generated_images = tf.reshape(decoder_output, [-1, 28, 28, 1])
        tensorboard_path, saved_model_path, log_path, folder_name = form_results()
        # bp()
        writer = tf.summary.FileWriter(logdir=tensorboard_path, graph=sess.graph)
        # Tensorboard visualization
        tf.summary.scalar(name='Autoencoder Loss', tensor=autoencoder_loss)
        tf.summary.scalar(name='Autoencoder Test Loss', tensor=autoencoder_loss_test)
        tf.summary.scalar(name='Discriminator Loss', tensor=dc_loss)
        tf.summary.scalar(name='Generator Loss', tensor=generator_loss)
        tf.summary.scalar(name='Autoencoder z Loss', tensor=encoder_l2loss)
        tf.summary.histogram(name='Encoder Distribution', values=encoder_output)
        tf.summary.histogram(name='Real Distribution', values=real_distribution)
        tf.summary.histogram(name='Gradient AE', values=var_grad_autoencoder)
        tf.summary.histogram(name='Gradient D', values=var_grad_discriminator)
        tf.summary.histogram(name='Gradient G', values=var_grad_generator)
        tf.summary.image(name='Input Images', tensor=input_images, max_outputs=10)
        tf.summary.image(name='Generated Images', tensor=generated_images, max_outputs=10)
        summary_op = tf.summary.merge_all()
        saver = tf.train.Saver()
    # Saving the model

    step = 0
    # with tf.Session() as sess:
    with open(log_path + '/log.txt', 'a') as log:
        log.write("input_dim: {}\n".format(input_dim))
        log.write("z_dim: {}\n".format(z_dim))
        log.write("batch_size: {}\n".format(batch_size))
        log.write("\n")

    for i in range(EPOCHS):
        b = 0
        for batch in tl.iterate.minibatches(inputs=X_train, targets=np.zeros(X_train.shape),
                                            batch_size=batch_size, shuffle=True):
            z_real_dist = np.random.normal(0, 1, (batch_size, z_dim)) * 1.
            z_real_dist = z_real_dist.astype("float32")

            batch_x, _ = batch
            batch_x = batch_x[:, :, :, np.newaxis]
            #lambda_x = np.max(lambda_grow_max / np.float(i), lambda_grow_max)
            sess.run(autoencoderl2_optimizer, feed_dict={x_input: batch_x, x_target: batch_x})
            if i < 20:
                # sess.run(autoencoder_optimizer, feed_dict={x_input: batch_x, x_target: batch_x})
                for t in range(10):
                    for _ in range(20):
                        eta1 = np.random.rand(batch_size, 1)  # sampling from uniform distribution
                        eta1 = eta1.astype("float32")
                        sess.run(discriminator_optimizer,
                                 feed_dict={x_input: batch_x, x_target: batch_x,
                                            real_distribution: z_real_dist, eta: eta1})
            else:
                # sess.run(autoencoderl2_optimizer, feed_dict={x_input: batch_x, x_target: batch_x})
                for _ in range(20):
                    eta1 = np.random.rand(batch_size, 1)  # sampling from uniform distribution
                    eta1 = eta1.astype("float32")
                    sess.run(discriminator_optimizer,
                             feed_dict={x_input: batch_x, x_target: batch_x,
                                        real_distribution: z_real_dist, eta: eta1})

            sess.run(generator_optimizer, feed_dict={x_input: batch_x, x_target: batch_x})
            if b % 50 == 0:
                a_loss, e_loss, d_loss, g_loss, a_grad, d_grad, g_grad, en_output, d_real_logits_, d_fake_logits_, de_output, summary = sess.run(
                    [autoencoder_loss, encoder_l2loss, dc_loss, generator_loss, var_grad_autoencoder,
                     var_grad_discriminator,
                     var_grad_generator, encoder_output, d_real_logits, d_fake_logits, decoder_output, summary_op],
                    feed_dict={x_input: batch_x, x_target: batch_x,
                               real_distribution: z_real_dist, eta: eta1})
                print(model_name)
                saver.save(sess, save_path=saved_model_path, global_step=step)
                writer.add_summary(summary, global_step=step)

                print("Epoch: {}, iteration: {}".format(i, b))
                print("Autoencoder Loss: {}".format(a_loss))
                print("Autoencoder enc Loss: {}".format(e_loss))
                print("Discriminator Loss: {}".format(d_loss))
                print("Generator Loss: {}".format(g_loss))
                with open(log_path + '/log.txt', 'a') as log:
                    log.write("Epoch: {}, iteration: {}\n".format(i, b))
                    log.write("Autoencoder Loss: {}\n".format(a_loss))
                    log.write("Autoencoder enc Loss: {}\n".format(e_loss))
                    log.write("Discriminator Loss: {}\n".format(d_loss))
                    log.write("Generator Loss: {}\n".format(g_loss))
            b += 1
            step += 1

        b = 0
        for batch in tl.iterate.minibatches(inputs=y_train, targets=np.zeros(y_train.shape),
                                            batch_size=batch_size, shuffle=True):
            z_real_dist = np.random.normal(0, 1, (batch_size, z_dim)) * 1.
            z_real_dist = z_real_dist.astype("float32")
            batch_x, _ = batch
            batch_x = batch_x[:, :, :, np.newaxis]
            eta1 = np.random.rand(batch_size, 1)
            if b % 20 == 0:
                a_loss, e_loss, d_loss, g_loss = sess.run(
                    [autoencoder_loss_test, encoder_l2loss_test, dc_loss_test, generator_loss_test],
                    feed_dict={x_input: batch_x, x_target: batch_x,
                               real_distribution: z_real_dist, eta: eta1})
                print("v_Epoch: {}, iteration: {}".format(i, b))
                print("v_Autoencoder Loss: {}".format(a_loss))
                print("v_Autoencoder enc Loss: {}".format(e_loss))
                print("v_Discriminator Loss: {}".format(d_loss))
                print("v_Generator Loss: {}".format(g_loss))
                with open(log_path + '/log.txt', 'a') as log:
                    log.write("v_Epoch: {}, iteration: {}\n".format(i, b))
                    log.write("v_Autoencoder Loss: {}\n".format(a_loss))
                    log.write("v_Autoencoder enc Loss: {}\n".format(e_loss))
                    log.write("v_Discriminator Loss: {}\n".format(d_loss))
                    log.write("v_Generator Loss: {}\n".format(g_loss))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='None', help='model to retrain on')
    parser.add_argument('--z_dim', type=str, default='None', help='model comment')
    args = parser.parse_args()
    train(z_dim=args.z_dim, model_name=args.model_name)
