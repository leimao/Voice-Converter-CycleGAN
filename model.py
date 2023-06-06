from params import *
import os
from module import discriminator, generator_gatedcnn
from utils import l1_loss, l2_loss, cross_entropy_loss
from datetime import datetime
import tensorflow as tf


class CycleGAN(tf.keras.Model):

    def __init__(self, num_features, discriminator, generator):
        super(CycleGAN, self).__init__()

        self.num_features = num_features
        self.input_shape = (None, num_features, None)

        self.discriminator = discriminator
        self.generator = generator

    def build_model(self):
        # Placeholders for real training samples
        self.input_A_real = tf.keras.layers.Input(shape=self.input_shape, name='input_A_real')
        self.input_B_real = tf.keras.layers.Input(shape=self.input_shape, name='input_B_real')
        # Placeholders for fake generated samples
        self.input_A_fake = tf.keras.layers.Input(shape=self.input_shape, name='input_A_fake')
        self.input_B_fake = tf.keras.layers.Input(shape=self.input_shape, name='input_B_fake')
        # Placeholder for test samples
        self.input_A_test = tf.keras.layers.Input(shape=self.input_shape, name='input_A_test')
        self.input_B_test = tf.keras.layers.Input(shape=self.input_shape, name='input_B_test')

        self.generation_B = self.generator(self.input_A_real, training=True)
        self.cycle_A = self.generator(self.generation_B, training=True)

        self.generation_A = self.generator(self.input_B_real, training=True)
        self.cycle_B = self.generator(self.generation_A, training=True)

        self.generation_A_identity = self.generator(self.input_A_real, training=True, reuse=True)
        self.generation_B_identity = self.generator(self.input_B_real, training=True, reuse=True)

        self.discrimination_A_fake = self.discriminator(self.input_A_fake, training=True)
        self.discrimination_B_fake = self.discriminator(self.input_B_fake, training=True)

        # Cycle loss
        self.cycle_loss = tf.keras.losses.MeanAbsoluteError(y_true=self.input_A_real, y_pred=self.cycle_A) + tf.keras.losses.MeanAbsoluteError(y_true=self.input_B_real, y_pred=self.cycle_B)

        # Identity loss
        self.identity_loss = tf.keras.losses.MeanAbsoluteError(y_true=self.input_A_real, y_pred=self.generation_A_identity) + tf.keras.losses.MeanAbsoluteError(y_true=self.input_B_real, y_pred=self.generation_B_identity)

        # Place holder for lambda_cycle and lambda_identity
        self.lambda_cycle = tf.keras.layers.Lambda(lambda x: x)(tf.constant(1.0))
        self.lambda_identity = tf.keras.layers.Lambda(lambda x: x)(tf.constant(1.0))

        # Generator loss
        # Generator wants to fool discriminator
        self.generator_loss_A2B = tf.keras.losses.BinaryCrossentropy(from_logits=True)(y_true=tf.ones_like(self.discrimination_B_fake), y_pred=self.discrimination_B_fake)
        self.generator_loss_B2A = tf.keras.losses.BinaryCrossentropy(from_logits=True)(y_true=tf.ones_like(self.discrimination_A_fake), y_pred=self.discrimination_A_fake)

         # Merge the two generators and the cycle loss
        self.generator_loss = (
            self.generator_loss_A2B
            + self.generator_loss_B2A
            + self.lambda_cycle * self.cycle_loss
            + self.lambda_identity * self.identity_loss
        )

        # Discriminator loss
        self.discrimination_input_A_real = self.discriminator(self.input_A_real, reuse=True)
        self.discrimination_input_B_real = self.discriminator(self.input_B_real, reuse=True)
        self.discrimination_input_A_fake = self.discriminator(self.input_A_fake, reuse=True)
        self.discrimination_input_B_fake = self.discriminator(self.input_B_fake, reuse=True)

        # Discriminator wants to classify real and fake correctly
        self.discriminator_loss_input_A_real = tf.keras.losses.BinaryCrossentropy(from_logits=True)(y_true=tf.ones_like(self.discrimination_input_A_real), y_pred=self.discrimination_input_A_real)
        self.discriminator_loss_input_A_fake = tf.keras.losses.BinaryCrossentropy(from_logits=True)(y_true=tf.zeros_like(self.discrimination_input_A_fake), y_pred=self.discrimination_input_A_fake)
        self.discriminator_loss_A = (self.discriminator_loss_input_A_real + self.discriminator_loss_input_A_fake) / 2

        self.discriminator_loss_input_B_real = tf.keras.losses.BinaryCrossentropy(from_logits=True)(y_true=tf.ones_like(self.discrimination_input_B_real), y_pred=self.discrimination_input_B_real)
        self.discriminator_loss_input_B_fake = tf.keras.losses.BinaryCrossentropy(from_logits=True)(y_true=tf.zeros_like(self.discrimination_input_B_fake), y_pred=self.discrimination_input_B_fake)
        self.discriminator_loss_B = (self.discriminator_loss_input_B_real + self.discriminator_loss_input_B_fake) / 2

        # Merge the two discriminators into one
        self.discriminator_loss = self.discriminator_loss_A + self.discriminator_loss_B

        # Categorize variables because we have to optimize the two sets of the variables separately
        trainable_variables = tf.trainable_variables()
        self.discriminator_vars = [var for var in trainable_variables if 'discriminator' in var.name]
        self.generator_vars = [var for var in trainable_variables if 'generator' in var.name]

        # Reserved for test
        self.generation_B_test = self.generator(self.input_A_test, reuse=True, scope_name='generator_A2B')
        self.generation_A_test = self.generator(self.input_B_test, reuse=True, scope_name='generator_B2A')

    def optimizer_initializer(self):
        self.generator_learning_rate = tf.keras.optimizers.Adam(learning_rate=0.0001)
        self.discriminator_learning_rate = tf.keras.optimizers.Adam(learning_rate=0.0001)
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=self.generator_learning_rate)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=self.discriminator_learning_rate)

    def train(self, input_A, input_B):
        generation_A, generation_B, generator_loss, _, generator_summaries = self.sess.run(
            [self.generation_A, self.generation_B, self.generator_loss, self.generator_optimizer, self.generator_summaries],
            feed_dict={
                self.input_A_real: input_A,
                self.input_B_real: input_B,
                self.generator_learning_rate: 0.0001
            }
        )
        self.writer.add_summary(generator_summaries, self.train_step)

        discriminator_loss, _, discriminator_summaries = self.sess.run(
            [self.discriminator_loss, self.discriminator_optimizer, self.discriminator_summaries],
            feed_dict={
                self.input_A_real: input_A,
                self.input_B_real: input_B,
                self.input_A_fake: generation_A,
                self.input_B_fake: generation_B,
                self.discriminator_learning_rate: 0.0001
            }
        )
        self.writer.add_summary(discriminator_summaries, self.train_step)

        self.train_step += 1

        return generator_loss, discriminator_loss


    def test(self, inputs, direction):
        if direction not in ['A2B', 'B2A']:
            raise Exception('Conversion direction must be "A2B" or "B2A".')

        if direction == 'A2B':
            generation = self.sess.run(self.generation_B_test, feed_dict={self.input_A_test: inputs})
        elif direction == 'B2A':
            generation = self.sess.run(self.generation_A_test, feed_dict={self.input_B_test: inputs})

        return generation


    def save(self, directory, filename):
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.saver.save(self.sess, os.path.join(directory, filename))
        return os.path.join(directory, filename)

    def load(self, filepath):
        self.saver.restore(self.sess, filepath)

    def summary(self):
        with tf.name_scope('generator_summaries'):
            cycle_loss_summary = tf.summary.scalar('cycle_loss', self.cycle_loss)
            identity_loss_summary = tf.summary.scalar('identity_loss', self.identity_loss)
            generator_loss_A2B_summary = tf.summary.scalar('generator_loss_A2B', self.generator_loss_A2B)
            generator_loss_B2A_summary = tf.summary.scalar('generator_loss_B2A', self.generator_loss_B2A)
            generator_loss_summary = tf.summary.scalar('generator_loss', self.generator_loss)
            generator_summaries = tf.summary.merge(
                [cycle_loss_summary, identity_loss_summary, generator_loss_A2B_summary, generator_loss_B2A_summary, generator_loss_summary]
            )

        with tf.name_scope('discriminator_summaries'):
            discriminator_loss_A_summary = tf.summary.scalar('discriminator_loss_A', self.discriminator_loss_A)
            discriminator_loss_B_summary = tf.summary.scalar('discriminator_loss_B', self.discriminator_loss_B)
            discriminator_loss_summary = tf.summary.scalar('discriminator_loss', self.discriminator_loss)
            discriminator_summaries = tf.summary.merge(
                [discriminator_loss_A_summary, discriminator_loss_B_summary, discriminator_loss_summary]
            )

        return generator_summaries, discriminator_summaries



# if __name__ == '__main__':
    
#     model = CycleGAN(num_features = 24)
#     print('Graph Compile Successeded.')