from params import *
import os
from module import discriminator, generator_gatedcnn
from utils import l1_loss, l2_loss
from datetime import datetime
import zipfile
import re
import tensorflow as tf
import tensorflow.compat.v1 as v1

v1.disable_v2_behavior()


class CycleGAN(object):

    def __init__(self, num_features, discriminator=discriminator, generator=generator_gatedcnn, mode='train'):

        self.num_features = num_features
        # [batch_size, num_features, num_frames]
        self.input_shape = [None, num_features, None]

        self.discriminator = discriminator
        self.generator = generator
        self.mode = mode

        self.build_model()
        self.optimizer_initializer()

        self.saver = v1.train.Saver()
        self.sess = v1.Session()
        self.sess.run(v1.global_variables_initializer())

        if self.mode == 'train':
            self.train_step = 0
            now = datetime.now()
            self.log_dir = os.path.join(
                log_dir, f"{model_prefix}:{now.strftime('%d-%B-%Y-%I%p')}")
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            self.writer = v1.summary.FileWriter(self.log_dir, v1.get_default_graph())
            self.generator_summaries, self.discriminator_summaries = self.summary()

    def build_model(self):

        # Placeholders for real training samples
        self.input_A_real = v1.placeholder(
            tf.float32, shape=self.input_shape, name='input_A_real')
        self.input_B_real = v1.placeholder(
            tf.float32, shape=self.input_shape, name='input_B_real')
        # Placeholders for fake generated samples
        self.input_A_fake = v1.placeholder(
            tf.float32, shape=self.input_shape, name='input_A_fake')
        self.input_B_fake = v1.placeholder(
            tf.float32, shape=self.input_shape, name='input_B_fake')
        # Placeholder for test samples
        self.input_A_test = v1.placeholder(
            tf.float32, shape=self.input_shape, name='input_A_test')
        self.input_B_test = v1.placeholder(
            tf.float32, shape=self.input_shape, name='input_B_test')

        self.generation_B = self.generator(
            inputs=self.input_A_real, reuse=False, scope_name='generator_A2B')
        self.cycle_A = self.generator(
            inputs=self.generation_B, reuse=False, scope_name='generator_B2A')

        self.generation_A = self.generator(
            inputs=self.input_B_real, reuse=True, scope_name='generator_B2A')
        self.cycle_B = self.generator(
            inputs=self.generation_A, reuse=True, scope_name='generator_A2B')

        self.generation_A_identity = self.generator(
            inputs=self.input_A_real, reuse=True, scope_name='generator_B2A')
        self.generation_B_identity = self.generator(
            inputs=self.input_B_real, reuse=True, scope_name='generator_A2B')

        self.discrimination_A_fake = self.discriminator(
            inputs=self.generation_A, reuse=False, scope_name='discriminator_A')
        self.discrimination_B_fake = self.discriminator(
            inputs=self.generation_B, reuse=False, scope_name='discriminator_B')

        # Cycle loss
        self.cycle_loss = l1_loss(y=self.input_A_real, y_hat=self.cycle_A) + \
            l1_loss(y=self.input_B_real, y_hat=self.cycle_B)

        # Identity loss
        self.identity_loss = l1_loss(y=self.input_A_real, y_hat=self.generation_A_identity) + \
            l1_loss(y=self.input_B_real, y_hat=self.generation_B_identity)

        # Place holder for lambda_cycle and lambda_identity
        self.lambda_cycle = tf.compat.v1.placeholder(
            tf.float32, shape=[], name='lambda_cycle')
        self.lambda_identity = tf.compat.v1.placeholder(
            tf.float32, shape=[], name='lambda_identity')

        # Generator loss
        # Generator wants to fool discriminator
        self.generator_loss_A2B = l2_loss(y=tf.ones_like(
            self.discrimination_B_fake), y_hat=self.discrimination_B_fake)
        self.generator_loss_B2A = l2_loss(y=tf.ones_like(
            self.discrimination_A_fake), y_hat=self.discrimination_A_fake)

        # Merge the two generators and the cycle loss
        self.generator_loss = self.generator_loss_A2B + self.generator_loss_B2A + \
            self.lambda_cycle * self.cycle_loss + self.lambda_identity * self.identity_loss

        # Discriminator loss
        self.discrimination_input_A_real = self.discriminator(
            inputs=self.input_A_real, reuse=True, scope_name='discriminator_A')
        self.discrimination_input_B_real = self.discriminator(
            inputs=self.input_B_real, reuse=True, scope_name='discriminator_B')
        self.discrimination_input_A_fake = self.discriminator(
            inputs=self.input_A_fake, reuse=True, scope_name='discriminator_A')
        self.discrimination_input_B_fake = self.discriminator(
            inputs=self.input_B_fake, reuse=True, scope_name='discriminator_B')

        # Discriminator wants to classify real and fake correctly
        self.discriminator_loss_input_A_real = l2_loss(y=tf.ones_like(
            self.discrimination_input_A_real), y_hat=self.discrimination_input_A_real)
        self.discriminator_loss_input_A_fake = l2_loss(y=tf.zeros_like(
            self.discrimination_input_A_fake), y_hat=self.discrimination_input_A_fake)
        self.discriminator_loss_A = (
            self.discriminator_loss_input_A_real + self.discriminator_loss_input_A_fake) / 2

        self.discriminator_loss_input_B_real = l2_loss(y=tf.ones_like(
            self.discrimination_input_B_real), y_hat=self.discrimination_input_B_real)
        self.discriminator_loss_input_B_fake = l2_loss(y=tf.zeros_like(
            self.discrimination_input_B_fake), y_hat=self.discrimination_input_B_fake)
        self.discriminator_loss_B = (
            self.discriminator_loss_input_B_real + self.discriminator_loss_input_B_fake) / 2

        # Merge the two discriminators into one
        self.discriminator_loss = self.discriminator_loss_A + self.discriminator_loss_B

        # Categorize variables because we have to optimize the two sets of the variables separately
        trainable_variables = v1.trainable_variables()
        self.discriminator_vars = [
            var for var in trainable_variables if 'discriminator' in var.name]
        self.generator_vars = [
            var for var in trainable_variables if 'generator' in var.name]
        # for var in t_vars: print(var.name)

        # Reserved for test
        self.generation_B_test = self.generator(
            inputs=self.input_A_test, reuse=True, scope_name='generator_A2B')
        self.generation_A_test = self.generator(
            inputs=self.input_B_test, reuse=True, scope_name='generator_B2A')

    def optimizer_initializer(self):

        self.generator_learning_rate = v1.placeholder(
            tf.float32, None, name='generator_learning_rate')
        self.discriminator_learning_rate = v1.placeholder(
            tf.float32, None, name='discriminator_learning_rate')
        self.discriminator_optimizer = v1.train.AdamOptimizer(learning_rate=self.discriminator_learning_rate, beta1=0.5).minimize(
            self.discriminator_loss, var_list=self.discriminator_vars)
        self.generator_optimizer = v1.train.AdamOptimizer(learning_rate=self.generator_learning_rate, beta1=0.5).minimize(
            self.generator_loss, var_list=self.generator_vars)

    def train(self, input_A, input_B, lambda_cycle, lambda_identity, generator_learning_rate, discriminator_learning_rate):

        generation_A, generation_B, generator_loss, _, generator_summaries = self.sess.run(
            [self.generation_A, self.generation_B, self.generator_loss, self.generator_optimizer, self.generator_summaries], \
            feed_dict = {self.lambda_cycle: lambda_cycle, self.lambda_identity: lambda_identity, self.input_A_real: input_A, self.input_B_real: input_B, self.generator_learning_rate: generator_learning_rate})

        self.writer.add_summary(generator_summaries, self.train_step)

        discriminator_loss, _, discriminator_summaries = self.sess.run([self.discriminator_loss, self.discriminator_optimizer, self.discriminator_summaries], \
            feed_dict = {self.input_A_real: input_A, self.input_B_real: input_B, self.discriminator_learning_rate: discriminator_learning_rate, self.input_A_fake: generation_A, self.input_B_fake: generation_B})

        self.writer.add_summary(discriminator_summaries, self.train_step)

        # Increment the training step
        self.train_step += 1

        return generator_loss, discriminator_loss

    def test(self, inputs, direction):

        if direction == 'A2B':
            generation = self.sess.run(self.generation_B_test, feed_dict={
                                       self.input_A_test: inputs})
        elif direction == 'B2A':
            generation = self.sess.run(self.generation_A_test, feed_dict={
                                       self.input_B_test: inputs})
        else:
            raise Exception('Conversion direction must be specified.')

        return generation

    def save(self, directory, filename, compressedname):

        if not os.path.exists(directory):
            os.makedirs(directory)

        # Save the session variables to a checkpoint file
        self.saver.save(self.sess, os.path.join(directory, filename))

        # Get the filename of the variables file
        variables_filename = ''
        for file_name in os.listdir(directory):
            if file_name.startswith(f'{model_prefix}.ckpt.data'):
                variables_filename = file_name
                break

        if variables_filename == '':
            raise ValueError("Variables file not found.")

        # Extract the suffix from the variables filename
        suffix_pattern = r'.*\.data-(\d+)-of-(\d+)'
        match = re.match(suffix_pattern, variables_filename)
        if match:
            suffix = match.group(1)
        else:
            raise ValueError("Unable to extract suffix from variables filename.")


        # Compress the checkpoint file using zip
        with zipfile.ZipFile(os.path.join(directory, compressedname), 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
            # Write the checkpoint file
            zipf.write(os.path.join(directory, 'checkpoint'), arcname='checkpoint')

            # Write the graph file
            zipf.write(os.path.join(directory, f'{model_prefix}.ckpt.meta'), arcname=f'{model_prefix}.ckpt.meta')

            # Write the variables file
            zipf.write(os.path.join(directory, variables_filename), arcname=f'{model_prefix}.ckpt.data-{suffix}-of-00001')
            zipf.write(os.path.join(directory, f'{model_prefix}.ckpt.index'), arcname=f'{model_prefix}.ckpt.index')

        # Delete the uncompressed checkpoint file
        tf.io.gfile.remove(os.path.join(directory, filename))


    def load(self, filepath,compressedpath):
        # Extract the compressed model file
        with zipfile.ZipFile(compressedpath, 'r') as zipf:
            zipf.extractall(filepath)
        self.saver.restore(self.sess, filepath+f'/{model_prefix}.ckpt')

        os.remove(filepath)


    def summary(self):

        with v1.name_scope('generator_summaries'):
            cycle_loss_summary =v1.summary.scalar('cycle_loss', self.cycle_loss)
            identity_loss_summary =v1.summary.scalar('identity_loss', self.identity_loss)
            generator_loss_A2B_summary =v1.summary.scalar('generator_loss_A2B', self.generator_loss_A2B)
            generator_loss_B2A_summary =v1.summary.scalar('generator_loss_B2A', self.generator_loss_B2A)
            generator_loss_summary =v1.summary.scalar('generator_loss', self.generator_loss)
            generator_summaries =v1.summary.merge([cycle_loss_summary, identity_loss_summary, generator_loss_A2B_summary, generator_loss_B2A_summary, generator_loss_summary])

        with v1.name_scope('discriminator_summaries'):
            discriminator_loss_A_summary =v1.summary.scalar('discriminator_loss_A', self.discriminator_loss_A)
            discriminator_loss_B_summary =v1.summary.scalar('discriminator_loss_B', self.discriminator_loss_B)
            discriminator_loss_summary =v1.summary.scalar('discriminator_loss', self.discriminator_loss)
            discriminator_summaries =v1.summary.merge([discriminator_loss_A_summary, discriminator_loss_B_summary, discriminator_loss_summary])

        return generator_summaries, discriminator_summaries
