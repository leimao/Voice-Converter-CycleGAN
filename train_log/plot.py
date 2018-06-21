
import os
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np




def main():

    prefix = 'run_20180619-011402-tag-'
    discriminator_loss_A_file = prefix + 'discriminator_summaries_discriminator_loss_A.csv'
    discriminator_loss_B_file = prefix + 'discriminator_summaries_discriminator_loss_B.csv'
    generator_loss_A2B_file = prefix + 'generator_summaries_generator_loss_A2B.csv'
    generator_loss_B2A_file = prefix + 'generator_summaries_generator_loss_B2A.csv'
    cycle_loss_file = prefix + 'generator_summaries_cycle_loss.csv'
    identity_loss_file = prefix + 'generator_summaries_identity_loss.csv'

    df_discriminator_loss_A = pd.read_csv(discriminator_loss_A_file)
    df_discriminator_loss_B = pd.read_csv(discriminator_loss_B_file)
    df_generator_loss_A2B = pd.read_csv(generator_loss_A2B_file)
    df_generator_loss_B2A = pd.read_csv(generator_loss_B2A_file)
    df_cycle_loss = pd.read_csv(cycle_loss_file)
    df_identity_loss = pd.read_csv(identity_loss_file)

    cutoff_length = np.min([len(df_discriminator_loss_A), len(df_discriminator_loss_B), len(df_generator_loss_A2B), len(df_generator_loss_B2A), len(df_cycle_loss), len(df_identity_loss)])

    step_discriminator_loss_A = df_discriminator_loss_A['Step'].values[:cutoff_length]
    loss_discriminator_loss_A = df_discriminator_loss_A['Value'].values[:cutoff_length]
    step_discriminator_loss_B = df_discriminator_loss_B['Step'].values[:cutoff_length]
    loss_discriminator_loss_B = df_discriminator_loss_B['Value'].values[:cutoff_length]

    step_generator_loss_A2B = df_generator_loss_A2B['Step'].values[:cutoff_length]
    loss_generator_loss_A2B = df_generator_loss_A2B['Value'].values[:cutoff_length]
    step_generator_loss_B2A = df_generator_loss_B2A['Step'].values[:cutoff_length]
    loss_generator_loss_B2A = df_generator_loss_B2A['Value'].values[:cutoff_length]

    step_cycle_loss = df_cycle_loss['Step'].values[:cutoff_length]
    loss_cycle_loss = df_cycle_loss['Value'].values[:cutoff_length]
    step_identity_loss = df_identity_loss['Step'].values[:cutoff_length]
    loss_identity_loss = df_identity_loss['Value'].values[:cutoff_length]


    # Plot
    iteration = step_discriminator_loss_A

    fig = plt.figure(figsize=(10,4))
    plt.rc('font', weight='bold')
    plt.plot(iteration, loss_discriminator_loss_A, color = 'r', clip_on = False, label = 'Discriminator A')
    plt.plot(iteration, loss_discriminator_loss_B, color = 'b', clip_on = False, label = 'Discriminator B')
    plt.plot(iteration, loss_generator_loss_A2B, color = 'g', clip_on = False, label = 'Generator A2B')
    plt.plot(iteration, loss_generator_loss_B2A, color = 'orange', clip_on = False, label = 'Generator B2A')
    plt.legend()
    plt.ylabel('Loss', fontsize = 16, fontweight = 'bold')
    plt.xlabel('Iteration', fontsize = 16, fontweight = 'bold')
    plt.xlim(iteration[0], iteration[-1])
    fig.savefig('discriminator_discriminator.png', format = 'png', dpi = 600, bbox_inches = 'tight')
    plt.close()


    fig = plt.figure(figsize=(10,4))
    plt.rc('font', weight='bold')
    plt.plot(iteration, loss_cycle_loss, color = 'r', clip_on = False, label = 'Cycle Consistency')
    plt.plot(iteration, loss_identity_loss, color = 'b', clip_on = False, label = 'Identity Mapping')
    plt.legend()
    plt.ylabel('Loss', fontsize = 16, fontweight = 'bold')
    plt.xlabel('Iteration', fontsize = 16, fontweight = 'bold')
    plt.xlim(iteration[0], iteration[-1])
    fig.savefig('cycle_identity.png', format = 'png', dpi = 600, bbox_inches = 'tight')
    plt.close()



if __name__ == '__main__':
    
    main()