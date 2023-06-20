#############################
# Training Parameters
#############################
num_of_epochs = 1
audio_sampling_rate = 16000

#############################
# Datsets Parameters
#############################
train_A_dir = '/content/drive/MyDrive/Dataset/Emotional Speech Audio/Audios/neutral'
train_B_dir = '/content/drive/MyDrive/Dataset/Emotional Speech Audio/Audios/happy'

#############################
# Model Parameters
#############################
model_prefix = 'model_exp'
model_dir =f'/content/drive/MyDrive/Model Assets/Models{model_prefix}'

#############################
# Training logs Parameters
#############################
train_logs__dir ='/content/drive/MyDrive/Model Assets/Train_Logs'

#############################
# Output Parameters
#############################
norm_dir = '/content/drive/MyDrive/Model Assets/Normalizations'
output_dir = './output'
log_dir = '/content/drive/MyDrive/Model Assets/Tensorboard_Logs'