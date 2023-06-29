#############################
# Training Parameters
#############################
num_of_epochs = 1000
audio_sampling_rate = 16000

#############################
# Datsets Parameters
#############################
train_A_dir = 'gs://audio_dataset_bucket/Audios/neutral'
train_B_dir = 'gs://audio_dataset_bucket/Audios/happy'

#############################
# Model Parameters
#############################
model_prefix = 'happy'
model_dir =f'./{model_prefix}'

#############################
# Training logs Parameters
#############################
train_logs__dir ='./Train_Logs'

#############################
# Output Parameters
#############################
norm_dir = './Normalizations'
output_dir = './output'
log_dir = './Tensorboard_Logs'