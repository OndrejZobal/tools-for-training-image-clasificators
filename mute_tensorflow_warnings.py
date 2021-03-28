import os

print(' [v] TensorFlow Messages are muted!')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Mute TensorFlow messages
