import matplotlib

matplotlib.use('Agg')
from model import load_model
from data_utils import data_utils

max_caption_len = 24
vocab_size = 38
batch_size = 3000
nb_epoch = 2
data_augmentation = True
img_rows, img_cols = 32, 100

X_train, X_test, Words_train, Words_test, y_train, y_test = data_utils()

model = load_model()

history = model.fit([X_train, Words_train], y_train, batch_size=32,
          validation_data=([X_test, Words_test], y_test),
          nb_epoch=1)

# train_model()
# plot the results and training losses
# test_model()
# Save the model


