import matplotlib

matplotlib.use( 'Agg' )
from model import load_model
from data_utils import data_utils
from plotting import plot
from train import train

# TODO Write the code for testing
# TODO Also think about the best way to test the accuracy of the model
# TODO Load and save all the training and testing data

max_caption_len = 24
vocab_size = 38
batch_size = 3000
nb_epoch = 2
data_augmentation = False
img_rows, img_cols = 32, 100

X_train, X_test, Words_train, Words_test, y_train, y_test = data_utils( )

model = load_model( )

past_history = train( model=model, mode=2)

plot(past_history)

