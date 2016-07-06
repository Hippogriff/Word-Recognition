import matplotlib

matplotlib.use('Agg')
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Input, \
    Embedding, GRU, TimeDistributed, Merge, RepeatVector, LSTM
from keras.layers import Convolution2D, MaxPooling2D
from keras.models import Model

# from data_utils import data_utils
max_caption_len = 24
vocab_size = 37
batch_size = 3000
nb_epoch = 2
data_augmentation = True
img_rows, img_cols = 32, 100
nb_classes = 2

inputs = Input(shape=(1, img_rows, img_cols,))
conv1 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(inputs)
conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')
conv2_1 = conv2(conv1)
conv2_2 = conv2(conv2_1)
maxpool1 = MaxPooling2D(pool_size=(2, 2))(conv2_2)

conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(maxpool1)
conv4 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')
conv4_1 = conv4(conv3)
conv4_2 = conv4(conv4_1)
maxpool2 = MaxPooling2D(pool_size=(2, 2))(conv4_2)

conv5 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(maxpool2)
conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')
conv6_1 = conv6(conv5)
conv6_2 = conv6(conv6_1)
maxpool3 = MaxPooling2D(pool_size=(2, 2))(conv6_2)

conv7 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(maxpool3)

conv8 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')
conv8_1 = conv8(conv7)
conv8_2 = conv8(conv8_1)

flat = Flatten()(conv8_2)
fc1 = Dense(1024, init='he_normal', activation='relu')(flat)
fc2 = Dense(1024, init='he_normal', activation='sigmoid')(fc1)
repeat = RepeatVector(1)(fc2)
image_model = Model(input=inputs, output=repeat)

embedding_model = Sequential()
embedding_model.add(Embedding(input_dim=vocab_size, output_dim=1024, input_length=max_caption_len))

model = Sequential()
model.add(Merge([image_model, embedding_model], mode='concat', concat_axis=1))

model.add(LSTM(output_dim=1024, return_sequences=True))
model.add(TimeDistributed(Dense(vocab_size)))

model.add(Activation('softmax'))
print (model.layers[-1].output_shape)

model.compile(loss='categorical_crossentropy', optimizer='adam')

print (model.summary())

