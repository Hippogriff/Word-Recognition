import matplotlib

matplotlib.use('Agg')
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Input, \
    Embedding, TimeDistributed, Merge, RepeatVector, LSTM, Dropout
from keras.layers import Convolution2D, MaxPooling2D
from keras.models import Model


def load_model():
    max_caption_len = 24
    vocab_size = 38
    img_rows, img_cols = 32, 100

    inputs = Input(shape=(1, img_rows, img_cols,))
    conv1 = Convolution2D(64, 3, 3, activation='relu',
                          border_mode='same')(inputs)
    drop1 = Dropout(0.5)(conv1)
    conv2 = Convolution2D(64, 3, 3, activation='relu',
                          border_mode='same')
    conv2_1 = conv2(drop1)
    drop2 = Dropout(.5)(conv2_1)
    conv2_2 = conv2(drop2)
    drop3 = Dropout(0.5)(conv2_2)
    maxpool1 = MaxPooling2D(pool_size=(2, 2))(drop3)

    conv3 = Convolution2D(128, 3, 3, activation='relu',
                          border_mode='same')(maxpool1)
    drop4 = Dropout(0.5)(conv3)
    conv4 = Convolution2D(128, 3, 3, activation='relu',
                          border_mode='same')
    conv4_1 = conv4(drop4)
    drop5 = Dropout(0.5)(conv4_1)

    conv4_2 = conv4(drop5)
    drop6 = Dropout(.5)(conv4_2)
    maxpool2 = MaxPooling2D(pool_size=(2, 2))(drop6)

    conv5 = Convolution2D(256, 3, 3, activation='relu',
                          border_mode='same')(maxpool2)
    drop7 = Dropout(conv5)
    conv6 = Convolution2D(256, 3, 3, activation='relu',
                          border_mode='same')
    conv6_1 = conv6(drop7)
    drop8 = Dropout(0.5)(conv6_1)
    conv6_2 = conv6(drop8)
    drop9 = Dropout(.5)(conv6_2)
    maxpool3 = MaxPooling2D(pool_size=(2, 2))(drop9)

    conv7 = Convolution2D(512, 3, 3, activation='relu',
                          border_mode='same')(maxpool3)
    drop10 = Dropout(0.5)(conv7)
    conv8 = Convolution2D(512, 3, 3, activation='relu',
                          border_mode='same')
    conv8_1 = conv8(drop10)
    drop11 = Dropout(0.5)(conv8_1)
    conv8_2 = conv8(drop11)
    drop12 = Dropout(0.5)(conv8_2)

    flat = Flatten()(drop12)
    fc1 = Dense(1024, init='he_normal', activation='relu')(flat)
    drop13 = Dropout(0.5)(fc1)
    fc2 = Dense(1024, init='he_normal', activation='softmax')(drop13)
    drop14 = Dropout(0.5)(fc2)
    repeat = RepeatVector(1)(drop14)

    image_model = Model(input=inputs, output=repeat)

    embedding_model = Sequential()
    embedding_model.add(Embedding(input_dim=vocab_size,
                                  output_dim=1024,
                                  input_length=max_caption_len))

    model = Sequential()
    model.add(Merge([image_model, embedding_model],
                    mode='concat',
                    concat_axis=1))
    model.add(LSTM(output_dim=1024, return_sequences=True,
                   dropout_U=0.2,
                   dropout_W=0.2))
    model.add(TimeDistributed(Dense(vocab_size)))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  matrics=['accuracy'])

    print (model.summary())
    return model
