import keras.backend as K
from context import context_output_shape, context_vector
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Input, \
    Embedding, TimeDistributed, Merge, RepeatVector, LSTM, Dropout
from keras.layers import Convolution2D, MaxPooling2D, Lambda
from keras.models import Model

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
drop7 = Dropout(0.5)(conv5)
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
conv8_2 = conv8(drop11) # shape of N, 512, 4, 12

print (conv8_2._keras_shape)

# I = Flatten()(a) # shape 24576
flat = Flatten()(conv8_2)
I = RepeatVector(24)(flat)

print ("Shape of I", I._keras_shape)

phi = TimeDistributed(Dense(12, init='he_normal', activation='relu'))(I)

print (phi._keras_shape)

image_model = Model(input=inputs, output=phi)
image_model.summary()
a = conv8_2 # shape of N, 512, 4, 12


language_model = Sequential()
language_model.add(Embedding(input_dim=vocab_size,
                              output_dim=1024,
                              input_length=max_caption_len))

language_model.add(LSTM(output_dim=1024, return_sequences=True,
               dropout_U=0.2,
               dropout_W=0.2))

s = language_model.layers[-1] # Shape N, 24, 1024
print (s.output_shape)

language_model.add(TimeDistributed(Dense(12, init='he_normal', activation='relu'))) # let us make the dimensions
# explicit
print (language_model.layers[-1].output_shape)

attention_model = Sequential()
attention_model.add(Merge([image_model, language_model], mode='sum'))
attention_model.add(Activation('tanh')) # we have got the tau

# I have little doubts about this, need to check whether is does the normalization over the
# 12 outputs or complete 24 outputs across time span.
attention_model.add((Activation('softmax'))) # we have ahpa weights

print (attention_model.layers[-1].output_shape) # None, 24, 12)

# attention_model.add(Lambda(function=context_vector, output_shape=context_output_shape, arguments=a))
alpha = attention_model.layers[-1].output
print(type(alpha))

final_model = Sequential()
final_model.add(Merge([attention_model.layers[-1], image_model.layers[22]], mode='mul'))

'''
tau = K.tanh(mapping_a + mapping_s)

tau_exp = K.exp(tau)
tau_ex_sum = K.sum(tau_exp)
alpha = tau_exp / tau_ex_sum
c_star = alpha*a
c_star_flatten = K.reshape(c_star, shape=(a.output_shape[1]*a.output_shape[2],
                                          a.output_shape[3]))
# This will be our input to the next layer of LSTM
c = K.sum(c_star_flatten)





joint_LSTM = LSTM(output_dim=1024, return_sequences=True,
               dropout_U=0.2,
               dropout_W=0.2)(c)



model.add(TimeDistributed(Dense(vocab_size)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              matrics=['accuracy'])

print (model.summary())

'''