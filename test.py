'''
File to test the model
Since there is no end-to-end way to test the model, it is done in iterative way.
The idea is to one character at a time. Update the input with newly found character.
The image should be processed once and the rest of the LSTM is processed multiple times,
untill the end of the word is encountered
The function should be made such that it can be used in stand alone way.
'''
# Evaluate generator can be used
import numpy as np

def test(model, X_test, y):
    """
    :param model: The
    :param X_test: trained model, shape: N, 1, 32, 100
    :param y: Target, Shape N, 25
    :return: Percentage accuracy, number of words classified correctly.
    """
    # Make the Words vector
    predicted_words = {}
    Words = np.zeros( (X_test.shape[0], 24) )
    output = np.zeros( (X_test.shape[0], 25) )
    indices = range(X_test.shape[0])
    for i in range(24):
        for j in indices:
            if output[j, i+1] != 37: # end of word hasn't sampled yet
                Words[j, i+1] = output[j, i+1]
            else: # end of the words has come and no more testing for that instance is required
                predicted_words[i] = Words[i, 2:i]
                indices.remove(j)
                break
        output = model.predict_classes([X_test[j], Words[j]])