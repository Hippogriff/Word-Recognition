import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import h5py


def plot(past_history):
    val_losses = past_history.history['val_loss']
    val_axis = np.arange(len(val_losses))
    losses_batch = past_history.history['loss']
    loss_batch_axis = np.arange(len(losses_batch))
    acc = past_history.history['acc']
    acc_axis = np.arange(len(acc))

    plt.subplot(1, 2, 1)
    plt.plot(val_axis, val_losses, color="blue", label="validation_loss")
    plt.subplot(1, 2, 2)
    plt.plot(loss_batch_axis, losses_batch, color="red", label="training_loss")
    plt.savefig('losses.png')
    plt.plot(acc_axis, acc, label="Accury v/s epochs")
    plt.savefig('accuracy.png')

    with h5py.File('losses_history.h5', 'w') as hf:
        hf.create_dataset('acc', data=acc)
        hf.create_dataset('val_losses', data=val_losses)
        hf.create_dataset('losses_batch', data=losses_batch)
