import numpy as np

import tensorflow as tf
from tensorflow.keras import layers

X_train = np.arange(10).reshape((10, 1))
y_train = np.array([1.0, 1.3, 3.1, 2.0, 5.0, 6.3,
                    6.6, 7.4, 8.0, 9.0])


X_train_norm = (X_train - np.mean(X_train)) / np.std(X_train)
ds_train = tf.data.Dataset.from_tensor_slices(
    (tf.cast(X_train_norm, tf.float32),
     tf.cast(y_train, tf.float32)))


class CustomModel(tf.keras.Model):
    def __init__(self):
        super(CustomModel, self).__init__()
        
        self.w = tf.Variable(0.0, name='weight')
        self.b = tf.Variable(0.0, name='bias')
        
    def call(self, x):
        return self.w * x + self.b
    
    
def loss_fn(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

def train(model, inputs, outputs, learning_rate):
    with tf.GradientTape() as tape:
        current_loss = loss_fn(model(inputs), outputs)
    dw, db = tape.gradient(current_loss, [model.w, model.b])
    model.w.assign_sub(learning_rate * dw)
    model.b.assign_sub(learning_rate * db)
    
    
tf.random.set_seed(42)
num_epochs = 200
log_steps = 100
learning_rate = 0.001
batch_size = 1
steps_per_epoch = int(np.ceil(len(y_train) / batch_size))

ds = ds_train.shuffle(buffer_size=len(y_train))
ds = ds.repeat(count=None)
ds = ds.batch(1)
ws, bs = [], []

if __name__ == "__main__":
    model = CustomModel()
    model.build(input_shape=(None, 1))
    model.summary()
    
    for i, batch in enumerate(ds):
        if i >= steps_per_epoch * num_epochs:
            break
        
        ws.append(model.w.numpy())
        bs.append(model.b.numpy())
        
        bx, by = batch
        loss_val = loss_fn(model(bx), by)
        
        train(model, bx, by, learning_rate=learning_rate)
        if i % log_steps == 0:
            print(f'Epoch {int(i / steps_per_epoch):4d} Step {i:2d} Loss {loss_val:6.4f}')
