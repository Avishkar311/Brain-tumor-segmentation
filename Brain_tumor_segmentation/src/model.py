import tensorflow as tf
from tensorflow.keras import layers, models


def attention_block(input_tensor, inter_channels):
    theta = layers.Conv3D(inter_channels, (1, 1, 1))(input_tensor)
    phi = layers.Conv3D(inter_channels, (1, 1, 1))(input_tensor)
    g = layers.Conv3D(inter_channels, (1, 1, 1))(input_tensor)

    attention = layers.Add()([theta, phi])
    attention = tf.nn.relu(attention)
    attention = layers.Conv3D(inter_channels, (1, 1, 1), activation='sigmoid')(attention)

    out = layers.Multiply()([attention, g])
    return out

def hybrid_3dcnn_attention_model(input_shape=(128, 128, 128, 1)):
    inputs = layers.Input(input_shape)

    conv1 = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(inputs)
    pool1 = layers.MaxPooling3D((2, 2, 2))(conv1)
    conv2 = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool1)
    pool2 = layers.MaxPooling3D((2, 2, 2))(conv2)
    attention = attention_block(pool2, 128)
    conv3 = layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same')(attention)
    pool3 = layers.MaxPooling3D((2, 2, 2))(conv3)

    flat = layers.Flatten()(pool3)
    dense1 = layers.Dense(512, activation='relu')(flat)
    outputs = layers.Dense(1, activation='sigmoid')(dense1)

    model = models.Model(inputs=inputs, outputs=outputs)
    
    return model

if __name__ == "__main__":
    model = hybrid_3dcnn_attention_model()
    model.summary()
