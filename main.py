import numpy as np
from PIL import Image
from tensorflow import keras


def Model(input_size):
    model = keras.Sequential()
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(32, activation="relu"))
    model.add(keras.layers.Dense(input_size, activation="sigmoid"))
    model.compile(keras.optimizers.Adam(), keras.losses.MeanSquaredError(), metrics=["accuracy"])
    return model

def Data():
    from keras.datasets import mnist
    (images, _), (_, _) = mnist.load_data()
    images = images.astype('float32') / 255.
    images = images.reshape(-1,28*28)
    return images

def FeatureExtractor(model, encoder_layer_index = 1):
    return keras.Model(inputs=model.inputs, outputs=model.get_layer(index=encoder_layer_index).output)

def ShowImage(model, images,input_shape, image_index = 0):
    img_data = [images[image_index]]
    img_data = np.array(img_data)
    img = Image.fromarray(   (model(img_data).numpy().reshape(input_shape) * 255.0).astype(np.uint8)    )
    img.show()



if __name__ == "__main__":
    size = (28,28)
    input_size = size[0]*size[1]*1
    input_shape = (size[0],size[1])

    model  = Model(input_size)
    images = Data()

    model.fit(  images, 
                images, 
                epochs=50,
                verbose=1,
                batch_size=256,
                shuffle=True
            )

    ShowImage(model,images,input_shape, image_index=0)

