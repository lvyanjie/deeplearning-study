import keras

def VGG16(input_shape=(224,224,3), classes = 1000):
    '''
    VGG16
    :param input_shape: tuple, input tensor shape
    :param classes: integer, classes defined by your dataset
    
    returns:  
    keras model
    '''
    x_input = keras.layers.Input(input_shape)
    
    #block1
    x = keras.layers.convolutional.Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu', padding = 'same', name = 'block1_conv1')(x_input)
    x = keras.layers.convolutional.Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu', padding = 'same', name = 'block1_conv2')(x)
    x = keras.layers.pooling.MaxPooling2D(pool_size = (2, 2), strides = (2, 2), name = 'block1_pool')(x)
    
    #block2
    x = keras.layers.convolutional.Conv2D(filters = 128, kernel_size = (3, 3), activation = 'relu', padding = 'same', name = 'block2_conv1')(x)
    x = keras.layers.convolutional.Conv2D(filters = 128, kernel_size = (3, 3), activation = 'relu', padding = 'same', name = 'block2_conv2')(x)
    x = keras.layers.pooling.MaxPooling2D(pool_size = (2, 2), strides = (2, 2), name = 'block2_pool')(x)
    
    #block3
    x = keras.layers.convolutional.Conv2D(filters = 256, kernel_size = (3, 3), activation = 'relu', padding = 'same', name = 'block3_conv1')(x)
    x = keras.layers.convolutional.Conv2D(filters = 256, kernel_size = (3, 3), activation = 'relu', padding = 'same', name = 'block3_conv2')(x)
    x = keras.layers.convolutional.Conv2D(filters = 256, kernel_size = (3, 3), activation = 'relu', padding = 'same', name = 'block3_conv3')(x)
    x = keras.layers.pooling.MaxPooling2D(pool_size = (2, 2), strides = (2, 2), name = 'block3_pool')(x)
    
    #block4
    x = keras.layers.convolutional.Conv2D(filters = 512, kernel_size = (3, 3), activation = 'relu', padding = 'same', name = 'block4_conv1')(x)
    x = keras.layers.convolutional.Conv2D(filters = 512, kernel_size = (3, 3), activation = 'relu', padding = 'same', name = 'block4_conv2')(x)
    x = keras.layers.convolutional.Conv2D(filters = 512, kernel_size = (3, 3), activation = 'relu', padding = 'same', name = 'block4_conv3')(x)
    x = keras.layers.pooling.MaxPooling2D(pool_size = (2, 2), strides = (2, 2), name = 'block4_pool')(x)
    
    #block5
    x = keras.layers.convolutional.Conv2D(filters = 512, kernel_size = (3, 3), activation = 'relu', padding = 'same', name = 'block5_conv1')(x)
    x = keras.layers.convolutional.Conv2D(filters = 512, kernel_size = (3, 3), activation = 'relu', padding = 'same', name = 'block5_conv2')(x)
    x = keras.layers.convolutional.Conv2D(filters = 512, kernel_size = (3, 3), activation = 'relu', padding = 'same', name = 'block5_conv3')(x)
    x = keras.layers.pooling.MaxPooling2D(pool_size = (2, 2), strides = (2, 2), name = 'block5_pool')(x)
    
    #flatten
    x = keras.layers.core.Flatten(name = 'flatten')(x)
    
    #fc1
    x = keras.layers.core.Dense(units = 4096, activation = 'relu', name = 'fc1')(x)
    #fc2
    x = keras.layers.core.Dense(units = 4096, activation = 'relu', name = 'fc2')(x)
    #classification
    x = keras.layers.core.Dense(units = classes, activation='softmax', name='predictions')(x)
    
    model = keras.models.Model(inputs = x_input, outputs = x, name = 'vgg16')
    return model