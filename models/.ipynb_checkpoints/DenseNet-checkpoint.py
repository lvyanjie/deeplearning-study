import keras

def conv_block(x, growth_rate, name):
    '''
    conv block
    :param x: input tensor
    :param growth_rate: integer, output feature map channel
    :param name: string, dense block name
    
    returns:
    output tensor
    '''
    x1 = keras.layers.normalization.BatchNormalization(axis = 3, name = name + '_0_bn')(x)
    x1 = keras.layers.Activation('relu', name = name + '_0_relu')(x1)
    x1 = keras.layers.convolutional.Conv2D(filters = 4 * growth_rate, kernel_size = (1, 1), strides = (1, 1), padding = 'same',
                                         kernel_initializer = 'glorot_uniform', use_bias = False, name = name + '_0_conv')(x1)
    
    x1 = keras.layers.normalization.BatchNormalization(axis = 3, name = name + '_1_bn')(x1)
    x1 = keras.layers.Activation('relu', name = name + '_1_relu')(x1)
    x1 = keras.layers.convolutional.Conv2D(filters = growth_rate, kernel_size = (3, 3), strides = (1, 1), padding = 'same',
                                          kernel_initializer = 'glorot_uniform', use_bias = False, name = name + '_1_conv')(x1)
    
    x = keras.layers.Concatenate(axis = 3, name = name + '_concat')([x, x1])#保留每层得到的feature map
    return x

#不同数量的conv_block组成了DenseBlock模块
def dense_block(x, blocks, name):
    '''
    dense block
    :param x: input tensor
    :param blocks: integer, number of bottleneck layers
    :param name: string , name of dense block
    
    returns:
    output tensor    
    '''
    for i in range(blocks):
        x = conv_block(x, 32, name = name + '_convblock'+str(i+1))
    
    return x

def transition_block(x, theta, name):
    '''
    trainsition block
    :param x: input tensor.
    :param theta: float, compression rate at transition layers.
    :param name: string, block label.
    
    returns:
    output tensor
    '''
    x = keras.layers.normalization.BatchNormalization(axis=3, epsilon=1.001e-5, name=name + '_bn')(x)#epsilon防止除零错误
    x = keras.layers.Activation('relu', name=name + '_relu')(x)
    x = keras.layers.convolutional.Conv2D(filters = int(keras.backend.int_shape(x)[3] * theta), kernel_size = (1, 1), strides = (1, 1), padding = 'same',
                                          kernel_initializer='glorot_uniform', use_bias=False, name=name + '_conv')(x)
    x = keras.layers.pooling.AveragePooling2D(pool_size = (2, 2), strides=2, name=name + '_pool')(x)
    return x

def DenseNet121(input_shape=(224,224,3), classes = 1000):
    '''
    DenseNet121
    :param input_shape:  tuple, input tensor shape
    :param classes: integer, classes defined by your dataset
    
    returns:   
    keras model
    '''
    x_input = keras.layers.Input(input_shape)
    
    x = keras.layers.convolutional.Conv2D(64, 7, strides=2, padding='same', use_bias=False, name='conv1/conv')(x_input)
    x = keras.layers.normalization.BatchNormalization(axis=3, epsilon=1.001e-5,name='conv1/bn')(x)
    x = keras.layers.Activation('relu', name='conv1/relu')(x)
    x = keras.layers.pooling.MaxPooling2D(3, strides=2, padding='same', name='pool1')(x)
    
    x = dense_block(x, 6, name='dense_block1')
    x = transition_block(x, 0.5, name = 'transition_block_1')
    
    x = dense_block(x, 12, name = 'dense_block_2')
    x = transition_block(x, 0.5, name = 'transition_block_2')
    
    x = dense_block(x, 24, name = 'dense_block_3')
    x = transition_block(x, 0.5, name = 'transition_block_3')
    
    x = dense_block(x, 16, name = 'dense_block_4')
    x = keras.layers.normalization.BatchNormalization(axis=3, epsilon=1.001e-5, name='bn')(x)
    
    #classification
    x = keras.layers.pooling.GlobalAveragePooling2D(name = 'global_avg_pooling')(x)
    x = keras.layers.core.Dense(classes, activation='softmax', name = 'classification')(x)
    
    model = keras.models.Model(inputs = x_input, outputs = x, name = 'DenseNet121')
    return model