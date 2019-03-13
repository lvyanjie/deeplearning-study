import keras

def identity_block(x, f, filters, stage, block):
    '''
    identity block
    :param x: input tensor (n, W, H, C)
    :param f: integer, conv kernel size
    :param filters: integer list, the number of filters in the conv layers
    :param stage: integer, position in the network(naem)
    :param block: string, name the layers, position in the network
    
    returns:
    :param x: output tensor(n, W, H, C)
    '''
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    F1, F2, F3 = filters
    
    x_shortcut = x#输入输出一致
    
    #conv1
    x = keras.layers.convolutional.Conv2D(filters = F1, kernel_size = (1, 1), strides = (1, 1), padding = 'valid',
                                          kernel_initializer = keras.initializers.glorot_uniform(seed=0), 
                                          name = conv_name_base + '2a')(x)
    x = keras.layers.normalization.BatchNormalization(axis = 3, name = bn_name_base + '2a')(x)
    x = keras.layers.Activation('relu')(x)
    
    #conv2
    x = keras.layers.convolutional.Conv2D(filters = F2, kernel_size = (f, f), strides = (1, 1), padding = 'same',
                                         kernel_initializer = keras.initializers.glorot_uniform(seed=0),
                                         name = conv_name_base + '2b')(x)
    x = keras.layers.normalization.BatchNormalization(axis = 3, name = bn_name_base + '2b')(x)
    x = keras.layers.Activation('relu')(x)
    
    #conv3
    x = keras.layers.convolutional.Conv2D(filters = F3, kernel_size = (1, 1), strides = (1, 1), padding = 'valid',
                                         kernel_initializer = keras.initializers.glorot_uniform(seed=0),
                                         name = conv_name_base + '2c')(x)
    x = keras.layers.normalization.BatchNormalization(axis = 3, name = bn_name_base + '2c')(x)
    
    #add
    x = keras.layers.Add()([x, x_shortcut])
    x = keras.layers.Activation('relu')(x)
    
    return x

def convolutional_block(x, f, filters, stage, block, stride=2):
    '''
    convolutional block
    :param x: input tensor (n, W, H, C)
    :param f: integer, conv kernel size
    :param filters: integer list, the number of filters in the conv layers
    :param stage: integer, position in the network(naem)
    :param block: string, name the layers, position in the network
    :param stride: integer, stride params to be usedf
    
    returns:
    :param x: output tensor(n, W, H, C)
    '''
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    F1, F2, F3 = filters
    
    x_shortcut = x#输入输出不一致
    
    #conv1
    x = keras.layers.convolutional.Conv2D(filters = F1, kernel_size = (1, 1), strides = (stride, stride), padding = 'valid',
                                         kernel_initializer = keras.initializers.glorot_uniform(seed=0),
                                         name = conv_name_base + '2a')(x)
    x = keras.layers.normalization.BatchNormalization(axis = 3, name = bn_name_base + '2a')(x)
    x = keras.layers.Activation('relu')(x)
    
    #conv2
    x = keras.layers.convolutional.Conv2D(filters = F2, kernel_size = (f, f), strides = (1, 1), padding = 'same',
                                         kernel_initializer = keras.initializers.glorot_uniform(seed=0),
                                         name = conv_name_base + '2b')(x)
    x = keras.layers.normalization.BatchNormalization(axis = 3, name = bn_name_base + '2b')(x)
    x = keras.layers.Activation('relu')(x)
    
    #conv3
    x = keras.layers.convolutional.Conv2D(filters = F3, kernel_size = (1, 1), strides = (1, 1), padding = 'valid',
                                         kernel_initializer = keras.initializers.glorot_uniform(seed=0),
                                         name = conv_name_base + '2c')(x)
    x = keras.layers.normalization.BatchNormalization(axis = 3, name = bn_name_base + '2c')(x)
    
    #shortcutx
    x_shortcut = keras.layers.convolutional.Conv2D(filters = F3, kernel_size = (1, 1), strides = (stride, stride), padding = 'valid',
                                                  kernel_initializer = keras.initializers.glorot_uniform(seed=0),
                                                  name = conv_name_base + '1')(x_shortcut)
    x_shortcut = keras.layers.normalization.BatchNormalization(axis = 3, name = bn_name_base + '1')(x_shortcut)
    
    #add
    x = keras.layers.Add()([x, x_shortcut])
    x = keras.layers.Activation('relu')(x)
    return x

def ResNet50(input_shape=(224,224,3), classes = 1000, use_dropout = True, dropout_rate = 0.2):
    '''
    resnet50
    :param input_shape:  tuple, input tensor shape
    :param classes: integer, classes defined by your dataset
    :param use_dropout: bool, use dropout or not
    :param dropout_rate: float, only valid if use_dropout is true
    
    returns:
    
    keras model
    
    '''
    
    x_input = keras.layers.Input(input_shape)
    x = keras.layers.convolutional.ZeroPadding2D(padding = (3, 3))(x_input)
    
    #stage1
    x = keras.layers.convolutional.Conv2D(filters = 64, kernel_size = (7, 7), strides = (2, 2), 
                                         kernel_initializer = keras.initializers.glorot_uniform(seed=0), name='conv1')(x)
    x = keras.layers.normalization.BatchNormalization(axis = 3, name = 'bn_conv1')(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.pooling.MaxPooling2D(pool_size = (3, 3), strides = (2, 2))(x)
    
    #stage2
    x = convolutional_block(x = x, f = 3, filters = [64, 64, 256], stage = 2, block = 'a', stride = 1)
    x = identity_block(x = x, f = 3, filters = [64, 64, 256], stage = 2, block = 'b')
    x = identity_block(x = x, f = 3, filters = [64, 64, 256], stage = 2, block = 'c')
    
    #stage3
    x = convolutional_block(x = x, f = 3, filters = [128, 128, 512], stage = 3, block = 'a', stride = 2)
    x = identity_block(x = x, f = 3, filters = [128, 128, 512], stage = 3, block = 'b')
    x = identity_block(x = x, f = 3, filters = [128, 128, 512], stage = 3, block = 'c')
    x = identity_block(x = x, f = 3, filters = [128, 128, 512], stage = 3, block = 'd')
    
    #stage4
    x = convolutional_block(x = x, f = 3, filters = [256, 256, 1024], stage = 4, block = 'a', stride = 2)
    x = identity_block(x = x, f = 3, filters = [256, 256, 1024], stage = 4, block = 'b')
    x = identity_block(x = x, f = 3, filters = [256, 256, 1024], stage = 4, block = 'c')
    x = identity_block(x = x, f = 3, filters = [256, 256, 1024], stage = 4, block = 'd')
    x = identity_block(x = x, f = 3, filters = [256, 256, 1024], stage = 4, block = 'e')
    x = identity_block(x = x, f = 3, filters = [256, 256, 1024], stage = 4, block = 'f')
    
    #stage5
    x = convolutional_block(x = x, f = 3, filters = [512, 512, 2048], stage = 5, block = 'a', stride = 2)
    x = identity_block(x = x, f = 3, filters = [512, 512, 2048], stage = 5, block = 'b')
    x = identity_block(x = x, f = 3, filters = [512, 512, 2048], stage = 5, block = 'c')
    
    #avgpool
    x = keras.layers.pooling.AveragePooling2D(pool_size = (2, 2), strides = (2, 2))(x)
    #flatten 
    x = keras.layers.core.Flatten(name = 'flatten')(x)
    
    #dropout
    if use_dropout:
        x = keras.layers.Dropout(dropout_rate)(x)
    
    #FC
    x = keras.layers.core.Dense(units = classes, activation='softmax', kernel_initializer='glorot_uniform', name = 'fc' + str(classes))(x)
    #create model
    model = keras.models.Model(inputs = x_input, outputs = x, name = 'ResNet50')
    
    return model