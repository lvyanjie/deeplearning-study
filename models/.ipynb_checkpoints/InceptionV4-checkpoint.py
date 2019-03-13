import keras

#conv_layer
def conv_layer(x, filters, kernel_size, padding = 'same', strides = (1, 1),name = None):
    '''
    conv + bn + relu
    :param x: tensor, tensor of previous layer
    :param filters: integer, number of kernel
    :param kernel_size: tumple, kernel's (row, col)
    :param padding: 'same' or 'valid'
    :param strides: tumple, stride of kernel
    :param name: layer's name
    
    returns:
    output tensor
    '''
    x = keras.layers.Conv2D(filters, kernel_size, strides = strides, padding = padding, use_bias = False, name = name + '_conv')(x)
    x = keras.layers.BatchNormalization(axis = 3, name = name + '_bn')(x)
    x = keras.layers.Activation('relu', name = name + '_relu')(x)
    
    return x

def inceptionA(x, name = None):
    '''
    inceptionA
    :param x: input tensor
    :param name: string
    
    returns:
    output tensor
    '''
    #branch0
    branch0 = conv_layer(x = x, filters = 64, kernel_size = (1, 1), name = name + '_branch0_conv1x1')
    branch0 = conv_layer(x = branch0, filters = 96, kernel_size = (3, 3), name = name + '_branch0_conv3x3_1')
    branch0 = conv_layer(x = branch0, filters = 96, kernel_size = (3, 3), name = name + '_branch0_conv3x3_2')
    
    #branch1
    branch1 = conv_layer(x = x, filters = 64, kernel_size = (1, 1), name = name + '_branch1_conv1x1')
    branch1 = conv_layer(x = branch1, filters = 96, kernel_size = (3, 3), name = name + '_branch1_conv3x3')
    
    #branch2
    branch2 = conv_layer(x = x, filters = 96, kernel_size = (1, 1), name = name + '_branch2_conv1x1')
    
    #branch3
    branch3 = keras.layers.AveragePooling2D(pool_size = (3, 3), strides = (1, 1),padding = 'same', name = name + '_branch3_pool')(x)
    branch3 = conv_layer(x = branch3, filters = 96, kernel_size = (1, 1), name = name + '_branch3_conv1x1')
    
    x = keras.layers.Concatenate(axis = 3, name = name + '_concat')([branch0, branch1, branch2, branch3])
    return x

def inceptionB(x, name = None):
    '''
    inceptionB
    :param x: input tensor
    :param name: string
    
    returns:
    output tensor
    '''
    #branch0
    branch0 = conv_layer(x = x, filters = 192, kernel_size = (1, 1), name = name + '_branch0_conv1x1')
    branch0 = conv_layer(x = branch0, filters = 192, kernel_size = (1, 7), name = name + '_branch0_1x7_1')
    branch0 = conv_layer(x = branch0, filters = 224, kernel_size = (7, 1), name = name + '_branch0_7x1_1')
    branch0 = conv_layer(x = branch0, filters = 224, kernel_size = (1, 7), name = name + '_branch0_1x7_2')
    branch0 = conv_layer(x = branch0, filters = 256, kernel_size = (7, 1), name = name + '_branch0_7x1_2')
    
    #branch1
    branch1 = conv_layer(x = x, filters = 192, kernel_size = (1, 1), name = name + '_branch1_conv1x1')
    branch1 = conv_layer(x = branch1, filters = 224, kernel_size = (7, 1), name = name + '_branch1_conv7x7_1')
    branch1 = conv_layer(x = branch1, filters = 256, kernel_size = (1, 7), name = name + '_branch1_conv7x7_2')
    
    #branch2
    branch2 = conv_layer(x = x, filters = 384, kernel_size = (1, 1), name = name + '_branch2_conv1x1')
    
    #branch3
    branch3 = keras.layers.AveragePooling2D(pool_size = (3, 3), strides = (1, 1), padding = 'same', name = name + '_branch3_pool')(x)
    branch3 = conv_layer(x = branch3, filters = 128, kernel_size = (1, 1), name = name + '_branch3_conv1x1')
    
    x = keras.layers.Concatenate(axis = 3, name = name + '_concat')([branch0, branch1, branch2, branch3])
    return x

def inceptionC(x, name = None):
    '''
    inceptionC
    :param x: input tensor
    :param name: string
    
    returns:
    output tensor
    '''
    #branch0
    branch0 = conv_layer(x = x, filters = 384, kernel_size = (1, 1), name = name + '_branch0_conv1x1')
    branch0 = conv_layer(x = branch0, filters = 448, kernel_size = (1, 3), name = name + '_branch0_conv1x3')
    branch0 = conv_layer(x = branch0, filters = 512, kernel_size = (3, 1), name = name + '_branch0_conv3x1')
    branch0_r = conv_layer(x = branch0, filters = 256, kernel_size = (1, 3), name = name + '_branch0_conv1x3_r')
    branch0_l = conv_layer(x = branch0, filters = 256, kernel_size = (3, 1), name = name + '_branch0_conv3x1_l')
    branch0 = keras.layers.Concatenate(axis =3, name = name + '_branch0_concat')([branch0_r, branch0_l])
    
    #branch1
    branch1 = conv_layer(x = x, filters = 384, kernel_size = (1, 1), name = name + '_branch1_conv1x1')
    branch1_l = conv_layer(x = branch1, filters = 256, kernel_size = (1, 3), name = name + '_branch1_conv1x3_l')
    branch1_r = conv_layer(x = branch1, filters = 256, kernel_size = (3, 1), name = name + '_branch1_conv3x1_r')
    branch1 = keras.layers.Concatenate(axis = 3, name = name + '_branch1_concat')([branch1_l, branch1_r])
    
    #branch2
    branch2 = conv_layer(x = x, filters = 256, kernel_size = (1, 1), name = name + '_branch2_conv1x1')
    
    #branch3
    branch3 = keras.layers.AveragePooling2D(pool_size = (3, 3), strides = (1, 1), padding = 'same', name = name + '_branch3_pool')(x)
    branch3 = conv_layer(x = branch3, filters = 256, kernel_size = (1, 1), name = name + '_branch3_conv1x1')
    
    x = keras.layers.Concatenate(axis = 3, name = name + '_concat')([branch0, branch1, branch2, branch3])
    return x

def reductionA(x, name = None):
    '''
    reduction A
    :param x: input tensor
    :param name: string
    
    returns:
    output tensor
    '''
    #branch0
    branch0 = conv_layer(x = x, filters = 384, kernel_size = (3, 3), strides = (2, 2), padding = 'valid', name = name + '_branch0_conv3x3')
    
    #branch1
    branch1 = conv_layer(x = x, filters = 192, kernel_size = (1, 1), name = name + '_branch1_conv1x1')
    branch1 = conv_layer(x = branch1, filters = 224, kernel_size = (3, 3), name = name + '_branch1_conv3x3_1')
    branch1 = conv_layer(x = branch1, filters = 256, kernel_size = (3, 3), strides = (2, 2), padding = 'valid', name = name + '_branch1_conv3x3_2')
    
    #branch3
    branch2 = keras.layers.MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = 'valid', name = name + '_branch2_pool')(x)
    
    x = keras.layers.Concatenate(axis = 3, name = name + '_concat')([branch0, branch1, branch2])
    return x

def reductionB(x, name = None):
    '''
    reduction B
    :param x: input tensor
    :param name: string
    
    returns:
    output tensor
    '''
    #branch0
    branch0 = conv_layer(x = x, filters = 192, kernel_size = (1, 1), name = name + '_branch0_conv1x1')
    branch0 = conv_layer(x = branch0, filters = 192, kernel_size = (3, 3), strides = (2, 2), padding = 'valid', name = name + '_branch0_conv3x3')
    
    #branch1
    branch1 = conv_layer(x = x, filters = 256, kernel_size = (1, 1), name = name + '_branch1_conv1x1')
    branch1 = conv_layer(x = branch1, filters = 256, kernel_size = (1, 7), name = name + '_branch1_conv7x7_1')
    branch1 = conv_layer(x = branch1, filters = 320, kernel_size = (7, 1), name = name + '_branch1_conv7x7_2')
    branch1 = conv_layer(x = branch1, filters = 320, kernel_size = (3, 3), strides = (2, 2), padding = 'valid', name = name + '_branch1_conv3x3')
    
    #branch2
    branch2 = keras.layers.MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = 'valid', name = name + '_branch2_pool')(x)
    
    x = keras.layers.Concatenate(axis = 3, name = name + '_concat')([branch0, branch1, branch2])
    return x

def InceptionV4(input_shape = (299, 299, 3), classes = 1000, use_dropout = True, dropout_rate = 0.2):
    '''
    Inception V4
    :param input_shape: tumple, input shape of network
    :param classes: integer, number of class of network
    :param use_dropout: bool, use dropout or not
    :param dropout_rate: float, only valid if use_dropout is True
    
    returns:
    keras model
    '''
    #stem
    x_input = keras.layers.Input(input_shape)
    
    x = conv_layer(x = x_input, filters = 32, kernel_size = (3, 3), strides = (2, 2), padding = 'valid', name = 'stem_block0_conv0')#149,149,32
    x = conv_layer(x = x, filters = 32, kernel_size = (3, 3), padding = 'valid', name = 'stem_block0_conv1')#147,147,32
    x = conv_layer(x = x, filters = 64, kernel_size = (3, 3), name = 'stem_block0_conv2')#147,147,64
    
    branch0 = conv_layer(x = x, filters = 96, kernel_size = (3, 3), strides = (2, 2), padding = 'valid', name = 'stem_block0_branch0')
    branch1 = keras.layers.MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = 'valid', name = 'stem_block0_branch1')(x)
    
    x = keras.layers.Concatenate(axis = 3, name = 'stem_block0_concat')([branch0, branch1])# 73, 73 160
    
    branch0 = conv_layer(x = x, filters = 64, kernel_size = (1, 1), name = 'stem_block1_branch0_conv0')
    branch0 = conv_layer(x = branch0, filters = 64, kernel_size = (7, 1), name = 'stem_block1_branch0_conv1')
    branch0 = conv_layer(x = branch0, filters = 64, kernel_size = (1, 7), name = 'stem_block1_branch0_conv2')
    branch0 = conv_layer(x = branch0, filters = 96, kernel_size = (3, 3), padding = 'valid', name = 'stem_block1_branch0_conv3')
    
    branch1 = conv_layer(x = x, filters = 64, kernel_size = (1, 1), name = 'stem_block1_branch1_conv0')
    branch1 = conv_layer(x = branch1, filters = 96, kernel_size = (3, 3), padding = 'valid', name = 'stem_block1_branch1_conv1')
    
    x = keras.layers.Concatenate(axis = 3, name = 'stem_block1_concat')([branch0, branch1])
    
    branch0 = keras.layers.MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = 'valid', name = 'stem_block2_branch0')(x)
    branch1 = conv_layer(x = x, filters = 192, kernel_size = (3, 3), strides = (2, 2), padding = 'valid', name = 'stem_block2_branch1')

    x = keras.layers.Concatenate(axis = 3, name = 'stem_block2_concat')([branch0, branch1])
    
    #4xinception-A
    for i in range(4):
        x = inceptionA(x = x, name = 'InceptionA'+str(i))
        
    #reductionA
    x = reductionA(x, name = 'ReductionA')
    
    #7xinception-B
    for i in range(7):
        x = inceptionB(x, name = 'InceptionB'+str(i))
    
    #reductionB
    x = reductionB(x, name = 'ReductionB')
    
    #3xinception-C
    for i in range(3):
        x = inceptionC(x, name = 'InceptionC'+str(i))
    
    x = keras.layers.GlobalAveragePooling2D(name = 'globalaveragepooling')(x)
    if use_dropout:
        x = keras.layers.Dropout(dropout_rate)(x)
    x = keras.layers.Dense(classes, activation = 'softmax', name = 'classificaiton')(x)
    
    model = keras.models.Model(inputs = x_input, outputs = x, name = 'inceptionv4')
    return model