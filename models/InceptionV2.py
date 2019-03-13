import keras

def conv_layer(x, filters, kernel_size, padding = 'same', strides = (1, 1), name = None):
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

## first inception module
def inceptionA(x,f_branch1_1x1,
              f_branch2_reduce3,
              f_branch2_3x3,
              f_branch3_reducedb3,
              f_branch3_db3,
              f_branch4_reducepool,
              name = None):
    '''
    inception A
    :param x: inpute tensor
    :param f_branch1_1x1: number of kernels 1x1
    :param f_branch2_reduce3: number of kernels 1x1 reducing 3x3
    :param f_branch2_3x3: number of kernels 3x3
    :param f_branch3_reducedb3: number of kernels 1x1 reducing double 3x3
    :param f_branch3_db3: list, number of kernels 3x3, 3x3
    :param f_branch3_reducepool: number of kernels 1x1 reducing pool
    
    returns:
    output tensor
    ''' 
    #conv1x1
    branch1 = conv_layer(x = x, filters = f_branch1_1x1, kernel_size = (1, 1), name = name + '_branch1_conv1x1')
    
    #conv 3x3
    branch2 = conv_layer(x = x, filters = f_branch2_reduce3, kernel_size = (1, 1), name = name + '_branch2_conv1x1')
    branch2 = conv_layer(x = branch2, filters = f_branch2_3x3, kernel_size = (3, 3), name = name + '_branch2_conv3x3')
    
    #conv double 3x3
    branch3 = conv_layer(x = x, filters = f_branch3_reducedb3, kernel_size = (1, 1), name = name + '_branch3_conv1x1')
    branch3 = conv_layer(x = branch3, filters = f_branch3_db3[0], kernel_size = (3, 3), name = name + '_branch3_conv3x3_1')
    branch3 = conv_layer(x = branch3, filters = f_branch3_db3[1], kernel_size = (3, 3), name = name + '_branch3_conv3x3_2')
    
    #pool
    branch4 = keras.layers.AveragePooling2D(pool_size = (3, 3), strides = (1, 1), padding = 'same', name = name + '_branch4_pool')(x)
    branch4 = conv_layer(x = branch4, filters = f_branch4_reducepool, kernel_size = (1, 1), padding = 'same', name = name + '_branch4_conv1x1')
    
    x = keras.layers.concatenate([branch1, branch2, branch3, branch4], axis = 3, name = name)
    return x

## second inception module
def inceptionB(x, f_branch1_reduce3,
              f_branch1_3x3,
              f_branch2_reducedb3,
              f_branch2_db3,
              name = None):
    '''
    inception B
    :param x: inpute tensor
    :parma f_branch1_reduce3: number of kernels 1x1 reducing 3x3
    :param f_branch1_3x3: number of kernels 3x3
    :param f_branch2_reducedb3: number of kernels 1x1 reducing double 3x3
    :param f_branch2_db3: list, number of kernels 3x3, 3x3
    
    returns:
    output tensor
    '''
    #conv1x1
    branch1 = conv_layer(x = x, filters = f_branch1_reduce3, kernel_size = (1, 1), name = name + '_branch1_conv1x1')
    branch1 = conv_layer(x = branch1, filters = f_branch1_3x3, kernel_size = (3, 3), strides = (2, 2), name = name + '_branch1_conv3x3')
    
    #conv double 3x3
    branch2 = conv_layer(x = x, filters = f_branch2_reducedb3, kernel_size = (1, 1), name = name + '_branch2_conv1x1')
    branch2 = conv_layer(x = branch2, filters = f_branch2_db3[0], kernel_size = (3, 3), name = name + '_branch2_conv3x3_1')
    branch2 = conv_layer(x = branch2, filters = f_branch2_db3[1], kernel_size = (3, 3), strides = (2, 2), name = name + '_branch2_conv3x3_2')
    
    #pool
    branch3 = keras.layers.MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = 'same', name = name + '_branch3_pool')(x)
    
    x = keras.layers.concatenate([branch1, branch2, branch3], axis = 3, name = name)
    return x

def inceptionv2(input_shape=(224,224,3), classes = 1000, use_dropout = True, dropout_rate = 0.2):
    '''
    inception v2
    :param input_shape: tumple, input shape of net
    :param classes: integer, number of class
    :param use_dropout: bool, use dropout or not
    :param dropout: float, only valid if use_dropout = True
    
    returns:
    keras model
    '''
    #224*224*3
    x_input = keras.layers.Input(input_shape)
    
    #conv1
    x = conv_layer(x = x_input, filters = 64, kernel_size = (7, 7), strides = (2, 2), name = 'conv1')
    x = keras.layers.MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = 'same', name = 'pool1')(x)
    
    #conv2
    x = conv_layer(x = x, filters = 192, kernel_size = (3, 3), name = 'conv2')
    x = keras.layers.MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = 'same', name = 'pool2')(x)
    
    #inception3a
    x = inceptionA(x = x,f_branch1_1x1 = 64,f_branch2_reduce3 = 64,f_branch2_3x3 = 64,f_branch3_reducedb3 = 64,f_branch3_db3 = [96, 96],f_branch4_reducepool = 32,name = 'inception3a')
    #inception3b
    x = inceptionA(x = x,f_branch1_1x1 = 64,f_branch2_reduce3 = 64,f_branch2_3x3 = 96,f_branch3_reducedb3 = 64,f_branch3_db3 = [96, 96],f_branch4_reducepool = 64,name = 'inception3b')
    #inception3c
    x = inceptionB(x = x, f_branch1_reduce3 = 128,f_branch1_3x3 = 160,f_branch2_reducedb3 = 64,f_branch2_db3 = [96, 96],name = 'inception3c')
    
    #inception4a
    x = inceptionA(x = x,f_branch1_1x1 = 224,f_branch2_reduce3 = 64,f_branch2_3x3 = 96,f_branch3_reducedb3 = 96,f_branch3_db3 = [128, 128],f_branch4_reducepool = 128,name = 'inception4a')
    #inception4b
    x = inceptionA(x = x,f_branch1_1x1 = 192,f_branch2_reduce3 = 96,f_branch2_3x3 = 128,f_branch3_reducedb3 = 96,f_branch3_db3 = [128, 128],f_branch4_reducepool = 128,name = 'inception4b')
    #inception4c
    x = inceptionA(x = x,f_branch1_1x1 = 160,f_branch2_reduce3 = 128,f_branch2_3x3 = 160,f_branch3_reducedb3 = 128,f_branch3_db3 = [160, 160],f_branch4_reducepool = 96,name = 'inception4c')
    #inception4d
    x = inceptionA(x = x,f_branch1_1x1 = 96,f_branch2_reduce3 = 128,f_branch2_3x3 = 192,f_branch3_reducedb3 = 160,f_branch3_db3 = [192, 192],f_branch4_reducepool = 96,name = 'inception4d')
    #inception4e
    x = inceptionB(x = x, f_branch1_reduce3 = 128,f_branch1_3x3 = 192,f_branch2_reducedb3 = 192,f_branch2_db3 = [256, 256],name = 'inception4e')
    
    #inception5a
    x = inceptionA(x = x,f_branch1_1x1 = 352,f_branch2_reduce3 = 192,f_branch2_3x3 = 320,f_branch3_reducedb3 = 160,f_branch3_db3 = [224, 224],f_branch4_reducepool = 128,name = 'inception5a')
    #inception5b
    x = inceptionA(x = x,f_branch1_1x1 = 352,f_branch2_reduce3 = 192,f_branch2_3x3 = 320,f_branch3_reducedb3 = 192,f_branch3_db3 = [224, 224],f_branch4_reducepool = 128,name = 'inception5b')
    
    #global avg pooling
    x = keras.layers.GlobalAveragePooling2D(name = 'global_average_pooling')(x)#1x1x1024
    if use_dropout:
        x = keras.layers.Dropout(dropout_rate, name = 'dropout')(x)
    
    x = keras.layers.Dense(classes, activation = 'softmax', name = 'classification')(x)
    model = keras.models.Model(inputs = x_input, outputs = x, name = 'inceptionv2')
    return model