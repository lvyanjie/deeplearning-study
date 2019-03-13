import keras

#conv_layer
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

#inception-A
def inceptionA(x, f_branch1_1x1,
              f_branch2_5x5,
              f_branch3_3x3,
              f_branch4_reducepool,
              f_branch2_reduce5,
              f_branch3_reduce3,
              name=None):
    '''
    inception A
    :param x: input tensor
    :param f_branch1_1x1: integer, number of kernel of 1*1
    :param f_branch2_5x5: integer, number of kernel of 5*5
    :param f_branch4_3x3: list, number of kernel of double 3*3
    :param branch2_reducepool: integer, number of reduce channel 1*1 of pool
    :param f_branch2_reduce5: integer, number of reduce channel 1*1 of 5*5 conv
    :param f_branch3_reduce3: integer, number of reduce channel 1*1 of double 3*3 conv
    :param name: string, layer's name
    
    returns:
    output tensor
    '''
    #conv 1*1
    branch1_1x1 = conv_layer(x = x, filters = f_branch1_1x1, kernel_size = (1, 1), name = name + '_branch1_1x1')
    
    #conv 5*5
    branch2_5x5 = conv_layer(x = x, filters = f_branch2_reduce5, kernel_size = (1, 1), name = name + '_branch2_1x1')
    branch2_5x5 = conv_layer(x = branch2_5x5, filters = f_branch2_5x5, kernel_size = (5, 5), name = name + '_branch2_5x5')
    
    #double conv 3*3
    branch3_3x3 = conv_layer(x = x, filters = f_branch3_reduce3, kernel_size = (1, 1), name = name + '_branch3_1x1')
    branch3_3x3 = conv_layer(x = branch3_3x3, filters = f_branch3_3x3[0], kernel_size = (3, 3), name = name + '_branch3_3x3_1')
    branch3_3x3 = conv_layer(x = branch3_3x3, filters = f_branch3_3x3[1], kernel_size = (3, 3), name = name + '_branch3_3x3_2')
    
    #pool
    branch4_pool = keras.layers.AveragePooling2D(pool_size = (3, 3), strides = (1, 1), padding='same', name = name + '_branch4_pool')(x)
    branch4_pool = conv_layer(x = branch4_pool, filters = f_branch4_reducepool, kernel_size = (1, 1), name = name + '_branch4_1x1')
    
    x = keras.layers.concatenate([branch1_1x1, branch2_5x5, branch3_3x3, branch4_pool], axis = 3, name = name)
    return x

#inception-B
def inceptionB(x, f_branch1_3x3,
               f_branch2_3x3,
               f_branch2_reduce3,
              name = None):
    '''
    inception B
    :param x: input tensor
    :param f_branch1_3x3: integer, number of kernel of 3*3
    :param f_branch2_3x3: list, number of kernel of double 3*3
    :param f_branch2_reduce3: integer, number of reduce channel 1*1 of double 3*3 conv
    
    returns:
    output tensor
    '''
    #conv_3
    branch1_3x3 = conv_layer(x = x, filters = f_branch1_3x3, kernel_size = (3, 3), strides=(2, 2), padding = 'valid', name = name + '_branch1_3x3')
    
    #double_conv33
    branch2_3x3 = conv_layer(x = x, filters = f_branch2_reduce3, kernel_size = (1, 1), name = name + '_branch2_1x1')
    branch2_3x3 = conv_layer(x = branch2_3x3, filters = f_branch2_3x3[0], kernel_size = (3, 3), name = name + '_branch2_3x3_1')
    branch2_3x3 = conv_layer(x = branch2_3x3, filters = f_branch2_3x3[1], kernel_size = (3, 3), strides = (2, 2), padding = 'valid', name = name + '_branch2_3x3_2')
    
    #pool
    branch3_pool = keras.layers.MaxPooling2D(pool_size = (3, 3), strides = (2, 2), name = name + '_branch3_pool')(x)
    
    x = keras.layers.concatenate([branch1_3x3, branch2_3x3, branch3_pool], axis=3, name = name)
    return x

#inception-C
def inceptionC(x, f_branch1_1x1,
               f_branch2_7x7,
               f_branch3_7x7,
               f_branch2_reduce7,
               f_branch3_reduce7,
               f_branch4_reducepool,
               name=None):
    '''
    inception C
    :param x: input tensor
    :param f_branch1_1x1: integer, number of kernels of conv 1*1
    :param f_branch2_7x7: list, number of kernels of conv 1*7 and 7*1
    :param f_branch3_7x7: list, number of kernels of conv 7*1, 1*7, 7*1, 1*7
    :param f_branch2_reduce7: integer, number of kernels of conv 1*1 reducing 1*7, 7*1
    :param f_branch3_reduce7: integer, number of kernels of conv 1*1 reducing 7*1, 1*7, 7*1, 1*7
    :param f_branch4_reducepool: integer, number of kernels of conv 1*1 reducing pooling
    
    returns:
    output tensor
    '''
    #conv1
    branch1_1x1 = conv_layer(x = x, filters = f_branch1_1x1, kernel_size = (1, 1), name = name + '_branch1_1x1')
    
    #conv7
    branch2_7x7 = conv_layer(x = x, filters = f_branch2_reduce7, kernel_size = (1, 1), name = name + '_branch2_1x1')
    branch2_7x7 = conv_layer(x = branch2_7x7, filters = f_branch2_7x7[0], kernel_size = (1, 7), name = name + '_branch2_1x7')
    branch2_7x7 = conv_layer(x = branch2_7x7, filters = f_branch2_7x7[1], kernel_size = (7, 1), name = name + '_branch2_7x1')
    
    #db_conv7
    branch3_7x7 = conv_layer(x = x, filters = f_branch3_reduce7, kernel_size = (1, 1), name = name + '_branch3_1x1')
    branch3_7x7 = conv_layer(x = branch3_7x7, filters = f_branch3_7x7[0], kernel_size = (7, 1), name = name + '_branch3_7x1_1')
    branch3_7x7 = conv_layer(x = branch3_7x7, filters = f_branch3_7x7[1], kernel_size = (1, 7), name = name + '_branch3_1x7_1')
    branch3_7x7 = conv_layer(x = branch3_7x7, filters = f_branch3_7x7[2], kernel_size = (7, 1), name = name + '_branch3_7x1_2')
    branch3_7x7 = conv_layer(x = branch3_7x7, filters = f_branch3_7x7[3], kernel_size = (1, 7), name = name + '_branch3_1x7_2')
    
    #pool
    branch4_pool = keras.layers.AveragePooling2D(pool_size = (3, 3), strides=(1, 1), padding='same', name = name + '_branch4_pool')(x)
    branch4_pool = conv_layer(x = branch4_pool, filters = f_branch4_reducepool, kernel_size = (1, 1), name = name + '_branch4_1x1')
    
    x = keras.layers.concatenate([branch1_1x1, branch2_7x7, branch3_7x7, branch4_pool], axis = 3, name = name)
    return x

#inception-D
def inceptionD(x, f_branch1_3x3,
              f_branch2_7x7,
              f_branch2_3x3,
              f_branch1_reduce3,
              f_branch2_reduce7,
              name = None):
    '''
    inception D
    :param x: input tensor
    :param f_branch1_3x3: integer, number of kernels 3*3 in branch 1
    :param f_branch2_7x7: list, number of kernels 1*7, 7*1 in branch 2
    :param f_branch2_3x3: integer, number of kernels 3*3 in branch 2
    :param f_branch1_reduce3: integer, number of kernels 1*1 reducing channels of 3*3
    :param f_branch2_reduce7: integer, number of kernels 1*1 reducing channels of 1*7, 7*1, 3*3
    
    returns:
    output tensor
    '''
    #branch1 conv3
    branch1_3x3 = conv_layer(x = x, filters = f_branch1_reduce3, kernel_size = (1, 1), name = name + '_branch1_1x1')
    branch1_3x3 = conv_layer(x = branch1_3x3, filters = f_branch1_3x3, kernel_size = (3, 3), strides = (2, 2), padding = 'valid', name = name + '_branch1_3x3')
    
    #branch2 conv7, conv3
    branch2_7x7 = conv_layer(x = x, filters = f_branch2_reduce7, kernel_size = (1, 1), name = name + '_branch2_1x1')
    branch2_7x7 = conv_layer(x = branch2_7x7, filters = f_branch2_7x7[0], kernel_size = (1, 7), name = name + '_branch2_1x7')
    branch2_7x7 = conv_layer(x = branch2_7x7, filters = f_branch2_7x7[1], kernel_size = (7, 1), name = name + '_branch2_7x1')
    branch2_3x3 = conv_layer(x = branch2_7x7, filters = f_branch2_3x3, kernel_size = (3, 3), strides = (2, 2), padding = 'valid', name = name + '_branch2_3x3')
    
    #pool
    branch3_pool = keras.layers.MaxPooling2D(pool_size = (3, 3), strides = (2, 2), name = name + '_branch3_pool')(x)
    
    x = keras.layers.concatenate([branch1_3x3, branch2_3x3, branch3_pool], axis = 3, name = name)
    return x

#inception-E
def inceptionE(x, f_branch1_1x1,
               f_branch2_3x3_sp,               
               f_branch3_3x3,
               f_branch3_3x3_sp,
               f_branch2_reduce3,
               f_branch3_reduce3,
               f_branch4_reducepool,
               name = None):
    '''
    inception E
    :param x: input tensor
    :param f_branch1_1x1: integer, number of kernels 1x1
    :param f_branch2_3x3_sp: list, number of kernels 3x1, 1x3
    :param f_branch3_3x3: integer, number of kernels 3x3
    :param f_branch3_3x3_sp: list, number of kernels 3x1, 1x3
    :param f_branch2_reduce3: integer, number of kernels 1x1 reducing channels of 3x1, 1x3
    :param f_branch3_reduce3: integer, number of kernels 1x1 reducing channels of 3x3
    :param f_branch4_reducepool: integer, number of kernels 1x1 reducing channels of pool
    
    returns:
    output tensor
    '''
    #branch1 conv1
    branch1_1x1 = conv_layer(x = x, filters = f_branch1_1x1, kernel_size = (1, 1), name = name + '_branch1_1x1')
    
    #branch2 conv 3x3
    branch2_1x1 = conv_layer(x = x, filters = f_branch2_reduce3, kernel_size = (3, 3), name = name + '_branch2_1x1')
    branch2_3x3_1 = conv_layer(x = branch2_1x1, filters = f_branch2_3x3_sp[0], kernel_size = (3, 1), name = name + '_branch2_3x3_1')
    branch2_3x3_2 = conv_layer(x = branch2_1x1, filters = f_branch2_3x3_sp[1], kernel_size = (1, 3), name = name + '_branch2_3x3_2')
    branch2_3x3 = keras.layers.concatenate([branch2_3x3_1, branch2_3x3_2], axis = 3, name = name + '_branch2_3x3')
    
    #branch3
    branch3_1x1 = conv_layer(x = x, filters = f_branch3_reduce3, kernel_size = (1, 1), name = name + '_branch3_1x1')
    branch3_3x3 = conv_layer(x = branch3_1x1, filters = f_branch3_3x3, kernel_size = (3, 3), name = name + '_branch3_3x3')
    branch3_3x3_1 = conv_layer(x = branch3_3x3, filters = f_branch3_3x3_sp[0], kernel_size = (3, 1), name = name + '_branch3_3x3_1')
    branch3_3x3_2 = conv_layer(x = branch3_3x3, filters = f_branch3_3x3_sp[1], kernel_size = (1, 3), name = name + '_branch3_3x3_2')
    branch_3x3_f = keras.layers.concatenate([branch3_3x3_1, branch3_3x3_2], axis = 3, name = name + '_branch3_3x3_final')
    
    #branch4
    branch4_pool = keras.layers.AveragePooling2D(pool_size = (3, 3), strides = (1, 1), padding = 'same', name = name + '_branch4_pool')(x)
    branch4_pool = conv_layer(x = branch4_pool, filters = f_branch4_reducepool, kernel_size = (1, 1), name = name + '_branch4_1x1')
    
    x = keras.layers.concatenate([branch1_1x1, branch2_3x3, branch_3x3_f, branch4_pool], axis = 3, name  = name)
    return x

def BN_auxiliary(x, classes):
    '''
    BN auxiliary
    :param x: input tensor(feature map)
    :param classes: number of output class
    
    returns:
    aux_logists
    '''
    x = keras.layers.AveragePooling2D(pool_size = (5, 5), strides = (3, 3), name = 'aux_average_pooling')(x)#5x5x768
    x = conv_layer(x = x, filters = 128, kernel_size = (1, 1), padding = 'same', strides = (1, 1), name = 'aux_conv1')#5x5x128
    x = conv_layer(x = x, filters = 768, kernel_size = (5, 5), padding = 'valid', strides = (1, 1), name = 'aux_conv2')#1x1x768
    x = keras.layers.Flatten(name = 'aux_flatten')(x)#768
    logits = keras.layers.Dense(units = classes, activation='softmax', name = 'aux_classification')(x)
    return logits

def Inceptionv3(input_shape=(299,299,3), classes = 1000, use_dropout = True, dropout_rate = 0.2, use_BN_auxiliary = True):
    '''
    inception v3
    :param input_shape: tumple, network input shape
    :param classes: integer, number of class
    :param use_dropout: bool, if true, dropout operation before Dense, else, only Dense
    :param dropout_rate: float, only valid if use_dropout = True
    :param use_BN_auxiliary: use BN_auxiliary or not
    
    returns:
    keras model
    '''
    x_input = keras.layers.Input(input_shape)
    
    x = conv_layer(x = x_input, filters = 32, kernel_size = (3, 3), strides = (2, 2), padding = 'valid', name = 'conv1_1')
    x = conv_layer(x = x, filters = 32, kernel_size = (3, 3), strides = (1, 1), padding = 'valid', name = 'conv1_2')
    x = conv_layer(x = x, filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = 'same', name = 'conv1_3')
    x = keras.layers.MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = 'valid', name = 'maxpool1')(x)
    
    x = conv_layer(x = x, filters = 80, kernel_size = (1, 1), strides = (1, 1), padding = 'valid', name = 'conv2_1')
    x = conv_layer(x = x, filters = 192, kernel_size = (3, 3), strides = (1, 1), padding = 'valid', name = 'conv2_2')
    x = keras.layers.MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = 'valid', name = 'maxpool2')(x)
    
    #inception A
    x = inceptionA(x = x, f_branch1_1x1 = 64,
                   f_branch2_5x5 = 64,
                   f_branch3_3x3 = [96, 96],
                   f_branch4_reducepool = 32,
                   f_branch2_reduce5 = 48,
                   f_branch3_reduce3 = 64,
                   name='inceptionA1')
    x = inceptionA(x = x, f_branch1_1x1 = 64,
                   f_branch2_5x5 = 64,
                   f_branch3_3x3 = [96, 96],
                   f_branch4_reducepool = 64,
                   f_branch2_reduce5 = 48,
                   f_branch3_reduce3 = 64,
                   name='inceptionA2')
    x = inceptionA(x = x, f_branch1_1x1 = 64,
                   f_branch2_5x5 = 64,
                   f_branch3_3x3 = [96, 96],
                   f_branch4_reducepool = 64,
                   f_branch2_reduce5 = 48,
                   f_branch3_reduce3 = 64,
                   name='inceptionA3')
    
    #inception B
    x = inceptionB(x = x, f_branch1_3x3 = 384,
                   f_branch2_3x3 = [96, 96],
                   f_branch2_reduce3 = 64,
                   name = 'inceptionB')
    
    #inception C
    x = inceptionC(x = x, f_branch1_1x1 = 192,
               f_branch2_7x7 = [128, 192],
               f_branch3_7x7 = [128, 128, 128, 192],
               f_branch2_reduce7 = 128,
               f_branch3_reduce7 = 128,
               f_branch4_reducepool = 192,
               name='inceptionC1')
    x = inceptionC(x = x, f_branch1_1x1 = 192,
               f_branch2_7x7 = [160, 192],
               f_branch3_7x7 = [160, 160, 160, 192],
               f_branch2_reduce7 = 160,
               f_branch3_reduce7 = 160,
               f_branch4_reducepool = 192,
               name='inceptionC2')
    x = inceptionC(x = x, f_branch1_1x1 = 192,
               f_branch2_7x7 = [160, 192],
               f_branch3_7x7 = [160, 160, 160, 192],
               f_branch2_reduce7 = 160,
               f_branch3_reduce7 = 160,
               f_branch4_reducepool = 192,
               name='inceptionC3')
    x = inceptionC(x = x, f_branch1_1x1 = 192,
               f_branch2_7x7 = [192, 192],
               f_branch3_7x7 = [192, 192, 192, 192],
               f_branch2_reduce7 = 192,
               f_branch3_reduce7 = 192,
               f_branch4_reducepool = 192,
               name='inceptionC4')
    
    #add BN_auxiliary
    if use_BN_auxiliary:
        logits = BN_auxiliary(x = x, classes = classes)
    
    #inception D
    x = inceptionD(x, f_branch1_3x3 = 320,
              f_branch2_7x7 = [192, 192],
              f_branch2_3x3 = 192,
              f_branch1_reduce3 = 192,
              f_branch2_reduce7 = 192,
              name = 'inceptionD')
    
    #inception E
    x = inceptionE(x = x, f_branch1_1x1 = 320,
               f_branch2_3x3_sp = [384, 384],               
               f_branch3_3x3 = 384,
               f_branch3_3x3_sp = [384, 384],
               f_branch2_reduce3 = 384,
               f_branch3_reduce3 = 448,
               f_branch4_reducepool = 192,
               name = 'inceptionE1')
    x = inceptionE(x = x, f_branch1_1x1 = 320,
               f_branch2_3x3_sp = [384, 384],               
               f_branch3_3x3 = 384,
               f_branch3_3x3_sp = [384, 384],
               f_branch2_reduce3 = 384,
               f_branch3_reduce3 = 448,
               f_branch4_reducepool = 192,
               name = 'inceptionE2')
    
    #global_avg_pool
    x = keras.layers.GlobalAveragePooling2D(name = 'global_avg_pooling')(x)
    #dropout
    if use_dropout:
        x = keras.layers.Dropout(dropout_rate, name = 'dropout')(x)#概率可修改为其他参数
    #fc
    x = keras.layers.Dense(units = classes, activation='softmax', name = 'classification')(x)
    
    model = keras.models.Model(inputs = x_input, outputs = x, name = 'inceptionv3')
    
    if use_BN_auxiliary:
        model = keras.models.Model(inputs = x_input, outputs = [x, logits], name = 'inceptionv3')
        
    return model