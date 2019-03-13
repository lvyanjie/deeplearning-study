import keras

def conv_bn(x,
            filters,
            kernel_size,
            strides=1,
            padding='same',
            activation=None,
            name=None):
    '''
    添加bn的卷积操作
    :param x: input tensor
    :param filters: integer，the number of kernel
    :param kernel_size: integer or tumple，the size of kernel
    :param strides: integer or tumple, default 1
    :param padding: string, 'same' or 'valid', 'same' default
    :param activation: string, if None, linear activation
    :param name: name of operation
    :return: output tensor
    '''
    x = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=False, name=name+'_conv2d')(x)
    x = keras.layers.BatchNormalization(axis=3, name=name + '_bn')(x)
    if activation is not None:
        x = keras.layers.Activation(activation, name=name+'_'+activation)(x)
    return x

def stem(x, name=None):
    '''
    stem structure
    :param x: input tensor
    :param name: operation name
    :return: output tensor
    '''
    x = conv_bn(x, filters=32, kernel_size=(3,3), strides=2, padding='valid', activation='relu', name=name+'_conv1')
    x = conv_bn(x, filters=32, kernel_size=(3,3), padding='valid', activation='relu', name=name+'_conv2')
    x = conv_bn(x, filters=64, kernel_size=(3,3), activation='relu', name=name+'_conv3')
    x = keras.layers.MaxPooling2D(pool_size=(3,3), strides=2, padding='valid', name=name+'_pool1')(x)
    x = conv_bn(x, filters=80, kernel_size=(1,1), padding='valid', activation='relu', name=name+'_conv4')
    x = conv_bn(x, filters=192, kernel_size=(3,3), padding='valid', activation='relu', name=name+'_conv5')
    x = keras.layers.MaxPooling2D(pool_size=(3,3), strides=2, padding='valid', name=name+'_pool')(x)
    return x

def inception_A(x, name=None):
    '''
    inception A block structure
    :param x: input tensor
    :param name: string , operation name 
    :return: output tensor
    '''
    branch_0 = conv_bn(x, filters=96, kernel_size=(1,1), activation='relu', name=name+'_branch0_convbn')
    branch_1 = conv_bn(x, filters=48, kernel_size=(1,1), activation='relu', name=name+'_branch1_convbn0')
    branch_1 = conv_bn(branch_1, filters=64, kernel_size=(5,5), activation='relu', name=name+'_branch1_convbn1')
    branch_2 = conv_bn(x, filters=64, kernel_size=(1,1), activation='relu', name=name+'_branch2_convbn0')
    branch_2 = conv_bn(branch_2, filters=96, kernel_size=(3,3), activation='relu', name=name+'_branch2_convbn1')
    branch_2 = conv_bn(branch_2, filters=96, kernel_size=(3,3), activation='relu', name=name+'_branch2_convbn2')
    branch_3 = keras.layers.AveragePooling2D(pool_size=(3,3), strides=1, padding='same', name=name+'_branch3_pool')(x)
    branch_3 = conv_bn(branch_3, filters=64, kernel_size=(1,1), activation='relu', name=name+'_branch3_convbn')
    x = keras.layers.Concatenate(axis=3, name=name + '_concat')([branch_0, branch_1, branch_2, branch_3])
    return x

def inception_resnet_A(x, scale=0.1, name=None):
    '''
    inception resnet A block structure
    :param x: input tensor
    :param scale: float, around 0.1
    :param name: operation name
    :return: output tensor
    '''
    branch_0 = conv_bn(x, filters=32, kernel_size=(1,1), activation='relu', name=name + '_branch0_convbn')
    
    branch_1 = conv_bn(x, filters=32, kernel_size=(1,1), activation='relu', name=name + '_branch1_convbn0')
    branch_1 = conv_bn(branch_1, filters=32, kernel_size=(3,3), activation='relu', name=name + '_branch1_convbn1')
    
    branch_2 = conv_bn(x, filters=32, kernel_size=(1,1), activation='relu', name=name + '_branch2_convbn0')
    branch_2 = conv_bn(branch_2, filters=48, kernel_size=(3,3), activation='relu', name=name + '_branch2_convbn1')
    branch_2 = conv_bn(branch_2, filters=64, kernel_size=(3,3), activation='relu', name=name + '_branch2_convbn2')
    
    x2 = keras.layers.Concatenate(axis=3, name=name+'_concat')([branch_0, branch_1,branch_2])
    
    x2 = conv_bn(x2, filters=320, kernel_size=(1,1), name=name+'_conv')
    x2 = keras.layers.Lambda(lambda x: x * scale, name=name+'_scale')(x2)
    
    x = keras.layers.Add(name=name+'_add')([x, x2])
    x = keras.layers.Activation('relu', name=name+'_relu')(x)
    return x

def inception_resnet_B(x, scale=0.1, name=None):
    '''
    inception resnet B structure
    :param x: input tensor
    :param scale:  float, around 0.1
    :param name: string， operation name
    :return: output tensor
    '''
    branch_0 = conv_bn(x, filters=192, kernel_size=(1,1), activation='relu', name=name + '_branch0_convbn')
    
    branch_1 = conv_bn(x, filters=128, kernel_size=(1,1), activation='relu', name=name + '_branch1_convbn0')
    branch_1 = conv_bn(branch_1, filters=160, kernel_size=(1,7), activation='relu', name=name+'_branch1_convbn1')
    branch_1 = conv_bn(branch_1, filters=192, kernel_size=(7,1), activation='relu', name=name+'_branch1_convbn2')
    
    x2 = keras.layers.Concatenate(axis=3, name=name+'_concat')([branch_0, branch_1])
    x2 = conv_bn(x2, filters=1088, kernel_size=(1,1), name=name+'_conv')
    x2 = keras.layers.Lambda(lambda x: x * scale, name=name+'_scale')(x2)
    
    x = keras.layers.Add(name=name+'_add')([x, x2])
    x = keras.layers.Activation('relu', name=name+'_relu')(x)
    return x

def inception_resnet_C(x, scale=0.1, name=None):
    '''
    inception resnet C structure
    :param x: input tensor
    :param scale: float, around 0.1
    :param name: string, operation name
    :return: output tensor
    '''
    branch_0 = conv_bn(x, filters=192, kernel_size=(1,1), activation='relu', name=name+'_branch0_convbn')
    
    branch_1 = conv_bn(x, filters=192, kernel_size=(1,1), activation='relu', name=name+'_branch1_convbn0')
    branch_1 = conv_bn(branch_1, filters=224, kernel_size=(1,3), activation='relu', name=name+'_branch1_convbn1')
    branch_1 = conv_bn(branch_1, filters=256, kernel_size=(3,1), activation='relu', name=name+'_branch1_convbn2')
    
    x2 = keras.layers.Concatenate(axis=3, name=name+'_concat')([branch_0, branch_1])
    x2 = conv_bn(x2, filters=2080, kernel_size=(1,1), name=name+'_conv')
    x2 = keras.layers.Lambda(lambda x: x * scale, name=name+'_scale')(x2)
    
    x = keras.layers.Add(name=name+'_add')([x, x2])
    x = keras.layers.Activation('relu', name=name+'_relu')(x)
    return x

def reduction_A(x, name=None):
    '''
    reduction A structure
    :param x: input tensor
    :param name: string, operation name
    :return: output tensor
    '''
    branch_0 = conv_bn(x, filters=384, kernel_size=(3,3), strides=2, padding='valid', activation='relu', name=name+'_branch0_convbn')
    
    branch_1 = conv_bn(x, filters=256, kernel_size=(1,1), activation='relu', name=name+'_branch1_convbn0')
    branch_1 = conv_bn(branch_1, filters=256, kernel_size=(3,3), activation='relu', name=name+'_branch1_convbn1')
    branch_1 = conv_bn(branch_1, filters=384, kernel_size=(3,3), strides=2, padding='valid', activation='relu', name=name+'_branch1_convbn2')
    
    branch_2 = keras.layers.MaxPooling2D(pool_size=(3,3), strides=2, padding='valid')(x)
    x = keras.layers.Concatenate(axis=3, name=name+'_concat')([branch_0, branch_1, branch_2])
    return x

def reduction_B(x, name=None):
    '''
    reduction B structure
    :param x: input tensor
    :param name: string, operation name
    :return: output tensor
    '''
    branch_0 = conv_bn(x, filters=256, kernel_size=(1,1), activation='relu', name=name+'_branch0_convbn0')
    branch_0 = conv_bn(branch_0, filters=384, kernel_size=(3,3), strides=2, padding='valid', activation='relu', name=name+'_branch0_convbn1')
    
    branch_1 = conv_bn(x, filters=256, kernel_size=(1,1), activation='relu', name=name+'_branch1_convbn0')
    branch_1 = conv_bn(branch_1, filters=288, kernel_size=(3,3), strides=2, padding='valid', activation='relu', name=name+'_branch1_convbn1')
    
    branch_2 = conv_bn(x, filters=256, kernel_size=(1,1), activation='relu', name=name+'_branch2_convbn0')
    branch_2 = conv_bn(branch_2, filters=288, kernel_size=(3,3), activation='relu', name=name+'_branch2_convbn1')
    branch_2 = conv_bn(branch_2, filters=320, kernel_size=(3,3), strides=2, padding='valid', activation='relu', name=name+'_branch2_convbn2')
    
    branch_3 = keras.layers.MaxPooling2D(pool_size=(3,3), strides=2, padding='valid')(x)
    
    x = keras.layers.Concatenate(axis=3, name=name+'_concat')([branch_0, branch_1, branch_2, branch_3])
    return x

def Inception_Resnet_v2(input_shape=(299,299,3),
                        keep_rate=0.8,
                        classes=1000):
    '''
    inception resnet v2 structure
    :param input_shape: tuple，input image of shape , (299,299,3) default
    :param keep_rate: float, Number of reserved nodes
    :param classes: integer, number of classes, 1000 default
    :return: keras model
    '''
    x_input = keras.layers.Input(shape=input_shape, name='input')
    
    x = stem(x_input, name='stem')
    x = inception_A(x, name='inception_A')
    for i in range(10):
        x = inception_resnet_A(x, scale=0.17, name='inception_resnet_A_'+str(i))
    x = reduction_A(x, name='reduction_A')
    for i in range(20):
        x = inception_resnet_B(x, scale=0.1, name='inception_resnet_B_'+str(i))
    x = reduction_B(x, name='reduction_B')
    for i in range(9):
        x = inception_resnet_C(x, scale=0.2, name='inception_resnet_C_'+str(i))
    x = inception_resnet_C(x, scale=1, name='inception_resnet_C_9')
    
    x = conv_bn(x, filters=1536, kernel_size=(1,1), activation='relu', name='conv')
    x = keras.layers.GlobalAveragePooling2D(name='globalaveragepooling')(x)
    x = keras.layers.Dropout(rate=1-keep_rate, name='dropout')(x)
    x = keras.layers.Dense(units=classes, activation='softmax', name='classification')(x)
    
    model = keras.models.Model(inputs=x_input, outputs = x, name='inception_resnet_v2')
    return model