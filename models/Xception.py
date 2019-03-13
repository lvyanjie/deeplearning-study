import keras

def middle_flow(x, block, name=None):
    '''
    middle flow block
    :param x: input tensor
    :param block: index of block for name
    :param name: operation name
    :return: output tensor
    '''
    shortcut = x
    x = keras.layers.Activation('relu', name=name+str(block)+'_relu0')(x)
    x = keras.layers.SeparableConv2D(filters=728, kernel_size=(3,3), padding='same', name=name+'_res'+str(block)+'_SepConv0')(x)
    x = keras.layers.BatchNormalization(name=name+str(block)+'_bn0')(x)
    x = keras.layers.Activation('relu', name=name+str(block)+'_relu1')(x)
    x = keras.layers.SeparableConv2D(filters=728, kernel_size=(3,3), padding='same', name=name+str(block)+'_SepConv1')(x)
    x = keras.layers.BatchNormalization(name=name+str(block)+'_bn1')(x)
    x = keras.layers.Activation('relu', name=name+str(block)+'_relu2')(x)
    x = keras.layers.SeparableConv2D(filters=728, kernel_size=(3,3), padding='same', name=name+str(block)+'_SepConv2')(x)
    x = keras.layers.BatchNormalization(name=name+str(block)+'_bn2')(x)
    x = keras.layers.Add(name=name+str(block)+'_add')([shortcut, x])
    return x

def Xception(input_shape=(299,299,3), classes = 1000, use_dropout = True, dropout_rate = 0.2):
    
    # Entry flow 1
    x_input = keras.layers.Input(shape=input_shape, name='input_layer')
    x = keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=2, padding='same', name='entryflow_conv0')(x_input)
    x = keras.layers.BatchNormalization(name='entryflow_bn0')(x)
    x = keras.layers.Activation('relu', name='entryflow_relu0')(x)  
    x = keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding='same', name='entryflow_conv1')(x)
    x = keras.layers.BatchNormalization(name='entryflow_bn1')(x)
    x = keras.layers.Activation('relu', name='entryflow_relu1')(x)
    
    # Entry flow 2
    shortcut = keras.layers.Conv2D(filters=128, kernel_size=(1,1), strides=2, padding='valid', name='entryflow_res0_shortcut')(x)
    shortcut = keras.layers.BatchNormalization(name='entryflow_res0_shortcutbn')(shortcut) 
    x = keras.layers.SeparableConv2D(filters=128, kernel_size=(3,3), padding='same', name='entryflow_res0_SepConv0')(x)
    x = keras.layers.BatchNormalization(name='entryflow_res0_bn0')(x)
    x = keras.layers.Activation('relu', name='entryflow_res0_relu0')(x)
    x = keras.layers.SeparableConv2D(filters=128, kernel_size=(3,3),padding='same', name='entryflow_res0_SepConv1')(x)
    x = keras.layers.BatchNormalization(name='entryflow_res0_bn1')(x)
    x = keras.layers.MaxPooling2D(pool_size=(3,3), strides=2, padding='same', name='entryflow_res0_pool')(x)
    x = keras.layers.Add(name='entryflow_res0_add')([shortcut, x])
    
    #Entry flow 3
    shortcut = keras.layers.Conv2D(filters=256, kernel_size=(1,1), strides=2, padding='same', name='entryflow_res1_shortcut')(x)
    shortcut = keras.layers.BatchNormalization(name='entryflow_res1_shortcutbn')(shortcut)
    x = keras.layers.Activation('relu', name='entryflow_res1_relu0')(x)
    x = keras.layers.SeparableConv2D(filters=256, kernel_size=(3,3), padding='same', name='entryflow_res1_SepConv0')(x)
    x = keras.layers.BatchNormalization(name='entryflow_res1_bn0')(x)
    x = keras.layers.Activation('relu', name='entryflow_res1_relu1')(x)
    x = keras.layers.SeparableConv2D(filters=256, kernel_size=(3,3), padding='same', name='entryflow_res1_SepConv1')(x)
    x = keras.layers.BatchNormalization(name='entryflow_res1_bn1')(x)
    x = keras.layers.MaxPooling2D(pool_size=(3,3), strides=2, padding='same', name='entryflow_res1_pool')(x)  
    x = keras.layers.Add(name='entryflow_res1_add')([shortcut, x])
    
    #Entry flow 4
    shortcut = keras.layers.Conv2D(filters=728, kernel_size=(1,1), strides=2, padding='same', name='entryflow_res2_shortcut')(x)
    shortcut = keras.layers.BatchNormalization(name='entryflow_res2_shortcutbn')(shortcut)
    x = keras.layers.Activation('relu', name='entryflow_res2_relu0')(x)
    x = keras.layers.SeparableConv2D(filters=728, kernel_size=(3,3), padding='same', name='entryflow_res2_SepConv0')(x)
    x = keras.layers.BatchNormalization(name='entryflow_res2_bn0')(x)
    x = keras.layers.Activation('relu', name='entryflow_res2_relu1')(x)
    x = keras.layers.SeparableConv2D(filters=728, kernel_size=(3,3), padding='same', name='entryflow_res2_SepConv1')(x)
    x = keras.layers.BatchNormalization(name='entryflow_res2_bn1')(x)
    x = keras.layers.MaxPooling2D(pool_size=(3,3), strides=2, padding='same', name='entryflow_res2_pool')(x)
    x = keras.layers.Add(name='entryflow_res2_add')([shortcut, x])
    
    #Middle flow
    #重复8次
    for i in range(8):
        x = middle_flow(x, i+3, name='middleflow')
    
    #Exit flow
    shortcut = keras.layers.Conv2D(filters=1024, kernel_size=(1,1), strides=2, padding='same', name='exitflow_shortcut')(x)
    shortcut = keras.layers.BatchNormalization(name='exitflow_shortcutbn')(shortcut)
    x = keras.layers.Activation('relu', name='exitflow_relu0')(x)
    x = keras.layers.SeparableConv2D(filters=728, kernel_size=(3,3), padding='same', name='exitflow_SepConv0')(x)
    x = keras.layers.BatchNormalization(name='exitflow_bn0')(x)
    x = keras.layers.Activation('relu', name='exitflow_relu1')(x)
    x = keras.layers.SeparableConv2D(filters=1024, kernel_size=(3,3), padding='same', name='exitflow_SepConv1')(x)
    x = keras.layers.BatchNormalization(name='exitflow_bn1')(x)
    x = keras.layers.MaxPooling2D(pool_size=(3,3), strides=2, padding='same', name='exitflow_pool')(x)   
    x = keras.layers.Add(name='exitflow_add')([shortcut, x])
    
    x = keras.layers.SeparableConv2D(filters=1536, kernel_size=(3,3), padding='same', name='exitflow_SepConv2')(x)
    x = keras.layers.BatchNormalization(name='exitflow_bn2')(x)
    x = keras.layers.Activation('relu', name='exitflow_relu2')(x)
    x = keras.layers.SeparableConv2D(filters=2048, kernel_size=(3,3), padding='same', name='exitflow_SepConv3')(x)
    x = keras.layers.BatchNormalization(name='exitflow_bn3')(x)
    x = keras.layers.Activation('relu', name='exitflow_relu3')(x)
    
    x = keras.layers.GlobalAveragePooling2D(name='exitflow_globalaveragepool')(x)
    
    #dense layer
    #dropout
    if use_dropout:
        x = keras.layers.Dropout(dropout_rate)(x)
        
    #dense-layer
    x = keras.layers.Dense(units = classes, activation='softmax', kernel_initializer='glorot_uniform', name = 'fc' + str(classes))(x)
    model = keras.models.Model(inputs = x_input, outputs = x, name = 'Xception')
    return model