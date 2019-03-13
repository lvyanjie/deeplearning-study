#inception module模块实现
import keras
import tensorflow as tf

def inception_module(x, 
                     filters_11,
                     filters_33,
                     filters_55,
                     reduce_filters_33,
                     reduce_filters_55,
                     reduce_filters_pool,
                     name):
    '''
    inception module
    :param x: input tensor
    :param filters_11: integer, number of 1*1 conv
    :param filters_33: integer, number of 3*3 conv
    :param filters_55: integer, number of 5*5 conv
    :param reduce_filters_33: integer, reduce channels of 3*3 conv
    :param reduce_filters_55: integer, reduce channels of 5*5 conv
    :param reduce_filters_pool: integer: reduce channels of pooling
    :param name: string
    
    returns:
    output tensor
    '''
    #1*1conv
    x_conv11 = keras.layers.Conv2D(filters = filters_11, kernel_size=(1,1),padding='same', name = name + '_conv_11')(x)
    x_conv11 = keras.layers.Activation('relu', name=name + '_relu_11')(x_conv11)
    
    #3*3 conv
    x_conv33 = keras.layers.Conv2D(filters = reduce_filters_33, kernel_size=(1,1),padding = 'same', name = name + '_reduce_33')(x)
    x_conv33 = keras.layers.Activation('relu', name = name + '_relu_reduce_33')(x_conv33)
    x_conv33 = keras.layers.Conv2D(filters = filters_33, kernel_size = (3,3),padding = 'same', name = name + '_conv_33')(x_conv33)
    x_conv33 = keras.layers.Activation('relu', name = name + '_relu_33')(x_conv33)
    
    #5*5 conv
    x_conv55 = keras.layers.Conv2D(filters = reduce_filters_55, kernel_size=(1,1),padding='same', name=name + '_reduce_55')(x)
    x_conv55 = keras.layers.Activation('relu', name = name + '_relu_reduce_55')(x_conv55)
    x_conv55 = keras.layers.Conv2D(filters = filters_55, kernel_size = (5, 5),padding = 'same', name = name + '_conv_55')(x_conv55)
    x_conv55 = keras.layers.Activation('relu', name = name + '_relu_55')(x_conv55)
    
    #maxpool
    x_pool = keras.layers.MaxPooling2D(pool_size=(3,3), strides=1, padding='same')(x)
    x_pool = keras.layers.Conv2D(filters = reduce_filters_pool, kernel_size=(1,1),padding = 'same', name = name + '_reduce_pool')(x_pool)
    x_pool = keras.layers.Activation('relu', name = name + '_relu_pool')(x_pool)
    
    x = keras.layers.Concatenate(axis=3, name = name + '_concat')([x_conv11, x_conv33, x_conv55, x_pool])
    return x
    
#定义LRN（局部响应归一化）层
#无训练参数
class LRN(keras.engine.Layer):
    def __init__(self, depth_radius=2,bias=1.0,alpha=1,beta=1, **kwargs):
        self.depth_radius = depth_radius
        self.bias = bias
        self.alpha = alpha
        self.beta = beta
        super(LRN, self).__init__(**kwargs)
 
    def build(self, input_shape):
        super(LRN, self).build(input_shape)
 
    def call(self, x):
        return tf.nn.lrn(x, depth_radius=self.depth_radius,bias=self.bias,alpha=self.alpha,beta=self.beta)
 
    def compute_output_shape(self, input_shape):
        return input_shape

def Inceptionv1(input_shape=(224,224,3), classes = 1000):
    '''
    inception v1

    :param input_shape: tuple, input shape
    :param classes: integer, number of classes
    
    returns:
    keras model
    '''
    x_input = keras.layers.Input(input_shape)
    
    x = keras.layers.Conv2D(filters = 64, kernel_size = (7, 7), strides=2, padding = 'same', name = 'conv1')(x_input)
    x = keras.layers.Activation('relu', name = 'relu1')(x)
    x = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same', name='max_pool1')(x)
    
    #LRN
    x = LRN(alpha=0.0001, bias = 1e-4, beta=0.75, name= 'lrn1')(x)#
    
    x = keras.layers.Conv2D(filters = 64, kernel_size = (1, 1), strides=1, padding='same', name = 'conv2_reduce')(x)
    x = keras.layers.Activation('relu', name = 'relu2_reduce')(x)
    x = keras.layers.Conv2D(filters = 192, kernel_size = (3, 3), strides = 1, padding = 'same', name = 'conv2')(x)
    x = keras.layers.Activation('relu', name = 'relu2')(x)
    
    #LRN
    x = LRN(alpha=0.0001, bias = 1e-4, beta=0.75, name='lrn2')(x)
    
    #maxpool
    x = keras.layers.MaxPooling2D(pool_size=(3, 3), strides = 2, padding = 'same', name = 'max_pool2')(x)
    
    #inception(3a)
    x = inception_module(x, filters_11 = 64,filters_33=128,filters_55=32,reduce_filters_33=96,reduce_filters_55=16,reduce_filters_pool=32,name='inception_3a')
    #inception(3b)
    x = inception_module(x, filters_11 = 128,filters_33=192,filters_55=96,reduce_filters_33=128,reduce_filters_55=32,reduce_filters_pool=64,name='inception_3b')
    #maxpool
    x = keras.layers.MaxPooling2D(pool_size=(3, 3), strides = 2, padding = 'same', name = 'max_pool3')(x)
    
    #inception(4a~4e)
    x = inception_module(x, filters_11 = 192, filters_33 = 208, filters_55 = 48, reduce_filters_33 = 96, reduce_filters_55 = 16, reduce_filters_pool = 64, name = 'inception_4a')
    x = inception_module(x, filters_11 = 160, filters_33 = 224, filters_55 = 64, reduce_filters_33 = 112, reduce_filters_55 = 24, reduce_filters_pool = 64, name = 'inception_4b')
    x = inception_module(x, filters_11 = 128, filters_33 = 256, filters_55 = 64, reduce_filters_33 = 128, reduce_filters_55 = 24, reduce_filters_pool = 64, name = 'inception_4c')
    x = inception_module(x, filters_11 = 112, filters_33 = 288, filters_55 = 64, reduce_filters_33 = 144, reduce_filters_55 = 32, reduce_filters_pool = 64, name = 'inception_4d')
    x = inception_module(x, filters_11 = 256, filters_33 = 320, filters_55 = 128, reduce_filters_33 = 160, reduce_filters_55 = 32, reduce_filters_pool = 128, name = 'inception_4e')
    
    #maxpool
    x = keras.layers.MaxPooling2D(pool_size=(3,3), strides=2, padding='same', name = 'max_pool4')(x)
    
    #inception(5a~5b)
    x = inception_module(x, filters_11 = 256, filters_33 = 320, filters_55 = 128, reduce_filters_33 = 160, reduce_filters_55 = 32, reduce_filters_pool = 128, name = 'inception_5a')
    x = inception_module(x, filters_11 = 384, filters_33 = 384, filters_55 = 128, reduce_filters_33 = 192, reduce_filters_55 = 48, reduce_filters_pool = 128, name = 'inception_5b')
    
    #global average pooling
    x = keras.layers.GlobalAveragePooling2D()(x)
    
    #dropout(40%)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(units = classes, activation='softmax', name = 'classification')(x)
    
    model = keras.models.Model(inputs=x_input, outputs = x, name = 'inceptionv1')
    return model