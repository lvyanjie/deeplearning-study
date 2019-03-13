#分组卷积定义
import keras

class GroupConv2D(keras.layers.Conv2D):
    """Group Conv2D

    # Arguments
        filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the convolution).
            multiple of group
        kernel_size: An integer or tuple/list of 2 integers, specifying the
            height and width of the 2D convolution window.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution
            along the height and width.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        padding: one of `"valid"` or `"same"` (case-insensitive).
        group: Integer，split feature map into group.
        output_padding: An integer or tuple/list of 2 integers,
            specifying the amount of padding along the height and width
            of the output tensor.
            Can be a single integer to specify the same value for all
            spatial dimensions.
            The amount of output padding along a given dimension must be
            lower than the stride along that same dimension.
            If set to `None` (default), the output shape is inferred.
        data_format: A string,
            one of `"channels_last"` or `"channels_first"`.
            The ordering of the dimensions in the inputs.
            `"channels_last"` corresponds to inputs with shape
            `(batch, height, width, channels)` while `"channels_first"`
            corresponds to inputs with shape
            `(batch, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
        dilation_rate: an integer or tuple/list of 2 integers, specifying
            the dilation rate to use for dilated convolution.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Currently, specifying any `dilation_rate` value != 1 is
            incompatible with specifying any stride value != 1.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to the kernel matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).

    # Input shape
        4D tensor with shape:
        `(batch, channels, rows, cols)`
        if `data_format` is `"channels_first"`
        or 4D tensor with shape:
        `(batch, rows, cols, channels)`
        if `data_format` is `"channels_last"`.

    # Output shape
        4D tensor with shape:
        `(batch, filters, new_rows, new_cols)`
        if `data_format` is `"channels_first"`
        or 4D tensor with shape:
        `(batch, new_rows, new_cols, filters)`
        if `data_format` is `"channels_last"`.
        `rows` and `cols` values might have changed due to padding.
        If `output_padding` is specified:

        ```
        new_rows = ((rows - 1) * strides[0] + kernel_size[0]
                    - 2 * padding[0] + output_padding[0])
        new_cols = ((cols - 1) * strides[1] + kernel_size[1]
                    - 2 * padding[1] + output_padding[1])
        ```

    # References
        https://blog.csdn.net/lyl771857509/article/details/84109695
    """
    def __init__(self, filters,
             kernel_size,
             strides=(1, 1),
             padding='valid',
             group=1,
             output_padding=None,
             data_format=None,
             dilation_rate=(1, 1),
             activation=None,
             use_bias=True,
             kernel_initializer='glorot_uniform',
             bias_initializer='zeros',
             kernel_regularizer=None,
             bias_regularizer=None,
             activity_regularizer=None,
             kernel_constraint=None,
             bias_constraint=None,
             **kwargs):
        super(GroupConv2D, self).__init__(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)
    
        self.output_padding = output_padding
        self.group = group #初始化
        if self.output_padding is not None:
            self.output_padding = keras.utils.conv_utils.normalize_tuple(
                self.output_padding, 2, 'output_padding')
            for stride, out_pad in zip(self.strides, self.output_padding):
                if out_pad >= stride:
                    raise ValueError('Stride ' + str(self.strides) + ' must be '
                                     'greater than output padding ' +
                                     str(self.output_padding))
    
    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim // self.group, self.filters)  #kernel size: (kernel_size[0], kernel_size[1], input_channels, output_channels)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = keras.engine.base_layer.InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_dim})
        self.built = True
        
    def call(self, inputs):
        '''
        组卷积的具体实现
        :param inputs: 
        :return: 
        ''' 
        group_conv = []#保存每组卷积计算featuremap
        output_channels = self.filters // self.group  #output channel
        if self.data_format == 'channels_first':#theano
            channel_axis = 1
        else:
            channel_axis = -1
        #对每个group分别进行卷积计算
        for i in range(self.group):
            if self.data_format == 'channels_first':
                x = inputs[:,i*output_channels:(i+1)*output_channels,:,:]
            else:
                x = inputs[:,:,:,i*output_channels:(i+1)*output_channels]
            #kernel shape: (kernel_size[0], kernel_size[1], input_channels, output_channels)
            outputs = keras.backend.conv2d(x, self.kernel[:,:,:,i*output_channels:(i+1)*output_channels], self.strides, self.padding, self.data_format, self.dilation_rate)
            
            if self.use_bias:
                outputs = keras.backend.bias_add(outputs, self.bias[i*output_channels:(i+1)*output_channels], self.data_format)          
            group_conv.append(outputs)

        outputs = keras.backend.concatenate(group_conv, axis=channel_axis)
        return outputs
    
    def get_config(self):
        config = super(keras.layers.Conv2D, self).get_config()
        config.pop('rank')
        config["group"] = self.group
        return config
    
def convolutional_block(x, f, filters, group, stage, block, stride=2):
    '''
    convolutional block
    :param x: input tensor (n, W, H, C)
    :param f: integer, conv kernel size
    :param filters: integer list, the number of filters in the conv layers
    :param group: integer, number of group convolution
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
    
    #conv2 group convolution
    x = GroupConv2D(filters = F2, kernel_size = (f, f), strides = (1, 1), padding = 'same', 
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

def identity_block(x, f, filters, group, stage, block):
    '''
    identity block
    :param x: input tensor (n, W, H, C)
    :param f: integer, conv kernel size
    :param filters: integer list, the number of filters in the conv layers
    :param group: integer, the number of group convolution
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
    x = GroupConv2D(filters = F2, kernel_size = (f, f), strides = (1, 1), padding = 'same', 
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

def ResNext50(input_shape=(224,224,3), classes = 1000, use_dropout = True, dropout_rate = 0.2):
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
                                         kernel_initializer = keras.initializers.glorot_uniform(seed=0), name='conv1')(x)#112x112x64
    x = keras.layers.normalization.BatchNormalization(axis = 3, name = 'bn_conv1')(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.pooling.MaxPooling2D(pool_size = (3, 3), strides = (2, 2))(x)#56x56x64
    
    #stage2
    x = convolutional_block(x = x, f = 3, filters = [128, 128, 256], group = 32, stage = 2, block = 'a', stride = 1)#56x56x256
    x = identity_block(x = x, f = 3, filters = [128, 128, 256], group = 32, stage = 2, block = 'b')#56x56x256
    x = identity_block(x = x, f = 3, filters = [128, 128, 256], group = 32, stage = 2, block = 'c')#56x56x256
    
    #stage3
    x = convolutional_block(x = x, f = 3, filters = [256, 256, 512], group = 32, stage = 3, block = 'a', stride = 2)#28x28x512
    x = identity_block(x = x, f = 3, filters = [256, 256, 512], group = 32, stage = 3, block = 'b')#28x28x512
    x = identity_block(x = x, f = 3, filters = [256, 256, 512], group = 32, stage = 3, block = 'c')#28x28x512
    x = identity_block(x = x, f = 3, filters = [256, 256, 512], group = 32, stage = 3, block = 'd')#28x28x512
    
    #stage4
    x = convolutional_block(x = x, f = 3, filters = [512, 512, 1024], group = 32, stage = 4, block = 'a', stride = 2)#14x14x1024
    x = identity_block(x = x, f = 3, filters = [512, 512, 1024], group = 32, stage = 4, block = 'b')#14x14x1024
    x = identity_block(x = x, f = 3, filters = [512, 512, 1024], group = 32, stage = 4, block = 'c')#14x14x1024
    x = identity_block(x = x, f = 3, filters = [512, 512, 1024], group = 32, stage = 4, block = 'd')#14x14x1024
    x = identity_block(x = x, f = 3, filters = [512, 512, 1024], group = 32, stage = 4, block = 'e')#14x14x1024
    x = identity_block(x = x, f = 3, filters = [512, 512, 1024], group = 32, stage = 4, block = 'f')#14x14x1024
    
    #stage5
    x = convolutional_block(x = x, f = 3, filters = [1024, 1024, 2048], group = 32, stage = 5, block = 'a', stride = 2)#7x7x2048
    x = identity_block(x = x, f = 3, filters = [1024, 1024, 2048], group = 32, stage = 5, block = 'b')#7x7x2048
    x = identity_block(x = x, f = 3, filters = [1024, 1024, 2048], group = 32, stage = 5, block = 'c')#7x7x2048
    
    #avgpool
    x = keras.layers.GlobalAveragePooling2D()(x)
    
    #dropout
    if use_dropout:
        x = keras.layers.Dropout(dropout_rate)(x)
    
    #FC
    x = keras.layers.core.Dense(units = classes, activation='softmax', kernel_initializer='glorot_uniform', name = 'fc' + str(classes))(x)
    #create model
    model = keras.models.Model(inputs = x_input, outputs = x, name = 'ResNet50')
    
    return model