# from tensorflow.keras.layers import (Dense, Dropout, Reshape,concatenate, Input, Add,Conv3D, Conv2D,Conv1D, Flatten, MaxPooling2D, Activation, SeparableConv2D, GlobalAveragePooling2D)
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from keras_cv_attention_models import attention_layers
# **2D CNN**
def CNN2D(WS, k, Num_Classes):
  input_layer = tf.keras.layers.Input((WS, WS, k))
  conv_layer1 = tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
  conv_layer2 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(conv_layer1)
  conv_layer3 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(conv_layer2)
  conv_layer4 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv_layer3)
  flatten_layer = tf.keras.layers.Flatten()(conv_layer4)
  dense_layer1 = tf.keras.layers.Dense(units=256, activation='relu')(flatten_layer)
  dense_layer1 = tf.keras.layers.Dropout(0.4)(dense_layer1)
  dense_layer2 = tf.keras.layers.Dense(units=128, activation='relu')(dense_layer1)
  dense_layer2 = tf.keras.layers.Dropout(0.4)(dense_layer2)
  output_layer = tf.keras.layers.Dense(units=Num_Classes, activation='softmax')(dense_layer2)
  model = Model(inputs=input_layer, outputs=output_layer, name='CNN2D')
  return model

####################### **Exception2D** #################################################

def entry_flow(inputs):
    x = tf.keras.layers.Conv2D(32, (3,3), strides = (2,2), padding = 'same')(inputs)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(64, 3, padding = 'same')(x)
    x = tf.keras.layers.Activation('relu')(x)
    previous_block_activation = x
    for size in [128, 256, 512]:
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.SeparableConv2D(size, (3, 3), padding = 'same')(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.SeparableConv2D(size, (3, 3), padding = 'same')(x)
        x = tf.keras.layers.MaxPooling2D((3, 3), strides = (2, 2), padding = 'same')(x)
        residual = tf.keras.layers.Conv2D(size, (1, 1), strides = (2, 2), padding = 'same')(previous_block_activation)
        x = tf.keras.layers.Add()([x, residual])
        previous_block_activation = x
    return x

def middle_flow(x, num_blocks = 8):
    previous_block_activation = x
    for _ in range(num_blocks):
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.SeparableConv2D(512, (3, 3), padding = 'same')(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.SeparableConv2D(512, (3,3), padding = 'same')(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.SeparableConv2D(512, (3,3), padding = 'same')(x)
        x = tf.keras.layers.Add()([x, previous_block_activation])
        previous_block_activation = x
    return x

def exit_flow(x,Num_Classes) :
    previous_block_activation = x
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.SeparableConv2D(512, (3,3), padding='same')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.SeparableConv2D(728, (3,3), padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((3,3), strides=(2,2), padding='same')(x)
    residual = tf.keras.layers.Conv2D(728, (1,1), strides=(2,2), padding='same')(previous_block_activation)
    x = tf.keras.layers.Add()([x, residual])
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.SeparableConv2D(512, (3, 3), padding = 'same')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.SeparableConv2D(728, (3, 3), padding = 'same')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(Num_Classes, activation='softmax')(x)
    return x

def Exception2D(WS, k, Num_Classes):
  inputs = tf.keras.layers.Input(shape=(WS, WS, k))
  outputs = exit_flow(middle_flow(entry_flow(inputs)),Num_Classes)
  model = Model(inputs, outputs, name='Exception2D')
  return model


########################## **(2+1)D Exception Net** #############################################################

class Conv2Plus1D(tf.keras.layers.Layer):
  def __init__(self, filters, kernel_size, padding):
    """
      A sequence of convolutional layers that first apply the convolution operation over the
      spatial dimensions, and then the temporal dimension.
      https://www.tensorflow.org/tutorials/video/video_classification
      https://arxiv.org/abs/1711.11248v3 (Original Paper for (2+1)D)
    """
    super().__init__()
    self.seq = tf.keras.Sequential([
        # Spatial decomposition
        tf.keras.layers.Conv3D(filters=filters, kernel_size=(1, kernel_size[1], kernel_size[2]),
                      padding=padding),
        # Temporal decomposition
        tf.keras.layers.Conv3D(filters=filters, kernel_size=(kernel_size[0], 1, 1),
                      padding=padding)
        ])

  def call(self, x):
    return self.seq(x)

def entry_block(inputs):
    x = tf.keras.layers.Conv3D(filters=32, kernel_size=(5, 5, 3), activation='relu', padding = 'same')(inputs)
    x = tf.keras.layers.Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu', padding = 'same')(x)
    previous_block_activation = x
    for size in [128, 256, 512]:
        # x = Activation('relu')(x)
        x = Conv2Plus1D(filters=size, kernel_size=(3, 3, 3), padding='same')(x)
        # x = Dropout(0.4)(x)
        # x = Activation('relu')(x)
        x = Conv2Plus1D(filters=size, kernel_size=(3, 3, 3), padding='same')(x)
        # x = Dropout(0.4)(x)
        x = tf.keras.layers.MaxPooling3D(pool_size=(3, 3, 3), padding = 'same')(x)
        residual = tf.keras.layers.Conv3D(size, kernel_size=(3, 3, 3), strides = (3, 3, 3), padding = 'same')(previous_block_activation)
        x = tf.keras.layers.Add()([x, residual])
        previous_block_activation = x
    return x

def middle_block(x, num_blocks = 8):
    previous_block_activation = x
    for _ in range(num_blocks):
        x = tf.keras.layers.Activation('relu')(x)
        x = Conv2Plus1D(filters=128, kernel_size=(5, 3, 3), padding='same')(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = Conv2Plus1D(filters=256, kernel_size=(5, 3, 3), padding='same')(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = Conv2Plus1D(filters=512, kernel_size=(5, 3, 3), padding='same')(x)
        x = tf.keras.layers.Add()([x, previous_block_activation])
        previous_block_activation = x
    return x

def exit_block(x,Num_Classes) :
    previous_block_activation = x
    x = tf.keras.layers.Activation('relu')(x)
    x = Conv2Plus1D(filters=512, kernel_size=(3, 3, 3), padding='same')(x)
    # x = Dropout(0.4)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = Conv2Plus1D(filters=728, kernel_size=(3, 3, 3), padding='same')(x)
    # x = Dropout(0.4)(x)
    x = tf.keras.layers.MaxPooling3D(pool_size=(3, 3, 3), padding = 'same')(x)
    residual = tf.keras.layers.Conv3D(728, kernel_size=(3, 3, 3), padding = 'same')(previous_block_activation)
    x = tf.keras.layers.Add()([x, residual])
    x = tf.keras.layers.Activation('relu')(x)
    x = Conv2Plus1D(filters=512, kernel_size=(3, 3, 3), padding='same')(x)
    # x = Dropout(0.4)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = Conv2Plus1D(filters=256, kernel_size=(3, 3, 3), padding='same')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.GlobalAveragePooling3D()(x)
    x = tf.keras.layers.Dense(Num_Classes, activation='softmax')(x)
    return x

def Xtreme_Exception(WS, k, Num_Classes):
  inputs = tf.keras.layers.Input(shape=(WS, WS, k, 1))
  outputs = exit_block(middle_block(entry_block(inputs)),Num_Classes)
  model = Model(inputs, outputs, name='Xtreme_Exception')
  return model


# **2D inception Net**

def IN2D(WS, k, Num_Classes):
  input_img = tf.keras.layers.Input(shape=(WS, WS, k))
  ## Block 1
  layer_1 = tf.keras.layers.Conv2D(30, (1, 1), padding = 'same', activation = 'relu')(input_img)
  layer_1 = tf.keras.layers.Conv2D(20, (1, 1), padding = 'same', activation = 'relu')(layer_1)
  layer_1 = tf.keras.layers.Conv2D(10, (3, 3), padding = 'same', activation = 'relu')(layer_1)
  ## Block 2
  layer_2 = tf.keras.layers.Conv2D(40, (1, 1), padding = 'same', activation = 'relu')(input_img)
  layer_2 = tf.keras.layers.Conv2D(20, (1, 1), padding = 'same', activation = 'relu')(layer_2)
  layer_2 = tf.keras.layers.Conv2D(10, (5, 5), padding = 'same', activation = 'relu')(layer_2)
  ## Block 3
  layer_3 = tf.keras.layers.MaxPooling2D((3, 3), strides = (1, 1), padding = 'same')(input_img)
  layer_3 = tf.keras.layers.Conv2D(20, (1, 1), padding = 'same', activation = 'relu')(layer_3)
  layer_3 = tf.keras.layers.Conv2D(10, (1, 1), padding = 'same', activation = 'relu')(layer_3)
  ## Concatination
  mid_1 = tf.keras.layers.concatenate([layer_1, layer_2, layer_3], axis = 3)
  ## Convolution
  layer_4 = tf.keras.layers.Conv2D(128, (1, 1), activation = 'relu')(mid_1)
    ## Classification Module
  flat_1 = tf.keras.layers.Flatten()(layer_4)
  dense_1 = tf.keras.layers.Dense(1200, activation = 'relu')(flat_1)
  dense_2 = tf.keras.layers.Dense(600, activation = 'relu')(dense_1)
  dense_3 = tf.keras.layers.Dense(150, activation = 'relu')(dense_2)
  output = tf.keras.layers.Dense(Num_Classes, activation = 'softmax')(dense_3)
  ## Medel
  model = Model([input_img], output, name='IN2D')
  return model

# **3D inception Net**

def IN3D(WS, k, Num_Classes):
  input_img = tf.keras.layers.Input(shape=(WS, WS, k, 1))
  ## Block 1
  layer_1 = tf.keras.layers.Conv3D(30, (5, 5, 7), activation='relu', padding='same')(input_img)
  layer_1 = tf.keras.layers.Conv3D(20, (3, 3, 5), activation='relu', padding='same')(layer_1)
  layer_1 = tf.keras.layers.Conv3D(10, (3, 3, 3), activation='relu', padding='same')(layer_1)
  ## Block 2
  layer_2 = tf.keras.layers.Conv3D(40, (5, 5, 7), activation='relu', padding='same')(input_img)
  layer_2 = tf.keras.layers.Conv3D(20, (3, 3, 5), activation='relu', padding='same')(layer_2)
  layer_2 = tf.keras.layers.Conv3D(10, (3, 3, 3), activation='relu', padding='same')(layer_2)
  ## Block 3
  layer_3 = tf.keras.layers.Conv3D(60, (5, 5, 7), activation='relu', padding='same')(input_img)
  layer_3 = tf.keras.layers.Conv3D(30, (3, 3, 5), activation='relu', padding='same')(layer_3)
  layer_3 = tf.keras.layers.Conv3D(10, (3, 3, 3), activation='relu', padding='same')(layer_3)
  ## Concatination
  mid_1 = tf.keras.layers.concatenate([layer_1, layer_2, layer_3], axis = 3)
  ## Convolution
  layer_4 = tf.keras.layers.Conv3D(128, (1, 1, 1), activation = 'relu')(mid_1)
  ## Classification Module
  flat_1 = tf.keras.layers.Flatten()(layer_4)
  dense_1 = tf.keras.layers.Dense(512, activation='relu')(flat_1)
  dense_1 = tf.keras.layers.Dropout(0.4)(dense_1)
  dense_2 = tf.keras.layers.Dense(128, activation='relu')(dense_1)
  dense_2 = tf.keras.layers.Dropout(0.4)(dense_2)
  dense_3 = tf.keras.layers.Dense(64, activation='relu')(dense_2)
  dense_3 = tf.keras.layers.Dropout(0.4)(dense_3)
  output = tf.keras.layers.Dense(Num_Classes, activation='softmax')(dense_3)
  model = Model([input_img], output, name='IN3D')
  return model

# **Hybrid Inception Net**

def HybIN(WS, k, Num_Classes):
  input_img = tf.keras.layers.Input(shape=(WS, WS, k, 1))
  ## Block 1
  layer_1 = tf.keras.layers.Conv3D(30, (5, 5, 7), activation='relu', padding='same')(input_img)
  layer_1 = tf.keras.layers.Conv3D(20, (3, 3, 5), activation='relu', padding='same')(layer_1)
  layer_1 = tf.keras.layers.Conv3D(10, (3, 3, 3), activation='relu', padding='same')(layer_1)
  layer_1_shape = layer_1.shape
  ## 2D Structure
  layer_1 = tf.keras.layers.Reshape((layer_1_shape[1], layer_1_shape[2], layer_1_shape[3]*layer_1_shape[4]))(layer_1)
  layer_1 = tf.keras.layers.MaxPooling2D((3,3), strides=(1,1), padding='same')(layer_1)
  layer_1 = tf.keras.layers.Conv2D(filters = 16, kernel_size = (1, 1), activation = 'relu', padding='same')(layer_1)
  layer_1 = tf.keras.layers.Conv2D(filters = 32, kernel_size = (1, 1), activation = 'relu', padding='same')(layer_1)
  layer_1 = tf.keras.layers.Conv2D(filters = 64, kernel_size = (1, 1), activation = 'relu', padding='same')(layer_1)
  ## Block 2
  layer_2 = tf.keras.layers.Conv3D(40, (5, 5, 7), activation = 'relu', padding='same')(input_img)
  layer_2 = tf.keras.layers.Conv3D(20, (3, 3, 5), activation = 'relu', padding='same')(layer_2)
  layer_2 = tf.keras.layers.Conv3D(10, (3, 3, 3), activation = 'relu', padding='same')(layer_2)
  ## 2D Structure
  layer_2_shape = layer_2.shape
  layer_2 = tf.keras.layers.Reshape((layer_2_shape[1], layer_2_shape[2], layer_2_shape[3]*layer_2_shape[4]))(layer_2)
  layer_2 = tf.keras.layers.MaxPooling2D((3,3), strides=(1,1), padding='same')(layer_2)
  layer_2 = tf.keras.layers.Conv2D(filters = 16, kernel_size = (1, 1), activation = 'relu', padding='same')(layer_2)
  layer_2 = tf.keras.layers.Conv2D(filters = 32, kernel_size = (1, 1), activation = 'relu', padding='same')(layer_2)
  layer_2 = tf.keras.layers.Conv2D(filters = 64, kernel_size = (1, 1), activation = 'relu', padding='same')(layer_2)
  ## Block 3
  layer_3 = tf.keras.layers.Conv3D(60, (5, 5, 7), activation='relu', padding='same')(input_img)
  layer_3 = tf.keras.layers.Conv3D(30, (3, 3, 5), activation='relu', padding='same')(layer_3)
  layer_3 = tf.keras.layers.Conv3D(10, (3, 3, 3), activation='relu', padding='same')(layer_3)
  ## 2D Structure
  layer_3_shape = layer_3.shape
  layer_3 = tf.keras.layers.Reshape((layer_3_shape[1], layer_3_shape[2], layer_3_shape[3]*layer_3_shape[4]))(layer_3)
  layer_3 = tf.keras.layers.MaxPooling2D((3,3), strides=(1,1), padding='same')(layer_3)
  layer_3 = tf.keras.layers.Conv2D(filters = 16, kernel_size=(1, 1), activation = 'relu', padding='same')(layer_3)
  layer_3 = tf.keras.layers.Conv2D(filters = 32, kernel_size=(1, 1), activation = 'relu', padding='same')(layer_3)
  layer_3 = tf.keras.layers.Conv2D(filters = 64, kernel_size = (1, 1), activation = 'relu', padding='same')(layer_3)
  ## Concatinate
  mid_1 = tf.keras.layers.concatenate([layer_1, layer_2, layer_3], axis = 3)
  ## Convolution
  layer_4 = tf.keras.layers.Conv2D(filters = 128, kernel_size = (1, 1), activation = 'relu', padding='same')(mid_1)
  ## Classification Model
  flat_1 = tf.keras.layers.Flatten()(layer_4)
  dense_1 = tf.keras.layers.Dense(512, activation='relu')(flat_1)
  dense_1 = tf.keras.layers.Dropout(0.4)(dense_1)
  dense_2 = tf.keras.layers.Dense(128, activation='relu')(dense_1)
  dense_2 = tf.keras.layers.Dropout(0.4)(dense_2)
  dense_3 = tf.keras.layers.Dense(64, activation='relu')(dense_2)
  dense_3 = tf.keras.layers.Dropout(0.4)(dense_3)
  output = tf.keras.layers.Dense(Num_Classes, activation='softmax')(dense_3)
  ## Medel
  model = Model([input_img], output, name='HybIN')
  return model

# **Hybrid CNN**

def HybCNN(WS, k, Num_Classes):
  input_layer = tf.keras.layers.Input((WS, WS, k, 1))
  conv_layer1 = tf.keras.layers.Conv3D(filters=8, kernel_size=(3, 3, 7), activation='relu', padding='same')(input_layer)
  conv_layer2 = tf.keras.layers.Conv3D(filters=16, kernel_size=(3, 3, 5), activation='relu', padding='same')(conv_layer1)
  conv_layer3 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(conv_layer2)
  ##conv_layer4 = Conv3D(filters=64, kernel_size=(3, 3, 1), activation='relu')(conv_layer3)
  ## Conv3D with kernel_size=(3, 3, 1) can be used as well instead to reshape and Con2D.
  conv3d_shape = conv_layer3.shape
  conv_layer3 = tf.keras.layers.Reshape((conv3d_shape[1], conv3d_shape[2], conv3d_shape[3]*conv3d_shape[4]))(conv_layer3)
  conv_layer4 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same')(conv_layer3)
  flatten_layer = tf.keras.layers.Flatten()(conv_layer4)
  dense_layer1 = tf.keras.layers.Dense(units=256, activation = 'relu')(flatten_layer)
  dense_layer1 = tf.keras.layers.Dropout(0.4)(dense_layer1)
  dense_layer2 = tf.keras.layers.Dense(units = 128, activation = 'relu')(dense_layer1)
  dense_layer2 = tf.keras.layers.Dropout(0.4)(dense_layer2)
  output_layer = tf.keras.layers.Dense(units = Num_Classes, activation = 'softmax')(dense_layer2)
  # Define the Model with Input and Output Layers
  model = Model(inputs = input_layer, outputs = output_layer, name='HybCNN')
  return model

# **A Fast and Compact 3D CNN**
def CNN3D(WS, k, Num_Classes):
  input_layer = tf.keras.layers.Input((WS, WS, k, 1))
  conv_layer1 = tf.keras.layers.Conv3D(filters=8, kernel_size=(3, 3, 7), activation='relu', padding='same')(input_layer)
  conv_layer2 = tf.keras.layers.Conv3D(filters=16, kernel_size=(3, 3, 5), activation='relu', padding='same')(conv_layer1)
  conv_layer3 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(conv_layer2)
  conv_layer4 = tf.keras.layers.Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu', padding='same')(conv_layer3)
  flatten_layer = tf.keras.layers.Flatten()(conv_layer4)
  dense_layer1 = tf.keras.layers.Dense(units=256, activation='relu')(flatten_layer)
  dense_layer2 = tf.keras.layers.Dense(units=128, activation='relu')(dense_layer1)
  output_layer = tf.keras.layers.Dense(units = Num_Classes, activation='softmax')(dense_layer2)
  model = Model(inputs=input_layer, outputs=output_layer, name='CNN3D')
  return model

# ## Attention GCN 

def AttentionGCN(WS, k, Num_Classes):
    input_layer = tf.keras.layers.Input(shape=(WS, WS, k), name='input')
    conv1 = tf.keras.layers.Conv2D(8, kernel_size=1, activation='relu', name='conv1', kernel_regularizer=regularizers.l2(0.0005))(input_layer)
    nn = attention_layers.cot_attention(conv1, kernel_size=3)
########## Graph-based global reasoning module
    graph_t = tf.keras.layers.Reshape((8, WS*WS))(conv1)
    squeezed_graph_t = tf.keras.layers.Conv1D(16, 1, activation='relu', name='squeezer', kernel_regularizer=regularizers.l2(0.0001))(graph_t)
    squeezed_graph = tf.keras.layers.Reshape((8, 16))(squeezed_graph_t)
    gconv = tf.keras.layers.Conv1D(64, 1, activation='relu', name='gconv', kernel_regularizer=regularizers.l2(0.0001))(squeezed_graph)
    gconv_t = tf.keras.layers.Reshape((8, 64))(gconv)
    unsqueezed_graph_transpose = tf.keras.layers.Conv1D(64, 1, activation='relu', name='unsqueezer', kernel_regularizer=regularizers.l2(0.0001))(gconv_t)
    unsqueezed_graph = tf.keras.layers.Reshape((64, 8))(unsqueezed_graph_transpose)
    glore = tf.keras.layers.Reshape((8, WS, WS))(unsqueezed_graph)
    block1 = tf.keras.layers.concatenate([conv1, glore], axis=-1)
    block12 = tf.keras.layers.concatenate([conv1, block1, nn], axis=-1)
    conv2 = tf.keras.layers.Conv2D(32, kernel_size=1, activation='relu', name='conv2', kernel_regularizer=regularizers.l2(0.0005))(block12)

    ########### Graph-based global reasoning module 2
    graph_t2 = tf.keras.layers.Reshape((32,WS*WS))(conv2)
    squeezed_graph_t2 = tf.keras.layers.Conv1D(64, 1, activation='relu', name='squeezer2', kernel_regularizer=regularizers.l2(0.0001))(graph_t2)
    squeezed_graph2 = tf.keras.layers.Reshape((64, 32))(squeezed_graph_t2)
    gconv2 = tf.keras.layers.Conv1D(32, 1, activation='relu', name='gconv2', kernel_regularizer=regularizers.l2(0.0001))(squeezed_graph2)
    gconv_transpose2 = tf.keras.layers.Reshape((64, 32))(gconv2)
    unsqueezed_graph_t2 = tf.keras.layers.Conv1D(8*8, 1, activation='relu', name='unsqueezer2', kernel_regularizer=regularizers.l2(0.0001))(gconv_transpose2)
    unsqueezed_graph2 = tf.keras.layers.Reshape((8*8, 64))(unsqueezed_graph_t2)
    glore2 = tf.keras.layers.Reshape((8, 8, 64))(unsqueezed_graph2)
    block2 = tf.keras.layers.concatenate([conv2, glore2], axis=-1) #################  you do not need to add the feature maps of the conv2 layer
    block21 = tf.keras.layers.concatenate([conv2, block2, nn], axis=-1) #################  you do not need to use the feature maps of the attention block2

    conv3 = tf.keras.layers.Conv2D(32, kernel_size=1, activation='relu', name='conv3', kernel_regularizer=regularizers.l2(0.0005))(block21)
    conv4 = tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu', name='conv4', kernel_regularizer=regularizers.l2(0.0005))(conv3)
    conv5 = tf.keras.layers.Conv2D(128, kernel_size=1, activation='relu', name='conv5', kernel_regularizer=regularizers.l2(0.0005))(conv4)

    flat = tf.keras.layers.Flatten()(conv5)
    fc1 = tf.keras.layers.Dense(100, activation='relu', name='fc1', kernel_regularizer=regularizers.l2(0.01))(flat)
    fc2 = tf.keras.layers.Dense(20, activation='relu', name='fc2', kernel_regularizer=regularizers.l2(0.01))(fc1)
    fc3 = tf.keras.layers.Dense(Num_Classes, activation='softmax', name='fc3', kernel_regularizer=regularizers.l2(0.0001))(fc2)
    model = Model(inputs=input_layer, outputs=fc3, name='AttentionGCN')
    return model