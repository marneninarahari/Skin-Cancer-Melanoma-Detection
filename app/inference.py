import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

from keras.preprocessing import image 
from tensorflow import GradientTape
import matplotlib.cm as cm


# def read_image(image):
#     return mpimg.imread(image)

def read_image(img_name):
    img = image.load_img(img_name, target_size=(224,224))
    img = image.img_to_array(img)
    img = img/255.0
    return img


def format_image(image):
    return tf.image.resize(image[tf.newaxis, ...], [224, 224]) / 255.0


def get_category(img):
    """Write a Function to Predict the Class Name

    Args:
        img [jpg]: image file

    Returns:
        [str]: Prediction
    """

    path = 'app/static/models/'
    tflite_model_file = 'converted_model.tflite'

    # Load TFLite model and allocate tensors.
    with open(path + tflite_model_file, 'rb') as fid:
        tflite_model = fid.read()

    # Interpreter interface for TensorFlow Lite Models.
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    # Gets model input and output details.
    input_index = interpreter.get_input_details()[0]
    output_index = interpreter.get_output_details()[0]
    #print('input_index', input_index)
    #print('output_index', output_index)

    print(img)
    input_img = read_image(img)
    input_img = np.expand_dims(input_img, axis=0)
    interpreter.resize_tensor_input(0, [1, 224, 224, 3])
    interpreter.allocate_tensors()
    #format_img = format_image(input_img)
    # Sets the value of the input tensor
#    interpreter.set_tensor(input_index, format_img)
    interpreter.set_tensor(input_index['index'], input_img)
    # Invoke the interpreter.
    interpreter.invoke()

    predictions_array = interpreter.get_tensor(output_index['index'])
    print(predictions_array) #.round().reshape(-1)
    #predicted_label = np.argmax(predictions_array)
    predicted_label = int(predictions_array.round().reshape(-1))
    print(predicted_label)

    #class_names = ['rock', 'paper', 'scissors']
    class_names = ['Benign', 'Malignant']

    return class_names[predicted_label]


def plot_category(img, current_time):
    """Plot the input image

    Args:
        img [jpg]: image file
    """
    read_img = mpimg.imread(img)
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(ROOT_DIR + f'/static/test_images/{current_time}')
    print(file_path)

    if os.path.exists(file_path):
        os.remove(file_path)

    plt.imsave(file_path, read_img)


# Heatmap functions
def make_gradcam_heatmap(img, model, last_conv_layer_name, classifier_layer_names):
  img_array = np.expand_dims(img, axis=0)
  last_conv_layer = model.get_layer(last_conv_layer_name)
  last_conv_layer_model = tf.keras.Model(model.inputs, last_conv_layer.output)
  
  classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
  x = classifier_input
  for layer_name  in classifier_layer_names:
    x = model.get_layer(layer_name)(x)
  classifier_model = tf.keras.Model(classifier_input, x)

  with tf.GradientTape() as tape:
      # Compute activations of the last conv layer and make the tape watch it
      last_conv_layer_output = last_conv_layer_model(img_array)
      tape.watch(last_conv_layer_output)
      # Compute class predictions
      preds = classifier_model(last_conv_layer_output)
      top_pred_index = tf.argmax(preds[0])
      top_class_channel = preds[:, top_pred_index]

  grads = tape.gradient(top_class_channel, last_conv_layer_output)

  # This is a vector where each entry is the mean intensity of the gradient
  # over a specific feature map channel
  pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

  # We multiply each channel in the feature map array
  # by "how important this channel is" with regard to the top predicted class
  last_conv_layer_output = last_conv_layer_output.numpy()[0]
  pooled_grads = pooled_grads.numpy()
  for i in range(pooled_grads.shape[-1]):
      last_conv_layer_output[:, :, i] *= pooled_grads[i]

  # The channel-wise mean of the resulting feature map
  # is our heatmap of class activation
  heatmap = np.mean(last_conv_layer_output, axis=-1)

  # For visualization purpose, we will also normalize the heatmap between 0 & 1
  heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
  
  return heatmap


# A function to display a heatmap
def show_heatmap(img):
   
  path = 'app/static/models/'
  model_filename = 'skin_cancer_vgg19_main_3.3_sgd.h5'
  new_model = tf.keras.models.load_model(path+model_filename)
  #print(new_model)
 
  img = read_image(img) # load and preprocess img file

  # Relevant layer names
  last_conv_layer_name = 'block5_conv4'
  classifier_layer_names = ['block5_pool', 'global_average_pooling2d', 'dense', 'dropout', 'dense_1']

  heatmap = make_gradcam_heatmap(img, new_model, 
                                 last_conv_layer_name, 
                                 classifier_layer_names)
  heatmap = np.uint8(255 * heatmap)
  img = img * 255.
  # We use jet colormap to colorize heatmap
  jet = cm.get_cmap("jet")

  # We use RGB values of the colormap
  jet_colors = jet(np.arange(256))[:, :3]
  jet_heatmap = jet_colors[heatmap]

  # We create an image with RGB colorized heatmap
  jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
  jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
  jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

  # Superimpose the heatmap on original image
  superimposed_img = jet_heatmap * 0.5 + img
  superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

  # Display Grad CAM
  #plt.subplots(1,1)
  #plt.title(f"Image of {img_class}")
  #plt.imshow(superimposed_img)

  return superimposed_img
