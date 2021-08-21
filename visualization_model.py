import segmentation_models
from keras.utils import plot_model
model = segmentation_models.Unet('senet154', input_shape=(None, None, 3), encoder_weights=None)
print(model.summary())
plot_model(model, to_file='senet154.png')