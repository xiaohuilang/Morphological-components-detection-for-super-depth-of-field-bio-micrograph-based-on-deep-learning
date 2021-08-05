# Morphological-components-detection-for-super-depth-of-field-bio-micrograph-based-on-deep-learning
Source code of "Morphological components detection for super-depth-of-field bio-micrograph based on deep learning"

# Requirements

Python 3.7

tensorflow == 2.0.0

keras == 2.3.1

keras_resnet == 0.2.0


# Notes

Modify keras_resnet\models\_2d.py

line 75-77:

 x = keras.layers.Concatenate(axis=0)(inputs)
 
 x = keras.layers.Conv2D(64, (7, 7), strides=(2, 2), use_bias=False, name="conv1", padding="same")(x)
 
 
 



