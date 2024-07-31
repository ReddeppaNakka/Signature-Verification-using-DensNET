# Import necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.densenet import DenseNet169, preprocess_input, decode_predictions
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Define paths to your training dataset, validation dataset, and test images
train_data_path = "C:/Users/REDDEPPA/OneDrive/Tài liệu/se_project/data/Train"
validation_data_path = "C:/Users/REDDEPPA/OneDrive/Tài liệu/se_project/data/Validation"
test_image_path = "C:/Users/REDDEPPA/OneDrive/Tài liệu/se_project/data/Test/real2.jpg"

# Load and preprocess training data without validation split
train_datagen = image.ImageDataGenerator(
    preprocessing_function=preprocess_input
)

# Load training data
train_generator = train_datagen.flow_from_directory(
    train_data_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    classes=['forged', 'real'],
    shuffle=True  # Shuffle the data
)

# Display the total count of training image
print("Total Training Images:", train_generator.samples)

# Load and preprocess validation data
validation_datagen = image.ImageDataGenerator(
    preprocessing_function=preprocess_input
)

# Load validation data
validation_generator = validation_datagen.flow_from_directory(
    validation_data_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    classes=['forged', 'real'],
    shuffle=True  # Shuffle the data
)

# Display the total count of validation images
print("Total Validation Images:", validation_generator.samples)

# Build DenseNet169 model
base_model = DenseNet169(weights='imagenet', include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
num_classes = 2
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model (you can replace this with your actual training code)
model.fit(train_generator, epochs=5, validation_data=validation_generator)

# Load and preprocess the test image
test_image = image.load_img(test_image_path, target_size=(224, 224))
test_image_array = image.img_to_array(test_image)
test_image_array = np.expand_dims(test_image_array, axis=0)
test_image_array = preprocess_input(test_image_array)
 
# Make predictions
predictions = model.predict(test_image_array)

# Decode and print predictions
class_labels = ['forged', 'real']
predicted_class = class_labels[np.argmax(predictions)]
accuracy_percentage = np.max(predictions) * 100

# Display the test image along with the predicted class and accuracy
plt.imshow(image.load_img(test_image_path))
plt.title(f'Predicted Class: {predicted_class}\nAccuracy: {accuracy_percentage:.2f}%')
plt.show()
