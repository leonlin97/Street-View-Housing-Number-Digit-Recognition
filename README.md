# Street-View-Housing-Number-Digit-Recognition

## Introduction
Recognizing objects in natural scenes using deep learning techniques is a captivating and challenging task with numerous real-world applications. This project focuses on the recognition of house numbers from street view images, a problem that has practical significance in improving geographic information systems and enhancing location-based services.

## Objective
My objective is to develop models that can accurately recognize the digits in these images. I will explore and compare the performance of Artificial Neural Networks (ANNs) and Convolutional Neural Networks (CNNs) to determine the most effective approach for this task. This project not only demonstrates the application of deep learning techniques in a practical context but also showcases the potential improvements in map quality and location accuracy that can be achieved through automated digit recognition.

## Dataset Information
In this project, I utilize the Street View House Numbers (SVHN) dataset, which comprises over 600,000 labeled digit images extracted from street-level photographs. This dataset is a benchmark in the field of image recognition and has been instrumental in advancements made by organizations such as Google for automatic address transcription, which aids in accurately pinpointing building locations.

## Data Preparation
### Visualiza some images in the dataset
![image](https://github.com/leonlin97/Street-View-Housing-Number-Digit-Recognition/assets/142073522/130988b0-7d64-4195-ac19-543965b28b9e)

### Data Preprocessing Steps
- Train(80%) / Test(20%) split
- Reshape (32x32). 2D for ANN models and 4D for CNN models
- Normalizattion (divided by 255.0)
- One-hot encode
- Flatten images if needed

## Model Development (ANN vs CVV)
### Overview of the Models Used
I experimented with various models to predict the house numbers depicted in the images. I started with Artificial Neural Networks (ANNs) and then advanced to Convolutional Neural Networks (CNNs) to leverage their capability to handle image data more effectively. Each model was trained, validated, and evaluated to determine its performance.

### 1. Artificial Neural Networks (ANNs)
#### 1st ANN Model:
- This model consists of a simple feedforward neural network with two hidden layers.
- Architecture: An input layer with 1024 nodes, two hidden layers with 64 and 32 nodes respectively, and an output layer with 10 nodes (corresponding to the digits 0-9).
- Activation Functions: ReLU for hidden layers and softmax for the output layer.
- Optimization: Adam optimizer with a learning rate of 0.001.
- Loss Function: Categorical Crossentropy.

```
def nn_model_1():
    model = Sequential()
    model.add(Input(shape=(1024,)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
```




