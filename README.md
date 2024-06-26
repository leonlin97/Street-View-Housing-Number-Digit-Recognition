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

## Model Development
### Overview of the Models Used
I experimented with various models to predict the house numbers depicted in the images. I started with Artificial Neural Networks (ANNs) and then advanced to Convolutional Neural Networks (CNNs) to leverage their capability to handle image data more effectively. Each model was trained, validated, and evaluated to determine its performance.

### Key Techniques and Algorithms
1. Artificial Neural Networks (ANNs):

    - Used for initial experimentation with digit recognition.
    - Applied dropout and batch normalization to improve performance and prevent overfitting.
    - Activation functions like ReLU and LeakyReLU were used for their ability to introduce non-linearity.

2. Convolutional Neural Networks (CNNs):

    - Employed for their effectiveness in handling image data through convolutional layers that capture spatial hierarchies.
    - Used convolutional layers with varying filter sizes to capture different features of the images.
    - Applied batch normalization to stabilize learning and dropout to prevent overfitting.
    - Max pooling layers were used to reduce the spatial dimensions, thereby reducing the number of parameters and computational cost.

3. Optimization and Loss Functions:

    - Adam optimizer was used for its efficiency and ability to handle sparse gradients.
    - Categorical Crossentropy was the loss function of choice, as it is well-suited for multi-class classification problems.

### 1. Artificial Neural Networks (ANNs)
#### 1st ANN Model:
- This model consists of a simple feedforward neural network with two hidden layers.
- Architecture: An input layer with 1024 nodes, two hidden layers with 64 and 32 nodes respectively, and an output layer with 10 nodes (corresponding to the digits 0-9).
- Activation Functions: ReLU for hidden layers and softmax for the output layer.
- Optimization: Adam optimizer with a learning rate of 0.001.
- Loss Function: Categorical Crossentropy.

```python
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
|![image](https://github.com/leonlin97/Street-View-Housing-Number-Digit-Recognition/assets/142073522/9a6c8e80-fc7a-409b-80d7-4cf6e0344a12) | ![image](https://github.com/leonlin97/Street-View-Housing-Number-Digit-Recognition/assets/142073522/41d61a35-e5d1-4635-a523-ac75db5f21b7) |
| --- | --- |



#### 2nd ANN Model:
- This model expands on the first by adding more layers and incorporating dropout and batch normalization for better performance.
- Architecture: An input layer with 1024 nodes, four hidden layers with 256, 128, 64, and 32 nodes respectively, and an output layer with 10 nodes.
- Additional Techniques: Dropout and Batch Normalization.
- Optimization: Adam optimizer with a learning rate of 0.0005.
- Loss Function: Categorical Crossentropy.

```python
def nn_model_2():
    model = Sequential()
    model.add(Input(shape=(1024,)))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(10, activation='softmax'))
    
    model.compile(optimizer=Adam(learning_rate=0.0005),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
```
|![image](https://github.com/leonlin97/Street-View-Housing-Number-Digit-Recognition/assets/142073522/dde4471d-5ca7-4e4a-84f6-6023f6a01e5d) | ![image](https://github.com/leonlin97/Street-View-Housing-Number-Digit-Recognition/assets/142073522/9c7bff16-1708-4529-ac51-5e50accc55fc) |
| --- | --- |

#### Prediction Accuracy of the 2nd ANN Model
![image](https://github.com/leonlin97/Street-View-Housing-Number-Digit-Recognition/assets/142073522/c0b4945c-619f-4081-be3c-d8125c741ad9)
- The overall accuracy of model 2 on pridicting the testing data is 73.7%.
- The model achieves a high precision and F1-score for most digits, especially for digits 1, 2, 4, and 7.
- Digits like 5 and 8 have lower precision and recall, indicating room for improvement.
- The confusion matrix shows that the model performs well on most digits, with high accuracy for digits like 0, 1, 2, and 4.
- Misclassifications are more frequent for digits 5 and 8, as seen from the off-diagonal values.


### 2. Convolutional Neural Networks (CNNs)
#### 1st CNN Model:
- This model introduces convolutional layers to handle image data more effectively.
- Architecture: Input layer with shape (32, 32, 1), two convolutional layers with 16 and 32 filters, max pooling, flatten layer, and two dense layers.
- Activation Functions: LeakyReLU for hidden layers and softmax for the output layer.
- Optimization: Adam optimizer with a learning rate of 0.001.
- Loss Function: Categorical Crossentropy.

```python
def cnn_model_1():
    model = Sequential()
    model.add(Input(shape=(32,32,1)))
    model.add(Conv2D(16, (3, 3), padding='same'))
    model.add(LeakyReLU(negative_slope=0.1))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(LeakyReLU(negative_slope=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(32))
    model.add(LeakyReLU(negative_slope=0.1))
    model.add(Dense(10, activation='softmax'))
    
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
```
|![image](https://github.com/leonlin97/Street-View-Housing-Number-Digit-Recognition/assets/142073522/44328279-e5f1-424d-a743-f26150ffec5b) | ![image](https://github.com/leonlin97/Street-View-Housing-Number-Digit-Recognition/assets/142073522/39c0d271-f411-4406-9fa9-72679271001d) |
| --- | --- |

#### 2nd CNN Model:
- This model builds upon the first by adding more convolutional layers, batch normalization, and dropout for better performance.
- Architecture: Input layer with shape (32, 32, 1), four convolutional layers with 16, 32, 32, and 64 filters, max pooling after every two convolutional layers, flatten layer, and two dense layers.
- Additional Techniques: Batch Normalization and Dropout.
- Activation Functions: LeakyReLU for hidden layers and softmax for the output layer.
- Optimization: Adam optimizer with a learning rate of 0.001.
- Loss Function: Categorical Crossentropy.

```python
def cnn_model_2():
    model = Sequential()
    model.add(Input(shape=(32,32,1)))
    model.add(Conv2D(16, (3, 3), padding='same'))
    model.add(LeakyReLU(negative_slope=0.1))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(LeakyReLU(negative_slope=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(LeakyReLU(negative_slope=0.1))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(LeakyReLU(negative_slope=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(32))
    model.add(LeakyReLU(negative_slope=0.1))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
```
|![image](https://github.com/leonlin97/Street-View-Housing-Number-Digit-Recognition/assets/142073522/1b72356e-0f0a-422d-8873-8e4c4eb93bf4) | ![image](https://github.com/leonlin97/Street-View-Housing-Number-Digit-Recognition/assets/142073522/3440a9ba-cdf6-47ed-8f01-ea62beed42f4) |
| --- | --- |

#### Prediction Accuracy of the 2nd CNN Model
![image](https://github.com/leonlin97/Street-View-Housing-Number-Digit-Recognition/assets/142073522/a1f5d9b9-5ff2-4285-bd19-3ac51116abc2)
- The chosen CNN model has an accuracy of 91.5%, which is far more better than the chosen ANN model.
- Within the predicted label, "Label 6" seems to have the poorest correct prediction comparing to others, followed by "Label 8". We can provide model with more specific label data to improve the overall performance.
- The confusion matrix shows that the model performs exceptionally well on most digits, with high accuracy for digits 0, 1, 2, 3, 4, 5, 7, and 9.
- Misclassifications are relatively low across all digits, indicating robust performance.
- The model achieves high precision and F1-scores for all digits, particularly excelling in digits like 2, 4, and 7.
- The lowest performance is seen in digit 6, though it still maintains a high level of accuracy.

## Selection of the Final Model: 2nd CNN Model
After thoroughly evaluating the performance of both Artificial Neural Networks (ANNs) and Convolutional Neural Networks (CNNs) on the SVHN dataset, we have chosen CNN Model 2 as the final solution for the following reasons:

- **Superior Accuracy:** CNN Model 2 achieved an overall accuracy of 91.5%, which is significantly higher compared to the ANN models and CNN Model 1. This indicates a stronger ability to generalize from the training data to unseen test data.
- **Enhanced Performance Metrics:** The precision, recall, and F1-scores for CNN Model 2 are consistently high across all digit classes. This uniform performance across classes is crucial for applications requiring high reliability and accuracy.
- **Robust Confusion Matrix:** The confusion matrix for CNN Model 2 shows low misclassification rates, indicating that the model correctly identifies digits with high confidence and minimal errors. This robustness is essential for practical deployment where accuracy is critical.
- **Advanced Architectural Features:** CNN Model 2 includes additional convolutional layers, batch normalization, and dropout layers, which contribute to its superior performance. These architectural enhancements help in capturing complex patterns in the image data and reducing overfitting.
