# Skin Cancer Image Classification

This repository contains code for a skin cancer image classification, which aims to classify skin lesion images into different categories of skin cancer. The project utilizes various Python libraries for data manipulation, visualization, machine learning model creation, and evaluation. The dataset used for this project is the **hmnist_28_28_L.csv** file. [Dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000?select=HAM10000_images_part_1)

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Code Overview](#code-overview)
- [Results](#results)
- [Conclusion](#conclusion)
- [Contact](#contact)

## Introduction

Skin cancer is a prevalent form of cancer, and early detection can significantly improve the prognosis for patients. This project focuses on developing a machine learning model to classify skin lesion images into different categories of skin cancer. The model leverages deep learning techniques to learn and recognize patterns from the images.

## Installation

To run this project, you need to have Python and the following libraries installed:

- numpy
- pandas
- pydot
- matplotlib
- seaborn
- sklearn
- tensorflow

You can install these libraries using the following command:

```bash
pip install numpy pandas pydot matplotlib seaborn scikit-learn tensorflow
```

## Usage

1. Clone this repository to your local machine.
2. Place the **hmnist_28_28_L.csv** dataset file in the appropriate location within the project directory.
3. Open and run the Jupyter Notebook or Python script that corresponds to this project.

## Dataset

The dataset used in this project is the **hmnist_28_28_L.csv** file, which contains skin lesion images along with their corresponding labels indicating the type of skin cancer. The dataset needs to be preprocessed before feeding it into the machine learning model.

## Code Overview

The project involves the following steps:

1. **Data Preprocessing:** The dataset is loaded using pandas. Data preprocessing steps include handling missing values, normalizing pixel values, and splitting the data into training and testing sets.

2. **Model Creation:** A deep learning model is built using TensorFlow. The model architecture can be customized as needed, including the choice of layers, activation functions, and optimizer.

3. **Model Training:** The model is trained using the training data. Training parameters such as batch size, number of epochs, and learning rate can be adjusted based on experimentation.

4. **Model Evaluation:** The trained model is evaluated using the testing data. Evaluation metrics such as confusion matrix and classification report are generated to assess the model's performance.

## Results

The project aims to achieve high accuracy and appropriate evaluation metrics in classifying skin lesion images. The results of the model's performance, including accuracy, precision, recall, and F1-score, are presented and analyzed in the project code.

## Conclusion

The skin cancer image classification project demonstrates the application of deep learning techniques to medical image analysis. The model's performance on the test dataset provides insights into its effectiveness in classifying skin cancer types based on lesion images.

## Contact

If you have any questions or suggestions regarding this project, feel free to contact Umair Khan at umairh1819@gmail.com .
