# Crowd Counting

This is a final course project of CS6140: Machine Learning from Northeastern University. Crowd counting in computer vision is a sophisticated task that involves estimating the number of people in images or videos of crowded scenes. This task is particularly challenging due to the variability of crowd densities, the presence of occlusions where individuals block each other, and the varying scales of people due to camera perspective. Additionally, it must be robust against diverse environmental conditions and varying lighting. Crowd counting has significant applications in public safety, urban planning, retail, and transportation, employing advanced machine learning techniques, especially deep learning models like convolutional neural networks. These models are trained on annotated datasets to accurately estimate crowd sizes, often producing density maps for precise analysis. This technology is crucial for real-time monitoring and management in various public and private sectors.

## Team Members
- Jihao Zhang
- Zeyang Wang

## Dataset
- **Source**: [Crowd Counting](https://www.kaggle.com/datasets/fmena14/crowd-counting/data) Dataset From Kaggle.
- **About**: The dataset is composed by 2000 RGB images of frames in a video (as inputs) and the object counting on every frame, this is the number of pedestrians (object) in the image. The images are 480x640 pixels at 3 channels of the same spot recorded by a webcam in a mall, but it has different number of person on every frame, is a problem of crowd counting.

## Algorithms & Models

### Neural Network Methods

#### About

Neural networks, particularly in the form of deep learning, have revolutionized the field of computer vision. Their ability to learn complex patterns from data makes them exceptionally powerful for visual recognition tasks. 

#### Rationality

- **Performance**: Neural networks have achieved state-of-the-art results in numerous computer vision tasks, such as image classification, object detection, and semantic segmentation. This high level of performance is a strong justification for their widespread adoption.
- **Generalization**: Neural networks have achieved state-of-the-art results in numerous computer vision tasks, such as image classification, object detection, and semantic segmentation. This high level of performance is a strong justification for their widespread adoption.
- **Scalability**: Neural networks scale well with the increasing amount of data and computational resources. The advent of GPUs and TPUs has significantly accelerated neural network training, making them feasible for large-scale applications.

#### Model Selections

##### CSRNet

- Specially designed for crowd counting tasks.
- Utilizes densely connected convolutional layers to better handle highly congested scenes.
- Suitable for estimating the density of crowds in images.

##### General CNN

- A broad category referring to any neural network built with convolutional layers.
- Can have various architectures, ranging from simple few-layered networks to complex multi-layered ones.

##### MobileNet

- Designed for vision applications on mobile and embedded devices, optimizing for speed and memory efficiency.
- Uses depthwise separable convolutions to reduce computation and model size.
- Ideal for real-time vision processing in resource-constrained environments.

##### VGG16

- A deep network with 16 layers, featuring repetitive convolutional and pooling layers.
- Though larger in size, it's simple, efficient, and easy to comprehend.
- Performs exceptionally well in image classification and feature extraction tasks.

## Implementation

You can directly download datasets from Kaggle using the provided [URL](https://www.kaggle.com/datasets/fmena14/crowd-counting/data). Once downloaded, you can fully execute the implementations as outlined in the crowd-counting.ipynb notebook.

## Findings

- **Model Architecture and Complexity**: Simpler models or those with the right inductive biases for the task might outperform more complex models like VGG16, which has a much deeper architecture.
- **Data Suitability**: The type and quality of the dataset play a significant role. For instance, if the dataset doesn't match CSRNet's expectations for crowd scenes, it may not perform well.
- **Training Regime**: The choice of hyperparameters, including learning rates, regularization, and the number of epochs, directly impacts the model's learning trajectory.
- T**ask Alignment**: The nature of the task and how well it aligns with the model's design intent (e.g., crowd counting for CSRNet) is crucial.

## Conclusion
The analysis of the four models—CSRNet, a general CNN, MobileNet, and VGG16—reveals a notable variance in performance on the crowd counting task. The General CNN emerged as the top performer, achieving high accuracy and maintaining low loss, suggesting a well-suited architecture and effective training strategy for the task. CSRNet, while specifically designed for crowd counting, exhibited unexpectedly low accuracy, which may point to issues with data compatibility or the need for further tuning of its hyperparameters and training regimen. MobileNet's performance was robust, but a drop in test accuracy indicates a potential overfitting problem. VGG16, the most complex model, showed significant overfitting, as evidenced by the highest test loss and lowest accuracy figures.

For future efforts, focusing on the data preprocessing, ensuring a diverse and representative dataset, and fine-tuning the models' hyperparameters could potentially improve results. Additionally, experimenting with data augmentation, regularization techniques, and extending the training period may address overfitting issues. Given CSRNet's specialized nature, ensuring task alignment or modifying the network to better suit the data and task could help in harnessing its full potential. Investigating the impact of different loss functions and evaluation metrics more aligned with the task's objectives may also provide insights into improving model performance.
