**PROJECT GOAL:**

The project aims to build an image classification model that can accurately classify images of food into 101 different categories. This is achieved using the Food101 dataset, TensorFlow, and Keras. It leverages the EfficientNetB0 model for its architecture.

**STEPS:**
**1.Data Loading and Preprocessing:**                                                  
  The Food101 dataset is loaded using TensorFlow Datasets (tensorflow_datasets).
  Images are preprocessed by resizing them to a consistent shape (224x224) and converting them to the appropriate data type (tf.float32).
  The dataset is split into training and testing sets.
  Data augmentation is applied during training to improve model generalization.

**2.Model Building:**
  The EfficientNetB0 model is used as the base model for feature extraction.
  Mixed precision is employed to enhance training speed and efficiency.
  A classification head is added to the base model, consisting of a global average pooling layer, a dense layer, and a softmax activation function.

**3.Training:**
  Feature Extraction: The base model's weights are initially frozen, and only the classification head is trained.
  Fine-tuning: After initial training, some of the top layers of the base model are unfrozen and trained along with the classification head. This allows the model to adapt more specifically to the food classification task.
  Checkpoints are saved during training to preserve the best-performing model.
  Learning rate reduction is employed to refine the model's learning process.

**4.Evaluating Results and Visualizing:**
  The trained model is evaluated on the test set using metrics such as accuracy, F1-score, and a confusion matrix.
  Loss curves are plotted to visualize the training and validation performance over epochs.
  A comparison of the feature extraction and fine-tuning stages is provided.

**5.Predicting on a Random Image:**
  A random image from the test set is selected, and the model predicts its class.
  The predicted class is compared to the true class to demonstrate the model's prediction capability.

**KEY HIGHLIGHTS:**
  **EfficientNetB0:** Utilizes a powerful pre-trained model known for its efficiency and accuracy.
  **Mixed Precision: **Employs mixed precision training to accelerate the process.
  **Data Augmentation**: Applies random transformations to training images to improve model robustness.
  F**eature Extraction & Fine-tuning:** Employs a two-stage training approach to optimize performance.
  **Comprehensive Evaluation:** Assesses the model's performance using multiple metrics and visualizations.
