# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 16:10:24 2025

@author: ariji
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dropout, Flatten, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class CIFAR10CNN:
    """
    A comprehensive CNN implementation for CIFAR-10 image classification
    with regularization techniques and detailed analysis capabilities.
    """
    
    def __init__(self):
        self.model = None
        self.history = None
        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                           'dog', 'frog', 'horse', 'ship', 'truck']
        self.x_train, self.y_train = None, None
        self.x_val, self.y_val = None, None
        self.x_test, self.y_test = None, None
        self.mean, self.std = None, None
        
    def load_and_preprocess_data(self, validation_split=0.1):
        """
        Load CIFAR-10 data and perform preprocessing including normalization
        and one-hot encoding.
        """
        print("Loading CIFAR-10 dataset...")
        (x_train_full, y_train_full), (x_test, y_test) = cifar10.load_data()
        
        # Split training data into train and validation sets
        x_train, x_val, y_train, y_val = train_test_split(
            x_train_full, y_train_full, 
            test_size=validation_split, 
            random_state=42, 
            stratify=y_train_full
        )
        
        print(f"Dataset shapes after splitting:")
        print(f"Training: {x_train.shape}, Validation: {x_val.shape}, Test: {x_test.shape}")
        
        # Convert to float32 for better performance
        x_train = x_train.astype('float32')
        x_val = x_val.astype('float32')
        x_test = x_test.astype('float32')
        
        # Normalize using training set statistics
        self.mean = np.mean(x_train, axis=(0, 1, 2, 3))
        self.std = np.std(x_train, axis=(0, 1, 2, 3))
        
        x_train = (x_train - self.mean) / self.std
        x_val = (x_val - self.mean) / self.std
        x_test = (x_test - self.mean) / self.std
        
        # One-hot encode labels
        y_train = to_categorical(y_train, 10)
        y_val = to_categorical(y_val, 10)
        y_test = to_categorical(y_test, 10)
        
        self.x_train, self.y_train = x_train, y_train
        self.x_val, self.y_val = x_val, y_val
        self.x_test, self.y_test = x_test, y_test
        
        print("Data preprocessing completed!")
        return self
    
    def visualize_samples(self, num_samples=25):
        """
        Visualize sample images from the training set
        """
        fig, axes = plt.subplots(5, 5, figsize=(12, 12))
        fig.suptitle('Sample CIFAR-10 Images', fontsize=16)
        
        for i in range(num_samples):
            ax = axes[i // 5, i % 5]
            # Denormalize for visualization
            img = self.x_train[i] * self.std + self.mean
            img = np.clip(img, 0, 255).astype('uint8')
            ax.imshow(img)
            ax.set_title(self.class_names[np.argmax(self.y_train[i])], fontsize=10)
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def create_data_generator(self):
        """
        Create data augmentation generator for training
        """
        return ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1,
            brightness_range=[0.9, 1.1],
            shear_range=0.1,
            fill_mode='nearest'
        )
    
    def build_model(self, weight_decay=0.0001):
        """
        Build the CNN architecture with regularization techniques:
        - L2 regularization
        - Batch Normalization
        - Dropout
        - Progressive filter size increase
        """
        model = Sequential([
            # First Convolutional Block
            Conv2D(32, (3, 3), padding='same', activation='relu', 
                   kernel_regularizer=l2(weight_decay), input_shape=(32, 32, 3)),
            BatchNormalization(),
            Conv2D(32, (3, 3), padding='same', activation='relu', 
                   kernel_regularizer=l2(weight_decay)),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Second Convolutional Block
            Conv2D(64, (3, 3), padding='same', activation='relu', 
                   kernel_regularizer=l2(weight_decay)),
            BatchNormalization(),
            Conv2D(64, (3, 3), padding='same', activation='relu', 
                   kernel_regularizer=l2(weight_decay)),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Third Convolutional Block
            Conv2D(128, (3, 3), padding='same', activation='relu', 
                   kernel_regularizer=l2(weight_decay)),
            BatchNormalization(),
            Conv2D(128, (3, 3), padding='same', activation='relu', 
                   kernel_regularizer=l2(weight_decay)),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Fourth Convolutional Block
            Conv2D(256, (3, 3), padding='same', activation='relu', 
                   kernel_regularizer=l2(weight_decay)),
            BatchNormalization(),
            Conv2D(256, (3, 3), padding='same', activation='relu', 
                   kernel_regularizer=l2(weight_decay)),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Classifier
            Flatten(),
            Dense(512, activation='relu', kernel_regularizer=l2(weight_decay)),
            BatchNormalization(),
            Dropout(0.5),
            Dense(10, activation='softmax')
        ])
        
        self.model = model
        print("Model architecture created!")
        return self
    
    def compile_model(self, learning_rate=0.001):
        """
        Compile the model with optimizer, loss function, and metrics
        """
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        print("Model compiled successfully!")
        return self
    
    def train_model(self, epochs=100, batch_size=32, save_best_model=True):
        """
        Train the model with callbacks for learning rate reduction,
        early stopping, and model checkpointing
        """
        # Create data generator
        datagen = self.create_data_generator()
        
        # Define callbacks
        callbacks = [
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                verbose=1
            )
        ]
        
        if save_best_model:
            callbacks.append(
                ModelCheckpoint(
                    'best_cifar10_model.h5',
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=1
                )
            )
        
        print("Starting model training...")
        self.history = self.model.fit(
            datagen.flow(self.x_train, self.y_train, batch_size=batch_size),
            epochs=epochs,
            validation_data=(self.x_val, self.y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        print("Training completed!")
        return self
    
    def plot_training_history(self):
        """
        Plot training and validation loss and accuracy curves
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        ax1.plot(self.history.history['loss'], label='Training Loss', color='navy')
        ax1.plot(self.history.history['val_loss'], label='Validation Loss', color='darkorange')
        ax1.set_title('Model Loss Evolution')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot(self.history.history['accuracy'], label='Training Accuracy', color='navy')
        ax2.plot(self.history.history['val_accuracy'], label='Validation Accuracy', color='darkorange')
        ax2.set_title('Model Accuracy Evolution')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def evaluate_model(self):
        """
        Evaluate the model on test set and return detailed metrics
        """
        print("Evaluating model on test set...")
        test_loss, test_accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        
        print(f"\nTest Results:")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        # Get predictions for detailed analysis
        y_pred = self.model.predict(self.x_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(self.y_test, axis=1)
        
        return test_accuracy, y_true_classes, y_pred_classes
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """
        Plot confusion matrix for detailed performance analysis
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
    
    def generate_classification_report(self, y_true, y_pred):
        """
        Generate detailed classification report
        """
        report = classification_report(y_true, y_pred, 
                                     target_names=self.class_names,
                                     output_dict=True)
        
        print("\nDetailed Classification Report:")
        print("=" * 60)
        for class_name in self.class_names:
            metrics = report[class_name]
            print(f"{class_name:12} - Precision: {metrics['precision']:.3f}, "
                  f"Recall: {metrics['recall']:.3f}, F1-Score: {metrics['f1-score']:.3f}")
        
        print("=" * 60)
        print(f"Overall Accuracy: {report['accuracy']:.3f}")
        print(f"Macro Avg F1-Score: {report['macro avg']['f1-score']:.3f}")
        print(f"Weighted Avg F1-Score: {report['weighted avg']['f1-score']:.3f}")
    
    def predict_sample_images(self, num_samples=10):
        """
        Make predictions on sample test images and visualize results
        """
        indices = np.random.choice(len(self.x_test), num_samples, replace=False)
        
        fig, axes = plt.subplots(2, 5, figsize=(15, 8))
        fig.suptitle('Sample Predictions vs Ground Truth', fontsize=16)
        
        for i, idx in enumerate(indices):
            ax = axes[i // 5, i % 5]
            
            # Denormalize image for visualization
            img = self.x_test[idx] * self.std + self.mean
            img = np.clip(img, 0, 255).astype('uint8')
            
            # Make prediction
            pred = self.model.predict(np.expand_dims(self.x_test[idx], axis=0))
            pred_class = np.argmax(pred)
            true_class = np.argmax(self.y_test[idx])
            confidence = np.max(pred)
            
            # Display image
            ax.imshow(img)
            color = 'green' if pred_class == true_class else 'red'
            ax.set_title(f'True: {self.class_names[true_class]}\n'
                        f'Pred: {self.class_names[pred_class]} ({confidence:.2f})',
                        color=color, fontsize=10)
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()

def main():
    """
    Main execution function demonstrating the complete case study workflow
    """
    print("CIFAR-10 CNN Case Study")
    print("=" * 50)
    
    # Initialize the CNN model
    cifar_cnn = CIFAR10CNN()
    
    # Load and preprocess data
    cifar_cnn.load_and_preprocess_data(validation_split=0.1)
    
    # Visualize sample images
    print("\nVisualizing sample training images...")
    cifar_cnn.visualize_samples(25)
    
    # Build and compile model
    cifar_cnn.build_model(weight_decay=0.0001)
    cifar_cnn.compile_model(learning_rate=0.001)
    
    # Display model summary
    print("\nModel Architecture:")
    cifar_cnn.model.summary()
    
    # Train the model
    print(f"\nStarting training process...")
    cifar_cnn.train_model(epochs=50, batch_size=32, save_best_model=True)
    
    # Plot training history
    print("\nPlotting training history...")
    cifar_cnn.plot_training_history()
    
    # Evaluate model
    test_acc, y_true, y_pred = cifar_cnn.evaluate_model()
    
    # Generate detailed analysis
    cifar_cnn.plot_confusion_matrix(y_true, y_pred)
    cifar_cnn.generate_classification_report(y_true, y_pred)
    
    # Show sample predictions
    print("\nShowing sample predictions...")
    cifar_cnn.predict_sample_images(10)
    
    print("\n" + "=" * 50)
    print("Case Study Completed Successfully!")
    print(f"Final Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()