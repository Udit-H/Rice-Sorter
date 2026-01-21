"""
Rice Grain Classification Model Training Script
4 Classes: black, brown, chalky, yellow
Based on EfficientNet-B0 Transfer Learning
"""

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

tf.random.set_seed(42)
np.random.seed(42)

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {len(tf.config.list_physical_devices('GPU'))} GPU(s)")

# ==================== CONFIGURATION ====================
BASE_PATH = './rice'  # Local path to rice folder
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE_INITIAL = 0.001
LEARNING_RATE_FINETUNE = 0.0001

# These will be set based on discovered folders
CLASS_NAMES = ['black', 'brown', 'chalky', 'yellow']  # Alphabetical order!
NUM_CLASSES = 4
LABEL_MAP = {0: 'black', 1: 'brown', 2: 'chalky', 3: 'yellow'}

# ==================== DATASET CREATION ====================
def create_datasets(base_path, validation_split=0.2, test_split=0.2, seed=42):
    print("Creating datasets from directory structure...")
    
    # Training + Validation set
    train_val_ds = tf.keras.utils.image_dataset_from_directory(
        base_path,
        validation_split=test_split,
        subset="training",
        seed=seed,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        shuffle=True
    )
    
    # Test set
    test_ds = tf.keras.utils.image_dataset_from_directory(
        base_path,
        validation_split=test_split,
        subset="validation",
        seed=seed,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        shuffle=True
    )
    
    class_names = train_val_ds.class_names
    print(f"Detected classes (alphabetical): {class_names}")
    
    # Split train_val into train and val
    val_batches = tf.data.experimental.cardinality(train_val_ds).numpy()
    val_size = max(1, int(validation_split * val_batches))
    
    val_ds = train_val_ds.take(val_size)
    train_ds = train_val_ds.skip(val_size)
    
    return train_ds, val_ds, test_ds, class_names

# ==================== PREPROCESSING ====================
def preprocess_for_efficientnet(image, label):
    """Just cast to float32 - NO normalization (matches inference code)"""
    image = tf.cast(image, tf.float32)
    return image, label

def augment_training_data(image, label):
    """Data augmentation for training"""
    if tf.random.uniform([]) > 0.5:
        image = tf.image.random_flip_left_right(image)
    if tf.random.uniform([]) > 0.5:
        image = tf.image.random_flip_up_down(image)
    if tf.random.uniform([]) > 0.6:
        k = tf.random.uniform([], 0, 4, dtype=tf.int32)
        image = tf.image.rot90(image, k=k)
    if tf.random.uniform([]) > 0.7:
        image = tf.image.random_brightness(image, max_delta=0.1 * 255)
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    image = tf.clip_by_value(image, 0.0, 255.0)
    return image, label

def optimize_dataset(ds, shuffle=False, augment=False):
    ds = ds.map(preprocess_for_efficientnet, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        ds = ds.map(augment_training_data, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=1000, reshuffle_each_iteration=True)
    ds = ds.cache()
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

# ==================== MODEL CREATION ====================
def create_efficientnet_classifier(num_classes):
    print("Building EfficientNet-B0 transfer learning model...")
    
    base_model = keras.applications.EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(*IMAGE_SIZE, 3)
    )
    base_model.trainable = False
    
    inputs = keras.Input(shape=(*IMAGE_SIZE, 3), name='input_layer')
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    x = layers.BatchNormalization(name='bn_features')(x)
    x = layers.Dropout(0.2, name='dropout_features')(x)
    x = layers.Dense(512, activation='relu', name='dense_512')(x)
    x = layers.BatchNormalization(name='bn_dense_512')(x)
    x = layers.Dropout(0.3, name='dropout_512')(x)
    x = layers.Dense(256, activation='relu', name='dense_256')(x)
    x = layers.BatchNormalization(name='bn_dense_256')(x)
    x = layers.Dropout(0.2, name='dropout_256')(x)
    
    outputs = layers.Dense(
        num_classes,
        activation='softmax',
        name='predictions',
        kernel_regularizer=keras.regularizers.l2(0.001)
    )(x)
    
    model = keras.Model(inputs, outputs, name='rice_efficientnet_4class')
    print(f"Model created with {num_classes} output classes")
    
    return model, base_model

# ==================== TRAINING ====================
def train_model():
    print("\n" + "="*60)
    print("RICE GRAIN CLASSIFICATION - 4 CLASS MODEL TRAINING")
    print("="*60)
    
    # Create datasets
    train_ds, val_ds, test_ds, detected_classes = create_datasets(BASE_PATH)
    
    # Verify classes
    print(f"\nClass mapping:")
    for i, name in enumerate(detected_classes):
        print(f"  {i}: {name}")
    
    # Preprocess datasets
    train_ds_processed = optimize_dataset(train_ds, shuffle=True, augment=True)
    val_ds_processed = optimize_dataset(val_ds, shuffle=False, augment=False)
    test_ds_processed = optimize_dataset(test_ds, shuffle=False, augment=False)
    
    # Calculate class weights for imbalanced data
    print("\nCalculating class distribution...")
    class_counts = np.zeros(len(detected_classes), dtype=int)
    for batch_images, batch_labels in train_ds_processed:
        for label in batch_labels:
            class_counts[np.argmax(label.numpy())] += 1
    
    total_samples = np.sum(class_counts)
    class_weights = {i: total_samples / (len(detected_classes) * count) if count > 0 else 1.0 
                     for i, count in enumerate(class_counts)}
    
    print(f"Training samples per class:")
    for i, name in enumerate(detected_classes):
        print(f"  {name}: {class_counts[i]} (weight: {class_weights[i]:.2f})")
    
    # Create model
    model, base_model = create_efficientnet_classifier(len(detected_classes))
    
    # Phase 1: Train with frozen base
    print("\n" + "="*60)
    print("PHASE 1: Training with Frozen Base Model")
    print("="*60)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE_INITIAL),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )
    
    callbacks_phase1 = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7),
        keras.callbacks.ModelCheckpoint('rice_model_phase1.keras', monitor='val_accuracy', 
                                         save_best_only=True, verbose=1)
    ]
    
    history1 = model.fit(
        train_ds_processed,
        validation_data=val_ds_processed,
        epochs=EPOCHS//2,
        callbacks=callbacks_phase1,
        class_weight=class_weights,
        verbose=1
    )
    
    print(f"\nPhase 1 Complete!")
    print(f"  Best Val Accuracy: {max(history1.history['val_accuracy']):.4f}")
    
    # Phase 2: Fine-tune
    print("\n" + "="*60)
    print("PHASE 2: Fine-tuning")
    print("="*60)
    
    base_model.trainable = True
    for layer in base_model.layers[:-60]:
        layer.trainable = False
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE_FINETUNE),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )
    
    callbacks_phase2 = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=6, min_lr=1e-8),
        keras.callbacks.ModelCheckpoint('rice_model_phase2.keras', monitor='val_accuracy',
                                         save_best_only=True, verbose=1)
    ]
    
    history2 = model.fit(
        train_ds_processed,
        validation_data=val_ds_processed,
        epochs=EPOCHS//2,
        callbacks=callbacks_phase2,
        class_weight=class_weights,
        verbose=1
    )
    
    print(f"\nPhase 2 Complete!")
    print(f"  Best Val Accuracy: {max(history2.history['val_accuracy']):.4f}")
    
    # Evaluate on test set
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    test_results = model.evaluate(test_ds_processed, verbose=1)
    print(f"\nTest Results:")
    print(f"  Loss: {test_results[0]:.4f}")
    print(f"  Accuracy: {test_results[1]:.4f}")
    
    # Save final model
    final_model_path = 'rice_4class_model.keras'
    model.save(final_model_path)
    print(f"\nModel saved to: {final_model_path}")
    
    # Save class mapping
    with open('rice_model_config.txt', 'w') as f:
        f.write("Rice Grain Classification Model - 4 Classes\n")
        f.write("="*50 + "\n\n")
        f.write("Class Mapping (IMPORTANT - alphabetical order!):\n")
        for i, name in enumerate(detected_classes):
            f.write(f"  {i}: {name}\n")
        f.write(f"\nTest Accuracy: {test_results[1]:.4f}\n")
        f.write(f"Input Size: {IMAGE_SIZE}\n")
        f.write("\nUsage:\n")
        f.write("  1. Resize image to 224x224\n")
        f.write("  2. Cast to float32 (no normalization!)\n")
        f.write("  3. model.predict(image)\n")
        f.write("  4. np.argmax(prediction) -> class index\n")
    
    print(f"Configuration saved to: rice_model_config.txt")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nCopy 'rice_4class_model.keras' and rename to 'efficientnet_rice_final_inference.keras'")
    print(f"Then update LABEL_MAP in process_image_updated.py to:")
    print(f"  LABEL_MAP = {{0: 'black', 1: 'brown', 2: 'chalky', 3: 'yellow'}}")
    
    return model, detected_classes

if __name__ == "__main__":
    train_model()
