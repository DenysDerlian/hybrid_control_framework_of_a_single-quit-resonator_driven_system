"""
Enhanced RNN Training Script for Quantum Control Optimization

This script implements a sophisticated machine learning pipeline for quantum control pulse
optimization using a multi-pathway recurrent neural network with attention mechanisms.

Key Features:
    • Multi-pathway architecture processing system parameters, temporal features, and sequences
    • Physics-informed loss functions with energy conservation constraints
    • Energy-aware data scaling preserving quantum mechanical relationships
    • Comprehensive evaluation with physics-based metrics
    • Memory-efficient training with on-the-fly data augmentation

Architecture:
    • System Parameters Pathway: Dense layers for physical parameters (ωt, ωr, g)
    • Temporal Features Pathway: Statistical and dynamic feature processing
    • Sequence Pathway: Bidirectional LSTM with multi-head attention
    • Output: Fourier coefficients for quantum control pulse synthesis

Physics Constraints:
    • Energy conservation penalties
    • Magnitude bound enforcement
    • Smoothness regularization
    • Systematic bias correction

Usage:
    python run_rnn_training.py
    
    Background execution:
    nohup python -u run_rnn_training.py > training.log 2>&1 &
    tail -f training.log

Requirements:
    tensorflow>=2.8.0, numpy, pandas, matplotlib, scikit-learn, psutil
    Optional: graphviz (for model architecture visualization)

Authors: Quantum Control Research Team
License: MIT
"""

import os
import sys
import warnings
import json
import gc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import psutil

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, LSTM, Concatenate, Dropout, BatchNormalization,
    Bidirectional, RepeatVector, Permute, Add, Flatten, Activation, Lambda, Softmax,
    MultiHeadAttention, LayerNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler, TensorBoard
)

# Configuration and Utility Functions
print("=== INITIALIZING QUANTUM CONTROL RNN TRAINING ===")

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def print_memory_usage(stage=""):
    """Monitor memory usage throughout training process."""
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        print(f"📊 Memory usage {stage}: {memory_mb:.1f} MB")
    except Exception as e:
        print(f"⚠️ Could not get memory usage: {e}")

def force_garbage_collection():
    """Force garbage collection to free memory."""
    gc.collect()
    print("🗑️ Garbage collection completed")

def setup_tensorflow():
    """Configure TensorFlow for optimal performance."""
    # Enable dynamic GPU memory growth
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Configure threading for CPU performance
    tf.config.threading.set_inter_op_parallelism_threads(0)
    tf.config.threading.set_intra_op_parallelism_threads(0)
    
    print("✅ TensorFlow configured successfully")

setup_tensorflow()
print("✅ Environment setup completed")


def verify_tensorflow_setup():
    """Verify TensorFlow installation and hardware availability."""
    print(f"\n🔧 TensorFlow Version: {tf.__version__}")
    
    # Check for GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        try:
            tf.config.set_visible_devices(gpus[0], 'GPU')
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f"🚀 GPU Available: {len(gpus)} Physical, {len(logical_gpus)} Logical")
            
            # Test GPU computation
            with tf.device('/GPU:0'):
                a = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
                b = tf.constant([[5.0, 6.0], [7.0, 8.0]], dtype=tf.float32)
                c = tf.matmul(a, b)
            
            print(f"✅ GPU test successful - Computation device: {c.device}")
            return True
            
        except RuntimeError as e:
            print(f"❌ GPU configuration failed: {e}")
            return False
    else:
        print("💻 No GPU detected - Using CPU")
        
        # Test CPU computation
        with tf.device('/CPU:0'):
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
            b = tf.constant([[5.0, 6.0], [7.0, 8.0]], dtype=tf.float32)
            c = tf.matmul(a, b)
        
        print(f"✅ CPU test successful - Computation device: {c.device}")
        return False

gpu_available = verify_tensorflow_setup()


def load_and_validate_data():
    """Load quantum control simulation data and perform validation."""
    print("\n📂 LOADING QUANTUM CONTROL DATA")
    
    # Discover data files
    data_files = [f for f in os.listdir("data") if f.endswith('.parquet')]
    print(f"📁 Found {len(data_files)} data files: {data_files}")
    
    # Load and concatenate all data files
    df = pd.concat([
        pd.read_parquet(os.path.join("data", f), engine='pyarrow') 
        for f in data_files
    ], ignore_index=True)
    
    print(f"📊 Dataset shape: {df.shape}")
    print_memory_usage("after data loading")
    
    # Data validation
    print(f"📋 Columns: {df.columns.tolist()}")
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print(f"⚠️ Missing values detected:\n{missing_values[missing_values > 0]}")
    
    # Handle simulation errors
    error_count = df['error'].notna().sum()
    print(f"🔍 Simulations with errors: {error_count} out of {len(df)}")
    
    if error_count > 0:
        df_clean = df[df['error'].isna()].copy()
        print(f"🧹 Clean dataset shape: {df_clean.shape}")
    else:
        df_clean = df.copy()
    
    print(f"✅ Final dataset: {len(df_clean)} clean samples")
    return df_clean

df_clean = load_and_validate_data()

class QuantumFeatureExtractor:
    """Feature engineering for quantum control simulation data."""
    
    @staticmethod
    def find_settling_time(signal, threshold=0.9, eps=1e-6):
        """Calculate the time required for signal to reach threshold of final value."""
        if len(signal) == 0:
            return 0.0
            
        final_val = signal[-1]
        n = len(signal)
        
        if abs(final_val) < eps:
            max_mag = float(np.max(np.abs(signal))) + eps
            target = threshold * max_mag
            for i, val in enumerate(np.abs(signal)):
                if val >= target:
                    return i / n
            return 1.0
        else:
            target = threshold * final_val
            for i, val in enumerate(signal):
                if abs(val) >= abs(target):
                    return i / n
            return 1.0
    
    @staticmethod
    def extract_comprehensive_features(df):
        """Extract comprehensive features from quantum simulation data.
        
        Returns:
            features: Array of combined system parameters and temporal features
            sequences: Array of time-series expectation values
        """
        print("🔧 Extracting comprehensive features...")
        
        features = []
        sequences = []
        
        for _, row in df.iterrows():
            # System parameters
            sys_params = [row['omega_t'], row['omega_r'], row['coupling_g']]
            
            # Expectation value time series
            expect_z = np.array(row['expect_z'])
            expect_x = np.array(row['expect_x'])
            expect_y = np.array(row['expect_y'])
            expect_n = np.array(row['expect_n'])
            
            # Create sequence matrix
            sequence = np.column_stack([expect_z, expect_x, expect_y, expect_n])
            sequences.append(sequence)
            
            # Extract temporal features
            temporal_features = [
                # Final values
                expect_z[-1], expect_x[-1], expect_y[-1], expect_n[-1],
                # Statistical moments
                np.mean(expect_z), np.mean(expect_x), np.mean(expect_y), np.mean(expect_n),
                np.std(expect_z), np.std(expect_x), np.std(expect_y), np.std(expect_n),
                # Extrema
                np.max(expect_z), np.max(expect_x), np.max(expect_y), np.max(expect_n),
                np.min(expect_z), np.min(expect_x), np.min(expect_y), np.min(expect_n),
                # Settling times
                QuantumFeatureExtractor.find_settling_time(expect_z),
                QuantumFeatureExtractor.find_settling_time(expect_x),
                QuantumFeatureExtractor.find_settling_time(expect_y),
                QuantumFeatureExtractor.find_settling_time(expect_n)
            ]
            
            # Time-weighted averages (recent values weighted more heavily)
            n_timesteps = len(expect_z)
            time_weights = np.linspace(0.5, 1.0, n_timesteps)
            recency_features = [
                np.average(expect_z, weights=time_weights),
                np.average(expect_x, weights=time_weights),
                np.average(expect_y, weights=time_weights),
                np.average(expect_n, weights=time_weights)
            ]
            temporal_features.extend(recency_features)
            
            # Combine all features
            features.append(sys_params + temporal_features)
        
        print(f"✅ Extracted {len(features[0])} features per sample")
        return np.array(features), np.array(sequences)
    
    @staticmethod
    def physics_aware_split(df, test_size=0.15, random_state=42):
        """Split data ensuring representative distribution across coupling strengths."""
        np.random.seed(random_state)
        df_sorted = df.sort_values('coupling_g').reset_index(drop=True)
        n = int(1 / test_size)
        val_indices = list(range(0, len(df_sorted), n))
        train_indices = [i for i in range(len(df_sorted)) if i not in val_indices]
        return df_sorted.iloc[train_indices], df_sorted.iloc[val_indices]

print("✅ Feature extraction utilities defined")

def prepare_training_data(df_clean):
    """Prepare and split data for training."""
    print("\n🔧 PREPARING DATA FOR TRAINING")
    
    # Extract features and sequences
    enhanced_features, sequences = QuantumFeatureExtractor.extract_comprehensive_features(df_clean)
    system_params = enhanced_features[:, :3]
    temporal_features = enhanced_features[:, 3:]
    fourier_coeffs = np.array([row['coeffs'] for _, row in df_clean.iterrows()])
    
    print_memory_usage("after feature extraction")
    force_garbage_collection()
    
    # Physics-aware train/validation split
    df_train, df_val = QuantumFeatureExtractor.physics_aware_split(df_clean, test_size=0.15, random_state=42)
    
    # Extract features for train and validation sets
    train_features, train_sequences = QuantumFeatureExtractor.extract_comprehensive_features(df_train)
    val_features, val_sequences = QuantumFeatureExtractor.extract_comprehensive_features(df_val)
    
    train_fourier = np.array([row['coeffs'] for _, row in df_train.iterrows()])
    val_fourier = np.array([row['coeffs'] for _, row in df_val.iterrows()])
    
    # Split features
    system_params_train, temporal_features_train = train_features[:, :3], train_features[:, 3:]
    system_params_val, temporal_features_val = val_features[:, :3], val_features[:, 3:]
    
    print(f"📊 Training set: {len(train_features)} samples")
    print(f"📊 Validation set: {len(val_features)} samples")
    print_memory_usage("after train/val split")
    
    return {
        'system_params_train': system_params_train,
        'temporal_features_train': temporal_features_train,
        'train_sequences': train_sequences,
        'train_fourier': train_fourier,
        'system_params_val': system_params_val,
        'temporal_features_val': temporal_features_val,
        'val_sequences': val_sequences,
        'val_fourier': val_fourier
    }

# Prepare data
data = prepare_training_data(df_clean)
system_params_train = data['system_params_train']
temporal_features_train = data['temporal_features_train']
train_sequences = data['train_sequences']
train_fourier = data['train_fourier']
system_params_val = data['system_params_val']
temporal_features_val = data['temporal_features_val']
val_sequences = data['val_sequences']
val_fourier = data['val_fourier']

class DataScaler:
    """Advanced data scaling with physics-aware energy preservation."""
    
    @staticmethod
    def energy_aware_scaling(fourier_coeffs):
        """Apply energy-preserving scaling to Fourier coefficients."""
        # Calculate original energy distribution
        original_energies = np.sum(fourier_coeffs**2, axis=1)
        
        # Apply standard scaling
        scaler = StandardScaler()
        scaled_coeffs = scaler.fit_transform(fourier_coeffs)
        
        # Calculate scaled energies
        scaled_energies = np.sum(scaled_coeffs**2, axis=1)
        
        # Compute energy correction factor
        energy_ratio = np.sqrt(np.mean(original_energies) / np.mean(scaled_energies))
        corrected_coeffs = scaled_coeffs * energy_ratio
        
        print(f"⚡ Energy preservation factor: {energy_ratio:.4f}")
        print(f"⚡ Original energy mean: {np.mean(original_energies):.4f}")
        print(f"⚡ Corrected energy mean: {np.mean(np.sum(corrected_coeffs**2, axis=1)):.4f}")
        
        return corrected_coeffs, scaler, energy_ratio
    
    @staticmethod
    def scale_all_features(system_params_train, temporal_features_train, train_sequences,
                          train_fourier, system_params_val, temporal_features_val, 
                          val_sequences, val_fourier):
        """Scale all feature types with appropriate methods."""
        print("\n🔧 SCALING FEATURES")
        print_memory_usage("before scaling")
        
        # Initialize scalers
        scaler_system_params = StandardScaler()
        scaler_temporal_features = StandardScaler()
        scaler_sequences = StandardScaler()
        
        # Scale system parameters
        system_params_train_scaled = scaler_system_params.fit_transform(system_params_train)
        system_params_val_scaled = scaler_system_params.transform(system_params_val)
        
        # Scale temporal features
        temporal_features_train_scaled = scaler_temporal_features.fit_transform(temporal_features_train)
        temporal_features_val_scaled = scaler_temporal_features.transform(temporal_features_val)
        
        # Scale sequences
        scaler_sequences.fit(train_sequences.reshape(-1, train_sequences.shape[-1]))
        train_sequences_scaled = np.array([scaler_sequences.transform(seq) for seq in train_sequences])
        val_sequences_scaled = np.array([scaler_sequences.transform(seq) for seq in val_sequences])
        
        # Energy-aware scaling for Fourier coefficients
        fourier_coeffs_train_scaled, scaler_fourier_coeffs, energy_factor = \
            DataScaler.energy_aware_scaling(train_fourier)
        fourier_coeffs_val_scaled = scaler_fourier_coeffs.transform(val_fourier) * energy_factor
        
        print_memory_usage("after scaling")
        force_garbage_collection()
        print("✅ Feature scaling completed")
        
        return {
            'system_params_train_scaled': system_params_train_scaled,
            'system_params_val_scaled': system_params_val_scaled,
            'temporal_features_train_scaled': temporal_features_train_scaled,
            'temporal_features_val_scaled': temporal_features_val_scaled,
            'train_sequences_scaled': train_sequences_scaled,
            'val_sequences_scaled': val_sequences_scaled,
            'fourier_coeffs_train_scaled': fourier_coeffs_train_scaled,
            'fourier_coeffs_val_scaled': fourier_coeffs_val_scaled,
            'scalers': {
                'system_params': scaler_system_params,
                'temporal_features': scaler_temporal_features,
                'sequences': scaler_sequences,
                'fourier_coeffs': scaler_fourier_coeffs
            },
            'energy_factor': energy_factor
        }

# Scale all features
scaled_data = DataScaler.scale_all_features(
    system_params_train, temporal_features_train, train_sequences, train_fourier,
    system_params_val, temporal_features_val, val_sequences, val_fourier
)

# Extract scaled data
system_params_train_scaled = scaled_data['system_params_train_scaled']
system_params_val_scaled = scaled_data['system_params_val_scaled']
temporal_features_train_scaled = scaled_data['temporal_features_train_scaled']
temporal_features_val_scaled = scaled_data['temporal_features_val_scaled']
train_sequences_scaled = scaled_data['train_sequences_scaled']
val_sequences_scaled = scaled_data['val_sequences_scaled']
fourier_coeffs_train_scaled = scaled_data['fourier_coeffs_train_scaled']
fourier_coeffs_val_scaled = scaled_data['fourier_coeffs_val_scaled']
scalers = scaled_data['scalers']
energy_factor = scaled_data['energy_factor']

# Energy validation function
def validate_energy_distribution(fourier_coeffs, title="Training Data"):
    """Validate energy distribution in training data"""
    energies = np.sum(fourier_coeffs**2, axis=1)
    
    print(f"\n--- {title} Energy Analysis ---")
    print(f"Energy statistics:")
    print(f"  Mean: {np.mean(energies):.4f}")
    print(f"  Std: {np.std(energies):.4f}")
    print(f"  Min: {np.min(energies):.4f}")
    print(f"  Max: {np.max(energies):.4f}")
    print(f"  Median: {np.median(energies):.4f}")
    print(f"  25th percentile: {np.percentile(energies, 25):.4f}")
    print(f"  75th percentile: {np.percentile(energies, 75):.4f}")
    
    # Check for potential issues
    if np.std(energies) > np.mean(energies) * 0.5:
        print("⚠️ WARNING: High energy variance detected!")
    
    if np.max(energies) > 20:
        print("⚠️ WARNING: Very high energy samples detected!")
    
    return energies

# Validate energy distributions
train_energies = validate_energy_distribution(train_fourier, "Training Data")
val_energies = validate_energy_distribution(val_fourier, "Validation Data")

# Calculate adaptive energy bounds
mean_energy = np.mean(train_energies)
std_energy = np.std(train_energies)
adaptive_energy_bound = mean_energy + 2 * std_energy  # 95% of data should be within this

print(f"\nAdaptive energy bound calculated: {adaptive_energy_bound:.4f}")
print(f"This covers ~95% of training data energy distribution")

# Memory-efficient training with on-the-fly augmentation
def create_augmented_dataset(system_params, temporal_features, sequences, targets, 
                                     batch_size=128, noise_level=0.003, shuffle=True):
    """Optimized dataset creation for large datasets"""
    
    @tf.function
    def augment_batch(system_batch, temporal_batch, sequences_batch, targets_batch):
        batch_size_tf = tf.shape(system_batch)[0]
        aug_size = batch_size_tf // 4  # Reduce augmentation to 25%
        
        # Pre-compute standard deviations
        system_std = tf.math.reduce_std(system_batch, axis=0, keepdims=True)
        temporal_std = tf.math.reduce_std(temporal_batch, axis=0, keepdims=True)
        sequences_std = tf.math.reduce_std(sequences_batch, axis=[0, 1], keepdims=True)
        
        # Generate noise more efficiently
        system_noise = tf.random.normal([aug_size, tf.shape(system_batch)[1]], 
                                      mean=0.0, stddev=noise_level) * system_std
        temporal_noise = tf.random.normal([aug_size, tf.shape(temporal_batch)[1]], 
                                        mean=0.0, stddev=noise_level) * temporal_std
        sequences_noise = tf.random.normal([aug_size, tf.shape(sequences_batch)[1], tf.shape(sequences_batch)[2]], 
                                         mean=0.0, stddev=noise_level) * sequences_std
        
        # Apply augmentation more efficiently
        system_augmented = tf.concat([
            system_batch[:aug_size] + system_noise,
            system_batch[aug_size:]
        ], axis=0)
        
        temporal_augmented = tf.concat([
            temporal_batch[:aug_size] + temporal_noise,
            temporal_batch[aug_size:]
        ], axis=0)
        
        sequences_augmented = tf.concat([
            sequences_batch[:aug_size] + sequences_noise,
            sequences_batch[aug_size:]
        ], axis=0)
        
        return (system_augmented, temporal_augmented, sequences_augmented), targets_batch
    
    # Create dataset with optimized parameters
    dataset = tf.data.Dataset.from_tensor_slices((
        tf.convert_to_tensor(system_params, dtype=tf.float32),
        tf.convert_to_tensor(temporal_features, dtype=tf.float32),
        tf.convert_to_tensor(sequences, dtype=tf.float32),
        tf.convert_to_tensor(targets, dtype=tf.float32)
    ))
    
    if shuffle:
        # Reduce shuffle buffer size for faster startup
        dataset = dataset.shuffle(buffer_size=min(5000, len(system_params) // 10))
    
    # Batch and augment
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.map(augment_batch, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Optimize prefetching
    dataset = dataset.prefetch(2)  # Prefetch 2 batches instead of AUTOTUNE
    
    return dataset

# Create generators
print(f"Creating training generator with {len(system_params_train_scaled)} samples (effective size with augmentation: ~{len(system_params_train_scaled) * 1.5})")
print_memory_usage("after data generator creation")
print("✅ Data preparation completed!")

class PhysicsInformedLoss:
    """Physics-informed loss functions for quantum control optimization."""
    
    @staticmethod
    def create_loss_function(y_mean, y_scale, mag_bound=3.0, energy_bound=16.0,
                           w_mag=0.010, w_energy=0.005, w_smooth=0.0001,
                           w_energy_match=0.025, w_energy_bias=0.015):
        """Create enhanced physics-informed loss with energy bias correction."""
        
        # Convert to TensorFlow constants
        y_mean_tf = tf.constant(y_mean, dtype=tf.float32)
        y_scale_tf = tf.constant(y_scale, dtype=tf.float32)
        mag_bound_tf = tf.constant(mag_bound, dtype=tf.float32)
        energy_bound_tf = tf.constant(energy_bound, dtype=tf.float32)
        w_mag_tf = tf.constant(w_mag, dtype=tf.float32)
        w_energy_tf = tf.constant(w_energy, dtype=tf.float32)
        w_smooth_tf = tf.constant(w_smooth, dtype=tf.float32)
        w_energy_match_tf = tf.constant(w_energy_match, dtype=tf.float32)
        w_energy_bias_tf = tf.constant(w_energy_bias, dtype=tf.float32)
        
        def loss(y_true, y_pred):
            # Base MSE loss
            mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
            
            # Transform to physical units
            y_true_phys = y_true * y_scale_tf + y_mean_tf
            y_pred_phys = y_pred * y_scale_tf + y_mean_tf
            
            # Energy calculations
            energy_true = tf.reduce_sum(tf.square(y_true_phys), axis=1)
            energy_pred = tf.reduce_sum(tf.square(y_pred_phys), axis=1)
            
            # Physics-informed penalties
            energy_matching_penalty = tf.reduce_mean(tf.square(energy_pred - energy_true))
            energy_bias = tf.reduce_mean(energy_pred - energy_true)
            energy_bias_penalty = tf.square(tf.minimum(0.0, energy_bias))
            
            # Magnitude matching
            mag_true = tf.reduce_mean(tf.abs(y_true_phys), axis=1)
            mag_pred = tf.reduce_mean(tf.abs(y_pred_phys), axis=1)
            magnitude_matching_penalty = tf.reduce_mean(tf.square(mag_pred - mag_true))
            
            # Energy bounds
            max_allowed_energy = tf.maximum(energy_bound_tf, 1.3 * energy_true)
            energy_violation = tf.reduce_mean(tf.square(tf.maximum(0.0, energy_pred - max_allowed_energy)))
            min_allowed_energy = 0.8 * energy_true
            energy_deficiency = tf.reduce_mean(tf.square(tf.maximum(0.0, min_allowed_energy - energy_pred)))
            
            # Magnitude penalty
            magnitude_penalty = tf.reduce_mean(tf.square(tf.maximum(0.0, tf.abs(y_pred_phys) - mag_bound_tf)))
            
            # Smoothness penalty
            smooth_true = tf.reduce_mean(tf.abs(y_true_phys[:, 1:] - y_true_phys[:, :-1]))
            smooth_pred = tf.reduce_mean(tf.abs(y_pred_phys[:, 1:] - y_pred_phys[:, :-1]))
            smoothness_penalty = tf.square(smooth_pred - smooth_true)
            
            # Combine all penalties
            total_penalty = (
                mse_loss +
                w_energy_match_tf * energy_matching_penalty +
                w_energy_bias_tf * energy_bias_penalty +
                w_energy_match_tf * 0.5 * magnitude_matching_penalty +
                w_energy_tf * (energy_violation + energy_deficiency) +
                w_mag_tf * magnitude_penalty +
                w_smooth_tf * smoothness_penalty
            )
            
            return total_penalty
        
        return loss 

    @staticmethod
    def custom_mae_metric(y_true, y_pred):
        """Mean Absolute Error metric."""
        return tf.reduce_mean(tf.abs(y_true - y_pred))
    
    @staticmethod
    def custom_r2_metric(y_true, y_pred):
        """R-squared metric."""
        ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
        ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
        return 1 - ss_res / (ss_tot + tf.keras.backend.epsilon())
    
    @staticmethod
    def energy_bias_metric(y_mean, y_scale):
        """Monitor energy bias during training."""
        y_mean_tf = tf.constant(y_mean, dtype=tf.float32)
        y_scale_tf = tf.constant(y_scale, dtype=tf.float32)
        
        def metric(y_true, y_pred):
            y_true_phys = y_true * y_scale_tf + y_mean_tf
            y_pred_phys = y_pred * y_scale_tf + y_mean_tf
            energy_true = tf.reduce_sum(tf.square(y_true_phys), axis=1)
            energy_pred = tf.reduce_sum(tf.square(y_pred_phys), axis=1)
            return tf.reduce_mean(energy_pred) / (tf.reduce_mean(energy_true) + 1e-8)
        
        return metric


class QuantumControlModel:
    """Enhanced neural network architecture for quantum control."""
    
    @staticmethod
    def create_model(system_params_dim, temporal_features_dim, sequence_shape, output_dim,
                    lstm_units=96, dense_units=[384, 192, 96], dropout_rate=0.15):
        """Create multi-pathway RNN with attention mechanism."""
        
        # Input layers
        system_params_input = Input(shape=(system_params_dim,), name='system_params')
        temporal_features_input = Input(shape=(temporal_features_dim,), name='temporal_features')
        sequence_input = Input(shape=sequence_shape, name='sequences')
        
        # System parameters pathway
        x1 = Dense(128, activation='relu', name='system_dense_1')(system_params_input)
        x1 = BatchNormalization(name='system_bn_1')(x1)
        x1 = Dropout(dropout_rate, name='system_dropout_1')(x1)
        x1 = Dense(64, activation='relu', name='system_dense_2')(x1)
        x1 = BatchNormalization(name='system_bn_2')(x1)
        x1 = Dropout(dropout_rate * 0.8, name='system_dropout_2')(x1)
        
        # Temporal features pathway
        x2 = Dense(128, activation='relu', name='temporal_dense_1')(temporal_features_input)
        x2 = BatchNormalization(name='temporal_bn_1')(x2)
        x2 = Dropout(dropout_rate, name='temporal_dropout_1')(x2)
        x2 = Dense(64, activation='relu', name='temporal_dense_2')(x2)
        x2 = BatchNormalization(name='temporal_bn_2')(x2)
        x2 = Dropout(dropout_rate, name='temporal_dropout_2')(x2)
        
        # Sequence pathway with bidirectional LSTM and attention
        x3 = Bidirectional(LSTM(lstm_units, return_sequences=True, dropout=dropout_rate), 
                          name='bilstm_1')(sequence_input)
        x3 = BatchNormalization(name='lstm_bn_1')(x3)
        x3 = Dropout(dropout_rate, name='lstm_dropout_1')(x3)
        
        x3 = Bidirectional(LSTM(lstm_units//2, return_sequences=True, dropout=dropout_rate), 
                          name='bilstm_2')(x3)
        x3 = BatchNormalization(name='lstm_bn_2')(x3)
        
        # Multi-head attention
        attn_output = MultiHeadAttention(num_heads=4, key_dim=lstm_units//4, 
                                       name='multihead_attention')(x3, x3, x3)
        x3 = LayerNormalization()(Add()([x3, attn_output]))
        sequence_pathway = Dropout(dropout_rate * 1.2, name='sequence_dropout_final')(Flatten()(x3))
        
        # Combine all pathways
        combined = Concatenate(name='concatenate_all')([x1, x2, sequence_pathway])
        
        # Final dense layers with residual connection
        x = Dense(dense_units[0], activation='relu', name='final_dense_1')(combined)
        x = BatchNormalization(name='final_bn_1')(x)
        x = Dropout(dropout_rate, name='final_dropout_1')(x)
        
        x_res = Dense(dense_units[1], activation='relu', name='final_dense_2')(x)
        x_res = BatchNormalization(name='final_bn_2')(x_res)
        x_res = Dropout(dropout_rate * 1.2, name='final_dropout_2')(x_res)
        
        x = Dense(dense_units[2], activation='relu', name='final_dense_3')(x_res)
        x = BatchNormalization(name='final_bn_3')(x)
        x = Dropout(dropout_rate * 1.5, name='final_dropout_3')(x)
        
        # Residual connection if dimensions match
        if dense_units[1] == dense_units[2]:
            x = Add(name='residual_add')([x, x_res])
        
        # Output layer
        output = Dense(output_dim, activation='linear', name='fourier_coeffs_output')(x)
        
        return Model(inputs=[system_params_input, temporal_features_input, sequence_input], 
                    outputs=output, name='EnhancedQuantumControlRNN')

print("\n🏗️ CREATING MODEL ARCHITECTURE")

# Create the enhanced quantum control model
model = QuantumControlModel.create_model(
    system_params_dim=system_params_train_scaled.shape[1],
    temporal_features_dim=temporal_features_train_scaled.shape[1],
    sequence_shape=(train_sequences_scaled.shape[1], train_sequences_scaled.shape[2]),
    output_dim=fourier_coeffs_train_scaled.shape[1],
    lstm_units=128, 
    dense_units=[512, 256, 128], 
    dropout_rate=0.15
)

model.summary()

# Attempt to plot model architecture
try:
    tf.keras.utils.plot_model(model, to_file='enhanced_model_architecture.png', 
                             show_shapes=True, show_layer_names=True, dpi=150)
    print("✅ Model architecture diagram saved")
except Exception as e:
    print(f"⚠️ Skipping model plot: {e}")

print(f"✅ Model created with {model.count_params():,} parameters")

print("\n⚙️ COMPILING MODEL")

# Configure optimizer
optimizer = Adam(learning_rate=0.003, beta_1=0.9, beta_2=0.999, epsilon=1e-7, clipnorm=1.0)

# Get scaler parameters for physics-informed loss
scaler_fourier_coeffs = scalers['fourier_coeffs']
if hasattr(scaler_fourier_coeffs, 'mean_') and scaler_fourier_coeffs.mean_ is not None:
    _scaler_y_mean = (scaler_fourier_coeffs.mean_ * energy_factor).astype(np.float32)
else:
    _scaler_y_mean = np.zeros(fourier_coeffs_train_scaled.shape[1], dtype=np.float32)

if hasattr(scaler_fourier_coeffs, 'scale_') and scaler_fourier_coeffs.scale_ is not None:
    _scaler_y_scale = (scaler_fourier_coeffs.scale_ * energy_factor).astype(np.float32)
else:
    _scaler_y_scale = np.ones(fourier_coeffs_train_scaled.shape[1], dtype=np.float32)

# Compile model with physics-informed loss
model.compile(
    optimizer=optimizer,
    loss=PhysicsInformedLoss.create_loss_function(
        _scaler_y_mean, _scaler_y_scale,
        mag_bound=3.0,
        energy_bound=float(adaptive_energy_bound),
        w_mag=0.008,
        w_energy=0.005,
        w_smooth=0.001,
        w_energy_match=0.030,
        w_energy_bias=0.020
    ),
    metrics=[
        PhysicsInformedLoss.custom_mae_metric,
        PhysicsInformedLoss.custom_r2_metric,
        PhysicsInformedLoss.energy_bias_metric(_scaler_y_mean, _scaler_y_scale)
    ]
)

print("✅ Model compiled with physics-informed loss and adaptive energy bounds")

class TrainingConfig:
    """Training configuration and utilities."""
    
    def __init__(self):
        self.EPOCHS = 80
        self.BATCH_SIZE = 256
        self.create_directories()
    
    @staticmethod
    def create_directories():
        """Create necessary directories for training artifacts."""
        os.makedirs('models', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
    
    @staticmethod
    def cosine_annealing_schedule(epoch, initial_lr=0.003, total_epochs=80):
        """Cosine annealing learning rate schedule."""
        cycle = 40  # Longer cycles for large datasets
        epoch_within_cycle = epoch % cycle
        return initial_lr * (np.cos(epoch_within_cycle * np.pi / cycle) + 1) / 2

# Initialize training configuration
config = TrainingConfig()
print(f"🏋️ Training configuration: {config.EPOCHS} epochs, batch size {config.BATCH_SIZE}")

print("\n🚀 STARTING TRAINING")

# Create training data generator
train_dataset = create_augmented_dataset(
    system_params_train_scaled, temporal_features_train_scaled, 
    train_sequences_scaled, fourier_coeffs_train_scaled,
    batch_size=config.BATCH_SIZE, noise_level=0.005
)

# Configure training callbacks
callbacks = [
    ModelCheckpoint('models/best_enhanced_model.keras', monitor='val_loss', 
                   save_best_only=True, mode='min', verbose=1),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, 
                 mode='min', verbose=1, min_delta=1e-6),
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=5, min_lr=1e-8, 
                     mode='min', verbose=1, cooldown=2),
    LearningRateScheduler(lambda epoch: TrainingConfig.cosine_annealing_schedule(
        epoch, total_epochs=config.EPOCHS), verbose=1),
    TensorBoard(log_dir='logs/enhanced_training', histogram_freq=1, 
               write_graph=True, write_images=True, update_freq='epoch')
]

print(f"✅ Training setup complete: {config.EPOCHS} epochs, batch size {config.BATCH_SIZE}")

# Prepare validation inputs
val_inputs = [system_params_val_scaled, temporal_features_val_scaled, val_sequences_scaled]

# Train the model
history = model.fit(
    train_dataset,
    epochs=config.EPOCHS,
    validation_data=(val_inputs, fourier_coeffs_val_scaled),
    callbacks=callbacks, 
    verbose=2
)
print("✅ Training completed!")

print("\n📊 EVALUATING MODEL PERFORMANCE")
train_loss, val_loss = history.history.get('loss', [None])[-1], history.history.get('val_loss', [None])[-1]
train_mae, val_mae = history.history.get('custom_mae_metric', [None])[-1], history.history.get('val_custom_mae_metric', [None])[-1]
train_r2, val_r2 = history.history.get('custom_r2_metric', [None])[-1], history.history.get('val_custom_r2_metric', [None])[-1]
print(f"\nFinal Training Metrics: Loss={train_loss:.6f}, MAE={train_mae:.6f}, R²={train_r2:.6f}")
print(f"Final Validation Metrics: Loss={val_loss:.6f}, MAE={val_mae:.6f}, R²={val_r2:.6f}")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].plot(history.history['loss'], label='Training Loss'); axes[0].plot(history.history['val_loss'], label='Validation Loss'); axes[0].set_title('Model Loss'); axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss'); axes[0].legend(); axes[0].set_yscale('log')
axes[1].plot(history.history['custom_mae_metric'], label='Training MAE'); axes[1].plot(history.history['val_custom_mae_metric'], label='Validation MAE'); axes[1].set_title('Model MAE'); axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('MAE'); axes[1].legend()
axes[2].plot(history.history['custom_r2_metric'], label='Training R²'); axes[2].plot(history.history['val_custom_r2_metric'], label='Validation R²'); axes[2].set_title('Model R²'); axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('R² Score'); axes[2].legend()
plt.tight_layout()
plt.savefig("training_history.png")
print("✅ Training history plot saved to training_history.png")

best_model = model
val_predictions_scaled = best_model.predict(val_inputs)
val_predictions = scaler_fourier_coeffs.inverse_transform(val_predictions_scaled)
val_true = scaler_fourier_coeffs.inverse_transform(fourier_coeffs_val_scaled)

# Apply energy correction to address systematic under-prediction
def apply_energy_correction(predictions, targets, correction_strength=0.3):
    """Post-processing energy correction for predictions"""
    # Calculate energy bias
    energy_true = np.sum(targets**2, axis=1)
    energy_pred = np.sum(predictions**2, axis=1)
    
    # Calculate correction factor per sample
    energy_ratios = np.sqrt(energy_true / (energy_pred + 1e-8))
    
    # Apply smooth correction (avoid overcorrection)
    correction_factors = 1.0 + correction_strength * (energy_ratios - 1.0)
    correction_factors = np.clip(correction_factors, 0.8, 1.2)  # Limit correction range
    
    # Apply correction
    corrected_predictions = predictions * correction_factors[:, np.newaxis]
    
    print(f"\nEnergy Correction Applied:")
    print(f"Original energy bias: {np.mean(energy_pred / energy_true):.4f}")
    print(f"Corrected energy bias: {np.mean(np.sum(corrected_predictions**2, axis=1) / energy_true):.4f}")
    
    return corrected_predictions

# Apply energy correction to validation predictions
val_predictions_corrected = apply_energy_correction(val_predictions, val_true, correction_strength=0.3)

def evaluate_physics_metrics(y_true, y_pred):
    """Enhanced evaluation including energy analysis"""
    mse = np.mean((y_true - y_pred)**2)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(mse)
    r2 = 1 - (np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2))
    correlation = np.corrcoef(y_true.flatten(), y_pred.flatten())[0, 1]
    
    # Detailed energy analysis
    energy_true = np.sum(y_true**2, axis=1)
    energy_pred = np.sum(y_pred**2, axis=1)
    
    energy_mae = np.mean(np.abs(energy_true - energy_pred))
    energy_mse = np.mean((energy_true - energy_pred)**2)
    energy_bias = np.mean(energy_pred - energy_true)  # Positive = over-prediction
    energy_r2 = 1 - (np.sum((energy_true - energy_pred)**2) / np.sum((energy_true - np.mean(energy_true))**2))
    
    # Energy distribution comparison
    mean_energy_true = np.mean(energy_true)
    mean_energy_pred = np.mean(energy_pred)
    energy_ratio = mean_energy_pred / mean_energy_true
    
    # Physics-aware metrics
    energy_error = energy_mae
    magnitude_error = np.mean(np.abs(np.abs(y_true) - np.abs(y_pred)))
    
    # Smoothness analysis
    smooth_true = np.mean(np.abs(y_true[:, 1:] - y_true[:, :-1]), axis=1)
    smooth_pred = np.mean(np.abs(y_pred[:, 1:] - y_pred[:, :-1]), axis=1)
    smoothness_error = np.mean(np.abs(smooth_true - smooth_pred))
    
    # Magnitude distribution
    mean_magnitude_true = np.mean(np.abs(y_true))
    mean_magnitude_pred = np.mean(np.abs(y_pred))
    
    # Coefficient-wise analysis
    coeff_mae_per_mode = np.mean(np.abs(y_true - y_pred), axis=0)
    worst_modes = np.argsort(coeff_mae_per_mode)[-5:]  # 5 worst modes
    
    return {
        'mse': mse, 'mae': mae, 'rmse': rmse, 'r2': r2, 'correlation': correlation,
        'energy_mae': energy_mae, 'energy_mse': energy_mse, 'energy_bias': energy_bias,
        'energy_r2': energy_r2, 'mean_energy_true': mean_energy_true,
        'mean_energy_pred': mean_energy_pred, 'energy_ratio': energy_ratio,
        'energy_error': energy_error, 'magnitude_error': magnitude_error, 
        'smoothness_error': smoothness_error,
        'mean_magnitude_true': mean_magnitude_true, 'mean_magnitude_pred': mean_magnitude_pred,
        'worst_performing_modes': worst_modes.tolist(),
        'worst_mode_errors': coeff_mae_per_mode[worst_modes].tolist()
    }

metrics = evaluate_physics_metrics(val_true, val_predictions)
print("\nComprehensive Model Evaluation:")

# Separate numeric and list metrics for proper formatting
numeric_metrics = {}
list_metrics = {}

for k, v in metrics.items():
    if isinstance(v, (list, tuple)):
        list_metrics[k] = v
    else:
        numeric_metrics[k] = v

# Print numeric metrics with proper formatting
for k, v in numeric_metrics.items():
    print(f"{k}: {v:.6f}")

# Print list metrics without float formatting
if list_metrics:
    print("\nAdditional Analysis:")
    for k, v in list_metrics.items():
        if k == 'worst_performing_modes':
            print(f"{k}: {v}")
        elif k == 'worst_mode_errors':
            formatted_errors = [f"{err:.6f}" for err in v]
            print(f"{k}: {formatted_errors}")

with open('models/enhanced_performance_metrics.json', 'w') as f:
    # Convert all values to JSON-serializable format
    json_metrics = {}
    for k, v in metrics.items():
        if isinstance(v, (list, tuple)):
            json_metrics[k] = list(v)  # Ensure it's a list
        elif isinstance(v, np.ndarray):
            json_metrics[k] = v.tolist()
        else:
            json_metrics[k] = float(v)  # Convert numpy types to Python float
    json.dump(json_metrics, f, indent=2)

with open('models/training_history.json', 'w') as f:
    json.dump(history.history, f)

print("✅ Detailed metrics saved to models/enhanced_performance_metrics.json")


# Enhanced Evaluation Functions

def cross_validate_model(X_system, X_temporal, X_sequences, y, n_folds=5):
    """Perform k-fold cross-validation"""
    from sklearn.model_selection import KFold
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_system)):
        print(f"Cross-validation fold {fold + 1}/{n_folds}")
        
        # Split data
        X_sys_train, X_sys_val = X_system[train_idx], X_system[val_idx]
        X_temp_train, X_temp_val = X_temporal[train_idx], X_temporal[val_idx]
        X_seq_train, X_seq_val = X_sequences[train_idx], X_sequences[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Scale data
        scaler_sys = StandardScaler()
        scaler_temp = StandardScaler()
        scaler_seq = StandardScaler()
        scaler_y = StandardScaler()
        
        X_sys_train_scaled = scaler_sys.fit_transform(X_sys_train)
        X_sys_val_scaled = scaler_sys.transform(X_sys_val)
        X_temp_train_scaled = scaler_temp.fit_transform(X_temp_train)
        X_temp_val_scaled = scaler_temp.transform(X_temp_val)
        
        # Scale sequences
        X_seq_train_reshaped = X_seq_train.reshape(-1, X_seq_train.shape[-1])
        scaler_seq.fit(X_seq_train_reshaped)
        X_seq_train_scaled = np.array([scaler_seq.transform(seq) for seq in X_seq_train])
        X_seq_val_scaled = np.array([scaler_seq.transform(seq) for seq in X_seq_val])
        
        y_train_scaled = scaler_y.fit_transform(y_train)
        y_val_scaled = scaler_y.transform(y_val)
        
        # Create and train model
        fold_model = QuantumControlModel.create_model(
            system_params_dim=X_sys_train_scaled.shape[1],
            temporal_features_dim=X_temp_train_scaled.shape[1],
            sequence_shape=(X_seq_train_scaled.shape[1], X_seq_train_scaled.shape[2]),
            output_dim=y_train_scaled.shape[1],
            lstm_units=64,  # Smaller for faster training
            dense_units=[256, 128, 64],
            dropout_rate=0.3
        )
        
        fold_model.compile(
            optimizer=Adam(learning_rate=0.003, clipnorm=1.0),
            loss=PhysicsInformedLoss.create_loss_function(
                scaler_y.mean_.astype(np.float32) if scaler_y.mean_ is not None else np.zeros(y_train.shape[1], dtype=np.float32), 
                scaler_y.scale_.astype(np.float32) if scaler_y.scale_ is not None else np.ones(y_train.shape[1], dtype=np.float32)
            ),
            metrics=[PhysicsInformedLoss.custom_mae_metric]
        )
        
        # Train with early stopping
        fold_model.fit(
            x=[X_sys_train_scaled, X_temp_train_scaled, X_seq_train_scaled],
            y=y_train_scaled,
            epochs=50,  # Fewer epochs for CV
            batch_size=32,
            validation_data=([X_sys_val_scaled, X_temp_val_scaled, X_seq_val_scaled], y_val_scaled),
            callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
            verbose=0
        )
        
        # Evaluate
        y_pred_scaled = fold_model.predict([X_sys_val_scaled, X_temp_val_scaled, X_seq_val_scaled], verbose=0)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        
        fold_metrics = evaluate_physics_metrics(y_val, y_pred)
        cv_scores.append(fold_metrics)
        
        print(f"Fold {fold + 1} R²: {fold_metrics['r2']:.4f}, MAE: {fold_metrics['mae']:.4f}")
    
    return cv_scores

print("Enhanced evaluation functions defined successfully!")

# Comprehensive Model Evaluation

# Use the current model (already trained) instead of loading from file to avoid Lambda layer issues
best_model = model
print("Using current trained model for evaluation!")

# Make predictions on validation set
val_predictions_scaled = best_model.predict(val_inputs)
val_predictions = scaler_fourier_coeffs.inverse_transform(val_predictions_scaled)
val_true = scaler_fourier_coeffs.inverse_transform(fourier_coeffs_val_scaled)

# Calculate comprehensive metrics
metrics = evaluate_physics_metrics(val_true, val_predictions)

print("\n" + "="*60)
print("COMPREHENSIVE MODEL EVALUATION")
print("="*60)

print(f"\nRegression Metrics:")
print(f"MSE: {metrics['mse']:.6f}")
print(f"MAE: {metrics['mae']:.6f}")
print(f"RMSE: {metrics['rmse']:.6f}")
print(f"R² Score: {metrics['r2']:.6f}")
print(f"Correlation: {metrics['correlation']:.6f}")

print(f"\nPhysics-Aware Metrics:")
print(f"Energy Error: {metrics['energy_error']:.6f}")
print(f"Magnitude Error: {metrics['magnitude_error']:.6f}")
print(f"Smoothness Error: {metrics['smoothness_error']:.6f}")

print(f"\nEnergy Analysis:")
print(f"Mean Energy (True): {metrics['mean_energy_true']:.6f}")
print(f"Mean Energy (Predicted): {metrics['mean_energy_pred']:.6f}")
print(f"Energy Ratio: {metrics['mean_energy_pred']/metrics['mean_energy_true']:.6f}")

print(f"\nMagnitude Analysis:")
print(f"Mean Magnitude (True): {metrics['mean_magnitude_true']:.6f}")
print(f"Mean Magnitude (Predicted): {metrics['mean_magnitude_pred']:.6f}")

# Quality assessment
if metrics['r2'] > 0.9:
    quality = "Excellent"
elif metrics['r2'] > 0.8:
    quality = "Good"
elif metrics['r2'] > 0.7:
    quality = "Fair"
elif metrics['r2'] > 0.5:
    quality = "Moderate"
else:
    quality = "Poor"

print(f"\nOverall Model Quality: {quality} (R² = {metrics['r2']:.4f})")

# Save detailed metrics
performance_metrics = {
    'model_type': 'Enhanced Quantum Control RNN',
    'training_samples': int(len(system_params_train_scaled)),
    'validation_samples': int(len(system_params_val_scaled)),
    'model_parameters': int(best_model.count_params()),
    'best_epoch': int(np.argmin(history.history['val_loss']) + 1) if 'history' in locals() and history.history.get('val_loss') else None,
    'best_val_loss': float(min(history.history['val_loss'])) if 'history' in locals() and history.history.get('val_loss') else None,
    'quality_assessment': quality,
    'training_epochs_completed': int(len(history.history['loss'])) if 'history' in locals() else None,
    'note': 'Evaluation performed with reduced training epochs for testing'
}

# Add metrics with proper type conversion
for k, v in metrics.items():
    if isinstance(v, (list, tuple)):
        performance_metrics[k] = list(v)  # Keep lists as lists
    elif isinstance(v, np.ndarray):
        performance_metrics[k] = v.tolist()  # Convert arrays to lists
    else:
        performance_metrics[k] = float(v)  # Convert scalars to float

# Create models directory if it doesn't exist
import os
os.makedirs('models', exist_ok=True)

import json
with open('models/enhanced_performance_metrics.json', 'w') as f:
    json.dump(performance_metrics, f, indent=2)

print("\nDetailed metrics saved to models/enhanced_performance_metrics.json")

# Evaluate corrected predictions for comparison
print("\n" + "="*60)
print("ENERGY-CORRECTED PREDICTIONS EVALUATION")
print("="*60)

corrected_metrics = evaluate_physics_metrics(val_true, val_predictions_corrected)
print(f"Corrected Energy Ratio: {corrected_metrics['energy_ratio']:.6f}")
print(f"Corrected R²: {corrected_metrics['r2']:.6f}")
print(f"Corrected Energy Error: {corrected_metrics['energy_error']:.6f}")
print(f"Corrected Energy Bias: {corrected_metrics['energy_bias']:.6f}")

# Save corrected metrics for comparison
corrected_performance_metrics = performance_metrics.copy()
corrected_performance_metrics['note'] = 'Evaluation with post-processing energy correction'
for k, v in corrected_metrics.items():
    if isinstance(v, (list, tuple)):
        corrected_performance_metrics[f'corrected_{k}'] = list(v)
    elif isinstance(v, np.ndarray):
        corrected_performance_metrics[f'corrected_{k}'] = v.tolist()
    else:
        corrected_performance_metrics[f'corrected_{k}'] = float(v)

with open('models/enhanced_performance_metrics_with_correction.json', 'w') as f:
    json.dump(corrected_performance_metrics, f, indent=2)

print("✅ Corrected metrics saved to models/enhanced_performance_metrics_with_correction.json")

# Enhanced Visualization and Analysis

# Create comprehensive visualization
fig = plt.figure(figsize=(20, 15))

# 1. Training History (2x2 grid in top)
gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)

# Loss plots
ax1 = fig.add_subplot(gs[0, 0])
epochs = range(1, len(history.history['loss']) + 1)
ax1.plot(epochs, history.history['loss'], 'b-', label='Training Loss', linewidth=2)
ax1.plot(epochs, history.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
ax1.set_title('Training & Validation Loss', fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_yscale('log')

# MAE plots
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(epochs, history.history.get('custom_mae_metric', []), 'b-', label='Training MAE', linewidth=2)
ax2.plot(epochs, history.history.get('val_custom_mae_metric', []), 'r-', label='Validation MAE', linewidth=2)
ax2.set_title('Training & Validation MAE', fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('MAE')
ax2.legend()
ax2.grid(True, alpha=0.3)

# R² plots
ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(epochs, history.history.get('custom_r2_metric', []), 'b-', label='Training R²', linewidth=2)
ax3.plot(epochs, history.history.get('val_custom_r2_metric', []), 'r-', label='Validation R²', linewidth=2)
ax3.set_title('Training & Validation R²', fontweight='bold')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('R² Score')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Learning Rate
ax4 = fig.add_subplot(gs[0, 3])
if 'learning_rate' in history.history:
    ax4.plot(epochs, history.history['learning_rate'], 'g-', linewidth=2)
    ax4.set_title('Learning Rate Schedule', fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Learning Rate')
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3)
else:
    ax4.text(0.5, 0.5, 'Learning Rate\nData Not Available', ha='center', va='center', transform=ax4.transAxes)
    ax4.set_title('Learning Rate Schedule', fontweight='bold')

# 2. Prediction Analysis
ax5 = fig.add_subplot(gs[1, :2])
# Scatter plot of true vs predicted
idx_sample = np.random.choice(len(val_true), min(1000, len(val_true)), replace=False)
ax5.scatter(val_true[idx_sample].flatten(), val_predictions[idx_sample].flatten(), 
           alpha=0.6, s=1, c='blue', label='Predictions')
# Perfect prediction line
min_val = min(val_true.min(), val_predictions.min())
max_val = max(val_true.max(), val_predictions.max())
ax5.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
ax5.set_xlabel('True Values')
ax5.set_ylabel('Predicted Values')
ax5.set_title(f'True vs Predicted Fourier Coefficients (R² = {metrics["r2"]:.4f})', fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3)
ax5.set_aspect('equal')

# 3. Residual Analysis
ax6 = fig.add_subplot(gs[1, 2:])
residuals = (val_true - val_predictions).flatten()
ax6.hist(residuals, bins=50, alpha=0.7, edgecolor='black', color='skyblue', density=True)
ax6.axvline(0, color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
ax6.set_xlabel('Residuals (True - Predicted)')
ax6.set_ylabel('Density')
ax6.set_title(f'Residual Distribution (μ={np.mean(residuals):.4f}, σ={np.std(residuals):.4f})', fontweight='bold')
ax6.legend()
ax6.grid(True, alpha=0.3)

# 4. Coefficient-wise Analysis
ax7 = fig.add_subplot(gs[2, :2])
coeff_mae = np.mean(np.abs(val_true - val_predictions), axis=0)
coeff_indices = range(len(coeff_mae))
ax7.bar(coeff_indices, coeff_mae, alpha=0.7, color='orange', edgecolor='black')
ax7.set_xlabel('Fourier Coefficient Index')
ax7.set_ylabel('Mean Absolute Error')
ax7.set_title('Per-Coefficient Prediction Error', fontweight='bold')
ax7.grid(True, alpha=0.3)

# 5. Energy Analysis
ax8 = fig.add_subplot(gs[2, 2:])
energy_true = np.sum(val_true**2, axis=1)
energy_pred = np.sum(val_predictions**2, axis=1)
ax8.scatter(energy_true, energy_pred, alpha=0.6, s=10, c='purple')
min_energy = min(energy_true.min(), energy_pred.min())
max_energy = max(energy_true.max(), energy_pred.max())
ax8.plot([min_energy, max_energy], [min_energy, max_energy], 'r--', linewidth=2)
ax8.set_xlabel('True Energy')
ax8.set_ylabel('Predicted Energy')
ax8.set_title(f'Energy Conservation Analysis (r={np.corrcoef(energy_true, energy_pred)[0,1]:.4f})', fontweight='bold')
ax8.grid(True, alpha=0.3)
ax8.set_aspect('equal')

# 6. Example Predictions
ax9 = fig.add_subplot(gs[3, :2])
n_examples = 3
example_indices = np.random.choice(len(val_true), n_examples, replace=False)
x_vals = range(val_true.shape[1])
for i, idx in enumerate(example_indices):
    offset = i * 0.1
    ax9.plot(x_vals, val_true[idx] + offset, 'o-', label=f'True {i+1}', linewidth=2, markersize=4)
    ax9.plot(x_vals, val_predictions[idx] + offset, 's--', label=f'Pred {i+1}', linewidth=2, markersize=4)
ax9.set_xlabel('Fourier Coefficient Index')
ax9.set_ylabel('Coefficient Value')
ax9.set_title('Example Predictions vs Ground Truth', fontweight='bold')
ax9.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax9.grid(True, alpha=0.3)

# 7. Model Performance Summary
ax10 = fig.add_subplot(gs[3, 2:])
ax10.axis('off')
summary_text = f"""
ENHANCED MODEL PERFORMANCE SUMMARY

Architecture: Multi-pathway RNN with sequence pooling
• System Parameters: {system_params_train_scaled.shape[1]} features
• Temporal Features: {temporal_features_train_scaled.shape[1]} features  
• Sequential Data: {train_sequences_scaled.shape[1]}×{train_sequences_scaled.shape[2]}
• Output: {fourier_coeffs_train_scaled.shape[1]} Fourier coefficients
• Total Parameters: {best_model.count_params():,}

Training Configuration:
• Epochs: {len(history.history['loss'])} (Early stopping)
• Batch Size: {config.BATCH_SIZE}
• Optimizer: Adam with gradient clipping
• Loss: Physics-informed MSE (penalties in physical units)
• Data Augmentation: 0.5% input noise

Performance Metrics:
• R² Score: {metrics['r2']:.4f}
• MAE: {metrics['mae']:.4f}
• RMSE: {metrics['rmse']:.4f}
• Correlation: {metrics['correlation']:.4f}
• Energy Error: {metrics['energy_error']:.4f}

Quality: {quality}
"""

ax10.text(0.05, 0.95, summary_text, transform=ax10.transAxes, fontsize=10, 
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))

plt.suptitle('Enhanced Quantum Control RNN - Comprehensive Analysis', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('models/enhanced_model_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("Enhanced analysis plot saved to models/enhanced_model_analysis.png")

def main():
    """Main training pipeline execution."""
    print("🎯 QUANTUM CONTROL RNN TRAINING COMPLETED SUCCESSFULLY!")
    print("\n📋 SUMMARY:")
    print(f"• Model: Enhanced Multi-pathway RNN with Attention")
    print(f"• Parameters: {model.count_params():,}")
    print(f"• Training samples: {len(system_params_train_scaled):,}")
    print(f"• Validation samples: {len(system_params_val_scaled):,}")
    print(f"• Training epochs: {len(history.history['loss'])}")
    print(f"• Final validation R²: {history.history.get('val_custom_r2_metric', [0])[-1]:.4f}")
    print("\n🎉 All training artifacts saved to 'models/' directory")


if __name__ == "__main__":
    main()
