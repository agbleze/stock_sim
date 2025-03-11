

#%%
import sys
import gc

# Example function to check if an object is still in RAM
def is_object_in_ram(obj_id):
    return any(obj_id == id(x) for x in gc.get_objects())

# Example function to get the size of an object
def get_object_size(obj):
    return sys.getsizeof(obj)

# Create a sample object
sample_list = [1, 2, 3, 4, 5]

# Get the ID of the sample object
sample_list_id = id(sample_list)

# Check if the object is in RAM
is_in_ram = is_object_in_ram(sample_list_id)
print(f"Is the sample_list object in RAM? {is_in_ram}")

# Get the size of the object
size_in_bytes = get_object_size(sample_list)
print(f"Size of the sample_list object in RAM: {size_in_bytes} bytes")

# Delete the object and force garbage collection
del sample_list
gc.collect()

# Check again if the object is in RAM
is_in_ram_after_deletion = is_object_in_ram(sample_list_id)
print(f"Is the sample_list object in RAM after deletion? {is_in_ram_after_deletion}")


# %%
################  clustering with garbage collection triggering based on memory usage ##############

import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from sklearn.decomposition import IncrementalPCA
import os
import gc
import psutil

# Load pre-trained model for extracting image embeddings
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

def extract_embeddings(image_paths):
    embeddings = []
    for img_path in image_paths:
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))
        img = np.expand_dims(img, axis=0)
        img = tf.keras.applications.resnet50.preprocess_input(img)
        embedding = model.predict(img)
        embeddings.append(embedding[0])
    return np.array(embeddings)

def compress_embeddings_incrementally(image_paths, n_components=32, batch_size=100, memory_threshold=75, output_dir="compressed_embeddings"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)

    for i in range(0, len(image_paths), batch_size):
        batch_embeddings = extract_embeddings(image_paths[i:i+batch_size])
        ipca.partial_fit(batch_embeddings)

    for i in range(0, len(image_paths), batch_size):
        batch_embeddings = extract_embeddings(image_paths[i:i+batch_size])
        compressed_batch = ipca.transform(batch_embeddings)
        np.save(os.path.join(output_dir, f"compressed_batch_{i}.npy"), compressed_batch)
        
        # Delete batch embeddings from RAM
        del batch_embeddings
        del compressed_batch

        # Check RAM usage and trigger garbage collection if above threshold
        memory_info = psutil.virtual_memory()
        if memory_info.percent > memory_threshold:
            gc.collect()
    
    # Optionally, clear the PCA model from RAM if it's no longer needed
    del ipca

# Example usage
image_paths = ['image1.jpg', 'image2.jpg', 'image3.jpg']  # Add your image paths here
compress_embeddings_incrementally(image_paths, n_components=32, batch_size=2, memory_threshold=75)


#%%
import os
import gc
import psutil
# %%
memory_info = psutil.virtual_memory()
# %%
memory_info
# %%
