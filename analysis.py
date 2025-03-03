
#%%
import torch
import clip
from PIL import Image
import os
from glob import glob
#%%
img_dir = "/home/lin/codebase/stock_app/src/stock_app/ionq_daily_plots"


img_list = glob(f"{img_dir}/*")


#%%
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

#%%

img=Image.open(img_list[0])

#%%
image = preprocess(img=img).unsqueeze(0).to(device)#
text = clip.tokenize(["bull market", "bear market", "buy", "sell"]).to(device)

#%%

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    # Compute similarity between the image and text embeddings
    #similarity = torch.cosine_similarity(image_features, text_features, dim=1)


#%%

def get_embeddings(img_list, model_type):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(model_type, device=device)

    embeddings = []
    for img_path in img_list:
        img = Image.open(img_path)
        image = preprocess(img=img).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image)
            embeddings.append(image_features)

    return embeddings
# %%


embeddings = get_embeddings(img_list=img_list, model_type="ViT-B/32")

# %%
import faiss
import numpy as np

#%%

d = embeddings[0].shape[1]
#%%

index = faiss.IndexFlatL2(d)
# %%
print(index.is_trained)
# %%
for emb in embeddings:
    emb_arr = np.array(emb.cpu()).astype(np.float32)
    index.add(x=emb_arr)
    
#%%


Image.open(img_list[0])


#%%
Image.open(img_list[10])

#%%

img10_emb = get_embeddings(img_list=[img_list[10]], model_type="ViT-B/32")

img10_emb_arr = np.array(img10_emb[0].cpu()).astype(np.float32)

#%%

distance, indices = index.search(img10_emb_arr, k=3)

#%%

class SimilarChartSearcher(object):
    def __init__(self, charts, model_type, index_type, similar_k):
        self.charts = charts
        self.model_type = model_type
        self.index_type = index_type
        self.similar_k = similar_k



# %%
import torch
import numpy as np

# Create a PyTorch tensor
tensor = torch.tensor([1, 2, 3, 4, 5])

# Convert the tensor to a NumPy array
numpy_array = tensor.numpy()

# Display the NumPy array
print(numpy_array)

# %%
import numpy as np
d = 64                           # dimension
nb = 100000                      # database size
nq = 10000                       # nb of queries
np.random.seed(1234)             # make reproducible
xb = np.random.random((nb, d)).astype('float32')
xb[:, 0] += np.arange(nb) / 1000.
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 1000.
# %%
import faiss                   # make faiss available
index = faiss.IndexFlatL2(d)   # build the index
print(index.is_trained)
index.add(xb)                  # add vectors to the index
print(index.ntotal)
# %%
faiss.index_cpu_to_gpus_list()
# %%
