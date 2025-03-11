
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
from typing import Union, Optional, List
from faiss import write_index, read_index
import json
class SimilarChartSearcher(object):
    def __init__(self, chart_paths, model_type, index_type, 
                 num_similar_charts: int,
                 #faiss_index_type, 
                 vector_index_save_path="vector_index.faiss",
                 save_charts_path="img_index.json"
                 ):
        self.chart_paths = chart_paths
        self.model_type = model_type
        self.index_type = index_type
        self.num_similar_charts = num_similar_charts
        #self.faiss_index_type = faiss_index_type
        self.vector_index_save_path = vector_index_save_path
        self.save_charts_path = save_charts_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(model_type, device=device)
        
    def get_embeddings(self, chart_paths):
        embeddings = []
        for img_path in chart_paths:
            img = Image.open(img_path)
            image = self.preprocess(img=img).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = model.encode_image(image)
                embeddings.append(image_features)

        return embeddings

    def get_faiss_index(self, index_type: Optional[Union[str, None]]=None):
        if index_type is None:
            index_type = self.index_type
        
        if hasattr(faiss, index_type):
            index = getattr(faiss, index_type)
            if callable(index):
                self.index = index
                return self.index
            else:
                raise ValueError(f"Index type: {index_type} is not a valid callable index in faiss")
            
        else:
            raise ValueError(f"Index type: {index_type} is not a valid index in faiss")
        
    def add_embeddings_to_store(self, #embeddings: Optional[List],
                                vector_index_save_path: Optional[Union[str, None]]=None,
                                save_charts_path: Optional[Union[str, None]]=None
                                ):
        #if not embeddings:
        if hasattr(self, "embeddings"):
            embeddings = self.embeddings
        else:
            print(f"Generating embeddings ...")
            embeddings = self.get_embeddings(chart_paths=self.chart_paths)
        if hasattr(self, "index"):
            index = self.index
        else:
            index = self.get_faiss_index(index_type=self.index_type)
        embedding_dimension = embeddings[0].shape[1]
        index = index(embedding_dimension)
        
        for emb in embeddings:
            emb_arr = np.array(emb.cpu()).astype(np.float32)
            index.add(x=emb_arr)
            if not vector_index_save_path:
                vector_index_save_path = self.vector_index_save_path
            write_index(index, vector_index_save_path)
        if not save_charts_path:
            save_charts_path = self.save_charts_path
        with open(save_charts_path, 'w') as f:
            json.dump(self.chart_paths, f)
            
            
    def get_similar_chart_paths(self, query_chart_path, num_similar_charts: int):     
        if not num_similar_charts:
            num_similar_charts = self.num_similar_charts
        query_embedding = self.get_embeddings(chart_paths=[query_chart_path])
        query_embedding = np.array(query_embedding[0].cpu()).astype(np.float32)
        distance, locs = index.search(x=query_embedding, k=num_similar_charts)
        self.similar_chart_paths = [self.chart_paths[loc] for loc in locs[0]]
        return self.similar_chart_paths



#%%

clip.available_models()
#%%
sim_searcher = SimilarChartSearcher(chart_paths=img_list, 
                                    model_type='ViT-B/32',
                                    index_type="IndexFlatL2",
                                    num_similar_charts=5
                                    )

#%%

sim_searcher.add_embeddings_to_store()


#%%

#sim_searcher.get_embeddings(chart_paths=[img_list[0]])[0].cpu().numpy()
#%%

simimgs = sim_searcher.get_similar_chart_paths(query_chart_path=img_list[0],
                                               num_similar_charts=5
                                               )



#%%


Image.open(img_list[0])


#%%

Image.open(simimgs[1])


#%%
import yfinance as yf
import plotly.express as px

stock = yf.Ticker("IONQ")
start="2025-03-06"
end="2025-03-07"
save_path = f"ionq_{start}_to_{end}.png"
#%%
hist = stock.history(start=start, 
                        end=end,
                    prepost=True,
                    interval='1m', 
                    period='8d',
                    )

#%%

fig = px.line(data_frame=hist, x=hist.index, 
            y="Close", #title=header, 
            template="plotly_dark"
            )


#os.path.join(savedir, f"ionq_{day}.png")
fig.write_image(save_path)


#%%

ionq_simimgs = sim_searcher.get_similar_chart_paths(query_chart_path=save_path,
                                                    num_similar_charts=5
                                                    )


#%%

Image.open(save_path)

#%%

Image.open(ionq_simimgs[2])

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
