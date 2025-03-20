

#%%

import yfinance as yf
import plotly.express as px

stock = yf.Ticker("IONQ")
start="2025-03-06"
end="2025-03-07"
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
save_path = f"ionq_{start}_to_{end}.png"

#os.path.join(savedir, f"ionq_{day}.png")
fig.write_image(save_path)


# %%
def get_most_unique_imgs(img_paths, similar_searcher):
    pass



#%%

from faiss import read_index

#%%

vector_index_path = "/home/lin/codebase/stock_sim/vector_index.faiss"

vector_rd = read_index(vector_index_path)

#%%

vector_rd
#%%
# TODO
'experiment with chroma'
# %%
