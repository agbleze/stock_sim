

#%%

import yfinance as yf
import plotly.express as px

stock = yf.Ticker("IONQ")
start="2025-03-05"
end="2025-03-06"
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
