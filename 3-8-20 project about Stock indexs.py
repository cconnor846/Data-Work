# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 15:28:35 2020

@author: Connor
"""
import pandas as pd
import seaborn as sns
import datetime as dt


import pandas_datareader.data as web
f = web.DataReader('^DJI', 'stooq').reset_index()
f_close = f[["Date", "Close"]]
f_close["Day_Dif"] = f["Close"].pct_change()


f_close["Category"] = pd.cut(f_close["Day_Dif"], bins = [-1, -.03, -.02, -.01, 0, .01, .02, .03, 1], include_lowest=True, labels = ["Greater than -3%", "-3% to -2%", "-2% to -1%", "-1% to 0%", "0% to 1%", "1% to 2%", "2% to 3%", "Greater than 3%"])
f_close["Month"] = f_close["Date"].dt.month
        
    
cat_group = f_close.groupby(["Category"]).count()["Day_Dif"].reset_index()

sns.countplot(x="Category", data = f_close)
sns.boxplot(x="Month", y = "Close", data=f_close)




