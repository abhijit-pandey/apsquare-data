#!/usr/bin/env python
# coding: utf-8

# In[78]:


import yfinance as yf
import datetime
import pandas as pd
import plotly.graph_objects as go
import math
import plotly.io as pio
import numpy as np
import pickle
import os
TODAY = datetime.datetime.today()
# TODAY = (TODAY-datetime.timedelta(days=2))
print(TODAY)


# In[79]:
def get_last_thursday(TODAY = datetime.datetime.today().date()):
    if isinstance(TODAY,str):
        TODAY = datetime.datetime.strptime(TODAY, '%Y-%m-%d').date()
    days_to_THU = 10 - TODAY.weekday()
    point_date = TODAY + datetime.timedelta(days=days_to_THU)
    while(True):
        if (point_date+datetime.timedelta(days=7)).month==TODAY.month:
            point_date = point_date+datetime.timedelta(days=7)
        else:
            Last_Thursday = point_date
            break
    return Last_Thursday

def get_nifty50_stocks():
    #PROCESS AND LOAD NIFTY 50 DATA
    nif50 = pd.read_html('https://www.moneyseth.com/blogs/Nifty-50-Stock-List-and-its-Weightage-in-Index')[1]
    new_header = nif50.iloc[0] #grab the first row for the header
    nif50 = nif50[1:] #take the data less the header row
    nif50.columns = new_header #set the header row as the df header
    nif50.drop(columns='Sl No', axis=1, inplace=True)
    return nif50

def get_data_date(SYM, START=None, END=None):
    if SYM[0]=='^' or '=' in SYM:
        NAME = f'{SYM}'
    else:
        NAME = f'{SYM}.NS'
    
    df = yf.download(NAME,
                 start=START,
                 end=END,
                 progress=False)
    df.reset_index(inplace=True)

    #RESISTANCE/SUPPORT FINDING
    df['Change'] = (df.Close - df.Close.shift(1))/df.Close
    df['Topline'] = df.apply(lambda x : x.Open if x.Open>x.Close else x.Close, axis=1)
    df['Bottomline'] = df.apply(lambda x : x.Open if x.Open<x.Close else x.Close, axis=1)

    #Create 200 period Moving Average
    df['Moving_Average_200'] = df.Close.rolling(window=200).mean().shift(1)
    df['Moving_Average_50'] = df.Close.rolling(window=50).mean().shift(1)
    df['Moving_Average_20'] = df.Close.rolling(window=20).mean().shift(1)
    
    df['ATR1'] = df.High-df.Low
    df['ATR2'] = abs(df.High-df.Close.shift(-1))
    df['ATR3'] = abs(df.Low-df.Close.shift(-1))
    
    
    df['ATR'] = df[['ATR1','ATR2', 'ATR3']].max(axis=1) # USE to get level area range
    df['200_Avg_ATR'] = df.ATR.rolling(window=200).mean()
    df.drop(columns=['ATR1','ATR2', 'ATR3'], inplace=True)
    
    return df

def get_price(SYM, DATE=None):
    if DATE is None:
        DATE = datetime.datetime.today().date()-datetime.timedelta(days=1)
    else:
        DATE = datetime.datetime.strptime(DATE,'%Y-%m-%d').date()
    df = get_data_date('INFY',str(DATE),str(DATE+datetime.timedelta(days=1)))
    return df.iloc[-1].Close

WIN_RATE = pd.read_csv('WinRate.csv')
def get_win_rate(df):
    global WIN_RATE
    if len(df)==0:
        df['Up_Probability'] = None
        return df
    df['Up_Probability'] = df.apply(lambda x :  WIN_RATE[(WIN_RATE.Trend==x.Trend)&(WIN_RATE.Trigger_Type==x.Trigger_Type)].Up_Probability.mean(),axis=1)
    return df

# In[80]:


def detect_doji(o,h,l,c):
    body = abs(o-c)
    rng  = abs(h-l)
    
    if body/rng <=0.25:
        return True
    return False

def detect_bullish_engulfing(po,pc,o,c):
    if po>pc and o<c and c>po and o < pc:
        return True
    return False

def detect_bearish_engulfing(po,pc,o,c):
    if po<pc and o>c and c<po:
        return True
    return False


# In[93]:


def get_rsi(df, cols=None, n=14):
    df['change'] = (df.Close - df.Close.shift(1))/df.Close.shift(1)
    df['up_change'] = df.change.apply(lambda x: x if x>0 else 0)
    df['up_mean_change'] = df.up_change.rolling(n).mean()
    df['down_change'] = df.change.apply(lambda x: x if x<0 else 0)
    df['down_mean_change'] = df.down_change.rolling(n).mean()
    df[f'rsi_{n}'] = 100-(100/(1+(df.up_mean_change/(-df.down_mean_change))))
    if cols is None:
        df.drop(columns=['up_change','up_mean_change','down_change','down_mean_change'], inplace=True)
        return df
    return df[cols]

def get_resistance_level(df):
    df['Topline-1'] = df.Topline - df.Topline.shift(-1)
    df['Topline-2'] = df.Topline - df.Topline.shift(-2)
    df['Topline-3'] = df.Topline - df.Topline.shift(-3)
    df['Topline-4'] = df.Topline - df.Topline.shift(-4)
    df['Topline+1'] = df.Topline - df.Topline.shift(1)
    df['Topline+2'] = df.Topline - df.Topline.shift(2)
    df['Topline+3'] = df.Topline - df.Topline.shift(3)
    df['Topline+4'] = df.Topline - df.Topline.shift(4)
    
    df['is_Resistance'] = (df['Topline-1']>=0)&(df['Topline-2']>=0)&(df['Topline+1']>=0)&(df['Topline+2']>=0)
    df['is_Resistance_strong'] = (df['is_Resistance'])&(df['Topline-3']>=0)&(df['Topline-4']>=0)&(df['Topline+3']>=0)&(df['Topline+4']>=0)
    df.drop(columns= ['Topline-1','Topline-2','Topline-3','Topline-4','Topline+1','Topline+2','Topline+3','Topline+4'], inplace=True)
    return df

def get_support_level(df):
    df['Bottomline-1'] = df.Bottomline - df.Bottomline.shift(-1)
    df['Bottomline-2'] = df.Bottomline - df.Bottomline.shift(-2)
    df['Bottomline-3'] = df.Bottomline - df.Bottomline.shift(-3)
    df['Bottomline-4'] = df.Bottomline - df.Bottomline.shift(-4)
    df['Bottomline+1'] = df.Bottomline - df.Bottomline.shift(1)
    df['Bottomline+2'] = df.Bottomline - df.Bottomline.shift(2)
    df['Bottomline+3'] = df.Bottomline - df.Bottomline.shift(3)
    df['Bottomline+4'] = df.Bottomline - df.Bottomline.shift(4)
    
    df['is_Support'] = (df['Bottomline-1']<=0)&(df['Bottomline-2']<=0)&(df['Bottomline+1']<=0)&(df['Bottomline+2']<=0)
    df['is_Support_strong'] = (df['is_Support'])&(df['Bottomline-3']<=0)&(df['Bottomline-4']<=0)&(df['Bottomline+3']<=0)&(df['Bottomline+4']<=0)
    df.drop(columns= ['Bottomline-1','Bottomline-2','Bottomline-3','Bottomline-4','Bottomline+1','Bottomline+2','Bottomline+3','Bottomline+4'], inplace=True)
    return df

#THIS NEEDS TO BE REFINED AND TESTED
def get_trend(df): 
    def get_strength(cng):
        cng = abs(cng)
        if cng < 0.12 :
            return 'Weak '
        elif cng <= 0.3:
            return 'Healthy '
        elif cng > 0.3:
            return 'Strong '
        else:
            return '-'
    df['cng'] = (df.Close - df.Close.shift(250))/df.Close.shift(250)
    df['Trend'] = df['cng'].apply(lambda x : get_strength(x)+('Up' if x>0 else ('Down' if x<0 else '-')))
    df.drop(columns=['cng'], inplace=True)
    return df

def which_level_support_probable(df):
    cp = df.iloc[-1].Close
    try:
        support_1 = df[(df.is_Support)&(df.Bottomline<cp)].iloc[-1]['Bottomline']
    except:
        return ('Not Sure', None)
    try:
        support_2 = df[(df.is_Support)&(df.Bottomline<cp)].iloc[-2]['Bottomline']
    except:
        return ('Not Sure', support_1)
    
    
    if abs(support_1-support_2) <= df[df.is_Support].iloc[-1]['ATR']:
        return ('Support_Level', support_1)
    if (abs(support_1-df[df.is_Support].iloc[-1]['Moving_Average_20']) <= df[df.is_Support].iloc[-1]['ATR'] and
       abs(support_2-df[df.is_Support].iloc[-2]['Moving_Average_20']) <= df[df.is_Support].iloc[-2]['ATR'] and
       df.iloc[-1]['Moving_Average_20']<cp):
        return ('Moving_Average_20', df.iloc[-1]['Moving_Average_20'])
    
    if (abs(support_1-df[df.is_Support].iloc[-1]['Moving_Average_50']) <= df[df.is_Support].iloc[-1]['ATR'] and
       abs(support_2-df[df.is_Support].iloc[-2]['Moving_Average_50']) <= df[df.is_Support].iloc[-2]['ATR'] and
       df.iloc[-1]['Moving_Average_50']<cp):
        return ('Moving_Average_50', df.iloc[-1]['Moving_Average_50'])
    
    if (abs(support_1-df[df.is_Support].iloc[-1]['Moving_Average_200']) <= df[df.is_Support].iloc[-1]['ATR'] and
       abs(support_2-df[df.is_Support].iloc[-2]['Moving_Average_200']) <= df[df.is_Support].iloc[-2]['ATR'] and
       df.iloc[-1]['Moving_Average_200']<cp):
        return ('Moving_Average_200', df.iloc[-1]['Moving_Average_200'])
    
    return ('Not Sure', support_1)

def which_level_resistance_probable(df):
    cp = df.iloc[-1].Close
    try:
        resistance_1 = df[(df.is_Resistance)&(cp<df.Topline)].iloc[-1]['Topline']
    except:
        return ('Not Sure', None)
    try:
        resistance_2 = df[(df.is_Resistance)&(cp<df.Topline)].iloc[-2]['Topline']
    except:
        return ('Not Sure', resistance_1)
    
    
    if abs(resistance_1-resistance_2) <= df[df.is_Resistance].iloc[-1]['ATR']:
        return ('Resistance_Level', resistance_1)
    if (abs(resistance_1-df[df.is_Resistance].iloc[-1]['Moving_Average_20']) <= df[df.is_Resistance].iloc[-1]['ATR'] and
       abs(resistance_2-df[df.is_Resistance].iloc[-2]['Moving_Average_20']) <= df[df.is_Resistance].iloc[-2]['ATR'] and 
       df.iloc[-1]['Moving_Average_20']>cp):
        return ('Moving_Average_20', df.iloc[-1]['Moving_Average_20'])
    
    if (abs(resistance_1-df[df.is_Resistance].iloc[-1]['Moving_Average_50']) <= df[df.is_Resistance].iloc[-1]['ATR'] and
       abs(resistance_2-df[df.is_Resistance].iloc[-2]['Moving_Average_50']) <= df[df.is_Resistance].iloc[-2]['ATR'] and
       df.iloc[-1]['Moving_Average_50']>cp):
        return ('Moving_Average_50', df.iloc[-1]['Moving_Average_50'])
    
    if (abs(resistance_1-df[df.is_Resistance].iloc[-1]['Moving_Average_200']) <= df[df.is_Resistance].iloc[-1]['ATR'] and
       abs(resistance_2-df[df.is_Resistance].iloc[-2]['Moving_Average_200']) <= df[df.is_Resistance].iloc[-2]['ATR'] and
       df.iloc[-1]['Moving_Average_200']>cp):
        return ('Moving_Average_200', df.iloc[-1]['Moving_Average_200'])
    
    return ('Not Sure', resistance_1)


# In[103]:


def detect_area_of_value(cp, ATR, sup_level, res_level, trend, sup_type='Not Sure', res_type='Not Sure'):
    if 'Up' in trend:
#         if sup_type=='Not Sure':
#             return False
        try:
            sr_range = abs(sup_level-res_level)
        except:
            sr_range = 0
        if ((sup_level+ATR) >= cp >= (sup_level-ATR)) and cp<=(sup_level+sr_range/2):
            return True
        
    if 'Down' in trend:
        if res_type=='Not Sure':
            return False
        try:
            sr_range = abs(sup_level-res_level)
        except:
            sr_range = 0
        if ((res_level-ATR) <= cp >= (res_level+ATR)) and cp>=(res_level-sr_range/2):
            return True
    return False


def detect_trigger(df):
    if 'Up' in df.iloc[-1].Trend:
        if detect_doji(df.iloc[-1].Open, df.iloc[-1].High, df.iloc[-1].Low, df.iloc[-1].Close):
            if min(df.iloc[-1].Open, df.iloc[-1].Close)>=((df.iloc[-1].High+df.iloc[-1].Low)/2):
                return (True, 'Up_Doji')
        if detect_bullish_engulfing(df.iloc[-2].Open, df.iloc[-2].Close, df.iloc[-1].Open, df.iloc[-1].Close):
            return (True, 'Bullish_Engulfing')
        if (df.iloc[-2].rsi_14 < 30) and (df.iloc[-2].rsi_14 < df.iloc[-1].rsi_14) and not detect_doji(df.iloc[-1].Open, df.iloc[-1].High, df.iloc[-1].Low, df.iloc[-1].Close):
            return (True, 'RSI')
        
    if 'Down' in df.iloc[-1].Trend:
        if detect_doji(df.iloc[-1].Open, df.iloc[-1].High, df.iloc[-1].Low, df.iloc[-1].Close):
            if max(df.iloc[-1].Open, df.iloc[-1].Close)<=((df.iloc[-1].High+df.iloc[-1].Low)/3):
                return (True, 'Down_Doji')
        if detect_bearish_engulfing(df.iloc[-2].Open, df.iloc[-2].Close, df.iloc[-1].Open, df.iloc[-1].Close):
            return (True, 'Bearish_Engulfing')
        if (df.iloc[-2].rsi_14 > 80) and (df.iloc[-2].rsi_14 > df.iloc[-1].rsi_14) and not detect_doji(df.iloc[-1].Open, df.iloc[-1].High, df.iloc[-1].Low, df.iloc[-1].Close):
            return (True, 'RSI')
        
    return (False, None)


# In[104]:


def process_df(df):
    df = get_resistance_level(df)
    df = get_support_level(df)
    df = get_trend(df)
    df = get_rsi(df)
    cp = df.iloc[-1].Close
    ATR = df.iloc[-1].ATR
    trend = df.iloc[-1].Trend
    MA20 = df.iloc[-1].Moving_Average_20
    MA50 = df.iloc[-1].Moving_Average_50
    MA200 = df.iloc[-1].Moving_Average_200
    sup_type, sup_level = which_level_support_probable(df)
    res_type, res_level = which_level_resistance_probable(df)
    is_in_area_of_value = detect_area_of_value(cp, ATR, sup_level, res_level, trend, sup_type, res_type)
    is_trigger, trigger_type = detect_trigger(df)
    
    return {'Close' : cp, 'ATR' : ATR, 'Trend' : trend, 'Support_Type' : sup_type, 'Support_Level' : sup_level,
            'Resistance_Type' : res_type, 'Resistance_Level' : res_level, 'In_Area_of_Value' : is_in_area_of_value,
            'Is_Trigger' : is_trigger, 'Trigger_Type' : trigger_type, 'Moving_Average_20':MA20, 'Moving_Average_50':MA50,
            'Moving_Average_200':MA200}


# In[105]:


def get_for_bt(d,t):
    if t.upper()=='NIFTY':
        t = '^NSEI'
    df = get_data_date(t,START=datetime.datetime.strptime(d,'%Y-%m-%d')-datetime.timedelta(days=450),END=datetime.datetime.strptime(d,'%Y-%m-%d')+datetime.timedelta(days=1)) #REMOVE TODAY.date() as END
    row = process_df(df)
    row['NAME'] = t
    row['Date'] = df.iloc[-1].Date
    row['Action'] = ('Buy' if ('Up' in row['Trend'] and row['In_Area_of_Value'] and row['Is_Trigger']) else(
                        'Sell' if ('Down' in row['Trend'] and row['In_Area_of_Value'] and row['Is_Trigger']) else None))
    row['Target'] = (row['Resistance_Level' if row['Resistance_Type']=='Not Sure' else row['Resistance_Type']]-0.2*row['ATR'] if row['Action']=='Buy' else(
                     row['Support_Level' if row['Support_Type']=='Not Sure' else row['Support_Type']]+0.2*row['ATR'] if row['Action']=='Sell' else None))
    
    row['Stoploss'] = (row['Resistance_Level' if row['Resistance_Type']=='Not Sure' else row['Resistance_Type']]+row['ATR'] if row['Action']=='Sell' else(
                         row['Support_Level' if row['Support_Type']=='Not Sure' else row['Support_Type']]-row['ATR'] if row['Action']=='Buy' else None))
    try:
        row['Expected_Within'] = int((abs(row['Close']-row['Target'])/((df.iloc[-1].ATR+df.iloc[-2].ATR+df.iloc[-3].ATR)/3))+1)
    except:
        row['Expected_Within'] = None
    try:  
        row['Risk_to_Reward'] = round((row['Close']-row['Stoploss'])/(row['Target']-row['Close']), 2)
    except:
        row['Risk_to_Reward'] = None
    row['Chart'] = f'<a class="w3-button" href="/chart/{t}" target="_blank"><button class="w3-button w3-white w3-border w3-border-green w3-round-large">Chart</button></a>'
    return pd.DataFrame([row])

def get_bt_for_date(d):
    nif50 = get_nifty50_stocks()['NSE Symbol'].to_list()
    f = pd.DataFrame()
    for t in nif50:
        f = f.append(get_for_bt(d,t))
    return f
# In[91]:


nif50 = get_nifty50_stocks()['NSE Symbol'].to_list()
nif50.extend(['^NSEI',"INR=X",'MCDOWELL-N'])

final = []
c = 0
t = len(nif50)
for eq in nif50:
    c += 1
    print(f'{c} out of {t} | Going for {eq}               ', end='\r')
    df = get_data_date(eq,str((TODAY-datetime.timedelta(days=450)).date()),str((TODAY+datetime.timedelta(days=1)).date())) #REMOVE TODAY.date() as END
    row = process_df(df)
    row['NAME'] = eq
    row['Date'] = df.iloc[-1].Date
    row['Action'] = ('Buy' if ('Up' in row['Trend'] and row['In_Area_of_Value'] and row['Is_Trigger']) else(
                        'Sell' if ('Down' in row['Trend'] and row['In_Area_of_Value'] and row['Is_Trigger']) else None))
    row['Target'] = (row['Resistance_Level' if row['Resistance_Type']=='Not Sure' else row['Resistance_Type']]-0.2*row['ATR'] if row['Action']=='Buy' else(
                     row['Support_Level' if row['Support_Type']=='Not Sure' else row['Support_Type']]+0.2*row['ATR'] if row['Action']=='Sell' else None))
    
    row['Stoploss'] = (row['Resistance_Level' if row['Resistance_Type']=='Not Sure' else row['Resistance_Type']]+row['ATR'] if row['Action']=='Sell' else(
                         row['Support_Level' if row['Support_Type']=='Not Sure' else row['Support_Type']]-row['ATR'] if row['Action']=='Buy' else None))
    try:
        row['Expected_Within'] = int((abs(row['Close']-row['Target'])/((df.iloc[-1].ATR+df.iloc[-2].ATR+df.iloc[-3].ATR)/3))+1)
    except:
        row['Expected_Within'] = None
    try:  
        row['Risk_to_Reward'] = round((row['Close']-row['Stoploss'])/(row['Target']-row['Close']), 2)
    except:
        row['Risk_to_Reward'] = None

    row['Chart'] = f'<a class="w3-button" href="/chart/{eq}" target="_blank"><button class="w3-button w3-white w3-border w3-border-green w3-round-large">Chart</button></a>'
    final.append(row)
df = pd.DataFrame(final)


# In[107]:


from flask import Flask, render_template, jsonify, request

app = Flask(__name__)
cols = ['Date','NAME', 'Support_Level', 'Resistance_Level','Action','Close','Target','Stoploss','Expected_Within','Up_Probability','Risk_to_Reward','Chart']

@app.route('/')
def home_null():
    return render_template('index_template.html', RECOM_TABLE='', HEADER= render_template('header_home.html'))
@app.route('/home')
def home():
    return render_template('index_template.html', RECOM_TABLE='', HEADER= render_template('header_home.html'))

@app.route('/options')
def options():
    d = request.args.get('Date')
    ed = request.args.get('EDate')
    N = ['^NSEI']+list(get_nifty50_stocks()['NSE Symbol'])
    f = {'Date' :[],'TICKER':[], 'Mean':[], 'Std':[], 'Within_1SD':[], 'Price':[], 'Up_Side':[], 'Down_Side':[]}
    if d is None or d=='':
        DATE = TODAY
    else:
        DATE = datetime.datetime.strptime(d,'%Y-%m-%d')
    if ed is None or ed=='':
        EXPIRY = get_last_thursday(DATE.date())
    else:
        EXPIRY = datetime.datetime.strptime(ed, '%Y-%m-%d').date()
    DAYS_TO_EXPIRY = (EXPIRY-DATE.date()).days
    TDE = np.busday_count(str(DATE.date()), str(EXPIRY))
    for s in N:
        print(f'Going for {s}           ',end='\r')
        dft = get_data_date(s, (DATE-datetime.timedelta(days=366)).date(), (DATE+datetime.timedelta(days=1)).date())
        f['Date'].append(dft.iloc[-1].Date)
        f['TICKER'].append(s)
        f['Mean'].append(round(dft.Change.mean()*100,2))
        f['Std'].append(round(dft.Change.std()*100,2)) 
        f['Price'].append(round(dft.iloc[-1].Close, 2))
        f['Up_Side'].append(round(dft.iloc[-1].Close*(1 + ((TDE*dft.Change.mean())+(TDE**0.5)*dft.Change.std())),2))
        f['Down_Side'].append(round(dft.iloc[-1].Close*(1 - ((TDE*dft.Change.mean())+(TDE**0.5)*dft.Change.std())),2))
        
        f['Within_1SD'].append(dft[(dft.Change.mean()-dft.Change.std()<=dft.Change)&(dft.Change<=dft.Change.mean()+dft.Change.std())].Change.count()/len(dft))
    dft = pd.DataFrame(f)
    s = dft.style.format().hide_index()
    s = s.set_table_attributes('class="w3-table w3-hoverable"').render()
    return render_template('index_template.html', RECOM_TABLE=s, 
                           HEADER= render_template('header_options.html'))

@app.route('/account')
def account():
    return render_template('index_template.html', RECOM_TABLE='', HEADER= render_template('Coming_Soon.html'))

@app.route('/recommendations')
def recommendations():
    global df
    dfs = get_win_rate(df[df.Action.notnull()].copy())[cols]
    for i in ['Support_Level','Target','Stoploss','Resistance_Level','Expected_Within','Close']:
        dfs[i] = dfs[i].astype(int)
    s = dfs.style.format().hide_index()
    s = s.set_table_attributes('class="w3-table w3-hoverable"').render()
    return render_template('index_template.html', RECOM_TABLE=s, 
                           HEADER= render_template('header_recommendations.html'))

@app.route('/backtest')
def backtest():
    t = request.args.get('Ticker')
    d = request.args.get('Date')
    if t is None and d is None:
        return render_template('index_template.html', RECOM_TABLE='', HEADER= render_template('header_backtest.html'))
    if d is not None and (t is None or t==''):
        dfs = get_bt_for_date(d)
        dfs = get_win_rate(dfs[dfs.Action.notnull()].copy())[cols]
    else:
        dfs = get_for_bt(d,t)
        dfs = get_win_rate(dfs)[cols]
#     print(dfs)
    
#     for i in ['Support_Level','Target','Stoploss','Resistance_Level','Expected_Within','Close']:
#         dfs[i] = dfs[i].astype(int)
    s = dfs.style.format().hide_index()
    s = s.set_table_attributes('class="w3-table w3-hoverable"').render()
    return render_template('index_template.html', RECOM_TABLE=s, HEADER= render_template('header_backtest.html'))


@app.route('/refresh')
def refresh():
    global df
    df = get_bt_for_date(str(TODAY.date()))
    return recommendations()

@app.route('/chart/<eq>')
def return_chart(eq):
    try:
        # if os.path.exists(f'templates/plots/{eq.upper()}.html'):
        #     return render_template('index_template.html', RECOM_TABLE='', HEADER= render_template(f'plots/{eq.upper()}.html'))
        pltdf = get_data_date(eq,str((TODAY-datetime.timedelta(days=370)).date()))
        fig = go.Figure(data=[go.Candlestick(x=pltdf['Date'],
                        open=pltdf['Open'],
                        high=pltdf['High'],
                        low=pltdf['Low'],
                        close=pltdf['Close'])])
        fig.add_hline(y=pltdf.iloc[-1].Close, line_color='orange', annotation_text='Current Price')
        fig.update_layout(xaxis_rangeslider_visible=True)
        pio.write_html(fig,file=f'templates/plots/{eq.upper()}.html')
        return render_template('index_template.html', RECOM_TABLE='', HEADER= render_template(f'plots/{eq.upper()}.html'))
    finally:
        os.remove(f'templates/plots/{eq.upper()}.html')
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002,debug=False)


# In[ ]:




