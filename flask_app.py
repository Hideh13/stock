from flask import Flask, render_template, request
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

# 포트폴리오 데이터 로드
Portfolio = pd.read_csv('port.csv')
cash = pd.read_csv('cash.csv')
korea = pd.read_csv('korea.csv')

spy_data = yf.download('SPY', period="6mo", interval="1d")['Close']
rate_o = 1375.92
origin = 11603782 

app = Flask(__name__)


    

@app.route('/')
def index():
    rate_list = yf.download("KRW=X" , start = "2024-03-01", end = "2030-03-01")["Close"]
    rate = rate_list[-1]
    
    portfolio, now = pofo_info()
    koreainfo, kow = korea_pofo()
    
    portfolio["percent"] = np.round(portfolio["value"]/(now+kow)*100,2)
    koreainfo["percent"] = np.round(koreainfo["value"]/(now+kow)*100,2)
    # 원그래프 생성
    valuelist = np.concatenate(portfolio["value"] , koreainfo["value"])
    labelist = np.concatenate(portfolio["Ticker" , koreainfo["Ticker"]])
    
    pie_chart = create_pie_chart(valuelist , labelist)
    pie_chart2 = create_pie_chart( [now + portfolio["value"][-2],kow + portfolio["value"][-1] ]  , ["달러" , "원화"] )
    cash =  portfolio[-2:].copy()
    
    
    
    #portfolio history
    hist = history(portfolio) + np.sum(portfolio["value"][-2:])/rate
    plt.figure(figsize=(10, 5))
    hist=(hist/hist[0]-1)*100
    shis = (spy_data/spy_data[0]-1)*100
    hist.plot()
    shis.plot()
    plt.ylabel("return percent(%)")
    plt.grid()
    plt.savefig(f'static/total_price.png')  # static 폴더에 저장
    plt.close()
    
    Prsi = calculate_rsi(hist , Bool = True)
    plt.figure(figsize=(10, 5))
    Prsi.plot( xlabel='Date')
    plt.grid()
    plt.savefig(f'static/total_rsi.png')  # static 폴더에 저장
    plt.close()
    
    current_return =np.round( np.sum(portfolio['Return']*portfolio["percent"])/100 ,2)
    total_return = np.round(((now / origin)-1)*100,2)
    
    cash["gaeiduc"] = np.round( cash["Return"]/(100+cash["Return"])*cash['value'],2)
    
    return render_template('index.html', 
                           portfolio=portfolio[:-2],
                           cash =cash, cash_rate= np.round(np.sum(portfolio["percent"][-2:]),2) , cash_value = np.round(np.sum(portfolio["value"][-2:]),2) , 
                           pie_chart=pie_chart,
                           pie_chat2 = pie_chart2,
                           total_return=total_return ,
                           current_return= current_return,
                           origin= np.round(origin),
                           now =np.round(now),
                           total_beta = calculate_beta( stock_data=  hist ),
                           koreainfo = koreainfo
                          )

def adding(pdd : pd.DataFrame , dic : dict) -> pd.DataFrame:
    return pd.concat(pdd,  pd.DateFrame(dic , index=[0]))

def korea_pofo():
    korea_info = pd.DataFrame()
    
    for k in len(korea):
        temp = {
            "Name" : korea["name"][k],
            "Ticker" : korea["number"][k],
            "Purchase Price" : korea["price"][k]
            "Number of Shares" : korea["having"][k]
        }
        korea_info = adding(korea_info , temp)
        
    korea_info["Current Price"] = korea_info["Ticker"].apply(lambda x: np.round(yf.download(x,period= "5d"  )["Close"][-1] ,2))
    korea_info["Return"] = np.round((korea_info['Current Price'] - korea_info['Purchase Price']) / korea_info['Purchase Price'] * 100,2)
    korea_info["value"] = np.round(korea_info['Number of Shares'] *korea_info["Current Price"] , 2)    
    

    return korea_info , np.sum(korea_info["value"])


def pofo_info():
    portfolio = Portfolio.copy()
    
    
    rate_list = yf.download("KRW=X" , start = "2024-03-01", end = "2030-03-01")["Close"]
    rate = round(rate_list[-1],2)
    portfolio['Name'] = portfolio['Ticker']
    
    # 현재 주가 및 베타값 
    portfolio['Current Price'] = portfolio['Ticker'].apply(lambda x: np.round(yf.download(x,period= "5d"  )["Close"][-1] ,2))
    portfolio['Convert Price'] = portfolio['Current Price']*rate
    portfolio['Beta'] = portfolio['Ticker'].apply(lambda x: calculate_beta(x))
    
    
    new_data1= {
        'Ticker' : "dollar($)",
		'Current Price' : 1,
        'Convert Price' : rate,
        'Purchase Price' : rate_o,
        'Beta': 0,
        'Number of Shares' : np.round(np.sum(cash["dollar"]),2)
    }
    
    new_data2= {
        'Ticker' : "won(₩)",
		'Current Price' : 1,
        'Convert Price' : 1,
        'Purchase Price' : 1,
        'Beta': 0,
        'Number of Shares' : np.sum(cash["won"])
    }
    portfolio = adding(portfolio,new_data1)
    portfolio = adding(portfolio,new_data2)
    # 수익률 계산
    
    portfolio['Return'] = np.round((portfolio['Current Price'] - portfolio['Purchase Price']) / portfolio['Purchase Price'] * 100,2)
    
    
    portfolio["value"] = np.round(portfolio['Number of Shares'] * portfolio['Convert Price'],2)
    
    
    now=  np.sum(portfolio["value"])
    
    return portfolio , now

def history(port):
    
    v = yf.download(port['Ticker'][0], period="6mo", interval="1d")['Close'] *port['percent'][0]
    
    for i in range(1, len(port)-2):
        v +=yf.download(port['Ticker'][i], period="6mo", interval="1d")['Close'] *port['percent'][i]
        
    return v 

def create_pie_chart(frequency, labels):
    # 원그래프 생성
    plt.figure(figsize=(8, 6))
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot()
    pie = ax.pie(frequency)
    total = np.sum(frequency)
    sum_pct = 0
    threshold= 5
    bbox_props = dict(boxstyle='square',fc='w',ec='w',alpha=0)
    config = dict(arrowprops=dict(arrowstyle='-'),bbox=bbox_props,va='center')
    
    cts = np.sum( frequency < threshold/100 * total )
    ind = 1
    
    for i,j in enumerate(labels):
        ang1, ang2 	= ax.patches[i].theta1	, ax.patches[i].theta2 ## 파이의 시작 각도와 끝 각도
        center, r 	= ax.patches[i].center	, ax.patches[i].r ## 원의 중심 좌표와 반지름길이
        
        if i < len(labels) - 1:
            sum_pct += float(f'{frequency[i]/total*100:.2f}')
            text = f'{labels[i]}\n{frequency[i]/total*100:.2f}%'
        else: 
            text = f'{labels[i]} \n {100-sum_pct:.2f}% '
            
            
        
        if frequency[i]/total*100 < threshold:
            ang = (ang1+ang2)/2 ## 중심각
            x = np.cos(np.deg2rad(ang)) 
            y = np.sin(np.deg2rad(ang))
            
            horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
            ax.annotate(text, xy=(x, y), xytext=(1.5*x, -1 + 2*ind/( cts+1)) ,horizontalalignment=horizontalalignment, **config)
            ind +=1
        else:
            x = (r/2)*np.cos(np.pi/180*((ang1+ang2)/2)) + center[0] ## 텍스트 x좌표
            y = (r/2)*np.sin(np.pi/180*((ang1+ang2)/2)) + center[1] ## 텍스트 y좌표
            ax.text(x,y,text,ha='center',va='center',fontsize=12)

    plt.title('Current Portfolio Composition')
    # 그래프를 이미지로 변환
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return f"data:image/png;base64,{graph_url}"

def calculate_beta(ticker = "" , stock_data = []):
    # 주식 및 S&P 500 지수의 과거 가격 데이터 로드
    if ticker:
    	stock_data = yf.download(ticker, period="6mo", interval="1d")['Close']
    
    # 데이터 정렬 및 결측치 제거
    data = pd.concat([stock_data, spy_data], axis=1)
    data.columns = ['Stock', 'SPY']
    data = data.dropna()
    
    # 일일 수익률 계산
    data['Stock Return'] = data['Stock'].pct_change()
    data['SPY Return'] = data['SPY'].pct_change()
    
    # 베타값 계산 (Covariance / Variance)
    covariance = np.cov(data['Stock Return'][1:], data['SPY Return'][1:])[0][1]
    variance = np.var(data['SPY Return'][1:])
    
    beta = round(covariance / variance,3)
    
    return beta
def calculate_rsi(data, window=14 , Bool = False):
    # 데이터의 수익률을 계산
    if Bool:
        delta=data.diff()
    else:
	    delta = data['Close'].diff()
    
    # 상승과 하락을 분리
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    # RSI 계산
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def stock_info(ticker):
    # 주식 데이터 다운로드
    if ticker == "cash1234":
        ticker = "KRW=X"
    
    stock_data = yf.download(ticker, period='6mo', interval='1d')['Close']
    stock_data[0]
    
    rsi = calculate_rsi(stock_data , Bool = True)
    
    # PER 및 Forward PER 가져오기
    stock = yf.Ticker(ticker)
    per = stock.info.get('trailingPE', 'N/A')  # PER
    forward_per = stock.info.get('forwardPE', 'N/A')  # Forward PER

    # 주가 추이 시각화
    plt.figure(figsize=(10, 5))
    stock_data.plot(title=f'{ticker} Stock Price - Last 6 Months', ylabel='Price (USD)', xlabel='Date')
    plt.grid()
    plt.savefig(f'static/{ticker}_stock_price.png')  # static 폴더에 저장
    plt.close()
    # rsi 시각화
    plt.figure(figsize=(10, 5))
    rsi.plot(title=f'{ticker} Stock RSI - Last 6 Months', xlabel='Date')
    plt.grid()
    plt.savefig(f'static/{ticker}_stock_rsi.png')  # static 폴더에 저장
    plt.close()
    
    return stock_data, per, forward_per

@app.route('/exch')
def exch():
    stock_data, per, forward_per = stock_info("KRW=X")
    return render_template('exch.html')


@app.route('/stock/<ticker>')
def stock(ticker):
    if ticker == "cash1234":
        ticker = "KRW=X"
    stock_data, per, forward_per = stock_info(ticker)
    return render_template('stock.html', ticker=ticker, per=per, forward_per=forward_per)
@app.route('/search', methods=['POST'])
def search():
    ticker = request.form['ticker'].upper()  # 대문자로 변환하여 검색
    try:
        stock_data, per, forward_per = stock_info(ticker)
        return render_template('stock.html', ticker=ticker, per=per, forward_per=forward_per)
    except Exception as e:
        return render_template('error.html')
