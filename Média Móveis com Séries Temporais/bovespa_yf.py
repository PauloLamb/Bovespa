import warnings
warnings.filterwarnings("ignore")

from pandas_datareader import data as pdr 
import yfinance as yf #yahoo finance
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import timedelta
from datetime import datetime
yf.pdr_override()

# importanto médido interno ARIMA

from statsmodels.tsa.arima.model import ARIMA

def MA_model_generation(ts, q):
    model = ARIMA(ts, order = [0, 0, q]) 
    
    # os parâmetros 0, 0, q correspondem a: 
    
    #p: The number of lag observations included in the model, also called the lag order.
    #d: The number of times that the raw observations are differenced, also called the degree of differencing.
    #q: The size of the moving average window, also called the order of moving average.

    # no modelo de médias móveis, por definição, p e d são zero.
    
    model_fit = model.fit()
    return model_fit

def captura_yahoofn (ticket,start,end):
    
    # ticket=ticket+'.SA'
    cotacao = pdr.get_data_yahoo(ticket, start, end)
    cotacao = cotacao.rename(columns={'Adj Close': 'Adj_Close'})
    serie=pd.DataFrame({"DATA":cotacao.index})
    serie.insert(1,'Fechamento',np.array(cotacao.Adj_Close))
    
    return serie

def gera_curva_MA (serie,grau):
    MA_otimo = MA_model_generation(serie.Fechamento, grau)
     
    return MA_otimo

def cria_dataset (serie,MA):
    serie_gerada = serie.copy()
    size = serie.shape[0]
    
    for i in range(size):
        serie_gerada.Fechamento[i]=MA.predict(i)[i]
        
    return serie_gerada

def mostra_curva(ticket,serie):
    plt.figure( figsize=(21, 7))

    plt.title(ticket)
    plt.plot(serie.DATA,serie.Fechamento)
    plt.xlabel("Data")
    plt.ylabel("Fechamento")

    plt.show()

def compara_series (ticket,original,calculada):
    plt.figure( figsize=(21, 7))

    plt.title(ticket+" - Fechamento e curva calculada")
    plt.plot(original.DATA,original.Fechamento)
    plt.plot(original.DATA,calculada.Fechamento, c='red')
    plt.legend(['valor original', 'Curva calculada']);
    plt.xlabel("Data")
    plt.ylabel("Valor")

    plt.show()

    print('Erro calculado = ',erro (original,calculada))

def gera_previsao (original,MA,qt_dias):
    size = original.shape[0]
    lastday = original.DATA[size-1]  
    período_previsto = qt_dias

    previsao = pd.DataFrame({"DATA":np.array([lastday+timedelta(i) for i in range (período_previsto)]),
                     "Fechamento":np.array(MA.predict(i)[i] for i in range (size, size + período_previsto))})
                     

    # ajustar todos os feriados aqui
    feriados = [datetime(2023, 9, 7),datetime(2023, 10, 12),datetime(2023, 11, 2),
               datetime(2023, 11, 15),datetime(2023, 12, 25),datetime(2023, 12, 29)]

    nextday=lastday
    for i in range (período_previsto):
        nextday=nextday+timedelta(1)
        if nextday in feriados:
            nextday=nextday+timedelta(1)
        if nextday.weekday()==5:     # descontando sábados e domingos - gambiarra
            nextday=nextday+timedelta(2) 
        previsao.DATA[i]=nextday
    return previsao

def erro (serie1,serie2):
    soma=0
    size=min(serie1.shape[0],serie2.shape[0])
    
    for i in range (size):
        dif=abs(serie1.Fechamento[i]-serie2.Fechamento[i])
        if dif<1:
            soma+=dif
        else:
            soma+=dif**2                 
        
    return soma

def mostra_previsao(acao,original,previsao):
    size = original.shape[0]
    lastday = original.DATA[size-1] 
    
    serie_MA = pd.concat([original, previsao], axis=0)
    serie_MA.reset_index(inplace=True, drop=True)
    
    plt.figure( figsize=(21, 7))
    plt.title(acao+" + previsão")
    plt.plot(serie_MA.DATA,serie_MA.Fechamento, c="blue")
    plt.legend(['Fechamento']);
    plt.vlines(lastday,min(original.Fechamento),max(original.Fechamento), color='green')
    plt.xlabel("DATA")
    plt.ylabel("VALORES")

    plt.show()