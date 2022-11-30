# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 11:15:30 2022

@author: snagarur
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil.relativedelta import relativedelta
import plotly.graph_objects as go
import plotly.express as px

st.title('Financial Dashboard')
st.write("Data Source: Yahoo Finance")

# Get the list of stock tickers from S&P500
ticker_list = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol']

# Add multiple choices box
ticker = st.sidebar.selectbox("Ticker", ticker_list)


tab1, tab2, tab3, tab4 = st.tabs(['Summary', 'Charts', 'Financials', 'Monte Carlo Sim'])
with tab1:

    # --- Select date time ---
    duration_list = ('1M', '6M', 'YTD', '1Y', '5Y', 'Max')
    duration = st.selectbox("Duration", duration_list)
    
    # Select start date
    end_date = datetime.today().date()
    if duration == '1M':
        start_date =  end_date - relativedelta(months=1)
    elif duration == '6M':
        start_date = end_date - relativedelta(months=6)
    elif duration == 'YTD':
        current_year = datetime.today().year
        current_year_string = str(current_year)
        start_date_string = current_year_string + '-01-01'
        start_date = datetime.strptime(start_date_string, '%Y-%m-%d')
        start_date = start_date.strftime('%Y-%m-%d')
    elif duration == '1Y':
        start_date = end_date - relativedelta(years=1)
    elif duration == '5Y':
        start_date = end_date - relativedelta(years=5)
    else:
        start_date = end_date - relativedelta(years=50)
    
    # --- Add a button ---
    get = st.button("Update", key="get")

    def getData():
        global get
        global ticker
        col1, col2 = st.columns(2)
        with col1:
            stock = yf.Ticker(ticker).info
            stock_price = {
                            'Previous Close' : [stock['previousClose']], 
                            'Open' : [stock['open']], 
                            'Bid' : [str(stock['bid'])],
                            'Ask' : [str(stock['ask'])],
                            "Day's Range" : [str(stock['dayLow']) + " - " + str(stock['dayHigh'])], 
                            'Volume' : [stock['volume']], 
                            'Avg. Volume' : [stock['averageVolume']]
                            }
            # Convert the dictionary to a dataframe
            stock_df = pd.DataFrame.from_dict(stock_price, orient='index')
            stock_df = stock_df.rename(columns={0:'Value'})
            stock_df = stock_df.astype(str)
            # Display table
            st.dataframe(stock_df)
            
        with col2:
            if len(ticker) > 0:
                st.write('Close price')
                fig, ax = plt.subplots()
                stock_df1 = yf.Ticker(ticker).history(interval='1d', start=start_date, end=end_date)
                ax.plot(stock_df1['Close'], label=ticker)
                plt.xticks(rotation=45)
                ax.legend()
                st.pyplot(fig)

    #Adding company profile
    def companyProfile():
        global ticker
        st.subheader('Company Profile')
        st.write('Sector: '+ yf.Ticker(ticker).info['sector'])
        st.write('Website: '+yf.Ticker(ticker).info['website'])
        st.write('Full Time Employees: ', yf.Ticker(ticker).info['fullTimeEmployees'])
        yf.Ticker(ticker).info['longBusinessSummary']

    def shareholders():
        global ticker
        major_shareholders = yf.Ticker(ticker).institutional_holders
        st.subheader("""*Institutional investors* for """ + ticker)
        if major_shareholders.empty == True:
            st.write("No data available at the moment")
        else:
            st.write(major_shareholders)

    # --- Show the above table and plot when the button is clicked ---
    if get:
        getData()
        companyProfile()
        shareholders()


with tab2:
    fig = go.Figure()  
    df = yf.Ticker(ticker).history(interval='1d', start=start_date, end=end_date)
    def Chart():
        tab1, tab2 = st.tabs(['Line','Candle'])
        with tab1:
            # line plot
            fig = px.line(df, x=df.index, y="Close")
            fig.update_layout(title = {'text':'Stock Price over time', 'y':1, 'x': 0.5, 'xanchor': 'center', 'yanchor':'top'}, 
                              yaxis_title = 'Stock Price', 
                              xaxis_rangeslider_visible = True)
            st.plotly_chart(fig)
        
        with tab2:
            # trace of stock price
            fig.add_trace(go.Candlestick(x=df.index, 
                                         open = df['Open'], 
                                         high = df['High'], 
                                         low = df['Low'], 
                                         close = df['Close'], 
                                         name='Stock Price'))
            # add colours
            colour = ['green' if row['Open'] - row['Close'] >= 0 else 'red' for index, row in df.iterrows()]
        
            # trace of volume
            fig.add_trace(go.Bar(x=df.index, 
                                 y= (df['Volume']/10000000), 
                                 marker_color = colour, 
                                 name = 'Volume'))
        
            # final chart
            fig.update_layout(title = {'text':'Stock Price over time', 'y':1, 'x': 0.5, 'xanchor': 'center', 'yanchor':'top'}, 
                              yaxis_title = 'Stock Price', 
                              xaxis_rangeslider_visible = False)
        
            return st.plotly_chart(fig, use_container_width=True)
    if get:
        Chart()
    
with tab3:
    def Financials():
        tab5, tab6, tab7 = st.tabs(['Income Statement', 'Balance Sheet', 'Cash Flow'])
        tick = yf.Ticker(ticker)
        with tab5:
            tab1, tab2 = st.tabs(['Annual', 'Quaterly'])
            with tab1:
                st.write("Annual Income Statement")
                annual_income = tick.financials
                annual_income
        
            with tab2:
                st.write("Quaterly Income Statement")
                quaterly_income = tick.quarterly_financials
                quaterly_income
        
        with tab6:
            tab1, tab2 = st.tabs(['Annual', 'Quaterly'])
            with tab1:
                st.write("Annual Balance Sheet")
                annual_balance = tick.balance_sheet
                annual_balance
                with tab2:    
                    st.write("Quaterly Balance Sheet")
                    quaterly_balance = tick.quarterly_balance_sheet
                    quaterly_balance   
       
        with tab7:
            tab1, tab2 = st.tabs(['Annual', 'Quaterly'])
            with tab1:
                st.write("Annual Cash Flow")
                annual_cash = tick.cashflow
                annual_cash
                with tab2:
                    st.write("Quaterly Cash Flow")
                    quaterly_cash = tick.quarterly_cashflow
                    quaterly_cash
    if get:
        Financials()
        
with tab4:
    np.random.seed(123)
    simulations = st.selectbox('No of Sim', (200, 500, 1000))
    time_horizon = st.selectbox('Time Horizon', (30, 60, 90))
    sim = st.button('Run Sim', key='sim')
    def MonteCarlo():
        stock_price = yf.Ticker(ticker).history(interval='1d', start= start_date, end= end_date)
        close_price = stock_price['Close']
        daily_return = close_price.pct_change()
        daily_volatility = np.std(daily_return)

        # Run the simulation
        simulation_df = pd.DataFrame()

        for i in range(simulations):
    
            # The list to store the next stock price
            next_price = []
    
            # Create the next stock price
            last_price = close_price[-1]
    
            for j in range(time_horizon):
                # Generate the random percentage change around the mean (0) and std (daily_volatility)
                future_return = np.random.normal(0, daily_volatility)
                # Generate the random future price
                future_price = last_price * (1 + future_return)
                # Save the price and go next
                next_price.append(future_price)
                last_price = future_price
                # Store the result of the simulation
                next_price_df = pd.Series(next_price).rename('sim' + str(i))
                simulation_df = pd.concat([simulation_df, next_price_df], axis=1)
                
        fig, ax = plt.subplots()
        fig.set_size_inches(15, 10, forward=True)
        plt.plot(simulation_df)
        plt.title('Monte Carlo simulation')
        plt.xlabel('Day')
        plt.ylabel('Price')
        plt.axhline(y=close_price[-1], color='red')
        plt.legend(['Current stock price is: ' + str(np.round(close_price[-1], 2))])
        ax.get_legend().legendHandles[0].set_color('red')

        st.pyplot(fig)
                
    if get:
        if sim:
            MonteCarlo()
    

    

    
































    