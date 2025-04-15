!pip install prophet gspread oauth2client google-api-python-client pandas
import pandas as pd
import numpy as np
from prophet import Prophet
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# 配置
SHEET_ID = "1Sdmd87m49Lfrw6lhsbub4RltAM0M-VYvJOw33Se-R_g"
SCOPES = ['https://www.googleapis.com/auth/spreadsheets',
          'https://www.googleapis.com/auth/gmail.send']

creds = ServiceAccountCredentials.from_json_keyfile_name('service.json', SCOPES)
gc = gspread.authorize(creds)
sh = gc.open_by_key(SHEET_ID)

# 获取销量数据
sales_df = pd.DataFrame(sh.worksheet('DailySales').get_all_records())
sales_df['Date'] = pd.to_datetime(sales_df['Date'])

# 单个 SKU 的预测逻辑
def forecast_sku(sku, horizon=14):
    df = sales_df[['Date', sku]].rename(columns={'Date':'ds', sku:'y'})
    if df['y'].sum() < 30:  # 数据不足直接返回均值
        yhat = df['y'].mean()
        return [max(int(yhat),1)]*horizon
    m = Prophet(seasonality_mode='additive', weekly_seasonality=True)
    m.fit(df)
    future = m.make_future_dataframe(periods=horizon)
    fcst = m.predict(future)[-horizon:]['yhat'].clip(lower=0).round().astype(int).tolist()
    return fcst

# 写回 Stock 表
stock_ws = sh.worksheet('Stock')
skus = stock_ws.col_values(1)[1:]  # 排除表头

safety, reorder = {}, {}

for sku in skus:
    pred = forecast_sku(sku)
    demand_14 = sum(pred)
    curr = int(stock_ws.cell(skus.index(sku)+2,5).value)  # CurrentStock
    safety[sku] = int(demand_14 * 0.7)  # 70% 服务水平
    if curr < safety[sku]:
        reorder[sku] = safety[sku]*2 - curr

# 写入 SafetyStock
for sku,val in safety.items():
    stock_ws.update_cell(skus.index(sku)+2,6,val)

# 写入 Forecast Sheet
fc_ws = sh.worksheet('Forecast') if 'Forecast' in [w.title for w in sh.worksheets()] else sh.add_worksheet('Forecast',1000,20)
fc_ws.clear()
fc_ws.update([['SKU','DemandNext14']] + [[k,v] for k,v in safety.items()])
