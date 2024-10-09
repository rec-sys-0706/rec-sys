import pandas as pd
from sqlalchemy import create_engine
from uuid import uuid4


csv_file_path = './240906_combined_papers.csv'
df = pd.read_csv(csv_file_path)

df['data_source'] = 'hf_paper'
df['uuid'] = df.apply(lambda row: str(uuid4()), axis=1)

server = 'DESKTOP-KTKE5PO'
database = 'rec_sys'
connection_string = f'mssql+pyodbc://{server}/{database}?trusted_connection=yes&driver=ODBC+Driver+17+for+SQL+Server'
engine = create_engine(connection_string)


table_name = 'item'
df.to_sql(table_name, con=engine, if_exists='append', index=False)


print("CSV 檔案成功匯入到 SQL Server!")

