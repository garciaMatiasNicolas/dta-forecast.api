from urllib.parse import urlparse
from database.db_engine import engine
import datetime
import pandas as pd
from .file_model import FileRefModel
import traceback


def obtain_file_route(route):
    parsed_url = urlparse(route)
    split_route = parsed_url.path.split('/')
    route_index = split_route.index('media')
    new_route = '/'.join(split_route[route_index:])
    return new_route

def validate_columns(dataframe: pd.DataFrame, required_columns: list) -> bool:
    missing_columns = [col for col in required_columns if col not in dataframe.columns]
    if missing_columns:
        return False, f"Faltan las siguientes columnas requeridas: {', '.join(missing_columns)}"
    return True, ""

def save_dataframe(route_file: str, file_name: str, model_type: str, wasSaved: bool, project_pk: int) -> str:
    models_allowed = ["historical_data", "historical_exogenous_variables", "projected_exogenous_variables", "stock_data"]
    
    cols_stock_data = ["Stock", "Sales Order Pending Deliverys", "Safety Lead Time (days)", "Safety stock (days)", 
    "Lead Time", "Price", "EOQ (Economical order quantity)", "Service Level", "Desv Est Lt Days", "Purchase Order", 
    "Lot Sizing", "ABC", "XYZ", "Purchase unit", "Make to order", "Slow moving"]

    cols_hsd = ["Starting Year", "Starting Period", "Periods Per Year", "Periods Per Cycle"]

    exog_cols = ["Variable"]

    base_cols = ["Family", "Region", "Salesman", "Client", "Category", "Subcategory", "SKU", "Description"]

    # Create dataframe with the Excel file
    if not wasSaved:
        new_route = obtain_file_route(route=route_file)
        dataframe = pd.read_excel(new_route)


        if model_type == 'historical_data':
            cols_hsd = base_cols + cols_hsd
            are_all_columns, message = validate_columns(dataframe=dataframe, required_columns=cols_hsd)

            if not are_all_columns:
                raise ValueError(message)

            date_columns = dataframe.iloc[:, 12:].columns
            not_date_columns = dataframe.iloc[:, :12].columns

            for col in not_date_columns:
                dataframe[col] = dataframe[col].fillna('null')
                dataframe[col] = dataframe[col].astype('str')

            exog_table = FileRefModel.objects.filter(project_id=project_pk, model_type_id=2).first()

            if exog_table is not None:
                exog_table = pd.read_sql_table(table_name=exog_table.file_name, con=engine)
                date_columns_exog_table = exog_table.iloc[:, 8:].columns

                if len(date_columns) != len(date_columns_exog_table):
                    raise ValueError("cols_exog_endog_not_match")

        elif model_type == 'historical_exogenous_variables' or model_type == 'projected_exogenous_variables':

            cols_exog = base_cols + exog_cols
            are_all_columns, message = validate_columns(dataframe=dataframe, required_columns=cols_exog)

            if not are_all_columns:
                raise ValueError(message)

            date_columns = dataframe.iloc[:, 9:].columns
            not_date_columns = dataframe.iloc[:, :9].columns

            endog_table = FileRefModel.objects.filter(project_id=project_pk, model_type_id=1).first()

            if endog_table is not None and model_type == 'historical_exogenous_variables':

                endog_table = pd.read_sql_table(table_name=endog_table.file_name, con=engine)
                date_columns_endog_table = endog_table.iloc[:, 12:].columns

                if len(date_columns) != len(date_columns_endog_table):
                    print("Longitud historica: ",len(date_columns_endog_table))
                    print("Longitud exogena: ", len(date_columns))
                    raise ValueError("cols_exog_endog_not_match")

        elif model_type == "stock_data":

            stock_cols = base_cols + cols_stock_data
            are_all_columns, message = validate_columns(dataframe=dataframe, required_columns=stock_cols)
            print(message)
            if not are_all_columns:
                raise ValueError(message)

            date_columns = dataframe.iloc[:, 8:].columns
            not_date_columns = dataframe.iloc[:, :8].columns

            for col in date_columns:
                dataframe[col] = dataframe[col].fillna(0)
                dataframe[col] = dataframe[col].astype('str')

            for col in not_date_columns:
                dataframe[col] = dataframe[col].fillna('null')
                dataframe[col] = dataframe[col].astype('str')

        else:
            raise ValueError("model_not_allowed")

        if model_type != 'stock_data':
            for date in date_columns:
                if isinstance(date, datetime.datetime):
                    dataframe[date] = dataframe[date].fillna(0.0)
                    dataframe[date] = dataframe[date].astype(float)
                    dataframe.rename(columns={date: date.strftime('%Y-%m-%d')}, inplace=True)

                else:
                    raise ValueError("columns_not_in_date_type")

        table_name = file_name
        dataframe = dataframe.applymap(lambda x: x.strip() if isinstance(x, str) else x)

        if model_type not in models_allowed:
            raise ValueError("model_not_allowed")

        else:
            if model_type == "historical_exogenous_variables" or model_type == "projected_exogenous_variables":
                try:
                    dataframe.to_sql(table_name, con=engine, if_exists='append', index=False)
                    print("TABLA GUARDADA")
                except Exception as Err:
                    print(Err)
                
            else:
                try:
                    dataframe.to_sql(table_name, con=engine, if_exists='replace', index=False)
                    print("TABLA GUARDADA")
                except Exception as Err:
                    print(Err)

            return "succeed"

    else:
        new_route = obtain_file_route(route=route_file)
        dataframe = pd.read_excel(new_route)
        dataframe.astype('str')
        dataframe = dataframe.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        dataframe.fillna(0, inplace=True)
        table_name = file_name

        if model_type == "historical_data":
            dataframe.to_sql(table_name, con=engine, if_exists='replace', index=False)
            return "succeed"
    

