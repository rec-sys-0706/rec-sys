# General functions(CRUD) for sql database
import logging
from server.utils import get_connection

class SQLError(BaseException):
    pass
# ---- General Functions ---- #
def select_all(table: str) -> list|None:
    """Select all columns from `table`"""
    conn = get_connection()
    query = f"SELECT * FROM {table}"
    try:
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        return rows
    except SQLError as error:
        logging.error(f"Failed to retrieve data from SQL table: {error}")
        return None
    finally:
        conn.close()

def select_by(table: str, columns: list[str], values: list[str]) -> list|None:
    """Select columns by condition `column`==`value`"""
    if len(columns) != len(values):
        raise ValueError("Length of `columns` and `values` must be the same")
    conn = get_connection()
    conditions = " AND ".join([f"{col} = ?" for col in columns])
    query = f"SELECT * FROM {table} WHERE {conditions}"
    try:
        cursor = conn.cursor()
        cursor.execute(query, values)
        rows = cursor.fetchall()
        return rows
    except SQLError as error:
        logging.error(f"Failed to retrieve data from SQL table: {error}")
        return None
    finally:
        conn.close()

def insert(table: str, data: dict) -> bool:
    """Inserts `data` into a specified table in the database"""
    conn = get_connection()
    columns = ', '.join(data.keys())
    placeholders = ', '.join(['?'] * len(data))
    query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
    try:
        cursor = conn.cursor()
        cursor.execute(query, tuple(data.values()))
        conn.commit()
        return True
    except SQLError as error:
        logging.error(f"Failed to insert data into SQL table: {error}")
        return False
    finally:
        conn.close()

def delete(table: str, columns: list[str], values: list[str]) -> bool:
    if len(columns) != len(values):
        raise ValueError("Length of `columns` and `values` must be the same")
    conn = get_connection()
    conditions = " AND ".join([f"{col} = ?" for col in columns])
    query = f"DELETE FROM {table} WHERE {conditions}"

    try:
        cursor = conn.cursor()
        cursor.execute(query, values)
        conn.commit()

        if cursor.rowcount == 0:
            raise SQLError(f"No rows deleted for {columns}: {values}")
        else:
            logging.info(f"Successfully deleted {cursor.rowcount} row(s).")
            return True

    except SQLError as error:
        logging.error(f"Failed to delete row from SQL table: {error}")
        return False
    finally:
        conn.close()

def update(table: str, column: str, value: str, data: dict) -> bool:
    """Update data where column = value in table."""
    conn = get_connection()
    clauses = ', '.join([f"{key} = ?" for key in data.keys()])
    query = f"UPDATE {table} SET {clauses} WHERE {column} = ?"
    try:
        cursor = conn.cursor()
        cursor.execute(query, (*data.values(), value))
        conn.commit()

        if cursor.rowcount == 0:
            raise SQLError(f"No rows updated for {column}: {value}")
        else:
            logging.info(f"Successfully updated {cursor.rowcount} row(s).")
            return True

    except SQLError as error:
        logging.error(f"Failed to update data in SQL table: {error}")
        return False
    finally:
        conn.close()

def get_headers(table_name):
    conn = get_connection()
    cursor = conn.cursor()
    query = f"""
    SELECT COLUMN_NAME 
    FROM INFORMATION_SCHEMA.COLUMNS 
    WHERE TABLE_NAME = '{table_name}'
    """
    cursor.execute(query)

    headers = [row.COLUMN_NAME for row in cursor.fetchall()]

    conn.close()
    return headers # , primary_key
# --------------------------- #

# TODO
# def upsert(table_name, new_table: DataFrame, keys: list=[]):
#     """
#     This function upserts data from a dataframe into a given SQLite table.

#     :param table_name: name of the table to upsert data into
#     :param new_table: pandas dataframe containing data to upsert
#     """
#     try:
#         conn = sqlite3.connect(DB)
#         cursor = conn.cursor()

#         _, primary_key = get_headers(table_name)
#         # keys.append(primary_key)

#         headers = new_table.columns.tolist()
#         columns = ", ".join(headers)
#         placeholders = ", ".join('?' * len(headers))
#         update_placeholders = ", ".join([f"{col} = EXCLUDED.{col}" for col in headers if (col not in keys)])
        
#         query = f'''
#         INSERT INTO {table_name} ({columns}, id)
#         VALUES ({placeholders}, ?)
#         ON CONFLICT({', '.join(keys)}) DO UPDATE SET
#         {update_placeholders};
#         '''
#         for row in new_table.itertuples(index=False, name=None): # for_through only values.
#             row += (str(uuid.uuid4()), )
#             cursor.execute(query, row)
#         conn.commit()
#     except sqlite3.Error as error:
#         print(f"Failed to update data in sqlite table: {error}")

#     finally:
#         if conn:
#             conn.close()


