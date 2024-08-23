import pyodbc

def connect_to_database():
    try:
        conn_str = (
            'DRIVER={SQL Server};'
            'SERVER=140.136.149.181,1433;' 
            'DATABASE=news;'  
            'UID=news;'  
            'PWD=123;' 
        )

        print("Attempting to connect to the database...")
        conn = pyodbc.connect(conn_str)
        print("Connection successful!")

        return conn

    except pyodbc.Error as e:
        print(f"Connection Error: {e}")
        return None

def create_record(conn, record):
    try:
        cursor = conn.cursor()
        query = """
        INSERT INTO news_table (Category, Subcategory, Title, Abstract, Content, URL, PublicationDate)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        cursor.execute(query, record['Category'], record['Subcategory'], record['Title'], 
                       record['Abstract'], record['Content'], record['URL'], record['PublicationDate'])
        conn.commit()
        print("Record created successfully.")

    except pyodbc.Error as e:
        print(f"Create Record Error: {e}")

    finally:
        if cursor:
            cursor.close()

def read_records(conn):
    try:
        cursor = conn.cursor()
        query = "SELECT Id, Category, Subcategory, Title, Abstract, Content, URL, PublicationDate FROM news_table"
        cursor.execute(query)
        rows = cursor.fetchall()
        print(f"Number of records fetched: {len(rows)}")
        for row in rows:
            print(row)

    except pyodbc.Error as e:
        print(f"Read Records Error: {e}")

    finally:
        if cursor:
            cursor.close()

def update_record(conn, record_id, updated_data):
    try:
        cursor = conn.cursor()
        query = """
        UPDATE news_table
        SET Category = ?, Subcategory = ?, Title = ?, Abstract = ?, Content = ?, URL = ?, PublicationDate = ?
        WHERE Id = ?
        """
        cursor.execute(query, updated_data['Category'], updated_data['Subcategory'], updated_data['Title'], 
                       updated_data['Abstract'], updated_data['Content'], updated_data['URL'], updated_data['PublicationDate'],
                       record_id)
        conn.commit()
        print("Record updated successfully.")

    except pyodbc.Error as e:
        print(f"Update Record Error: {e}")

    finally:
        if cursor:
            cursor.close()

def delete_record(conn, record_id):
    try:
        cursor = conn.cursor()
        query = "DELETE FROM news_table WHERE Id = ?"
        cursor.execute(query, record_id)
        conn.commit()
        print("Record deleted successfully.")

    except pyodbc.Error as e:
        print(f"Delete Record Error: {e}")

    finally:
        if cursor:
            cursor.close()

def main():
    conn = connect_to_database()
    if conn:
        # Create a record
        new_record = {
            'Category': 'Technology',
            'Subcategory': 'AI',
            'Title': 'The Future of AI',
            'Abstract': 'An overview of AI advancements.',
            'Content': 'AI is rapidly evolving...',
            'URL': 'https://example.com/ai-future',
            'PublicationDate': '2024-08-19'
        }
        create_record(conn, new_record)

        # Read records
        read_records(conn)

        # Update a record
        updated_record = {
            'Category': 'Technology',
            'Subcategory': 'AI',
            'Title': 'The Future of AI - Updated',
            'Abstract': 'An updated overview of AI advancements.',
            'Content': 'AI is evolving even faster...',
            'URL': 'https://example.com/ai-future-updated',
            'PublicationDate': '2024-08-20'
        }
        update_record(conn, 1, updated_record)

        # Optionally delete a record
        # delete_record(conn, 1)

        conn.close()
        print("Connection closed.")

if __name__ == "__main__":
    main()
