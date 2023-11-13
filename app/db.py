#!/usr/bin/python
import psycopg2
from configparser import ConfigParser
from psycopg2.extras import RealDictCursor


def config(db, section="postgresql"):
    # create a parser
    parser = ConfigParser()
    # read config file
    if db == "postgres":
        parser.read("./etc/postgres.ini")
    else:
        parser.read("./etc/ignition.ini")
    # get section, default to postgresql
    db = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            db[param[0]] = param[1]
    else:
        raise Exception("Section {0} not found in the {1} file".format(section, db))
    return db


def call_db(sql):
    conn = None
    try:
        params = config(db="postgres")
        conn = psycopg2.connect(**params)
        cur = conn.cursor()
        cur.execute(sql)
        row = cur.fetchall()
        cur.close()
        return row
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()


def call_db_json(sql):
    conn = None
    try:
        params = config(db="postgres")
        conn = psycopg2.connect(**params)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(sql)
        row = cur.fetchall()
        cur.close()
        return row
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()


def call_ignition(sql):
    conn = None
    try:
        params = config(db="ignition")
        conn = psycopg2.connect(**params)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(sql)
        row = cur.fetchall()
        cur.close()
        return row
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()

def call_ignition_array(sql):
    conn = None
    try:
        params = config(db="ignition")
        conn = psycopg2.connect(**params)
        cur = conn.cursor()
        cur.execute(sql)
        row = cur.fetchall()
        cur.close()
        return row
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()


def update_ignition(sql):
    conn = None
    try:
        params = config(db="ignition")
        conn = psycopg2.connect(**params)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(sql)
        conn.commit()
        return "success"
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()


def update_ignition_with_values(sql, vals):
    conn = None
    print(vals)
    try:
        params = config(db="ignition")
        conn = psycopg2.connect(**params)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(sql, vals)
        conn.commit()
        return "success"
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()


def update_db(sql):
    conn = None
    try:
        params = config(db="postgres")
        conn = psycopg2.connect(**params)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(sql)
        conn.commit()
        return "success"
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()


def insert_many_with_df(db, df, table):
    # Create a list of tupples from the dataframe values
    params = config(db=db)
    conn = psycopg2.connect(**params)
    tuples = [tuple(x) for x in df.to_numpy()]
    # Comma-separated dataframe columns
    cols = ",".join(list(df.columns))
    num_cols = len(df.columns)
    # SQL query to execute
    # query  = "INSERT INTO %s(%s) VALUES(%%s,%%s,%%s,%%s,%%s,%%s,%%s,%%s,%%s,%%s,%%s,%%s,%%s,%%s)" % (table, cols)
    string = str(["%%s"] * num_cols)[1:-1].replace("'", "")
    query = f"INSERT INTO %s(%s) VALUES({string})" % (table, cols)
    cursor = conn.cursor()    
    try:
        cursor.executemany(query, tuples)
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print("Error: %s" % error)
        conn.rollback()
        cursor.close()
        return 1    
    cursor.close()
    return "success"


def upsert_many_with_df(db, df, table):
    # Create a list of tupples from the dataframe values
    params = config(db=db)
    conn = psycopg2.connect(**params)
    tuples = [tuple(x) for x in df.to_numpy()]
    # Comma-separated dataframe columns
    cols = ",".join(list(df.columns))
    num_cols = len(df.columns)
    # SQL query to execute
    # query  = "INSERT INTO %s(%s) VALUES(%%s,%%s,%%s,%%s,%%s,%%s,%%s,%%s,%%s,%%s,%%s,%%s,%%s,%%s)" % (table, cols)
    string = str(["%%s"] * num_cols)[1:-1].replace("'", "")
    query = f"INSERT INTO %s(%s) VALUES({string})" % (table, cols)
    cursor = conn.cursor()
    try:
        cursor.executemany(query, tuples)
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print("Error: %s" % error)
        conn.rollback()
        cursor.close()
        return 1
    cursor.close()
    return "success"

def force_cancel_rail(hk):
    close_rails_sql = f"""
        UPDATE manufacturing_orders SET order_status = 99
            WHERE reference_number = '{hk}'
                AND item_description LIKE '%RAIL%'
                AND order_status != (55)
            RETURNING order_number, order_status
        """
    return update_db(close_rails_sql)
