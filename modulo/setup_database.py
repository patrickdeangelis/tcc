import sqlite3

con = sqlite3.connect("database.db")
cur = con.cursor()
cur.execute("create table lang (name, first_appeared)")

con.close()

