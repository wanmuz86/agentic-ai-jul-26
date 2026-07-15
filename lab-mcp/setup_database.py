#code to create a mock database

import sqlite3

conn = sqlite3.connect("company.db")
cursor = conn.cursor()

cursor.execute("DROP TABLE IF EXISTS employees")

cursor.execute("""
CREATE TABLE employees (
   id INTEGER PRIMARY KEY,
   name TEXT NOT NULL,
   department TEXT NOT NULL,
   role TEXT NOT NULL,
   email TEXT NOT NULL,
   manager TEXT NOT NULL
)
""")

employees = [
   (1, "Ali Bin Ahmad", "Data Team", "Data Analyst", "ali@company.com", "Sarah Abdullah"),
   (2, "Sarah Abdullah", "HR", "HR Manager", "sarah@company.com", "Nora Hassan"),
   (3, "John Tan", "Finance", "Finance Executive", "john@company.com", "Lim Wei"),
   (4, "Aisyah Rahman", "IT", "System Engineer", "aisyah@company.com", "Sarah Abdullah"),
]

cursor.executemany("""
INSERT INTO employees (id, name, department, role, email, manager)
VALUES (?, ?, ?, ?, ?, ?)
""", employees)

conn.commit()
conn.close()

print("company.db created successfully.")
