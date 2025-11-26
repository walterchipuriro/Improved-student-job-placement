import sqlite3

DB_NAME = "users.db"  # You can change this name if needed


# --------------------------------------------------------
# 1. Initialize the database (create users table if missing)
# --------------------------------------------------------
def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT DEFAULT 'student'
        )
    """)

    conn.commit()
    conn.close()


# --------------------------------------------------------
# 2. Verify user credentials for login
# --------------------------------------------------------
def verify_user(username, password):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    user = cursor.fetchone()

    conn.close()
    return user  # Returns tuple if found, None if not



def register_user(username, password, role="student"):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    try:
        cursor.execute(
            "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
            (username, password, role)
        )
        conn.commit()
        return True, "Registration successful."

    except sqlite3.IntegrityError:
        return False, "User already exists."

    except Exception as e:
        return False, f"Registration failed: {str(e)}"

    finally:
        conn.close()
