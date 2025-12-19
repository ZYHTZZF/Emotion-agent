# storage.py
# storage.py
import os
import sqlite3
import json
import datetime
import hashlib

# 把数据库放在当前目录下的 data 子目录里
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

DB_PATH = os.path.join(DATA_DIR, "risk_history.db")


def _get_conn():
    return sqlite3.connect(DB_PATH)


def _hash_password(email: str, password: str) -> str:
    """
    非高强度，但够本地 demo 用：
    使用 email + ':' + password 做一层 SHA256。
    """
    combo = (email.strip().lower() + ":" + password).encode("utf-8")
    return hashlib.sha256(combo).hexdigest()


def init_db():
    conn = _get_conn()
    c = conn.cursor()

    # 用户表：保存邮箱 + 密码哈希
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            email TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL
        )
        """
    )

    # 评估记录表
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS risk_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL,
            ts TEXT NOT NULL,
            score REAL,
            level TEXT,
            neg_prob REAL,
            hit_words TEXT,
            answers TEXT
        )
        """
    )
    conn.commit()
    conn.close()


# 模块导入时自动建表
init_db()


def register_or_check_user(email: str, password: str):
    """
    登录/注册逻辑：

    - 如果 email 尚不存在：创建用户，返回 (created=True, password_ok=True)
    - 如果 email 已存在：
        · 密码正确 → (created=False, password_ok=True)
        · 密码错误 → (created=False, password_ok=False)
    """
    email = (email or "").strip().lower()
    password = (password or "").strip()
    if not email or not password:
        return False, False

    conn = _get_conn()
    c = conn.cursor()
    c.execute("SELECT password_hash FROM users WHERE email = ?", (email,))
    row = c.fetchone()

    if row is None:
        # 新用户：注册
        pwd_hash = _hash_password(email, password)
        c.execute(
            "INSERT INTO users (email, password_hash) VALUES (?, ?)",
            (email, pwd_hash),
        )
        conn.commit()
        conn.close()
        return True, True  # created, ok
    else:
        # 老用户：校验密码
        pwd_hash = _hash_password(email, password)
        ok = (row[0] == pwd_hash)
        conn.close()
        return False, ok  # not created, ok?


def save_risk_session(email: str, risk: dict, answers: list[str]):
    """
    保存一次评估结果：
    - email: 当前登录邮箱
    - risk: compute_risk_score 返回的 dict
    - answers: 每一题用户原始回答的列表
    """
    email = (email or "").strip().lower()
    if not email:
        return

    ts = datetime.datetime.now().isoformat(timespec="seconds")
    hit_words_json = json.dumps(risk.get("hit_words", []), ensure_ascii=False)
    answers_json = json.dumps(answers or [], ensure_ascii=False)

    conn = _get_conn()
    c = conn.cursor()
    c.execute(
        """
        INSERT INTO risk_sessions (email, ts, score, level, neg_prob, hit_words, answers)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            email,
            ts,
            risk.get("score"),
            risk.get("level"),
            risk.get("neg_prob"),
            hit_words_json,
            answers_json,
        ),
    )
    conn.commit()
    conn.close()


def list_risk_sessions(email: str):
    """
    列出某个用户的所有评估摘要（不含详细回答）：
    返回 list[dict]，按时间倒序。
    """
    email = (email or "").strip().lower()
    if not email:
        return []

    conn = _get_conn()
    c = conn.cursor()
    c.execute(
        """
        SELECT id, ts, score, level
        FROM risk_sessions
        WHERE email = ?
        ORDER BY ts DESC, id DESC
        """,
        (email,),
    )
    rows = c.fetchall()
    conn.close()

    result = []
    for r in rows:
        result.append(
            {
                "id": r[0],
                "ts": r[1],
                "score": r[2],
                "level": r[3],
            }
        )
    return result


def get_risk_session(email: str, session_id: int):
    """
    获取某个用户的一次完整评估详情：
    如果不是该用户的记录，返回 None。
    """
    email = (email or "").strip().lower()
    conn = _get_conn()
    c = conn.cursor()
    c.execute(
        """
        SELECT id, ts, score, level, neg_prob, hit_words, answers
        FROM risk_sessions
        WHERE email = ? AND id = ?
        """,
        (email, session_id),
    )
    row = c.fetchone()
    conn.close()

    if row is None:
        return None

    hit_words = []
    answers = []
    try:
        hit_words = json.loads(row[5]) if row[5] else []
    except Exception:
        pass
    try:
        answers = json.loads(row[6]) if row[6] else []
    except Exception:
        pass

    return {
        "id": row[0],
        "ts": row[1],
        "score": row[2],
        "level": row[3],
        "neg_prob": row[4],
        "hit_words": hit_words,
        "answers": answers,
    }
