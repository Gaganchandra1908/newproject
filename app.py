"""
Professional Streamlit Broadband Subscription Portal
Enhanced with modern UI/UX, advanced analytics, and comprehensive demo data
Features:
- Modern card-based UI design with enhanced user dashboard
- Semi-circular progress indicator for plan expiry
- ML-based plan recommendations
- Plan comparison functionality
- Expiry reminders system
- Detailed usage analytics
- Bulk plan upload for admins
- Comprehensive billing history
- Real-world sample data
Run:
pip install streamlit pandas scikit-learn numpy python-dateutil plotly joblib
streamlit run app.py
"""
import streamlit as st
import sqlite3
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import hashlib
import uuid
from dateutil import parser
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
import io
import math
# ML Model Imports
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# ---------------------------
# Configuration & Styling
# ---------------------------
DB_PATH = os.path.join(os.path.dirname(__file__), "broadband.db")
SALT = "broadband_demo_salt"
MOCK_DATA_CREATED_FLAG = "mock_data_created"
DB_MIGRATION_FLAG = "db_migrated_v3"

# Custom CSS for modern UI including semi-circular progress
def load_css():
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Custom Cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    .plan-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .plan-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px 0 rgba(0, 0, 0, 0.15);
        border-color: #667eea;
    }
    
    .recommended-plan {
        border: 2px solid #10b981;
        background: linear-gradient(135deg, #ecfdf5 0%, #f0fdf4 100%);
    }
    
    .current-plan-card {
        background: linear-gradient(135deg, #f8fafc 0%, #ffffff 100%);
        border-left: 4px solid #667eea;
        padding: 2rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* Expiry Warning */
    .expiry-warning {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border: 2px solid #f59e0b;
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .expiry-critical {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border: 2px solid #ef4444;
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Status Badges */
    .status-active {
        background-color: #10b981;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    
    .status-inactive {
        background-color: #f59e0b;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    
    .status-cancelled {
        background-color: #ef4444;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    
    .status-expired {
        background-color: #6b7280;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    
    /* Usage Progress Cards */
    .usage-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        border: 1px solid #e5e7eb;
    }
    
    .usage-exceeded {
        border-color: #ef4444;
        background: linear-gradient(135deg, #fef2f2 0%, #ffffff 100%);
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background-color: #f8fafc;
    }
    
    /* Button Styling */
    .stButton > button {
        border-radius: 8px;
        border: none;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px 0 rgba(102, 126, 234, 0.4);
    }
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        color: #1f2937;
    }
    
    /* Metrics */
    [data-testid="metric-container"] {
        background: white;
        border: 1px solid #e5e7eb;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
    }
    
    /* Comparison Table */
    .comparison-table {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* File Upload */
    .uploadedFile {
        border-radius: 8px;
        border: 2px dashed #d1d5db;
        padding: 1rem;
    }
    
    /* Alert Styles */
    .alert-info {
        background-color: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
    
    .alert-warning {
        background-color: #fefce8;
        border-left: 4px solid #eab308;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
    
    .alert-success {
        background-color: #f0fdf4;
        border-left: 4px solid #22c55e;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
    
    .alert-danger {
        background-color: #fef2f2;
        border-left: 4px solid #ef4444;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# ---------------------------
# Database Utilities
# ---------------------------
def safe_to_datetime(s, *, utc=True, drop_na=True):
    """Robust datetime conversion"""
    dt = pd.to_datetime(s, errors='coerce', utc=utc)
    try:
        dt = dt.dt.tz_convert(None)
    except Exception:
        try:
            dt = dt.dt.tz_localize(None)
        except Exception:
            pass
    if drop_na:
        dt = dt[~dt.isna()]
    return dt

def utcnow_naive():
    """Naive current UTC timestamp"""
    return pd.Timestamp.utcnow().tz_localize(None)

def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def exec_query(query, params=(), fetch=False):
    conn = get_conn()
    c = conn.cursor()
    c.execute(query, params)
    if fetch:
        rows = c.fetchall()
        conn.close()
        return rows
    conn.commit()
    conn.close()

def exec_query_safe(query, params=(), fetch=False):
    """Execute query with error handling for missing columns"""
    try:
        return exec_query(query, params, fetch)
    except sqlite3.OperationalError as e:
        if "no such column" in str(e):
            if fetch:
                return []
            return None
        raise e

def df_from_query(query, params=()):
    rows = exec_query(query, params, fetch=True)
    if not rows:
        return pd.DataFrame()
    cols = rows[0].keys()
    data = [tuple(r) for r in rows]
    return pd.DataFrame(data, columns=cols)

def row_to_dict(row):
    if row is None:
        return None
    return {k: row[k] for k in row.keys()}

def column_exists(table_name, column_name):
    """Check if a column exists in a table"""
    try:
        result = exec_query(f"PRAGMA table_info({table_name})", fetch=True)
        columns = [row[1] for row in result]
        return column_name in columns
    except:
        return False

def add_column_if_not_exists(table_name, column_name, column_type, default_value=None):
    """Add a column to a table if it doesn't exist"""
    if not column_exists(table_name, column_name):
        try:
            default_clause = f" DEFAULT {default_value}" if default_value is not None else ""
            exec_query(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}{default_clause}")
            return True
        except Exception as e:
            print(f"Error adding column {column_name} to {table_name}: {e}")
            return False
    return False

# ---------------------------
# Schema & Database Migration
# ---------------------------
def create_tables():
    conn = get_conn()
    c = conn.cursor()
    
    # Create base tables (original schema)
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            username TEXT UNIQUE,
            password_hash TEXT,
            role TEXT,
            name TEXT,
            email TEXT,
            address TEXT,
            phone TEXT,
            is_autopay_enabled INTEGER DEFAULT 0
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS plans (
            id INTEGER PRIMARY KEY,
            name TEXT,
            speed_mbps INTEGER,
            data_limit_gb REAL,
            price REAL,
            validity_days INTEGER,
            description TEXT
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS subscriptions (
            id INTEGER PRIMARY KEY,
            user_id INTEGER,
            plan_id INTEGER,
            start_date TEXT,
            end_date TEXT,
            status TEXT,
            auto_renew INTEGER DEFAULT 0,
            FOREIGN KEY(user_id) REFERENCES users(id),
            FOREIGN KEY(plan_id) REFERENCES plans(id)
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS payments (
            id INTEGER PRIMARY KEY,
            subscription_id INTEGER,
            user_id INTEGER,
            amount REAL,
            payment_date TEXT,
            status TEXT,
            bill_month INTEGER,
            bill_year INTEGER,
            FOREIGN KEY(subscription_id) REFERENCES subscriptions(id),
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS usage (
            id INTEGER PRIMARY KEY,
            user_id INTEGER,
            date TEXT,
            data_used_gb REAL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT)
    ''')
    
    # Create new tables for enhanced features
    c.execute('''
        CREATE TABLE IF NOT EXISTS plan_comparisons (
            id INTEGER PRIMARY KEY,
            user_id INTEGER,
            plan_ids TEXT,
            created_date TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    ''')
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS notifications (
            id INTEGER PRIMARY KEY,
            user_id INTEGER,
            message TEXT,
            notification_type TEXT,
            is_read INTEGER DEFAULT 0,
            created_date TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    ''')
    
    conn.commit()
    conn.close()

def migrate_database():
    """Add new columns to existing tables"""
    if meta_get(DB_MIGRATION_FLAG) == '1':
        return
    
    # Add new columns to users table
    add_column_if_not_exists('users', 'city', 'TEXT')
    add_column_if_not_exists('users', 'state', 'TEXT')
    add_column_if_not_exists('users', 'signup_date', 'TEXT')
    add_column_if_not_exists('users', 'last_login', 'TEXT')
    add_column_if_not_exists('users', 'notification_preferences', 'TEXT', "'email,sms'")
    
    # Add new columns to plans table
    add_column_if_not_exists('plans', 'plan_type', 'TEXT', "'basic'")
    add_column_if_not_exists('plans', 'is_unlimited', 'INTEGER', '0')
    add_column_if_not_exists('plans', 'created_date', 'TEXT')
    add_column_if_not_exists('plans', 'features', 'TEXT')
    add_column_if_not_exists('plans', 'upload_speed_mbps', 'INTEGER')
    
    # Add new columns to subscriptions table
    add_column_if_not_exists('subscriptions', 'created_date', 'TEXT')
    add_column_if_not_exists('subscriptions', 'cancelled_date', 'TEXT')
    add_column_if_not_exists('subscriptions', 'cancellation_reason', 'TEXT')
    add_column_if_not_exists('subscriptions', 'renewal_count', 'INTEGER', '0')
    
    # Add new columns to payments table
    add_column_if_not_exists('payments', 'payment_method', 'TEXT', "'credit_card'")
    add_column_if_not_exists('payments', 'late_fee', 'REAL', '0')
    add_column_if_not_exists('payments', 'discount', 'REAL', '0')
    add_column_if_not_exists('payments', 'tax_amount', 'REAL', '0')
    add_column_if_not_exists('payments', 'transaction_id', 'TEXT')
    
    # Add new columns to usage table
    add_column_if_not_exists('usage', 'peak_hour_usage', 'REAL')
    add_column_if_not_exists('usage', 'off_peak_usage', 'REAL')
    add_column_if_not_exists('usage', 'upload_usage', 'REAL')
    add_column_if_not_exists('usage', 'average_speed', 'REAL')
    
    # Create support tickets table
    try:
        conn = get_conn()
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS support_tickets (
                id INTEGER PRIMARY KEY,
                user_id INTEGER,
                subject TEXT,
                description TEXT,
                category TEXT,
                status TEXT,
                priority TEXT,
                created_date TEXT,
                resolved_date TEXT,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        ''')
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error creating support_tickets table: {e}")
    
    # Update existing users with signup_date if missing
    try:
        users_without_signup = exec_query(
            "SELECT id FROM users WHERE signup_date IS NULL OR signup_date = ''", 
            fetch=True
        )
        if users_without_signup:
            default_date = (datetime.utcnow() - timedelta(days=30)).isoformat()
            for user_row in users_without_signup:
                exec_query(
                    "UPDATE users SET signup_date = ? WHERE id = ?", 
                    (default_date, user_row[0])
                )
    except Exception as e:
        print(f"Error updating signup dates: {e}")
    
    meta_set(DB_MIGRATION_FLAG, '1')

def hash_password(password: str) -> str:
    salt = SALT + uuid.uuid4().hex
    h = hashlib.sha256((salt + password).encode()).hexdigest()
    return f"{salt}${h}"

def verify_password(password: str, stored: str) -> bool:
    try:
        salt, h = stored.split('$')
    except Exception:
        return False
    calc = hashlib.sha256((salt + password).encode()).hexdigest()
    return calc == h

def ensure_default_admin():
    r = exec_query("SELECT * FROM users WHERE username = ?", ("admin",), fetch=True)
    if len(r) == 0:
        pw = hash_password("admin123")
        signup_date = (datetime.utcnow() - timedelta(days=365)).isoformat()
        if column_exists('users', 'signup_date'):
            exec_query(
                "INSERT INTO users (username, password_hash, role, name, email, signup_date, city, state) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                ("admin", pw, "admin", "Administrator", "admin@example.com", signup_date, "Mumbai", "Maharashtra"),
            )
        else:
            exec_query(
                "INSERT INTO users (username, password_hash, role, name, email) VALUES (?, ?, ?, ?, ?)",
                ("admin", pw, "admin", "Administrator", "admin@example.com"),
            )

def meta_get(k):
    r = exec_query("SELECT v FROM meta WHERE k = ?", (k,), fetch=True)
    return r[0][0] if r else None

def meta_set(k, v):
    exec_query("INSERT OR REPLACE INTO meta (k, v) VALUES (?, ?)", (k, v))

def create_comprehensive_mock_data():
    """Create rich, realistic demo data for meaningful analytics"""
    if meta_get(MOCK_DATA_CREATED_FLAG) == '1':
        return
    
    # Enhanced plans with more realistic variety
    plans = [
        ("Basic Starter", 25, 50, 299, 30, "Perfect for light browsing and social media", "basic", 0, "Email Support", 5),
        ("Home Essential", 50, 100, 499, 30, "Great for small families", "basic", 0, "Phone Support, Basic Wi-Fi", 10),
        ("Family Connect", 100, 200, 699, 30, "Ideal for families with streaming", "standard", 0, "24/7 Support, Dual-band Wi-Fi", 20),
        ("Power User", 300, 500, 999, 30, "High-speed for gaming and streaming", "premium", 0, "Priority Support, Gaming Mode", 50),
        ("Pro Unlimited", 500, 1000, 1499, 30, "Professional use with high speeds", "premium", 1, "VIP Support, Static IP", 100),
        ("Unlimited Elite", 1000, 2000, 1999, 30, "Ultimate speed and data", "elite", 1, "Dedicated Support, Enterprise Features", 200),
        ("Student Special", 50, 75, 399, 30, "Affordable plan for students", "basic", 0, "Student Support, Study Mode", 10),
        ("Business Basic", 200, 300, 1299, 30, "Small business package", "premium", 0, "Business Support, Fixed IP", 40),
        ("Enterprise", 1500, 5000, 2999, 30, "Large business solution", "elite", 1, "Enterprise Support, SLA", 300),
        ("Gaming Pro", 800, 1500, 1799, 30, "Optimized for gaming", "premium", 0, "Gaming Support, Low Latency", 150),
    ]
    
    today = datetime.utcnow().date()
    for i, p in enumerate(plans):
        created_date = (today - timedelta(days=300 - i*20)).isoformat()
        if column_exists('plans', 'plan_type') and column_exists('plans', 'features') and column_exists('plans', 'upload_speed_mbps'):
            exec_query(
                "INSERT INTO plans (name, speed_mbps, data_limit_gb, price, validity_days, description, plan_type, is_unlimited, created_date, features, upload_speed_mbps) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (*p, created_date),
            )
        else:
            exec_query(
                "INSERT INTO plans (name, speed_mbps, data_limit_gb, price, validity_days, description) VALUES (?, ?, ?, ?, ?, ?)",
                p[:6],
            )
    
    # Create diverse user base with realistic patterns
    cities = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata", "Hyderabad", "Pune", "Ahmedabad", "Jaipur", "Lucknow"]
    states = ["Maharashtra", "Delhi", "Karnataka", "Tamil Nadu", "West Bengal", "Telangana", "Maharashtra", "Gujarat", "Rajasthan", "Uttar Pradesh"]
    
    # Define user profiles with different behavior patterns
    user_profiles = [
        ("Professional", "heavy", "reliable", "low", 0.8),
        ("Family", "moderate", "reliable", "medium", 0.7),
        ("Student", "moderate", "unreliable", "low", 0.9),
        ("Senior", "light", "reliable", "high", 0.3),
        ("Gamer", "heavy", "reliable", "medium", 0.85),
        ("Remote Worker", "heavy", "reliable", "low", 0.75),
        ("Casual User", "light", "unreliable", "low", 0.4),
        ("Streamer", "heavy", "reliable", "medium", 0.9),
        ("Small Business", "heavy", "reliable", "medium", 0.6),
        ("Tech Enthusiast", "heavy", "reliable", "low", 0.95),
    ]
    
    users_data = []
    for i in range(100):  # Increased to 100 users for more realistic data
        username = f"user{i+1:03d}"
        profile = user_profiles[i % len(user_profiles)]
        name = f"{profile[0]} User {i+1}"
        email = f"user{i+1:03d}@example.com"
        city = cities[i % len(cities)]
        state = states[i % len(states)]
        # Stagger signup dates over past 2 years
        signup_days_ago = np.random.randint(1, 730)
        signup_date = (today - timedelta(days=signup_days_ago)).isoformat()
        users_data.append((username, name, email, city, state, signup_date, profile))
    
    for udata in users_data:
        pw = hash_password("password")
        existing = exec_query("SELECT id FROM users WHERE username = ?", (udata[0],), fetch=True)
        if not existing:
            if column_exists('users', 'city') and column_exists('users', 'signup_date'):
                exec_query(
                    "INSERT INTO users (username, password_hash, role, name, email, city, state, signup_date, is_autopay_enabled, notification_preferences) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (udata[0], pw, "user", udata[1], udata[2], udata[3], udata[4], udata[5], 1 if udata[6][1] == "reliable" else 0, "email,sms" if udata[6][1] == "reliable" else "email"),
                )
            else:
                exec_query(
                    "INSERT INTO users (username, password_hash, role, name, email, is_autopay_enabled) VALUES (?, ?, ?, ?, ?, ?)",
                    (udata[0], pw, "user", udata[1], udata[2], 1 if udata[6][1] == "reliable" else 0),
                )
    
    # Get plan and user IDs
    plan_rows = exec_query("SELECT id, price, data_limit_gb FROM plans", fetch=True)
    plan_data = [(r[0], r[1], r[2]) for r in plan_rows]
    user_rows = exec_query("SELECT id FROM users WHERE role = 'user'", fetch=True)
    user_data = [(r[0],) for r in user_rows]
    
    # Create realistic subscription and usage history
    for uid_tuple in user_data:
        uid = uid_tuple[0]
        user_profile_data = exec_query("SELECT city, state, signup_date FROM users WHERE id = ?", (uid,), fetch=True)
        if user_profile_data:
            city, state, signup_date = user_profile_data[0]
            profile_index = int(uid) % len(user_profiles)
            usage_pattern = user_profiles[profile_index][1]
            payment_pattern = user_profiles[profile_index][2]
            support_pattern = user_profiles[profile_index][3]
            tech_savviness = user_profiles[profile_index][4]
            
            # Create multiple subscriptions per user (subscription history)
            num_subscriptions = np.random.randint(1, 5)  # 1-4 subscriptions
            subscription_start = datetime.fromisoformat(signup_date)
            
            for sub_idx in range(num_subscriptions):
                # Choose plan based on usage pattern and tech savviness
                if usage_pattern == "light":
                    suitable_plans = [p for p in plan_data if p[2] <= 200]  # Up to 200GB
                elif usage_pattern == "moderate":
                    suitable_plans = [p for p in plan_data if 100 <= p[2] <= 1000]  # 100-1000GB
                else:  # heavy
                    suitable_plans = [p for p in plan_data if p[2] >= 300]  # 300GB+
                
                if not suitable_plans:
                    suitable_plans = plan_data
                
                plan_id, plan_price, plan_limit = random.choice(suitable_plans)
                
                # Determine subscription dates
                if sub_idx == 0:
                    start_date = subscription_start + timedelta(days=np.random.randint(1, 7))
                else:
                    start_date = previous_end + timedelta(days=np.random.randint(1, 30))
                
                duration_days = np.random.randint(28, 35) if payment_pattern == "reliable" else np.random.randint(15, 32)
                end_date = start_date + timedelta(days=duration_days)
                previous_end = end_date
                
                # Determine status
                if end_date > datetime.utcnow():
                    status = 'active'
                elif sub_idx == num_subscriptions - 1 and end_date <= datetime.utcnow():
                    status = 'expired'
                else:
                    status = np.random.choice(['cancelled', 'expired'], p=[0.3, 0.7])
                
                # Create subscription
                renewal_count = max(0, sub_idx)
                if column_exists('subscriptions', 'created_date') and column_exists('subscriptions', 'renewal_count'):
                    exec_query(
                        "INSERT INTO subscriptions (user_id, plan_id, start_date, end_date, status, auto_renew, created_date, renewal_count) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (uid, plan_id, start_date.isoformat(), end_date.isoformat(), status, 
                         1 if payment_pattern == "reliable" else 0, start_date.isoformat(), renewal_count),
                    )
                else:
                    exec_query(
                        "INSERT INTO subscriptions (user_id, plan_id, start_date, end_date, status, auto_renew) VALUES (?, ?, ?, ?, ?, ?)",
                        (uid, plan_id, start_date.isoformat(), end_date.isoformat(), status, 
                         1 if payment_pattern == "reliable" else 0),
                    )
                
                sub_id = exec_query("SELECT last_insert_rowid()", fetch=True)[0][0]
                
                # Create multiple payments for this subscription (monthly billing)
                current_payment_date = start_date
                while current_payment_date < end_date:
                    payment_status = 'paid' if payment_pattern == "reliable" or np.random.random() < 0.85 else 'failed'
                    payment_method = np.random.choice(['credit_card', 'debit_card', 'upi', 'net_banking'], p=[0.35, 0.25, 0.3, 0.1])
                    
                    # Calculate taxes and discounts
                    base_amount = plan_price
                    tax_amount = base_amount * 0.18  # 18% GST
                    discount = 0
                    if renewal_count > 0:  # Loyalty discount
                        discount = base_amount * 0.05
                    
                    total_amount = base_amount + tax_amount - discount
                    
                    if column_exists('payments', 'payment_method') and column_exists('payments', 'tax_amount'):
                        exec_query(
                            "INSERT INTO payments (subscription_id, user_id, amount, payment_date, status, payment_method, bill_month, bill_year, tax_amount, discount, transaction_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            (sub_id, uid, total_amount, current_payment_date.isoformat(), payment_status, payment_method, 
                             current_payment_date.month, current_payment_date.year, tax_amount, discount, f"TXN{uuid.uuid4().hex[:8].upper()}"),
                        )
                    else:
                        exec_query(
                            "INSERT INTO payments (subscription_id, user_id, amount, payment_date, status, bill_month, bill_year) VALUES (?, ?, ?, ?, ?, ?, ?)",
                            (sub_id, uid, base_amount, current_payment_date.isoformat(), payment_status, 
                             current_payment_date.month, current_payment_date.year),
                        )
                    
                    current_payment_date += timedelta(days=30)
                
                # Create usage data for this subscription period
                if status in ['active', 'expired']:
                    usage_days = min((datetime.utcnow().date() - start_date.date()).days, 
                                   (end_date.date() - start_date.date()).days)
                    
                    # Set base usage based on pattern and plan
                    if usage_pattern == "light":
                        base_daily = np.random.uniform(0.5, 2.0) * tech_savviness
                    elif usage_pattern == "moderate":
                        base_daily = np.random.uniform(2.0, 6.0) * tech_savviness
                    else:  # heavy
                        base_daily = np.random.uniform(6.0, 15.0) * tech_savviness
                    
                    for day_offset in range(usage_days):
                        usage_date = (start_date + timedelta(days=day_offset)).date()
                        if usage_date <= datetime.utcnow().date():
                            
                            # Add weekly patterns (higher usage on weekends)
                            weekend_factor = 1.4 if usage_date.weekday() >= 5 else 1.0
                            
                            # Add monthly patterns (higher usage mid-month)
                            month_factor = 1.2 if 10 <= usage_date.day <= 20 else 0.9
                            
                            # Add occasional spikes
                            spike_factor = np.random.uniform(2.0, 4.0) if np.random.random() < 0.08 else 1.0
                            
                            daily_usage = np.clip(
                                np.random.normal(base_daily * weekend_factor * month_factor * spike_factor, 
                                               base_daily * 0.3),
                                0.1, plan_limit * 0.8  # Cap at 80% of plan limit per day
                            )
                            
                            # Split into peak and off-peak hours
                            peak_usage = daily_usage * np.random.uniform(0.6, 0.8)
                            off_peak_usage = daily_usage - peak_usage
                            upload_usage = daily_usage * np.random.uniform(0.1, 0.3)
                            avg_speed = plan_limit * np.random.uniform(0.7, 0.95)  # 70-95% of advertised speed
                            
                            if column_exists('usage', 'peak_hour_usage') and column_exists('usage', 'upload_usage'):
                                exec_query(
                                    "INSERT INTO usage (user_id, date, data_used_gb, peak_hour_usage, off_peak_usage, upload_usage, average_speed) VALUES (?, ?, ?, ?, ?, ?, ?)",
                                    (uid, usage_date.isoformat(), daily_usage, peak_usage, off_peak_usage, upload_usage, avg_speed),
                                )
                            else:
                                exec_query(
                                    "INSERT INTO usage (user_id, date, data_used_gb) VALUES (?, ?, ?)",
                                    (uid, usage_date.isoformat(), daily_usage),
                                )
                
                # Create support tickets based on support pattern
                if support_pattern == "high" or (support_pattern == "medium" and np.random.random() < 0.4):
                    num_tickets = np.random.randint(1, 3) if support_pattern == "high" else 1
                    for _ in range(num_tickets):
                        ticket_date = start_date + timedelta(days=np.random.randint(1, min(30, (end_date - start_date).days)))
                        categories = ['billing', 'technical', 'service', 'plan_change', 'connection_issue', 'speed_complaint']
                        ticket_category = np.random.choice(categories)
                        ticket_priority = np.random.choice(['low', 'medium', 'high'], p=[0.5, 0.3, 0.2])
                        ticket_status = np.random.choice(['resolved', 'closed'], p=[0.8, 0.2])
                        
                        subjects = {
                            'billing': ['Billing inquiry', 'Payment issue', 'Invoice clarification'],
                            'technical': ['Connection problem', 'Speed issue', 'Equipment malfunction'],
                            'service': ['Service interruption', 'Installation query', 'Account update'],
                            'plan_change': ['Plan upgrade request', 'Plan downgrade', 'Plan comparison'],
                            'connection_issue': ['No internet', 'Frequent disconnection', 'Slow connection'],
                            'speed_complaint': ['Speed not as promised', 'Slow during peak hours', 'Upload speed issue']
                        }
                        
                        subject = np.random.choice(subjects[ticket_category])
                        
                        if column_exists('support_tickets', 'created_date'):
                            exec_query(
                                "INSERT INTO support_tickets (user_id, subject, description, category, status, priority, created_date, resolved_date) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                                (uid, subject, f"Customer reported issue with {ticket_category}", ticket_category, 
                                 ticket_status, ticket_priority, ticket_date.isoformat(), 
                                 (ticket_date + timedelta(days=np.random.randint(1, 7))).isoformat() if ticket_status == 'resolved' else None),
                            )
    
    # Create notifications for users (expiry reminders, etc.)
    active_subscriptions = exec_query(
        "SELECT s.*, u.notification_preferences FROM subscriptions s JOIN users u ON s.user_id = u.id WHERE s.status = 'active'",
        fetch=True
    )
    
    for sub in active_subscriptions:
        end_date = datetime.fromisoformat(sub[4])  # end_date is index 4
        days_until_expiry = (end_date.date() - datetime.utcnow().date()).days
        
        # Create expiry reminder notifications
        if 1 <= days_until_expiry <= 7:
            notification_message = f"Your broadband plan expires in {days_until_expiry} day{'s' if days_until_expiry > 1 else ''}. Renew now to avoid interruption."
            
            if column_exists('notifications', 'created_date'):
                exec_query(
                    "INSERT INTO notifications (user_id, message, notification_type, created_date) VALUES (?, ?, ?, ?)",
                    (sub[1], notification_message, 'expiry_reminder', datetime.utcnow().isoformat()),  # user_id is index 1
                )
    
    meta_set(MOCK_DATA_CREATED_FLAG, '1')


def generate_usage_for_user(user_id, days=60):
    """Generate random usage data for a given user"""
    conn = get_conn()
    cur = conn.cursor()

    today = datetime.today().date()
    for i in range(days):
        date = today - timedelta(days=i)
        data_used = round(random.uniform(1, 10), 2)  # 1–10 GB/day
        peak = round(data_used * random.uniform(0.5, 0.8), 2)
        off_peak = round(data_used - peak, 2)
        upload = round(data_used * random.uniform(0.1, 0.3), 2)
        avg_speed = round(random.uniform(20, 100), 2)  # Mbps

        cur.execute("""
            INSERT INTO usage (user_id, date, data_used_gb, peak_hour_usage, off_peak_usage, upload_usage, average_speed)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (user_id, date.isoformat(), data_used, peak, off_peak, upload, avg_speed))

    conn.commit()
    conn.close()


def populate_usage_for_all_users(days=60):
    """Populate usage for all users if they don't already have data"""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE role='user'")
    user_ids = [row[0] for row in cur.fetchall()]
    conn.close()

    for uid in user_ids:
        existing = exec_query("SELECT COUNT(*) FROM usage WHERE user_id = ?", (uid,), fetch=True)[0][0]
        if existing == 0:  # Only populate if no usage exists
            generate_usage_for_user(uid, days)
            print(f"✅ Inserted {days} days of usage for user {uid}")


# ---------------------------
# Business Logic
# ---------------------------
def signup(username, password, name, email):
    try:
        pw = hash_password(password)
        signup_date = utcnow_naive().isoformat()
        if column_exists('users', 'signup_date'):
            exec_query(
                "INSERT INTO users (username, password_hash, role, name, email, signup_date, city, state) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (username, pw, 'user', name, email, signup_date, 'Mumbai', 'Maharashtra'),
            )
        else:
            exec_query(
                "INSERT INTO users (username, password_hash, role, name, email) VALUES (?, ?, ?, ?, ?)",
                (username, pw, 'user', name, email),
            )
        return True, "User created successfully"
    except Exception as e:
        return False, str(e)

def signin(username, password):
    r = exec_query("SELECT * FROM users WHERE username = ?", (username,), fetch=True)
    if not r:
        return False, "No such user"
    row = r[0]
    if verify_password(password, row[2]):
        # Update last login if column exists
        if column_exists('users', 'last_login'):
            exec_query("UPDATE users SET last_login = ? WHERE id = ?", (utcnow_naive().isoformat(), row[0]))
        return True, row_to_dict(row)
    return False, "Invalid credentials"

def get_user_by_id(uid):
    r = exec_query("SELECT * FROM users WHERE id = ?", (uid,), fetch=True)
    return row_to_dict(r[0]) if r else None

def get_all_plans():
    rows = exec_query("SELECT * FROM plans ORDER BY price ASC", fetch=True)
    return [row_to_dict(r) for r in rows]

def get_plan(plan_id):
    r = exec_query("SELECT * FROM plans WHERE id = ?", (plan_id,), fetch=True)
    return row_to_dict(r[0]) if r else None

def get_user_active_subscription(user_id):
    r = exec_query(
        "SELECT s.*, p.name AS plan_name, p.data_limit_gb, p.price FROM subscriptions s JOIN plans p ON s.plan_id = p.id WHERE s.user_id = ? AND s.status = 'active' ORDER BY s.start_date DESC LIMIT 1",
        (user_id,),
        fetch=True,
    )
    return row_to_dict(r[0]) if r else None

def get_user_subscription_history(user_id):
    """Get all subscription history for a user"""
    query = """
        SELECT s.*, p.name AS plan_name, p.data_limit_gb, p.price, p.speed_mbps
        FROM subscriptions s 
        JOIN plans p ON s.plan_id = p.id 
        WHERE s.user_id = ? 
        ORDER BY s.start_date DESC
    """
    rows = exec_query(query, (user_id,), fetch=True)
    return [row_to_dict(r) for r in rows]

def subscribe_user_to_plan(user_id, plan_id, auto_renew=1):
    # Cancel any existing active subscription
    exec_query("UPDATE subscriptions SET status = 'cancelled' WHERE user_id = ? AND status = 'active'", (user_id,))
    
    today = datetime.utcnow().date()
    plan = get_plan(plan_id)
    end = today + timedelta(days=plan['validity_days'])
    
    if column_exists('subscriptions', 'created_date'):
        exec_query(
            "INSERT INTO subscriptions (user_id, plan_id, start_date, end_date, status, auto_renew, created_date) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (user_id, plan_id, today.isoformat(), end.isoformat(), 'active', auto_renew, utcnow_naive().isoformat()),
        )
    else:
        exec_query(
            "INSERT INTO subscriptions (user_id, plan_id, start_date, end_date, status, auto_renew) VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, plan_id, today.isoformat(), end.isoformat(), 'active', auto_renew),
        )

def create_payment(subscription_id, user_id, amount, status='paid', payment_method='credit_card'):
    now = utcnow_naive()
    tax_amount = amount * 0.18
    total_amount = amount + tax_amount
    
    if column_exists('payments', 'payment_method') and column_exists('payments', 'tax_amount'):
        exec_query(
            "INSERT INTO payments (subscription_id, user_id, amount, payment_date, status, payment_method, bill_month, bill_year, tax_amount, transaction_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (subscription_id, user_id, total_amount, now.isoformat(), status, payment_method, now.month, now.year, tax_amount, f"TXN{uuid.uuid4().hex[:8].upper()}"),
        )
    else:
        exec_query(
            "INSERT INTO payments (subscription_id, user_id, amount, payment_date, status, bill_month, bill_year) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (subscription_id, user_id, amount, now.isoformat(), status, now.month, now.year),
        )

def get_usage_for_user(user_id, days=30):
    query = "SELECT date, data_used_gb"
    if column_exists('usage', 'peak_hour_usage'):
        query += ", peak_hour_usage, off_peak_usage"
    if column_exists('usage', 'upload_usage'):
        query += ", upload_usage, average_speed"
    query += f" FROM usage WHERE user_id = ? ORDER BY date DESC LIMIT {days}"
    
    rows = exec_query(query, (user_id,), fetch=True)
    if not rows:
        return pd.DataFrame(columns=['date', 'data_used_gb'])
    cols = rows[0].keys()
    data = [tuple(r) for r in rows]
    return pd.DataFrame(data, columns=cols)

def get_user_notifications(user_id, limit=10):
    """Get recent notifications for a user"""
    if not column_exists('notifications', 'created_date'):
        return []
    
    rows = exec_query(
        "SELECT * FROM notifications WHERE user_id = ? ORDER BY created_date DESC LIMIT ?",
        (user_id, limit),
        fetch=True
    )
    return [row_to_dict(r) for r in rows]

def mark_notification_read(notification_id):
    """Mark a notification as read"""
    if column_exists('notifications', 'is_read'):
        exec_query("UPDATE notifications SET is_read = 1 WHERE id = ?", (notification_id,))

def check_expiry_reminders(user_id):
    """Check if user has any expiry reminders"""
    subscription = get_user_active_subscription(user_id)
    if not subscription:
        return []
    
    try:
        end_date = datetime.fromisoformat(subscription['end_date'])
        days_until_expiry = (end_date.date() - datetime.utcnow().date()).days
        
        reminders = []
        if days_until_expiry <= 7 and days_until_expiry > 0:
            reminders.append({
                'type': 'warning',
                'message': f"Your plan expires in {days_until_expiry} day{'s' if days_until_expiry > 1 else ''}!",
                'days': days_until_expiry
            })
        elif days_until_expiry <= 0:
            reminders.append({
                'type': 'critical',
                'message': "Your plan has expired!",
                'days': days_until_expiry
            })
        
        return reminders
    except:
        return []

def save_plan_comparison(user_id, plan_ids):
    """Save plan comparison for user"""
    if column_exists('plan_comparisons', 'created_date'):
        exec_query(
            "INSERT INTO plan_comparisons (user_id, plan_ids, created_date) VALUES (?, ?, ?)",
            (user_id, ','.join(map(str, plan_ids)), utcnow_naive().isoformat())
        )

def get_plan_comparison_history(user_id, limit=5):
    """Get recent plan comparisons for user"""
    if not column_exists('plan_comparisons', 'created_date'):
        return []
    
    rows = exec_query(
        "SELECT * FROM plan_comparisons WHERE user_id = ? ORDER BY created_date DESC LIMIT ?",
        (user_id, limit),
        fetch=True
    )
    return [row_to_dict(r) for r in rows]

def bulk_create_plans_from_csv(csv_data):
    """Create plans from CSV data"""
    try:
        df = pd.read_csv(io.StringIO(csv_data))
        
        # Expected columns: name, speed_mbps, data_limit_gb, price, validity_days, description, plan_type, features, upload_speed_mbps
        required_columns = ['name', 'speed_mbps', 'data_limit_gb', 'price', 'validity_days', 'description']
        
        for col in required_columns:
            if col not in df.columns:
                return False, f"Missing required column: {col}"
        
        created_count = 0
        for _, row in df.iterrows():
            try:
                # Check if plan already exists
                existing = exec_query("SELECT id FROM plans WHERE name = ?", (row['name'],), fetch=True)
                if existing:
                    continue
                
                # Set default values for optional columns
                plan_type = row.get('plan_type', 'standard')
                features = row.get('features', '')
                upload_speed = row.get('upload_speed_mbps', row['speed_mbps'] // 10)
                is_unlimited = 1 if row.get('is_unlimited', '').lower() in ['true', '1', 'yes'] else 0
                
                if column_exists('plans', 'plan_type') and column_exists('plans', 'features'):
                    exec_query(
                        "INSERT INTO plans (name, speed_mbps, data_limit_gb, price, validity_days, description, plan_type, is_unlimited, features, upload_speed_mbps, created_date) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (row['name'], row['speed_mbps'], row['data_limit_gb'], row['price'], 
                         row['validity_days'], row['description'], plan_type, is_unlimited, 
                         features, upload_speed, utcnow_naive().isoformat())
                    )
                else:
                    exec_query(
                        "INSERT INTO plans (name, speed_mbps, data_limit_gb, price, validity_days, description) VALUES (?, ?, ?, ?, ?, ?)",
                        (row['name'], row['speed_mbps'], row['data_limit_gb'], row['price'], row['validity_days'], row['description'])
                    )
                
                created_count += 1
                
            except Exception as e:
                print(f"Error creating plan from row: {e}")
                continue
        
        return True, f"Successfully created {created_count} plans"
        
    except Exception as e:
        return False, f"Error processing CSV: {str(e)}"

# ---------------------------
# ML Model Functions (Enhanced)
# ---------------------------
def collect_training_data():
    """Collect data for training the recommendation model"""
    query = """
    SELECT 
        u.id as user_id,
        u.city,
        u.state,
        u.signup_date,
        s.id as subscription_id,
        s.start_date,
        s.end_date,
        p.id as plan_id,
        p.name as plan_name,
        p.plan_type,
        p.speed_mbps,
        p.data_limit_gb,
        p.price
    FROM subscriptions s
    JOIN users u ON s.user_id = u.id
    JOIN plans p ON s.plan_id = p.id
    WHERE s.status IN ('active', 'expired')
    """
    
    subscriptions_df = df_from_query(query)
    
    # Get usage features
    usage_features = []
    for _, row in subscriptions_df.iterrows():
        user_id = row['user_id']
        usage_df = get_usage_for_user(user_id, days=90)
        
        if not usage_df.empty:
            avg_daily = usage_df['data_used_gb'].mean()
            max_daily = usage_df['data_used_gb'].max()
            std_daily = usage_df['data_used_gb'].std()
            total_monthly = usage_df['data_used_gb'].sum() * (30/len(usage_df))
            
            # Calculate usage patterns
            weekday_usage = 0
            weekend_usage = 0
            if 'date' in usage_df.columns:
                usage_df['date'] = pd.to_datetime(usage_df['date'])
                usage_df['weekday'] = usage_df['date'].dt.weekday
                weekday_usage = usage_df[usage_df['weekday'] < 5]['data_used_gb'].mean()
                weekend_usage = usage_df[usage_df['weekday'] >= 5]['data_used_gb'].mean()
            
            usage_features.append({
                'user_id': user_id,
                'avg_daily_usage': avg_daily,
                'max_daily_usage': max_daily,
                'usage_std': std_daily,
                'estimated_monthly_usage': total_monthly,
                'weekday_avg': weekday_usage,
                'weekend_avg': weekend_usage,
                'usage_consistency': 1 - (std_daily / avg_daily if avg_daily > 0 else 0)
            })
    
    if usage_features:
        usage_df = pd.DataFrame(usage_features)
        training_data = pd.merge(subscriptions_df, usage_df, on='user_id', how='left')
    else:
        training_data = subscriptions_df.copy()
        for col in ['avg_daily_usage', 'max_daily_usage', 'usage_std', 'estimated_monthly_usage', 
                   'weekday_avg', 'weekend_avg', 'usage_consistency']:
            training_data[col] = 0
    
    # Fill missing values
    numeric_cols = ['avg_daily_usage', 'max_daily_usage', 'usage_std', 'estimated_monthly_usage', 
                   'weekday_avg', 'weekend_avg', 'usage_consistency']
    for col in numeric_cols:
        training_data[col].fillna(0, inplace=True)
    
    return training_data

def engineer_features(df):
    """Create features for the ML model"""
    df = df.copy()
    
    # Handle date columns
    date_columns = ['signup_date', 'start_date']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Calculate user tenure
    if 'signup_date' in df.columns:
        current_date = pd.Timestamp.now()
        df['days_since_signup'] = (current_date - df['signup_date']).dt.days
    else:
        df['days_since_signup'] = 0
    
    # Create usage-based features
    df['usage_to_limit_ratio'] = df['estimated_monthly_usage'] / df['data_limit_gb']
    df['price_per_gb'] = df['price'] / df['data_limit_gb']
    df['speed_efficiency'] = df['speed_mbps'] / df['price']
    df['weekend_weekday_ratio'] = df['weekend_avg'] / (df['weekday_avg'] + 0.001)  # Avoid division by zero
    
    # Categorize users based on usage patterns
    def categorize_user(row):
        monthly_usage = row['estimated_monthly_usage']
        consistency = row['usage_consistency']
        
        if monthly_usage < 50:
            return 'light'
        elif monthly_usage < 200:
            return 'moderate'
        elif monthly_usage < 500:
            return 'heavy'
        else:
            return 'extreme'
    
    df['usage_category'] = df.apply(categorize_user, axis=1)
    
    # Set target variable
    if 'plan_type' in df.columns:
        df['plan_category'] = df['plan_type']
    else:
        df['plan_category'] = 'basic'
    
    return df

def train_recommendation_model():
    """Train the ML model for plan recommendation"""
    training_data = collect_training_data()
    if training_data.empty:
        st.error("Not enough data to train the model")
        return None
    
    training_data = engineer_features(training_data)
    
    # Define features and target
    feature_columns = [
        'avg_daily_usage', 'max_daily_usage', 'usage_std',
        'estimated_monthly_usage', 'days_since_signup',
        'weekday_avg', 'weekend_avg', 'usage_consistency',
        'city', 'state'
    ]
    
    target_column = 'plan_category'
    
    X = training_data[feature_columns]
    y = training_data[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Preprocessing
    numeric_features = ['avg_daily_usage', 'max_daily_usage', 'usage_std', 
                      'estimated_monthly_usage', 'days_since_signup',
                      'weekday_avg', 'weekend_avg', 'usage_consistency']
    categorical_features = ['city', 'state']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    # Create model pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=150, random_state=42, max_depth=10))
    ])
    
    # Train model
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("Training enhanced recommendation model...")
    progress_bar.progress(25)
    
    model.fit(X_train, y_train)
    progress_bar.progress(75)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_percent = accuracy * 100
    
    progress_bar.progress(100)
    status_text.text("Model training completed!")
    
    st.success(f"Model accuracy: {accuracy_percent:.1f}%")
    
    # Save model
    joblib.dump(model, 'plan_recommendation_model.pkl')
    return model

def ml_recommendation_for_user(user_id, num_recommendations=3):
    """Enhanced ML-based plan recommendation"""
    if not os.path.exists('plan_recommendation_model.pkl'):
        return advanced_recommendation_for_user(user_id, num_recommendations)
    
    model = joblib.load('plan_recommendation_model.pkl')
    user = get_user_by_id(user_id)
    if not user:
        return []
    
    usage_df = get_usage_for_user(user_id, days=90)
    
    # Prepare features
    if not usage_df.empty:
        avg_daily = usage_df['data_used_gb'].mean()
        max_daily = usage_df['data_used_gb'].max()
        std_daily = usage_df['data_used_gb'].std()
        total_monthly = usage_df['data_used_gb'].sum() * (30/len(usage_df))
        
        # Calculate weekday/weekend patterns
        if len(usage_df) > 7:  # Enough data for pattern analysis
            usage_df['date'] = pd.to_datetime(usage_df['date'])
            usage_df['weekday'] = usage_df['date'].dt.weekday
            weekday_avg = usage_df[usage_df['weekday'] < 5]['data_used_gb'].mean()
            weekend_avg = usage_df[usage_df['weekday'] >= 5]['data_used_gb'].mean()
        else:
            weekday_avg = avg_daily
            weekend_avg = avg_daily
        
        usage_consistency = 1 - (std_daily / avg_daily if avg_daily > 0 else 0)
    else:
        avg_daily = max_daily = std_daily = total_monthly = 0
        weekday_avg = weekend_avg = usage_consistency = 0
    
    # Calculate user tenure
    signup_date = pd.to_datetime(user.get('signup_date', pd.Timestamp.now()))
    days_since_signup = (pd.Timestamp.now() - signup_date).days
    
    # Create feature dataframe
    features = pd.DataFrame({
        'avg_daily_usage': [avg_daily],
        'max_daily_usage': [max_daily],
        'usage_std': [std_daily],
        'estimated_monthly_usage': [total_monthly],
        'days_since_signup': [days_since_signup],
        'weekday_avg': [weekday_avg],
        'weekend_avg': [weekend_avg],
        'usage_consistency': [usage_consistency],
        'city': [user.get('city', 'Unknown')],
        'state': [user.get('state', 'Unknown')]
    })
    
    # Predict plan category
    predicted_category = model.predict(features)[0]
    
    # Get plans and score them
    all_plans = get_all_plans()
    scored_plans = []
    
    for plan in all_plans:
        score = 0
        
        # Category match bonus
        if plan.get('plan_type', 'basic') == predicted_category:
            score += 40
        
        # Usage suitability
        if total_monthly > 0:
            if plan['data_limit_gb'] >= total_monthly * 1.2:  # 20% buffer
                score += 30
            elif plan['data_limit_gb'] >= total_monthly:
                score += 20
            else:
                score += 10 * (plan['data_limit_gb'] / total_monthly)
        
        # Price efficiency
        price_per_gb = plan['price'] / plan['data_limit_gb']
        if price_per_gb < 5:  # Good value
            score += 20
        elif price_per_gb < 10:
            score += 10
        
        # Speed adequacy
        if plan['speed_mbps'] >= max_daily * 10:  # 10x daily usage as speed
            score += 10
        
        scored_plans.append((plan, score))
    
    # Sort by score and return top recommendations
    scored_plans.sort(key=lambda x: x[1], reverse=True)
    return [plan for plan, score in scored_plans[:num_recommendations]]

def advanced_recommendation_for_user(user_id, num_recommendations=3):
    """Enhanced rule-based recommendation engine"""
    usage_df = get_usage_for_user(user_id, days=60)
    plans = get_all_plans()
    
    if usage_df.empty:
        # For new users, recommend based on popular/starter plans
        return sorted(plans, key=lambda x: x['price'])[:num_recommendations]
    
    usage_df['data_used_gb'] = pd.to_numeric(usage_df['data_used_gb'], errors='coerce')
    
    # Calculate usage statistics
    avg_daily = usage_df['data_used_gb'].dropna().mean()
    max_daily = usage_df['data_used_gb'].dropna().max()
    std_daily = usage_df['data_used_gb'].dropna().std()
    
    # Estimate monthly need with growth factor
    monthly_est = avg_daily * 30
    peak_monthly_est = max_daily * 30
    growth_factor = 1.2 + (std_daily / avg_daily if avg_daily > 0 else 0) * 0.1
    
    target_limit = max(monthly_est * growth_factor, peak_monthly_est * 1.1)
    
    # Score plans
    scored_plans = []
    for plan in plans:
        # Capacity score
        if plan['data_limit_gb'] >= target_limit:
            capacity_score = 1.0 - ((plan['data_limit_gb'] - target_limit) / target_limit) * 0.1
        else:
            capacity_score = 0.5 * (plan['data_limit_gb'] / target_limit)
        
        # Price efficiency
        price_per_gb = plan['price'] / plan['data_limit_gb']
        all_prices_per_gb = [p['price'] / p['data_limit_gb'] for p in plans]
        min_price_per_gb = min(all_prices_per_gb)
        price_score = min_price_per_gb / price_per_gb if price_per_gb > 0 else 0
        
        # Speed adequacy
        required_speed = avg_daily * 8  # 8 Mbps per GB daily usage (rough estimate)
        speed_score = min(1.0, plan['speed_mbps'] / max(required_speed, 25))
        
        # Combined score
        total_score = (capacity_score * 0.5) + (price_score * 0.3) + (speed_score * 0.2)
        scored_plans.append((plan, total_score))
    
    # Return top recommendations
    best_plans = sorted(scored_plans, key=lambda x: x[1], reverse=True)
    return [plan for plan, score in best_plans[:num_recommendations]]


# -------- Admin CRUD Helpers (Users & Plans) --------
def admin_create_user(username, password, name, email, role='user', city=None, state=None, phone=None, address=None):
    # Enforce unique username; return (ok, msg)
    existing = exec_query("SELECT id FROM users WHERE username = ?", (username,), fetch=True)
    if existing:
        return False, "Username already exists."
    pw = hash_password(password)
    cols = ['username','password_hash','role','name','email']
    vals = [username, pw, role, name, email]
    # Optional columns
    if column_exists('users','city'):
        cols += ['city']; vals += [city or '']
    if column_exists('users','state'):
        cols += ['state']; vals += [state or '']
    if column_exists('users','phone'):
        cols += ['phone']; vals += [phone or '']
    if column_exists('users','address'):
        cols += ['address']; vals += [address or '']
    if column_exists('users','signup_date'):
        cols += ['signup_date']; vals += [utcnow_naive().isoformat()]
    placeholders = ",".join(["?"]*len(vals))
    exec_query(f"INSERT INTO users ({','.join(cols)}) VALUES ({placeholders})", tuple(vals))
    return True, "User created."

def admin_update_user(user_id, **kwargs):
    # Only allow known columns
    allowed = {'username','name','email','role','city','state','phone','address','is_autopay_enabled','notification_preferences'}
    sets = []
    vals = []
    for k,v in kwargs.items():
        if k in allowed and (column_exists('users', k) or k in {'username','name','email','role'}):
            sets.append(f"{k} = ?")
            vals.append(v)
    if not sets:
        return False, "No valid fields to update."
    vals.append(user_id)
    exec_query(f"UPDATE users SET {', '.join(sets)} WHERE id = ?", tuple(vals))
    return True, "User updated."

def admin_delete_user(user_id):
    # Prevent delete if user has subscriptions/payments
    deps = exec_query("SELECT COUNT(*) FROM subscriptions WHERE user_id = ?", (user_id,), fetch=True)[0][0]
    if deps and deps > 0:
        return False, "Cannot delete: user has subscriptions. Cancel/delete those first."
    exec_query("DELETE FROM users WHERE id = ?", (user_id,))
    return True, "User deleted."

def admin_create_plan(name, speed_mbps, data_limit_gb, price, validity_days, description='', plan_type='basic', is_unlimited=0, features='', upload_speed_mbps=None):
    existing = exec_query("SELECT id FROM plans WHERE name = ?", (name,), fetch=True)
    if existing:
        return False, "Plan name already exists."
    cols = ['name','speed_mbps','data_limit_gb','price','validity_days','description']
    vals = [name, int(speed_mbps), float(data_limit_gb), float(price), int(validity_days), description or '']
    if column_exists('plans','plan_type'):
        cols += ['plan_type']; vals += [plan_type or 'basic']
    if column_exists('plans','is_unlimited'):
        cols += ['is_unlimited']; vals += [1 if is_unlimited else 0]
    if column_exists('plans','features'):
        cols += ['features']; vals += [features or '']
    if column_exists('plans','upload_speed_mbps'):
        cols += ['upload_speed_mbps']; vals += [int(upload_speed_mbps) if upload_speed_mbps is not None else int(speed_mbps)//10]
    if column_exists('plans','created_date'):
        cols += ['created_date']; vals += [utcnow_naive().isoformat()]
    placeholders = ",".join(["?"]*len(vals))
    exec_query(f"INSERT INTO plans ({','.join(cols)}) VALUES ({placeholders})", tuple(vals))
    return True, "Plan created."

def admin_update_plan(plan_id, **kwargs):
    allowed = {'name','speed_mbps','data_limit_gb','price','validity_days','description','plan_type','is_unlimited','features','upload_speed_mbps'}
    sets = []; vals = []
    for k,v in kwargs.items():
        if k in allowed and (column_exists('plans', k) or k in {'name','speed_mbps','data_limit_gb','price','validity_days','description'}):
            sets.append(f"{k} = ?")
            if k in {'speed_mbps','upload_speed_mbps','validity_days'} and v is not None:
                vals.append(int(v))
            elif k in {'data_limit_gb','price'} and v is not None:
                vals.append(float(v))
            else:
                vals.append(v)
    if not sets:
        return False, "No valid fields to update."
    vals.append(plan_id)
    exec_query(f"UPDATE plans SET {', '.join(sets)} WHERE id = ?", tuple(vals))
    return True, "Plan updated."

def admin_delete_plan(plan_id):
    deps = exec_query("SELECT COUNT(*) FROM subscriptions WHERE plan_id = ?", (plan_id,), fetch=True)[0][0]
    if deps and deps > 0:
        return False, "Cannot delete: plan is referenced in subscriptions."
    exec_query("DELETE FROM plans WHERE id = ?", (plan_id,))
    return True, "Plan deleted."


# ---------------------------
# UI Components (Enhanced)
# ---------------------------
def render_metric_card(title, value, delta=None, delta_color="normal"):
    if delta:
        delta_html = f"<div style='color: {'green' if delta_color == 'normal' else 'red'}; font-size: 0.8rem;'>{delta}</div>"
    else:
        delta_html = ""
    
    st.markdown(f"""
    <div class="metric-card">
        <h4 style="color: black; margin: 0;">{title}</h4>
        <h2 style="color: black; margin: 0.5rem 0;">{value}</h2>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

def render_partial_circle_progress(days_left, start_date, end_date, pct_visible=80):
    """
    Donut-style circle with missing slice at bottom.
    pct_visible = percent of circle filled (e.g. 80 means 80% arc shown).
    Color by days_left:
      - 30–16 = green
      - 15–6 = orange
      - 5–0 = red
    """
    # Pick color by days_left
    if days_left >= 16:
        bar_color = "#10b981"   # green
    elif days_left >= 6:
        bar_color = "#f59e0b"   # orange
    else:
        bar_color = "#ef4444"   # red

    visible = pct_visible
    hidden = 100 - pct_visible
    values = [visible, hidden]

    import plotly.graph_objects as go
    fig = go.Figure(go.Pie(
        values=values,
        hole=0.70,
        sort=False,
        direction="clockwise",
        rotation=220,   # <-- shift so missing slice is at bottom center
        textinfo="none",
        hoverinfo="skip",
        marker=dict(colors=[bar_color, "rgba(0,0,0,0)"]),
        showlegend=False
    ))

    # Center label
    fig.add_annotation(
        x=0.5, y=0.5,
        text=f"<b>{int(days_left)}</b><br>days left",
        showarrow=False, font=dict(size=16, color=bar_color), align="center"
    )

    # Dates
    fig.add_annotation(x=0.5, y=0.05, text=f"Started: {start_date}",
                       showarrow=False, font=dict(size=10, color="#6b7280"))
    fig.add_annotation(x=0.5, y=0.95, text=f"Expires: {end_date}",
                       showarrow=False, font=dict(size=10, color="#6b7280"))

    fig.update_layout(
                        margin=dict(l=0, r=0, t=0, b=0),
                        height=250,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)"
                    )
    return fig





def render_plan_card(plan, is_current=False, is_recommended=False, show_actions=True, current_user_id=None):
    card_class = "current-plan-card" if is_current else "recommended-plan" if is_recommended else "plan-card"
    
    status_badge = ""
    if is_current:
        status_badge = '<span class="status-active">Current Plan</span>'
    elif is_recommended:
        status_badge = '<span class="status-active">Recommended</span>'
    
    # Enhanced plan features display
    features_text = plan.get('features', '')
    upload_speed = plan.get('upload_speed_mbps', plan['speed_mbps'] // 10)
    
    st.markdown(f"""
    <div class="{card_class}">
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
            <div>
                <h3 style="margin: 0; color: #1f2937;">{plan['name']} {status_badge}</h3>
                <p style="color: #6b7280; margin: 0.5rem 0;">{plan['description']}</p>
                {f"<p style='color: #4f46e5; font-size: 0.9rem; margin: 0.5rem 0;'><strong>Features:</strong> {features_text}</p>" if features_text else ""}
            </div>
            <div style="text-align: right;">
                <h2 style="margin: 0; color: #667eea;">₹{plan['price']}</h2>
                <p style="color: #6b7280; margin: 0;">/month</p>
            </div>
        </div>
        <div style="margin: 1rem 0;">
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1rem;">
                <div><strong>Download:</strong> {plan['speed_mbps']} Mbps</div>
                <div><strong>Upload:</strong> {upload_speed} Mbps</div>
                <div><strong>Data:</strong> {'Unlimited' if plan.get('is_unlimited') else f"{plan['data_limit_gb']} GB"}</div>
                <div><strong>Validity:</strong> {plan['validity_days']} days</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if show_actions and not is_current:
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button(f"Subscribe to {plan['name']}", key=f"sub_{plan['id']}_{current_user_id}", use_container_width=True):
                subscribe_user_to_plan(current_user_id, plan['id'])
                create_payment(None, current_user_id, plan['price'])
                st.success(f"Successfully subscribed to {plan['name']}!")
                st.rerun()
        with col2:
            if st.button("Add to Compare", key=f"comp_{plan['id']}_{current_user_id}", use_container_width=True):
                if 'comparison_plans' not in st.session_state:
                    st.session_state['comparison_plans'] = []
                if plan['id'] not in [p['id'] for p in st.session_state['comparison_plans']]:
                    st.session_state['comparison_plans'].append(plan)
                    st.success(f"Added {plan['name']} to comparison!")
                else:
                    st.info("Plan already in comparison list!")
        with col3:
            if st.button("View Details", key=f"det_{plan['id']}_{current_user_id}", use_container_width=True):
                st.session_state[f'show_plan_details_{plan["id"]}'] = True

def render_plan_card(
    plan,
    is_current: bool = False,
    is_recommended: bool = False,
    show_actions: bool = True,
    current_user_id: int | None = None,
    section: str = "test"
        ):
    card_class = "current-plan-card" if is_current else "recommended-plan" if is_recommended else "plan-card"

    status_badge = ""
    if is_current:
        status_badge = '<span class="status-active">Current Plan</span>'
    elif is_recommended:
        status_badge = '<span class="status-active">Recommended</span>'

    features_text = plan.get('features', '')
    upload_speed = plan.get('upload_speed_mbps', plan['speed_mbps'] // 10)

    st.markdown(f"""
    <div class="{card_class}">
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
            <div>
                <h3 style="margin: 0; color: #1f2937;">{plan['name']} {status_badge}</h3>
                <p style="color: #6b7280; margin: 0.5rem 0;">{plan['description']}</p>
                {f"<p style='color: #4f46e5; font-size: 0.9rem; margin: 0.5rem 0;'><strong>Features:</strong> {features_text}</p>" if features_text else ""}
            </div>
            <div style="text-align: right;">
                <h2 style="margin: 0; color: #667eea;">₹{plan['price']}</h2>
                <p style="color: #6b7280; margin: 0;">/month</p>
            </div>
        </div>
        <div style="margin: 1rem 0;">
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1rem;">
                <div><strong>Download:</strong> {plan['speed_mbps']} Mbps</div>
                <div><strong>Upload:</strong> {upload_speed} Mbps</div>
                <div><strong>Data:</strong> {'Unlimited' if plan.get('is_unlimited') else f"{plan['data_limit_gb']} GB"}</div>
                <div><strong>Validity:</strong> {plan['validity_days']} days</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if show_actions and not is_current:
        # ensure section-specific unique keys
        sec = section or "default"
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button(
                f"Subscribe to {plan['name']}",
                key=f"{sec}_sub_{plan['id']}_{current_user_id}",
                use_container_width=True
            ):
                subscribe_user_to_plan(current_user_id, plan['id'])
                create_payment(None, current_user_id, plan['price'])
                st.success(f"Successfully subscribed to {plan['name']}!")
                st.rerun()
        with col2:
            if st.button(
                "Add to Compare",
                key=f"{sec}_comp_{plan['id']}_{current_user_id}",
                use_container_width=True
            ):
                if 'comparison_plans' not in st.session_state:
                    st.session_state['comparison_plans'] = []
                if plan['id'] not in [p['id'] for p in st.session_state['comparison_plans']]:
                    st.session_state['comparison_plans'].append(plan)
                    st.success(f"Added {plan['name']} to comparison!")
                else:
                    st.info("Plan already in comparison list!")
        with col3:
            if st.button(
                "View Details",
                key=f"{sec}_det_{plan['id']}_{current_user_id}",
                use_container_width=True
            ):
                st.session_state[f'show_plan_details_{plan["id"]}'] = True


def render_expiry_reminder(reminder):
    """Render expiry reminder with appropriate styling"""
    if reminder['type'] == 'critical':
        st.markdown(f"""
        <div class="expiry-critical">
            <h4 style="margin: 0; color: #dc2626;">⚠️ Plan Expired!</h4>
            <p style="margin: 0.5rem 0; color: #991b1b;">{reminder['message']}</p>
        </div>
        """, unsafe_allow_html=True)
    elif reminder['type'] == 'warning':
        st.markdown(f"""
        <div class="expiry-warning">
            <h4 style="margin: 0; color: #d97706;">⏰ Plan Expiring Soon!</h4>
            <p style="margin: 0.5rem 0; color: #92400e;">{reminder['message']}</p>
        </div>
        """, unsafe_allow_html=True)

def render_usage_analytics(user_id):
    """Render detailed usage analytics"""
    usage_df = get_usage_for_user(user_id, days=30)
    
    if usage_df.empty:
        st.info("No usage data available yet.")
        return
    
    usage_df['date'] = pd.to_datetime(usage_df['date'])
    usage_df = usage_df.sort_values('date')
    
    # Calculate statistics
    total_usage = usage_df['data_used_gb'].sum()
    avg_daily = usage_df['data_used_gb'].mean()
    max_daily = usage_df['data_used_gb'].max()
    
    # Usage metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Usage (30d)", f"{total_usage:.1f} GB")
    with col2:
        st.metric("Daily Average", f"{avg_daily:.1f} GB")
    with col3:
        st.metric("Peak Day", f"{max_daily:.1f} GB")
    with col4:
        projected_monthly = avg_daily * 30
        st.metric("Projected Monthly", f"{projected_monthly:.1f} GB")
    
    # Usage trend chart
    fig = px.line(usage_df, x='date', y='data_used_gb',
                 title="Daily Usage Trend (Last 30 Days)",
                 labels={'data_used_gb': 'Usage (GB)', 'date': 'Date'})
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Peak vs Off-peak usage (if available)
    if 'peak_hour_usage' in usage_df.columns and 'off_peak_usage' in usage_df.columns:
        st.subheader("Peak vs Off-Peak Usage")
        
        peak_avg = usage_df['peak_hour_usage'].mean()
        off_peak_avg = usage_df['off_peak_usage'].mean()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Peak Hours Avg", f"{peak_avg:.1f} GB")
        with col2:
            st.metric("Off-Peak Avg", f"{off_peak_avg:.1f} GB")
        
        # Peak vs off-peak chart
        usage_comparison = pd.DataFrame({
            'Type': ['Peak Hours', 'Off-Peak'],
            'Average Usage': [peak_avg, off_peak_avg]
        })
        
        fig = px.bar(usage_comparison, x='Type', y='Average Usage',
                    title="Peak vs Off-Peak Usage Comparison",
                    color='Type')
        st.plotly_chart(fig, use_container_width=True)

def render_plan_comparison():
    """Render plan comparison functionality"""
    st.subheader("Compare Plans")
    
    # Initialize comparison list in session state
    if 'comparison_plans' not in st.session_state:
        st.session_state['comparison_plans'] = []
    
    # Plan search and add functionality
    all_plans = get_all_plans()
    
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_plan_name = st.selectbox(
            "Search and add plans to compare:",
            options=["Select a plan..."] + [p['name'] for p in all_plans],
            key="plan_search_select"
        )
    with col2:
        if st.button("Add Plan", disabled=(selected_plan_name == "Select a plan...")):
            selected_plan = next((p for p in all_plans if p['name'] == selected_plan_name), None)
            if selected_plan and selected_plan not in st.session_state['comparison_plans']:
                st.session_state['comparison_plans'].append(selected_plan)
                st.success(f"Added {selected_plan['name']} to comparison!")
                st.rerun()
            elif selected_plan in st.session_state['comparison_plans']:
                st.info("Plan already in comparison!")
    
    # Display comparison table
    if st.session_state['comparison_plans']:
        st.markdown("### Comparison Table")
        
        # Create comparison dataframe
        comparison_data = []
        for plan in st.session_state['comparison_plans']:
            comparison_data.append({
                'Plan Name': plan['name'],
                'Price (₹)': plan['price'],
                'Speed (Mbps)': plan['speed_mbps'],
                'Upload (Mbps)': plan.get('upload_speed_mbps', plan['speed_mbps'] // 10),
                'Data Limit': 'Unlimited' if plan.get('is_unlimited') else f"{plan['data_limit_gb']} GB",
                'Validity': f"{plan['validity_days']} days",
                'Type': plan.get('plan_type', 'basic').title(),
                'Price/GB': f"₹{plan['price']/plan['data_limit_gb']:.2f}" if not plan.get('is_unlimited') else 'N/A'
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Clear comparison button
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("Clear All", type="secondary"):
                st.session_state['comparison_plans'] = []
                st.rerun()
        
        # Save comparison (if user is logged in)
        with col2:
            if st.button("Save Comparison", key="save_comparison"):
                if st.session_state.get('user'):
                    plan_ids = [p['id'] for p in st.session_state['comparison_plans']]
                    save_plan_comparison(st.session_state['user']['id'], plan_ids)
                    st.success("Comparison saved!")
    else:
        st.info("Add plans to compare them side by side.")

def render_billing_history(user_id):
    """Render comprehensive billing history"""
    st.subheader("Billing History")
    
    # Get payment history with enhanced details
    payment_query = """
    SELECT 
        p.amount, p.payment_date, p.status, p.bill_month, p.bill_year,
        s.start_date, s.end_date,
        pl.name as plan_name
    """
    
    if column_exists('payments', 'payment_method'):
        payment_query += ", p.payment_method"
    if column_exists('payments', 'transaction_id'):
        payment_query += ", p.transaction_id"
    if column_exists('payments', 'tax_amount'):
        payment_query += ", p.tax_amount, p.discount"
    
    payment_query += """
    FROM payments p
    LEFT JOIN subscriptions s ON p.subscription_id = s.id
    LEFT JOIN plans pl ON s.plan_id = pl.id
    WHERE p.user_id = ?
    ORDER BY p.payment_date DESC
    LIMIT 50
    """
    
    payments_df = df_from_query(payment_query, (user_id,))
    
    if payments_df.empty:
        st.info("No billing history found.")
        return
    
    # Payment statistics
    total_paid = payments_df[payments_df['status'] == 'paid']['amount'].sum()
    failed_payments = len(payments_df[payments_df['status'] == 'failed'])
    avg_monthly = payments_df[payments_df['status'] == 'paid']['amount'].mean()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Paid", f"₹{total_paid:,.0f}")
    with col2:
        st.metric("Failed Payments", failed_payments)
    with col3:
        st.metric("Average Payment", f"₹{avg_monthly:,.0f}")
    
    # Payment history table
    st.markdown("### Payment Records")
    
    # Format dates and amounts for display
    display_df = payments_df.copy()
    display_df['payment_date'] = pd.to_datetime(display_df['payment_date']).dt.strftime('%Y-%m-%d')
    display_df['amount'] = display_df['amount'].apply(lambda x: f"₹{x:,.0f}")
    
    # Select columns to display
    display_columns = ['payment_date', 'plan_name', 'amount', 'status']
    if 'payment_method' in display_df.columns:
        display_columns.append('payment_method')
    if 'transaction_id' in display_df.columns:
        display_columns.append('transaction_id')
    
    st.dataframe(display_df[display_columns], use_container_width=True)
    
    # Payment trends chart
    if len(payments_df) > 3:
        st.markdown("### Payment Trends")
        
        # Monthly payment trends
        payments_df['payment_date'] = pd.to_datetime(payments_df['payment_date'])
        payments_df['month_year'] = payments_df['payment_date'].dt.to_period('M')
        
        monthly_trends = payments_df[payments_df['status'] == 'paid'].groupby('month_year')['amount'].sum().reset_index()
        monthly_trends['month_year'] = monthly_trends['month_year'].astype(str)
        
        fig = px.line(monthly_trends, x='month_year', y='amount',
                     title="Monthly Payment Trends",
                     labels={'amount': 'Amount (₹)', 'month_year': 'Month'})
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Enhanced User Dashboard
# ---------------------------
def user_dashboard(user):
    st.title("🏠 My Dashboard")
    st.markdown(f"Welcome back, **{user['name']}**!")
    
    # Check for expiry reminders first
    reminders = check_expiry_reminders(user['id'])
    for reminder in reminders:
        render_expiry_reminder(reminder)
    
    # Get current subscription
    current_sub = get_user_active_subscription(user['id'])
    
    # Section 1: Current Plan with Semi-circular Progress
    st.markdown("### 📶 Your Current Plan")
    
    if current_sub:
        current_plan = get_plan(current_sub['plan_id'])
        
        # Calculate days remaining and percentage
        try:
            start_date = datetime.fromisoformat(current_sub['start_date']).date()
            end_date = datetime.fromisoformat(current_sub['end_date']).date()
            today = datetime.utcnow().date()
            
            total_days = (end_date - start_date).days
            days_passed = (today - start_date).days
            days_remaining = (end_date - today).days
            
            percentage = min(100, max(0, (days_passed / total_days) * 100))
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                fig = render_partial_circle_progress(
                    days_left=max(0, days_remaining),
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d'),
                    pct_visible=80   # <-- controls how much of the circle is shown
                )
                st.plotly_chart(fig, use_container_width=True)



            
            with col2:
                render_plan_card(current_plan, is_current=True, show_actions=False)
        
        except Exception as e:
            render_plan_card(current_plan, is_current=True, show_actions=False)
        
        # Usage overview for current plan
        st.markdown("### 📊 Usage Overview")
        render_usage_analytics(user['id'])
        
    else:
        st.info("🎯 You don't have an active plan. Choose one below to get started!")

    st.markdown("### 📈 Usage Insights & Smart Recommendations")
    
    usage_df = get_usage_for_user(user['id'], days=60)
    if not usage_df.empty:
        # Usage pattern analysis
        avg_daily = usage_df['data_used_gb'].mean()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Usage Pattern Analysis")
            if avg_daily < 2:
                pattern = "Light User"
                recommendation = "You're a light user. Consider our Basic plans for cost savings."
            elif avg_daily < 5:
                pattern = "Moderate User"
                recommendation = "You have moderate usage. Standard plans offer good value."
            else:
                pattern = "Heavy User"
                recommendation = "You're a heavy user. Premium plans with higher limits suit you best."
            
            st.info(f"**Pattern:** {pattern}")
            st.success(f"**Recommendation:** {recommendation}")
        
        with col2:
            # Usage vs Plan comparison
            if current_sub:
                current_plan = get_plan(current_sub['plan_id'])
                monthly_usage = avg_daily * 30
                usage_ratio = monthly_usage / current_plan['data_limit_gb'] * 100
                
                st.markdown("#### Plan Utilization")
                st.progress(min(usage_ratio / 100, 1.0))
                st.write(f"You're using {usage_ratio:.1f}% of your plan limit")
                
                if usage_ratio > 80:
                    st.warning("Consider upgrading to a higher limit plan!")
                elif usage_ratio < 30:
                    st.info("You might save money with a lower limit plan.")
    
    # Section 2: ML Recommended Plans (2 best plans)
    st.markdown("### 🎯 Recommended Plans for You")
    
    recommended_plans = ml_recommendation_for_user(user['id'], num_recommendations=2)
    
    if recommended_plans:
        cols = st.columns(2)
        for i, plan in enumerate(recommended_plans):
            with cols[i]:
                # render_plan_card(plan, is_recommended=True, current_user_id=user['id'])
                render_plan_card(
                                    plan,
                                    is_recommended=True,
                                    current_user_id=user['id'],
                                    section="recommended"
                                )


    else:
        st.info("No recommendations available at the moment.")

    st.markdown("---")
    # st.markdown("### 📈 Usage Insights & Smart Recommendations")
    
    # usage_df = get_usage_for_user(user['id'], days=60)
    # if not usage_df.empty:
    #     # Usage pattern analysis
    #     avg_daily = usage_df['data_used_gb'].mean()
        
    #     col1, col2 = st.columns(2)
        
    #     with col1:
    #         st.markdown("#### Usage Pattern Analysis")
    #         if avg_daily < 2:
    #             pattern = "Light User"
    #             recommendation = "You're a light user. Consider our Basic plans for cost savings."
    #         elif avg_daily < 5:
    #             pattern = "Moderate User"
    #             recommendation = "You have moderate usage. Standard plans offer good value."
    #         else:
    #             pattern = "Heavy User"
    #             recommendation = "You're a heavy user. Premium plans with higher limits suit you best."
            
    #         st.info(f"**Pattern:** {pattern}")
    #         st.success(f"**Recommendation:** {recommendation}")
        
    #     with col2:
    #         # Usage vs Plan comparison
    #         if current_sub:
    #             current_plan = get_plan(current_sub['plan_id'])
    #             monthly_usage = avg_daily * 30
    #             usage_ratio = monthly_usage / current_plan['data_limit_gb'] * 100
                
    #             st.markdown("#### Plan Utilization")
    #             st.progress(min(usage_ratio / 100, 1.0))
    #             st.write(f"You're using {usage_ratio:.1f}% of your plan limit")
                
    #             if usage_ratio > 80:
    #                 st.warning("Consider upgrading to a higher limit plan!")
    #             elif usage_ratio < 30:
    #                 st.info("You might save money with a lower limit plan.")
    
    
    # Section 3: All Available Plans
    st.markdown("### 📋 All Available Plans")
    
    # Plan filters
    col1, col2, col3 = st.columns(3)
    with col1:
        price_filter = st.selectbox("Filter by Price", 
                                   ["All", "Under ₹500", "₹500-₹1000", "Above ₹1000"])
    with col2:
        speed_filter = st.selectbox("Filter by Speed", 
                                   ["All", "Up to 100 Mbps", "100-500 Mbps", "500+ Mbps"])
    with col3:
        type_filter = st.selectbox("Filter by Type",
                                  ["All", "Basic", "Standard", "Premium", "Elite"])
    
    # Get and filter plans
    all_plans = get_all_plans()
    filtered_plans = all_plans.copy()
    
    # Apply filters
    if price_filter != "All":
        if price_filter == "Under ₹500":
            filtered_plans = [p for p in filtered_plans if p['price'] < 500]
        elif price_filter == "₹500-₹1000":
            filtered_plans = [p for p in filtered_plans if 500 <= p['price'] <= 1000]
        else:
            filtered_plans = [p for p in filtered_plans if p['price'] > 1000]
    
    if speed_filter != "All":
        if speed_filter == "Up to 100 Mbps":
            filtered_plans = [p for p in filtered_plans if p['speed_mbps'] <= 100]
        elif speed_filter == "100-500 Mbps":
            filtered_plans = [p for p in filtered_plans if 100 < p['speed_mbps'] <= 500]
        else:
            filtered_plans = [p for p in filtered_plans if p['speed_mbps'] > 500]
    
    if type_filter != "All":
        filtered_plans = [p for p in filtered_plans if p.get('plan_type', 'basic').lower() == type_filter.lower()]
    
    # Display filtered plans
    if filtered_plans:
        for plan in filtered_plans:
            is_current_plan = current_sub and plan['id'] == current_sub['plan_id']
            # render_plan_card(plan, is_current=is_current_plan, current_user_id=user['id'])
            render_plan_card(
                                plan,
                                is_current=is_current_plan,
                                current_user_id=user['id'],
                                section="all"
                            )

    else:
        st.warning("No plans match your filter criteria.")
    
    # Section 4: Plan Comparison
    st.markdown("---")
    render_plan_comparison()
    
    # Section 5: Usage Insights and Recommendations
    st.markdown("---")
   
    
    # Section 6: Previous Buying History
    st.markdown("---")
    render_billing_history(user['id'])
    
    # Additional sections for subscription history
    st.markdown("---")
    st.markdown("### 📋 Subscription History")
    
    subscription_history = get_user_subscription_history(user['id'])
    if subscription_history:
        history_data = []
        for sub in subscription_history:
            history_data.append({
                'Plan': sub['plan_name'],
                'Start Date': sub['start_date'],
                'End Date': sub['end_date'],
                'Status': sub['status'].title(),
                'Price': f"₹{sub['price']}",
                'Speed': f"{sub['speed_mbps']} Mbps",
                'Data': f"{sub['data_limit_gb']} GB"
            })
        
        history_df = pd.DataFrame(history_data)
        st.dataframe(history_df, use_container_width=True)
    else:
        st.info("No subscription history available.")

# ---------------------------
# Enhanced Admin Dashboard
# ---------------------------
def admin_dashboard(user):
    st.title("🛠️ Admin Dashboard")
    st.markdown(f"Welcome back, **{user['name']}**!")
    
    # Quick stats overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_users = exec_query("SELECT COUNT(*) FROM users WHERE role = 'user'", fetch=True)[0][0]
        render_metric_card("Total Users", total_users)
    
    with col2:
        active_subs = exec_query("SELECT COUNT(*) FROM subscriptions WHERE status = 'active'", fetch=True)[0][0]
        render_metric_card("Active Subscriptions", active_subs)
    
    with col3:
        monthly_revenue_query = "SELECT COALESCE(SUM(amount), 0) FROM payments WHERE status = 'paid'"
        if column_exists('payments', 'payment_date'):
            monthly_revenue_query += " AND payment_date >= date('now', '-30 days')"
        monthly_revenue = exec_query(monthly_revenue_query, fetch=True)[0][0]
        render_metric_card("Monthly Revenue", f"₹{monthly_revenue:,.0f}")
    
    with col4:
        if column_exists('support_tickets', 'status'):
            support_tickets = exec_query("SELECT COUNT(*) FROM support_tickets WHERE status IN ('open', 'in_progress')", fetch=True)[0][0]
        else:
            support_tickets = 0
        render_metric_card("Open Tickets", support_tickets)

    # Main dashboard tabs
    tabs = st.tabs(["📊 Analytics", "🤖 ML Model", "📋 Plans Management", "👥 User Management", "🎫 Support", "⚙️ Settings"])
    
    with tabs[0]:
        render_analytics_dashboard()
    
    with tabs[1]:
        render_ml_model_management()
    
    with tabs[2]:
        render_enhanced_plans_management()
    
    with tabs[3]:
        render_user_management()
    
    with tabs[4]:
        render_support_management()
    
    with tabs[5]:
        render_admin_settings()

def render_analytics_dashboard():
    st.header("📊 Business Analytics")
    
    # Revenue Analytics
    st.subheader("💰 Revenue Analytics")
    
    try:
        revenue_query = "SELECT DATE(payment_date) as date, SUM(amount) as daily_revenue, COUNT(*) as transaction_count FROM payments WHERE status = 'paid'"
        if column_exists('payments', 'payment_date'):
            revenue_query += " AND payment_date >= date('now', '-90 days')"
        revenue_query += " GROUP BY DATE(payment_date) ORDER BY date"
        
        revenue_data = df_from_query(revenue_query)
        
        if not revenue_data.empty :
            revenue_data['date'] = pd.to_datetime(revenue_data['date'], errors='coerce')
            revenue_data = revenue_data.dropna(subset=['date'])
            
            if not revenue_data.empty:
                fig = px.line(revenue_data, x='date', y='daily_revenue', 
                             title="Daily Revenue Trend (Last 90 Days)",
                             labels={'daily_revenue': 'Revenue (₹)', 'date': 'Date'})
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No valid revenue data available for chart")
        else:
            st.info("No revenue data available for chart")
    except Exception as e:
        st.error(f"Error rendering revenue analytics: {str(e)}")
    
    # User Growth Analytics
    st.subheader("📈 User Growth")
    
    try:
        if column_exists('users', 'signup_date'):
            user_growth = df_from_query("""
                SELECT 
                    DATE(signup_date) as signup_date,
                    COUNT(*) as new_users
                FROM users 
                WHERE role = 'user' AND signup_date IS NOT NULL AND signup_date >= date('now', '-90 days')
                GROUP BY DATE(signup_date)
                ORDER BY signup_date
            """)
            
            if not user_growth.empty:
                user_growth['signup_date'] = pd.to_datetime(user_growth['signup_date'], errors='coerce')
                user_growth = user_growth.dropna(subset=['signup_date'])
                
                if not user_growth.empty:
                    user_growth['cumulative_users'] = user_growth['new_users'].cumsum()
                    
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    fig.add_trace(
                        go.Bar(x=user_growth['signup_date'], y=user_growth['new_users'], name="New Users"),
                        secondary_y=False,
                    )
                    fig.add_trace(
                        go.Scatter(x=user_growth['signup_date'], y=user_growth['cumulative_users'], 
                                  name="Cumulative Users", line=dict(color='orange')),
                        secondary_y=True,
                    )
                    
                    fig.update_yaxes(title_text="New Users", secondary_y=False)
                    fig.update_yaxes(title_text="Cumulative Users", secondary_y=True)
                    fig.update_layout(title="User Registration Trend")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No valid user growth data available for chart")
            else:
                st.info("No user growth data available for chart")
        else:
            st.info("User signup dates not available - run database migration")
    except Exception as e:
        st.error(f"Error rendering user growth analytics: {str(e)}")
    
    # Plan Performance
    st.subheader("📋 Plan Performance")
    
    try:
        plan_stats = df_from_query("""
            SELECT 
                p.name as plan_name,
                COUNT(s.id) as subscription_count,
                SUM(CASE WHEN pay.status = 'paid' THEN pay.amount ELSE 0 END) as total_revenue
            FROM plans p
            LEFT JOIN subscriptions s ON p.id = s.plan_id
            LEFT JOIN payments pay ON s.id = pay.subscription_id
            GROUP BY p.id, p.name
            ORDER BY subscription_count DESC
        """)
        
        if not plan_stats.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(plan_stats, x='plan_name', y='subscription_count',
                            title="Plan Popularity (Total Subscriptions)")
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            
            # with col2:
            #     revenue_data = plan_stats[plan_stats['total_revenue'] > 0]
            #     if not revenue_data.empty:
            #         fig = px.pie(revenue_data, values='total_revenue', names='plan_name',
            #                     title="Revenue Distribution by Plan")
            #         st.plotly_chart(fig, use_container_width=True)
            #     else:
            #         st.info("No revenue data available for pie chart")
            
            with col2:
                revenue_data = plan_stats.groupby('plan_name', as_index=False)['total_revenue'].sum()

                if revenue_data['total_revenue'].sum() > 0:
                    # Revenue pie
                    fig = px.pie(
                        revenue_data,
                        values='total_revenue',
                        names='plan_name',
                        title="Revenue Distribution by Plan",
                        hole=0.35
                    )
                    fig.update_traces(textinfo='percent+label', hovertemplate='%{label}<br>₹%{value:,.0f}')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    # Fallback: subscription share pie
                    fig = px.pie(
                        plan_stats,
                        values='subscription_count',
                        names='plan_name',
                        title="Subscription Share by Plan",
                        hole=0.35
                    )
                    fig.update_traces(textinfo='percent+label', hovertemplate='%{label}<br>%{value} subs')
                    st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("No plan statistics available")
    except Exception as e:
        st.error(f"Error rendering plan performance: {str(e)}")

def render_ml_model_management():
    st.header("🤖 Machine Learning Model Management")
    
    # Model status
    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists('plan_recommendation_model.pkl'):
            st.success("✅ ML Model: Active")
        else:
            st.warning("⚠️ ML Model: Not Trained")
    
    with col2:
        model_size = 0
        if os.path.exists('plan_recommendation_model.pkl'):
            model_size = os.path.getsize('plan_recommendation_model.pkl') / (1024 * 1024)  # MB
            st.info(f"Model Size: {model_size:.2f} MB")
    
    # Training section
    st.subheader("Model Training")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Train New Model", use_container_width=True):
            with st.spinner("Training enhanced recommendation model..."):
                model = train_recommendation_model()
                if model:
                    st.success("Model trained successfully!")
                    st.rerun()
    
    with col2:
        if st.button("Evaluate Current Model", use_container_width=True):
            if os.path.exists('plan_recommendation_model.pkl'):
                evaluate_model()
            else:
                st.error("No model found to evaluate. Please train a model first.")
    
    # Model performance metrics
    if os.path.exists('plan_recommendation_model.pkl'):
        st.subheader("Model Performance")
        evaluate_model()

def render_enhanced_plans_management():

    st.header("📋 Enhanced Plans Management")

    # Current plans overview
    plans_df = df_from_query("SELECT * FROM plans ORDER BY price ASC")
    if not plans_df.empty:
        st.subheader("Current Plans")
        st.dataframe(plans_df, use_container_width=True)

    # ---------- Single Plan CRUD ----------
    st.subheader("➕ Create / ✏️ Edit / 🗑️ Delete Plan")
    action = st.radio("Action", ["Create", "Edit", "Delete"], horizontal=True, key="plan_action")
    if action == "Create":
        with st.form("create_plan_form", clear_on_submit=True):
            cols = st.columns(3)
            with cols[0]:
                name = st.text_input("Plan Name")
                speed = st.number_input("Download Speed (Mbps)", min_value=1, value=50)
                upload = st.number_input("Upload Speed (Mbps)", min_value=0, value=5)
            with cols[1]:
                data_limit = st.number_input("Data Limit (GB)", min_value=1.0, value=100.0, step=1.0)
                price = st.number_input("Price (₹)", min_value=1.0, value=499.0, step=1.0)
                validity = st.number_input("Validity (days)", min_value=1, value=30, step=1)
            with cols[2]:
                plan_type = st.selectbox("Plan Type", ["basic","standard","premium","elite"])
                is_unl = st.checkbox("Unlimited Data?")
                features = st.text_input("Features (comma-separated)", value="")
            desc = st.text_area("Description", value="")
            submitted = st.form_submit_button("Create Plan")
        if submitted:
            ok, msg = admin_create_plan(name, speed, data_limit, price, validity, desc, plan_type, int(is_unl), features, upload)
            (st.success if ok else st.error)(msg)
            if ok: st.rerun()
    elif action == "Edit":
        plans = df_from_query("SELECT id, name FROM plans ORDER BY name ASC")
        if plans.empty:
            st.info("No plans to edit.")
        else:
            sel = st.selectbox("Select Plan", options=plans['name'].tolist())
            pid = int(plans.loc[plans['name']==sel, 'id'].iloc[0])
            current = df_from_query("SELECT * FROM plans WHERE id = ?", (pid,))
            row = current.iloc[0].to_dict()
            with st.form("edit_plan_form"):
                cols = st.columns(3)
                with cols[0]:
                    name = st.text_input("Plan Name", value=row.get('name',''))
                    speed = st.number_input("Download Speed (Mbps)", min_value=1, value=int(row.get('speed_mbps',50)))
                    upload = st.number_input("Upload Speed (Mbps)", min_value=0, value=int(row.get('upload_speed_mbps',row.get('speed_mbps',50)//10)))
                with cols[1]:
                    data_limit = st.number_input("Data Limit (GB)", min_value=1.0, value=float(row.get('data_limit_gb',100.0)), step=1.0)
                    price = st.number_input("Price (₹)", min_value=1.0, value=float(row.get('price',499.0)), step=1.0)
                    validity = st.number_input("Validity (days)", min_value=1, value=int(row.get('validity_days',30)), step=1)
                with cols[2]:
                    plan_type = st.selectbox("Plan Type", ["basic","standard","premium","elite"], index=max(0, ["basic","standard","premium","elite"].index(str(row.get('plan_type','basic')))) if str(row.get('plan_type','basic')) in ["basic","standard","premium","elite"] else 0)
                    is_unl = st.checkbox("Unlimited Data?", value=bool(row.get('is_unlimited',0)))
                    features = st.text_input("Features (comma-separated)", value=str(row.get('features','') or ''))
                desc = st.text_area("Description", value=str(row.get('description','') or ''))
                submitted = st.form_submit_button("Update Plan")
            if submitted:
                ok, msg = admin_update_plan(pid, name=name, speed_mbps=speed, upload_speed_mbps=upload, data_limit_gb=data_limit, price=price, validity_days=validity, plan_type=plan_type, is_unlimited=int(is_unl), features=features, description=desc)
                (st.success if ok else st.error)(msg)
                if ok: st.rerun()
    else:  # Delete
        plans = df_from_query("SELECT id, name FROM plans ORDER BY name ASC")
        if plans.empty:
            st.info("No plans to delete.")
        else:
            sel = st.selectbox("Select Plan to Delete", options=plans['name'].tolist(), key="plan_del_sel")
            pid = int(plans.loc[plans['name']==sel, 'id'].iloc[0])
            if st.button("Delete Plan", type="secondary"):
                ok, msg = admin_delete_plan(pid)
                (st.success if ok else st.error)(msg)
                if ok: st.rerun()

    # ---------- Bulk CSV Upload (existing) ----------
    st.subheader("📤 Bulk Plan Upload")
    st.markdown("**CSV Template Format:** name,speed_mbps,data_limit_gb,price,validity_days,description,plan_type,is_unlimited,features,upload_speed_mbps")
    uploaded_file = st.file_uploader("Choose CSV file", type="csv", key="bulk_plans_upload")
    if uploaded_file is not None:
        try:
            csv_content = uploaded_file.read().decode('utf-8')
            preview_df = pd.read_csv(io.StringIO(csv_content))
            st.subheader("📋 Preview Uploaded Data")
            st.dataframe(preview_df, use_container_width=True)
            ok, msg = bulk_create_plans_from_csv(csv_content)
            (st.success if ok else st.error)(msg)
            if ok: st.rerun()
        except Exception as e:
            st.error(f"CSV error: {e}")

def render_user_management():

    st.header("👥 User Management")
    # Quick stats
    col1, col2 = st.columns(2)
    with col1:
        total_users = exec_query("SELECT COUNT(*) FROM users WHERE role = 'user'", fetch=True)[0][0]
        st.metric("Total Users", total_users)
    with col2:
        active_users = exec_query("SELECT COUNT(DISTINCT user_id) FROM subscriptions WHERE status = 'active'", fetch=True)[0][0] if column_exists('subscriptions', 'status') else 0
        st.metric("Active Users", active_users)

    st.subheader("➕ Create New User")
    with st.form("create_user_form", clear_on_submit=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            username = st.text_input("Username")
            name = st.text_input("Full Name")
        with c2:
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
        with c3:
            role = st.selectbox("Role", ["user","admin"])
            city = st.text_input("City")
            state = st.text_input("State")
        submitted = st.form_submit_button("Create User")
    if submitted:
        ok, msg = admin_create_user(username, password, name, email, role=role, city=city, state=state)
        (st.success if ok else st.error)(msg)
        if ok: st.rerun()

    st.subheader("✏️ Edit / 🗑️ Delete User")
    users_df = df_from_query("SELECT id, username, name, email, role, city, state FROM users ORDER BY id DESC")
    if users_df.empty:
        st.info("No users found.")
    else:
        st.dataframe(users_df, use_container_width=True)
        sel_name = st.selectbox("Select user by username", options=users_df['username'].tolist(), key="user_edit_sel")
        uid = int(users_df.loc[users_df['username']==sel_name, 'id'].iloc[0])
        current = df_from_query("SELECT * FROM users WHERE id = ?", (uid,)).iloc[0].to_dict()
        with st.form("edit_user_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                new_username = st.text_input("Username", value=current.get('username',''))
                name = st.text_input("Full Name", value=current.get('name',''))
            with c2:
                email = st.text_input("Email", value=current.get('email',''))
                role = st.selectbox("Role", ["user","admin"], index=0 if current.get('role','user')=='user' else 1)
            with c3:
                city = st.text_input("City", value=str(current.get('city','') or ''))
                state = st.text_input("State", value=str(current.get('state','') or ''))
            update_btn = st.form_submit_button("Update User")
        del_btn = st.button("Delete Selected User", type="secondary")
        if update_btn:
            # If username changed, ensure it's unique (handled by update attempt via helper)
            ok, msg = admin_update_user(uid, username=new_username, name=name, email=email, role=role, city=city, state=state)
            (st.success if ok else st.error)(msg)
            if ok: st.rerun()
        if del_btn:
            ok, msg = admin_delete_user(uid)
            (st.success if ok else st.error)(msg)
            if ok: st.rerun()

    # st.header("🎫 Support Ticket Management")
    
    # if column_exists('support_tickets', 'status'):
    #     # Support statistics
    #     ticket_stats = df_from_query("""
    #         SELECT status, COUNT(*) as count
    #         FROM support_tickets
    #         GROUP BY status
    #     """)
        
    #     if not ticket_stats.empty:
    #         cols = st.columns(min(4, len(ticket_stats)))
    #         for i, (_, row) in enumerate(ticket_stats.iterrows()):
    #             with cols[i % len(cols)]:
    #                 st.metric(f"{row['status'].title()} Tickets", row['count'])
        
    #     # Category breakdown
    #     category_stats = df_from_query("""
    #         SELECT category, COUNT(*) as count
    #         FROM support_tickets
    #         GROUP BY category
    #         ORDER BY count DESC
    #     """)
        
    #     if not category_stats.empty:
    #         st.subheader("Tickets by Category")
    #         fig = px.bar(category_stats, x='category', y='count',
    #                     title="Support Tickets by Category")
    #         st.plotly_chart(fig, use_container_width=True, key="tickets_by_category")

        
    #     # Recent tickets
    #     # recent_tickets = df_from_query("""
    #     #     SELECT 
    #     #         st.id, st.subject, st.category, st.status, st.priority, st.created_date,
    #     #         u.name as user_name, u.email as user_email
    #     #     FROM support_tickets st
    #     #     JOIN users u ON st.user_id = u.id
    #     #     ORDER BY st.created_date DESC
    #     #     LIMIT 50
    #     # """)

    #     # Categorize tickets into Resolved / Not Resolved / Ongoing
    #     recent_tickets = df_from_query("""
    #         SELECT 
    #             st.id,
    #             st.subject,
    #             st.category,
    #             CASE 
    #                 WHEN LOWER(st.status) IN ('resolved','solved','completed') THEN 'Resolved'
    #                 WHEN LOWER(st.status) IN ('closed','rejected','cancelled','won''t fix','invalid') THEN 'Not Resolved'
    #                 WHEN LOWER(st.status) IN ('open','in_progress','in progress','pending','assigned','acknowledged') THEN 'Ongoing'
    #                 ELSE 'Other'
    #             END AS ticket_group,
    #             st.status,
    #             st.priority,
    #             st.created_date,
    #             u.name AS user_name,
    #             u.email AS user_email
    #         FROM support_tickets st
    #         JOIN users u ON st.user_id = u.id
    #         ORDER BY st.created_date DESC
    #         LIMIT 50
    #     """)

        
    #     if not recent_tickets.empty:
    #         st.subheader("Recent Support Tickets")
    #         st.dataframe(recent_tickets, use_container_width=True)
    #     else:
    #         st.info("No support tickets found")
    # else:
    #     st.info("Support ticket system not available - run database migration to enable")

def render_support_management():
    st.header("🎫 Support Ticket Management")

    # Guard: table/columns present?
    if not column_exists('support_tickets', 'status'):
        st.info("Support ticket system not available - run database migration to enable")
        return

    # --------- Quick Stats ---------
    ticket_stats = df_from_query("""
        SELECT LOWER(status) AS status, COUNT(*) AS count
        FROM support_tickets
        GROUP BY LOWER(status)
    """)
    if not ticket_stats.empty:
        cols = st.columns(min(4, len(ticket_stats)))
        for i, (_, row) in enumerate(ticket_stats.iterrows()):
            with cols[i % len(cols)]:
                st.metric(f"{row['status'].title()} Tickets", row['count'])

    # --------- Category Breakdown ---------
    category_stats = df_from_query("""
        SELECT category, COUNT(*) AS count
        FROM support_tickets
        GROUP BY category
        ORDER BY count DESC
    """)
    if not category_stats.empty:
        st.subheader("Tickets by Category")
        fig = px.bar(category_stats, x='category', y='count', title="Support Tickets by Category")
        st.plotly_chart(fig, use_container_width=True)

    # --------- Tabs: Resolved / Not Resolved / Ongoing ---------
    st.subheader("Browse Tickets by Status")
    tab_resolved, tab_not_resolved, tab_ongoing = st.tabs(["Resolved", "Not Resolved", "Ongoing"])

    # You can tweak these buckets to match your real statuses
    resolved_statuses = tuple(s.lower() for s in ("resolved", "solved", "completed"))
    not_resolved_statuses = tuple(s.lower() for s in ("closed", "rejected", "cancelled", "won't fix", "invalid"))
    ongoing_statuses = tuple(s.lower() for s in ("open", "in_progress", "in progress", "pending", "assigned", "acknowledged"))

    def list_tickets(where_sql: str, params: tuple):
        q = f"""
            SELECT 
                st.id,
                st.subject,
                st.category,
                st.status,
                st.priority,
                st.created_date,
                st.resolved_date,
                u.name AS user_name,
                u.email AS user_email
            FROM support_tickets st
            JOIN users u ON st.user_id = u.id
            {where_sql}
            ORDER BY st.created_date DESC
            LIMIT 200
        """
        df = df_from_query(q, params)
        if df.empty:
            st.info("No tickets found for this tab.")
        else:
            st.dataframe(df, use_container_width=True)

    with tab_resolved:
        if len(resolved_statuses) == 0:
            st.info("No 'resolved' statuses configured.")
        else:
            placeholders = ",".join(["?"] * len(resolved_statuses))
            list_tickets(f"WHERE LOWER(st.status) IN ({placeholders})", resolved_statuses)

    with tab_not_resolved:
        if len(not_resolved_statuses) == 0:
            st.info("No 'not resolved' statuses configured.")
        else:
            placeholders = ",".join(["?"] * len(not_resolved_statuses))
            list_tickets(f"WHERE LOWER(st.status) IN ({placeholders})", not_resolved_statuses)

    with tab_ongoing:
        if len(ongoing_statuses) == 0:
            st.info("No 'ongoing' statuses configured.")
        else:
            placeholders = ",".join(["?"] * len(ongoing_statuses))
            list_tickets(f"WHERE LOWER(st.status) IN ({placeholders})", ongoing_statuses)


def render_admin_settings():
    st.header("⚙️ Admin Settings")
    
    st.subheader("🔧 System Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("Database Status: ✅ Connected")
        st.info("Migration Status: " + ("✅ Complete" if meta_get(DB_MIGRATION_FLAG) == '1' else "⚠️ Pending"))
        
        total_plans = exec_query("SELECT COUNT(*) FROM plans", fetch=True)[0][0]
        st.info(f"Total Plans: {total_plans}")
    
    with col2:
        if st.button("Generate Sample Data", help="Reset and generate new sample data"):
            meta_set(MOCK_DATA_CREATED_FLAG, '0')
            create_comprehensive_mock_data()
            st.success("Sample data regenerated!")
            st.rerun()
        
        if st.button("Run Database Migration", help="Add new columns to existing tables"):
            meta_set(DB_MIGRATION_FLAG, '0')
            migrate_database()
            st.success("Database migration completed!")
            st.rerun()
        
        if st.button("Optimize Database", help="Run database optimization"):
            try:
                exec_query("VACUUM")
                exec_query("REINDEX")
                st.success("Database optimized!")
            except Exception as e:
                st.error(f"Error optimizing database: {str(e)}")
    
    st.subheader("📊 System Statistics")
    
    stats_col1, stats_col2, stats_col3 = st.columns(3)
    
    with stats_col1:
        total_subscriptions = exec_query("SELECT COUNT(*) FROM subscriptions", fetch=True)[0][0]
        total_payments = exec_query("SELECT COUNT(*) FROM payments", fetch=True)[0][0]
        st.metric("Total Subscriptions", total_subscriptions)
        st.metric("Total Payments", total_payments)
    
    with stats_col2:
        if column_exists('usage', 'data_used_gb'):
            total_usage = exec_query("SELECT COALESCE(SUM(data_used_gb), 0) FROM usage", fetch=True)[0][0]
            st.metric("Total Data Usage", f"{total_usage:.0f} GB")
        
        if column_exists('support_tickets', 'id'):
            total_tickets = exec_query("SELECT COUNT(*) FROM support_tickets", fetch=True)[0][0]
            st.metric("Total Support Tickets", total_tickets)
    
    with stats_col3:
        db_size = os.path.getsize(DB_PATH) / (1024 * 1024) if os.path.exists(DB_PATH) else 0
        st.metric("Database Size", f"{db_size:.2f} MB")
        
        if os.path.exists('plan_recommendation_model.pkl'):
            model_size = os.path.getsize('plan_recommendation_model.pkl') / (1024 * 1024)
            st.metric("ML Model Size", f"{model_size:.2f} MB")

def evaluate_model():
    """Evaluate the ML model performance"""
    if not os.path.exists('plan_recommendation_model.pkl'):
        st.error("No model found to evaluate")
        return
    
    try:
        model = joblib.load('plan_recommendation_model.pkl')
        training_data = collect_training_data()
        
        if training_data.empty:
            st.error("Not enough data to evaluate model")
            return
        
        training_data = engineer_features(training_data)
        
        feature_columns = [
            'avg_daily_usage', 'max_daily_usage', 'usage_std',
            'estimated_monthly_usage', 'days_since_signup',
            'weekday_avg', 'weekend_avg', 'usage_consistency',
            'city', 'state'
        ]
        
        # Check which features are actually available
        available_features = [col for col in feature_columns if col in training_data.columns]
        
        if not available_features:
            st.error("No suitable features available for evaluation")
            return
        
        target_column = 'plan_category'
        
        X = training_data[available_features]
        y = training_data[target_column]
        
        # Make predictions
        y_pred = model.predict(X)
        
        # Display metrics
        st.subheader("Model Performance Metrics")
        
        accuracy = accuracy_score(y, y_pred)
        accuracy_percent = accuracy * 100
        
        st.metric("Model Accuracy", f"{accuracy_percent:.1f}%")
        
        # Classification report
        report = classification_report(y, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format("{:.3f}"))
        
        # Feature importance if available
        if hasattr(model.named_steps['classifier'], 'feature_importances_'):
            st.subheader("Feature Importance")
            
            # Get feature names after preprocessing
            numeric_features = [col for col in ['avg_daily_usage', 'max_daily_usage', 'usage_std', 
                              'estimated_monthly_usage', 'days_since_signup',
                              'weekday_avg', 'weekend_avg', 'usage_consistency'] if col in available_features]
            categorical_features = [col for col in ['city', 'state'] if col in available_features]
            
            if categorical_features:
                try:
                    cat_encoder = model.named_steps['preprocessor'].named_transformers_['cat']
                    cat_feature_names = cat_encoder.get_feature_names_out(categorical_features)
                    all_features = numeric_features + list(cat_feature_names)
                except:
                    all_features = available_features
            else:
                all_features = numeric_features
            
            importances = model.named_steps['classifier'].feature_importances_
            
            if len(all_features) == len(importances):
                feature_importance = pd.DataFrame({
                    'Feature': all_features,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)
                
                fig = px.bar(feature_importance.head(10), x='Importance', y='Feature', 
                           orientation='h', title="Top 10 Feature Importances")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Feature names and importances don't match. Skipping feature importance chart.")
        
    except Exception as e:
        st.error(f"Error evaluating model: {str(e)}")

# ---------------------------
# Main Application
# ---------------------------
def main():
    st.set_page_config(
        page_title="Enhanced Broadband Portal",
        layout='wide',
        initial_sidebar_state="expanded",
        page_icon="📡"
    )
    
    load_css()
    create_tables()
    migrate_database()
    ensure_default_admin()
    create_comprehensive_mock_data()
    populate_usage_for_all_users(days=60)

    
    if 'user' not in st.session_state:
        st.session_state['user'] = None

    # Sidebar Authentication
    with st.sidebar:
        st.title("📡 Broadband Portal")
        
        if st.session_state['user'] is None:
            st.markdown("### Welcome Back")
            
            tab1, tab2 = st.tabs(["Sign In", "Sign Up"])
            
            with tab1:
                with st.form("signin_form"):
                    username = st.text_input("Username", placeholder="Enter your username")
                    password = st.text_input("Password", type='password', placeholder="Enter your password")
                    submit = st.form_submit_button("Sign In", use_container_width=True)
                    
                    if submit and username and password:
                        ok, res = signin(username, password)
                        if ok:
                            st.session_state['user'] = res
                            st.success("Welcome back!")
                            st.rerun()
                        else:
                            st.error(res)
            
            with tab2:
                with st.form("signup_form"):
                    new_username = st.text_input("Choose Username", placeholder="Enter username")
                    new_password = st.text_input("Password", type='password', placeholder="Create password")
                    full_name = st.text_input("Full Name", placeholder="Enter your full name")
                    email = st.text_input("Email", placeholder="Enter your email")
                    submit_signup = st.form_submit_button("Create Account", use_container_width=True)
                    
                    if submit_signup and new_username and new_password and full_name and email:
                        ok, msg = signup(new_username, new_password, full_name, email)
                        if ok:
                            st.success("Account created! Please sign in.")
                        else:
                            st.error(msg)
        else:
            user = st.session_state['user']
            st.markdown("---")
            st.markdown(f"**👤 {user['name']}**")
            st.markdown(f"*{user['role'].title()}*")
            st.markdown("---")
            
            if st.button("🚪 Logout", use_container_width=True):
                st.session_state['user'] = None
                st.rerun()

    # Main Content Area
    if st.session_state['user'] is None:
        # Landing page for non-authenticated users
        st.markdown("""
        <div style="text-align: center; padding: 4rem 0;">
            <h1 style="font-size: 3rem; margin-bottom: 1rem;">📡 Welcome to Enhanced Broadband Portal</h1>
            <p style="font-size: 1.2rem; color: #6b7280; margin-bottom: 2rem;">
                Your gateway to intelligent internet plans with AI-powered recommendations
            </p>
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        border-radius: 12px; padding: 2rem; color: white; margin: 2rem 0;">
                <h3>Why Choose Our Enhanced Service?</h3>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 2rem; margin-top: 1rem;">
                    <div>🚀 High-Speed Internet</div>
                    <div>🤖 AI-Powered Recommendations</div>
                    <div>📊 Usage Analytics</div>
                    <div>💰 Smart Plan Comparison</div>
                    <div>🛡️ Reliable Service</div>
                    <div>📱 Mobile-First Design</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show sample plans
        st.markdown("### 📋 Featured Plans")
        plans = get_all_plans()[:4]
        
        if len(plans) >= 2:
            cols = st.columns(2)
            for i, plan in enumerate(plans):
                with cols[i % 2]:
                    render_plan_card(plan, show_actions=False)
        
        return

    user = st.session_state['user']
    if user['role'] == 'admin':
        admin_dashboard(user)
    else:
        user_dashboard(user)

if __name__ == '__main__':
    main()