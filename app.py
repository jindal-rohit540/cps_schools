"""
Student Performance Analytics — Kaggle Dataset
Run: streamlit run app.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
exec(open(os.path.join(os.path.dirname(__file__), "dashboard", "app.py")).read())
