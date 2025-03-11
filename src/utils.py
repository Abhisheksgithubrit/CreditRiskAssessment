# src/utils.py

import streamlit as st
import pandas as pd

def display_metrics(accuracy, report):
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.highlight_max(axis=1))
