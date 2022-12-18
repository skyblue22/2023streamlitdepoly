# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 19:07:35 2022

@author: Hyeonji Oh
"""

# import libraries needed
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from apps import home, individualprediction, contact_us, how_to_use

# loading navigation bar stylesheet
st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)

# create navigation bar
st.markdown("""
<nav class="navbar fixed-top navbar-expand-lg navbar-dark" style="background-color: #153888;">
  <a class="navbar-brand" href="#" target="_blank">LASF prediction model</a>
</nav>
""", unsafe_allow_html=True)

PAGES = {"Home":home, "How to use":how_to_use, "Individual prediction":individualprediction, "Contact us":contact_us}
selection = st.radio('Go to', list(PAGES.keys()))
page = PAGES[selection]
page.app()