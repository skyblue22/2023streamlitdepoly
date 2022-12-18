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
import pickle
import lifelines

# loading navigation bar style
st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)

# create navigation bar
st.markdown("""
<nav class="navbar fixed-top navbar-expand-lg navbar-dark" style="background-color: #153888;">
  <a class="navbar-brand" href="#" target="_blank">LASF prediction model</a>
""", unsafe_allow_html=True)


def app():
    st.title('Welcome.')
    st.write('This webpage will provide you an individualized prediction for loss of autonomy in swallowing function in ALS patients.')
    st.write('We expect our models to facilitate personalized advance care planning of gastrostomy placement for patients with ALS.')
    st.write('Probability prediction curves are freely available by 3 different machine-learning algorithms.')
    st.write('To access to our service, please register and login.')
