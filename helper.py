import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import os
import pandas as pd
from model_funcs_opp import predictions


import numpy as np

if 'currently_picked' not in st.session_state:
    st.session_state['currently_picked']=[]


def hero_selected(ind):
    if len(st.session_state['currently_picked'])<9:
        st.session_state['currently_picked'].append(ind)
        for i in st.session_state['currently_picked']:
            col6.image(".\\heroes\\"+hero_df[hero_df['sid']==i]['image_names'].values[0])
    else:
        for i in st.session_state['currently_picked']:
            col6.image(".\\heroes\\"+hero_df[hero_df['sid']==i]['image_names'].values[0])
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def loss3(y_true,y_pred):
    fun=keras.losses.CategoricalCrossentropy()

    
    preds=tf.math.top_k(y_pred,3).indices
    answer=tf.cast(tf.argmax(y_true),tf.int32)
    return tf.where(answer in preds,0,tf.convert_to_tensor(fun(y_true,y_pred),tf.float32))


st.title("Dota 2 Draft Recommendation System")
for i in range(4):
    st.write('\n')
hero_df=pd.read_csv('hero_df.csv')
hero_list=os.listdir(".\\heroes\\")
hero=0
col1,col2,col3,col4,col5,gap,col6,col7,col8=st.columns([2,2,2,2,2,1,2,1,1])
col1.text('Select')
col2.text('the')

col3.text('Currently')
col4.text('Selected')
col5.text('Heroes')


for i in range(24):
    for j in [col1,col2,col3,col4,col5]:
        j.image('.\\heroes\\'+hero_df[hero_df['sid']==hero]['image_names'].values[0])
        j.button(hero_df.loc[hero_df['sid']==hero]['localized_name'].values[0],key=hero,on_click=hero_selected,args=[hero])
        hero+=1

local_css("style.css")


col6.subheader("Currently Picked")
for i in range(5):
    col8.write('\n')




clear=col8.button("Clear")
predict=col8.button("Predict",key='prediction')
if clear:
    if len(st.session_state['currently_picked'])==0:
        del st.session_state['currently_picked']
    else: 
        st.session_state['currently_picked'].pop()
        for i in st.session_state['currently_picked']:
                col6.image(".\\heroes\\"+hero_df[hero_df['sid']==i]['image_names'].values[0])


if predict:
    name_list=[]
    for j in st.session_state['currently_picked']:
        name_list.append(hero_df[hero_df['sid']==j]['localized_name'].values[0])
    
    ans=predictions(np.array(name_list))
    for i in st.session_state['currently_picked']:
        col6.image(".\\heroes\\"+hero_df[hero_df['sid']==i]['image_names'].values[0])

    for i in range(3):
        col6.write('\n')
    col6.write('Predictions')

    for i in range(3):
        col6.write('\n')
    for _ in ans:
        col6.image(".\\heroes\\"+hero_df[hero_df['sid']==_]['image_names'].values[0])

    st.session_state['currently_picked']=[]

# st.write(st.session_state['currently_picked'])














