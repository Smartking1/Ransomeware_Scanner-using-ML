import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import pickle
import seaborn as sns

loaded_model= pickle.load(open("malware_predictor.sav","rb"))

st.title('Prediction of Ransomware vs Goodware webapp')

st.image('image.jpg', width=500)

st.title('Case Study on Detection of Ransomware and Goodwares')
data= pd.read_csv('Annex.csv')
data2= data.drop(columns=['family'],axis=1)
st.write('Shape of dataset', data.shape)
menu=st.sidebar.radio('Menue',['Home','Prediction family'])
if menu=='Home':
    st.image('image.webp',width=550)
    st.header('Tabular data of ransomware and goodware')
    if st.checkbox('Tabular Data'):
        st.table(data.head(5))
st.header('Statistical summary of dataframe')
if st.checkbox('Statistics'):
    st.table(data.describe())
    
st.header('correlaton graph')
if st.checkbox('Heatmap'):
    fig,ax=plt.subplots(figsize=(50,50))
    sns.heatmap(data2.corr(),annot=True, cmap='coolwarm')
    st.pyplot(fig)
st.title('Graphs')
graph=st.selectbox('Several Graphs',["scatter plot","bar chart","histogram"])
if graph== 'scatter plot':
    fig,ax=plt.subplots(figsize=(5,2.5))
    sns.scatterplot(data=data, x='hosts',y='requests', hue='family')
    st.pyplot(fig)

if graph== 'bar chart':
    fig,ax=plt.subplots(figsize=(5,2.5))
    sns.countplot(data=data, x='family')
    plt.title('count of ransomware and goodware faimly',fontsize=10)
    st.pyplot(fig)
if graph== 'histogram':
    fig,ax=plt.subplots(figsize=(5,2.5))
    sns.distplot(data['domains'], color=sns.color_palette()[0], kde=True)
    plt.title('Distribution of domains')
    st.pyplot(fig)

if menu=="Prediction family":
   def make_prediction(input_data):
    

     #changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] ==0):
        return 'ransom ware'
    elif (prediction[0]==1):
        return 'good ware'
    else:
        return 'ransom ware'



   def main():
    st.header('Prediction of Ransomware vs Goodware Web App')
    
    
    # getting the input data from the user
    
    proc_pid= st.text_input('proc_pid')
    file= st.text_input('file')
    urls= st.text_input('urls')
    type= st.text_input('type')
    name= st.text_input('name')
    ext_urls= st.text_input('ext_urls')
    path=st.text_input('path')
    program= st.text_input('program')
    info= st.text_input('info')
    positives=st.text_input('positives')
    families=st.text_input('families')
    description=st.text_input('description')
    sign_name=st.text_input('sign_name')
    sign_stacktrace=st.text_input('ign_stacktrace')
    arguments=st.text_input('argumen')
    api=st.text_input('api')
    category=st.text_input('category')
    imported_dll_count=st.text_input('imported_dll_count')
    dll= st.text_input('dll')
    pe_res_name=st.text_input('pe_res_name')
    filetype=st.text_input('filetype')
    pe_sec_name=st.text_input('pe_sec_name')
    entropy=st.text_input('entropy')
    hosts=st.text_input('hosts')
    requests=st.text_input('requests')
    mitm=st.text_input('mitm')
    domains=st.text_input('domains')
    dns_servers=st.text_input('dns_servers')
    tcp=st.text_input('tcp')
    udp=st.text_input('udp')
    dead_hosts=st.text_input('dead_hosts')
    proc=st.text_input('proc')
    beh_command_line=st.text_input('beh_command_line')
    process_path=st.text_input('process_path')
    tree_command_line=st.text_input('tree_command_line')
    children=st.text_input('children')
    tree_process_name=st.text_input('ree_process_name')
    command_line=st.text_input('command_li')
    regkey_read=st.text_input('regkey_read')
    directory_enumerated=st.text_input('directory_enumerated')
    regkey_opened=st.text_input('regkey_opened')
    file_created=st.text_input('ile_created')
    wmi_query=st.text_input('wmi_query')
    dll_loaded=st.text_input('dll_loaded')
    regkey_written=st.text_input('regkey_written')
    file_read=st.text_input('file_read')
    apistats=st.text_input('apistats')
    errors=st.text_input('errors')
    action=st.text_input('action')
    log=st.text_input('log')
    
    
   
    
    
    # code for Prediction
    performance = ''
    
    # creating a button for Prediction
    
    if st.button('Result'):
        performance = make_prediction([proc_pid,file,urls,type,name,ext_urls,path,
program,info,positives,families,description,sign_name,sign_stacktrace,
arguments,api,category,imported_dll_count,dll, pe_res_name ,filetype,pe_sec_name,
entropy,hosts,requests,mitm,domains,dns_servers,tcp,udp,dead_hosts,proc,beh_command_line,
process_path,tree_command_line,children,tree_process_name,
command_line,regkey_read,directory_enumerated,regkey_opened,file_created,
wmi_query,dll_loaded,regkey_written,file_read,apistats,errors,action,log])
        
    st.success(performance)


if __name__ == '__main__':
    main()


       

       
       
    
       
    
       


       





    
            





                        
