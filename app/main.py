import streamlit as st
import base64
import os
import pandas as pd
import pickle
import plotly.graph_objects as go
import numpy as np

def clean_data():
    data = pd.read_csv("Data/data.csv")
    # data = data['diagnosis'].fillna(data['diagnosis'].mean())
    data = data.drop(["Unnamed: 32","id"],axis=1)
    data['diagnosis'] = data['diagnosis'].map({"M" : 1,"B" : 0})
    # print(data.head())
    return data

def add_sidebar():
    st.sidebar.header("By: AIMAD BOUYA")
    st.sidebar.header("Cell Nuclei Measurements")

    data = clean_data()

    input_dicts = {}

    slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)","texture_se"),
        ("Perimeter (se)","perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)","area_worst"),
        ("Smothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst")
    ]

    for label, key in slider_labels:
        input_dicts[key] = st.sidebar.slider(
            label=label,
            min_value=float(0),
            max_value=float(data[key].max()),
            value=float(data[key].mean())
            )
      
    return input_dicts


def get_scaled_values(input_dict):
    data = clean_data()

    X = data.drop(['diagnosis'],axis=1)
    
    scaled_dict = {}

    for key,value in input_dict.items():
        max_val = X[key].max()
        min_val = X[key].min()
        scaled_value = (value - min_val) / (max_val - min_val)

        scaled_dict[key] = scaled_value

    return scaled_dict


def radar_chart(input_data):

    input_data = get_scaled_values(input_data)

    categories = ['Area','Perimeter','Texture',
                  'Radius','Fractal Dimension',
                   'Symmetry','Concave Points','Concavity',
                   'Compactness','Smoothness']
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r = [
            input_data['area_mean'],input_data['perimeter_mean'],
            input_data['texture_mean'],input_data['radius_mean'],
            input_data['fractal_dimension_mean'], input_data['symmetry_mean'],
            input_data['concave points_mean'],input_data['concavity_mean'],
            input_data['compactness_mean'],input_data['smoothness_mean']
        ],
        theta=categories,
        fill='toself',
        name='Mean Value'
    ))
    fig.add_trace(go.Scatterpolar(
        r = [
            input_data['area_se'],input_data['perimeter_se'],
            input_data['texture_se'],input_data['radius_se'],
            input_data['fractal_dimension_se'], input_data['symmetry_se'],
            input_data['concave points_se'],input_data['concavity_se'],
            input_data['compactness_se'],input_data['smoothness_se']
        ],
        theta=categories,
        fill='toself',
        name='Standard Error'
    ))
    fig.add_trace(go.Scatterpolar(
        r = [
            input_data['area_worst'],input_data['perimeter_worst'],
            input_data['texture_worst'],input_data['radius_worst'],
            input_data['fractal_dimension_worst'], input_data['symmetry_worst'],
            input_data['concave points_worst'],input_data['concavity_worst'],
            input_data['compactness_worst'],input_data['smoothness_worst']
        ],
        theta=categories,
        fill='toself',
        name='Worst Value'
    ))

    fig.update_layout(
        polar = dict(
            radialaxis=dict(
                visible=True,
                range=[0,1]
            )),
            showlegend=True,
            
    )

    return fig


def add_predictions(input_data):
    model = pickle.load(open('model/model.pkl','rb'))
    sc = pickle.load(open('model/scaler.pkl','rb'))

    input_array = np.array(list(input_data.values())).reshape(1,-1)
    input_array_scaled = sc.transform(input_array)

    prediction = model.predict(input_array_scaled)
    
    if prediction[0] == 0:
        st.markdown(f"""<div class='container2'>
                    <h2> Cell Cluster predictions </h2>
                    <p>The result is : </p>
                    <p id='resultB'> Benign</p>
                    <p>Probability of being benign : <span id='benign'>{model.predict_proba(input_array_scaled)[0][0]} </span> </p>
                    <p>Probability of being Malicious : <span id='malicious'>{model.predict_proba(input_array_scaled)[0][1]} </span> </p>
                    <p>This app can assist medical professionals in making a diagnosis, but should not be used as a substitude for a professional diagnosis.</p>
                </div>""",unsafe_allow_html=True)
    else:

        
        st.markdown(f"""<div class='container2'>
                    <h2> Cell Cluster predictions </h2>
                    <p>The result is : </p>
                    <p id='resultM'>Malicious</p>
                    <p>Probability of being benign : <span id='benign'>{model.predict_proba(input_array_scaled)[0][0]} </span> </p>
                    <p>Probability of being Malicious : <span id='malicious'>{model.predict_proba(input_array_scaled)[0][1]} </span> </p>
                    <p>This app can assist medical professionals in making a diagnosis, but should not be used as a substitude for a professional diagnosis.</p>
                </div>""",unsafe_allow_html=True)
def main():
    st.set_page_config(
        page_title="Breast Cancer Predictor",
        page_icon=":female-doctor:",
        layout="wide",
        initial_sidebar_state="expanded"
    )


    # Adding a sidebar, and return the values of each column from the slider
    input_data = add_sidebar()

    # st.write(input_data)
    
    # Path to the image (relative to the script location)
    image_path = "images/breast_cancer.png"

    
    # Read the image and encode it in base64
    try:
        with open(image_path,"rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        st.error(f"Image not found at : {os.path.abspath(image_path)}")
        return
    
    # Custom CSS for styling the page
    custom_css = """
    <style>
 
            /* h1 */
        h1{
            text-align:center;
            position:relative;
            bottom:50px;
        }
        /* Style for the image */
        #breast-cancer-image {
            width: 100%; /* Make the image responsive */
            max-width: 400px; /* Set a maximum width */
            border-radius: 10px; /* Rounded corners */
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); /* Add a shadow */
            display: block; /* Center the image */
            margin: 0 auto; /* Center the image */
            transition: transform 0.9s ease-out; /* Smooth hover effect */
            position:relative;
            bottom:60px;
        }

        /* Hover effect */
        #breast-cancer-image:hover {
            transform: scale(1.05); /* Slightly enlarge the image on hover */
            filter: drop-shadow(0px 0px 20px #ff69b4); /* Add a glow effect */
        }

        .container2{
            background: linear-gradient(to left,red,white);
            padding:15px 20px;
            border-radius: 18px;
            width:290px;
            transition: transform 0.3s ease-in-out,filter 0.3s ease-in;
           
            }
        .container2 h2{
            text-align:center;
            color:#000;
          
        }
        .container2 p{
                color:#000;
                font-weight:600;
                
            }

        .container2:hover{
            transform: scale(1.05);
            filter: drop-shadow(0px 0px 14px #fff);
           
        }
        #benign{
            color: rgb(0,210,0);
            padding: 3px 9px;
            background-color: #000;
            border-radius: 9px;
        }
        #malicious{
            color: rgb(210,0,0);
            padding: 3px 9px;
            background-color: #000;
            border-radius: 9px;
        }

        #resultM{
            background-color: rgb(210,0,0);
            padding:3px 6px;
            color:#fff;
            font-weight:600;
            width:120px;
            border-radius:19px;
            text-align:center;
            margin-left:64px;
        }

        #resultB{
                background-color: rgb(0,210,0);
            padding:3px 6px;
            color:#fff;
            font-weight:600;
            width:120px;
            border-radius:19px;
            text-align:center;
            margin-left:64px;
            }
    </style>
    """
    
    # Display the image using markdown and apply CSS
    with st.container():
        st.markdown("<h1>Breast Cancer Predictor</h1>",unsafe_allow_html=True)
        
        # Inject custom CSS
        st.markdown(custom_css, unsafe_allow_html=True)
        
        # Display the image with base64 encoding
        st.markdown(
            f'<img src="data:image/png;base64,{encoded_image}" alt="Breast Cancer Image" id="breast-cancer-image">',
            unsafe_allow_html=True
        )
        st.markdown("""<p>Welcome to the Breast Cancer Predictor app! 
                    This tool leverages machine learning algorithms to predict the likelihood 
                    of breast cancer based on various input features. By inputting relevant data,
                    users can receive a prediction that helps in early detection and timely medical 
                    intervention. The result will be predicted based on the slider in the sidebar,
                    allowing users to fine-tune the prediction by adjusting different input values.</p>"""
                    ,unsafe_allow_html=True)
   
    col1,col2 = st.columns([4,1])

    with col1:
       
        radar = radar_chart(input_data)
        st.plotly_chart(radar)

    with col2:
      add_predictions(input_data)
      st.markdown(custom_css,unsafe_allow_html=True)

if __name__ == '__main__':
    main()
    
