import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv2D, Flatten, MaxPooling2D, Input
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from keras.models import Model
import plotly.graph_objects as go

# Streamlit 앱을 wide mode와 dark mode로 설정
st.set_page_config(layout="wide", initial_sidebar_state="expanded", page_title="Data Analysis and Modeling with Streamlit")

# 사이드바에 페이지 선택 메뉴 추가
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Model Configuration", "Model Results"])

def preprocess_and_select_features(train_df, test_df, scaling_option, feature_selection_option, pca_components=None, autoencoder_params=None):
    numeric_features = train_df.select_dtypes(include=['float64', 'int64']).columns
    if scaling_option == "Normalization":
        scaler = StandardScaler()
    elif scaling_option == "MinMax":
        scaler = MinMaxScaler()
    elif scaling_option == "Robust":
        scaler = RobustScaler()
    else:
        scaler = None
    
    if scaler is not None:
        train_df[numeric_features] = scaler.fit_transform(train_df[numeric_features])
        test_df[numeric_features] = scaler.transform(test_df[numeric_features])

    X_train = train_df.drop('result', axis=1)
    y_train = train_df['result']
    X_test = test_df.drop('result', axis=1)
    y_test = test_df['result']
    
    if feature_selection_option == "PCA":
        pca = PCA(n_components=pca_components)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
    elif feature_selection_option == "AutoEncoder":
        input_dim = X_train.shape[1]
        encoding_dim = autoencoder_params['encoding_dim']
        epochs = autoencoder_params['epochs']
        batch_size = autoencoder_params['batch_size']
        input_layer = Input(shape=(input_dim,))
        encoder_layer = Dense(encoding_dim, activation='relu')(input_layer)
        decoder_layer = Dense(input_dim, activation='sigmoid')(encoder_layer)
        autoencoder = Model(inputs=input_layer, outputs=decoder_layer)
        autoencoder.compile(optimizer='adam', loss='mse')
        autoencoder.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, shuffle=True, validation_split=0.2)
        encoder = Model(inputs=input_layer, outputs=encoder_layer)
        X_train = encoder.predict(X_train)
        X_test = encoder.predict(X_test)
    
    # 이진 분류를 위해 y_train 및 y_test의 클래스 레이블을 이산형 값으로 변환
    y_train = np.where(y_train > 0.5, 1, 0)
    y_test = np.where(y_test > 0.5, 1, 0)
    
    return X_train, y_train, X_test, y_test

def train_model(model_option, X_train, y_train):
    if model_option == "Logistic Regression":
        model = LogisticRegression()
    elif model_option == "Random Forest":
        model = RandomForestClassifier()
    elif model_option == "XGBoost":
        model = XGBClassifier()
    elif model_option == "RNN":
        model = Sequential()
        model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], 1)))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        y_train = to_categorical(y_train)
    elif model_option == "CNN":
        # CNN의 input_shape는 4차원이어야 합니다: (samples, height, width, channels)
        if int(np.sqrt(X_train.shape[1])) ** 2 == X_train.shape[1]:
            X_train = X_train.reshape((X_train.shape[0], int(np.sqrt(X_train.shape[1])), int(np.sqrt(X_train.shape[1])), 1))
        else:
            raise ValueError("Input shape for CNN must be square. Please choose another feature selection method or preprocess the data accordingly.")
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        y_train = to_categorical(y_train)
    
    model.fit(X_train, y_train)
    return model

if page == "Home":
    st.title("Welcome to Data Analysis and Modeling with Streamlit")
    st.write("Use the sidebar to navigate to different pages.")
    
elif page == "Model Configuration":
    # Train 데이터 파일 업로드
    train_file = st.file_uploader("Choose a CSV file for training data", type="csv")

    # Test 데이터 파일 업로드
    test_file = st.file_uploader("Choose a CSV file for testing data", type="csv")

    if train_file is not None and test_file is not None:
        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)
        st.write("Train Data")
        st.write(train_df.head())
        st.write("Test Data")
        st.write(test_df.head())

        # 'result' 열이 없는 경우 임시로 생성 (예를 들어 임의의 이진 분류로)
        if 'result' not in train_df.columns:
            train_df['result'] = np.random.randint(0, 2, train_df.shape[0])
            st.warning("'result' column not found in training data. An example 'result' column has been added for demonstration purposes.")
        if 'result' not in test_df.columns:
            test_df['result'] = np.random.randint(0, 2, test_df.shape[0])
            st.warning("'result' column not found in testing data. An example 'result' column has been added for demonstration purposes.")

        # 다섯 개의 컬럼을 생성하여 각각 모델 설정, 전처리 결과 및 그래프를 표시
        col1, col2, col3 = st.columns(3)

        with col1:
            # 모델 1 설정
            st.subheader("Model 1 Configuration")
            scaling_option1 = st.radio("Choose a scaling method for Model 1", ("None", "Normalization", "MinMax", "Robust"), key='model1_scaling')
            feature_selection_option1 = st.radio("Choose a feature selection method for Model 1", ("None", "PCA", "AutoEncoder"), key='model1_feature')
            if feature_selection_option1 == "PCA":
                pca_components1 = st.number_input("Number of PCA components for Model 1", min_value=1, max_value=train_df.shape[1], value=2)
            else:
                pca_components1 = None
            if feature_selection_option1 == "AutoEncoder":
                with st.expander("AutoEncoder Settings for Model 1"):
                    encoding_dim1 = st.number_input("Encoding Dimension", min_value=1, max_value=train_df.shape[1], value=32)
                    epochs1 = st.number_input("Number of Epochs", min_value=1, value=50)
                    batch_size1 = st.number_input("Batch Size", min_value=1, value=256)
                    autoencoder_params1 = {
                        'encoding_dim': encoding_dim1,
                        'epochs': epochs1,
                        'batch_size': batch_size1
                    }
            else:
                autoencoder_params1 = None
            model_option1 = st.selectbox("Choose first model", ("Logistic Regression", "Random Forest", "XGBoost", "RNN", "CNN"), key='model1')

            # 모델 1 전처리 결과
            X_train1, y_train1, X_test1, y_test1 = preprocess_and_select_features(train_df.copy(), test_df.copy(), scaling_option1, feature_selection_option1, pca_components1, autoencoder_params1)
            st.session_state['model_option1'] = model_option1
            st.session_state['X_train1'] = X_train1
            st.session_state['y_train1'] = y_train1
            st.session_state['X_test1'] = X_test1
            st.session_state['y_test1'] = y_test1
            st.write("Data after preprocessing and feature selection for Model 1")
            st.write(pd.DataFrame(X_train1).head())

        with col2:
            # 모델 2 설정
            st.subheader("Model 2 Configuration")
            scaling_option2 = st.radio("Choose a scaling method for Model 2", ("None", "Normalization", "MinMax", "Robust"), key='model2_scaling')
            feature_selection_option2 = st.radio("Choose a feature selection method for Model 2", ("None", "PCA", "AutoEncoder"), key='model2_feature')
            if feature_selection_option2 == "PCA":
                pca_components2 = st.number_input("Number of PCA components for Model 2", min_value=1, max_value=train_df.shape[1], value=2, key='pca2')
            else:
                pca_components2 = None
            if feature_selection_option2 == "AutoEncoder":
                with st.expander("AutoEncoder Settings for Model 2"):
                    encoding_dim2 = st.number_input("Encoding Dimension", min_value=1, max_value=train_df.shape[1], value=32, key='ae_dim2')
                    epochs2 = st.number_input("Number of Epochs", min_value=1, value=50, key='ae_epochs2')
                    batch_size2 = st.number_input("Batch Size", min_value=1, value=256, key='ae_batch2')
                    autoencoder_params2 = {
                        'encoding_dim': encoding_dim2,
                        'epochs': epochs2,
                        'batch_size': batch_size2
                    }
            else:
                autoencoder_params2 = None
            model_option2 = st.selectbox("Choose second model", ("Logistic Regression", "Random Forest", "XGBoost", "RNN", "CNN"), key='model2')

            # 모델 2 전처리 결과
            X_train2, y_train2, X_test2, y_test2 = preprocess_and_select_features(train_df.copy(), test_df.copy(), scaling_option2, feature_selection_option2, pca_components2, autoencoder_params2)
            st.session_state['model_option2'] = model_option2
            st.session_state['X_train2'] = X_train2
            st.session_state['y_train2'] = y_train2
            st.session_state['X_test2'] = X_test2
            st.session_state['y_test2'] = y_test2
            st.write("Data after preprocessing and feature selection for Model 2")
            st.write(pd.DataFrame(X_train2).head())

        with col3:
            # 모델 학습
            model1 = train_model(model_option1, X_train1, y_train1)
            model2 = train_model(model_option2, X_train2, y_train2)

            # 예측 및 평가
            y_pred1 = model1.predict(X_test1)
            y_pred2 = model2.predict(X_test2)
            if model_option1 in ["RNN", "CNN"]:
                y_pred1 = np.argmax(y_pred1, axis=1)
            if model_option2 in ["RNN", "CNN"]:
                y_pred2 = np.argmax(y_pred2, axis=1)

            # 평가 지표 계산
            accuracy1 = accuracy_score(y_test1, y_pred1)
            accuracy2 = accuracy_score(y_test2, y_pred2)
            f1_1 = f1_score(y_test1, y_pred1)
            f1_2 = f1_score(y_test2, y_pred2)
            recall1 = recall_score(y_test1, y_pred1)
            recall2 = recall_score(y_test2, y_pred2)
            precision1 = precision_score(y_test1, y_pred1)
            precision2 = precision_score(y_test2, y_pred2)

            # 평가 지표 시각화
            st.subheader("Model Results")
            metrics = ['Accuracy', 'F1 Score', 'Recall', 'Precision']
            model1_scores = [accuracy1, f1_1, recall1, precision1]
            model2_scores = [accuracy2, f1_2, recall2, precision2]

            fig = go.Figure(data=[
                go.Bar(name='Model 1', x=metrics, y=model1_scores),
                go.Bar(name='Model 2', x=metrics, y=model2_scores)
            ])
            fig.update_layout(barmode='group', title='Model Comparison', yaxis_title='Score')
            st.plotly_chart(fig)

elif page == "Model Results":
    if 'model_option1' in st.session_state and 'X_train1' in st.session_state and 'y_train1' in st.session_state:
        model_option1 = st.session_state['model_option1']
        X_train1 = st.session_state['X_train1']
        y_train1 = st.session_state['y_train1']
        X_test1 = st.session_state['X_test1']
        y_test1 = st.session_state['y_test1']

    if 'model_option2' in st.session_state and 'X_train2' in st.session_state and 'y_train2' in st.session_state:
        model_option2 = st.session_state['model_option2']
        X_train2 = st.session_state['X_train2']
        y_train2 = st.session_state['y_train2']
        X_test2 = st.session_state['X_test2']
        y_test2 = st.session_state['y_test2']

    col1, col2, col3, col4, col5 = st.columns(5)

    

