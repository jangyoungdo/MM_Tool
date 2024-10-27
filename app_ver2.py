import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Flatten, Dense, Conv1D, BatchNormalization, ReLU, GlobalAveragePooling1D, Input
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from tensorflow.keras.models import Model
import plotly.graph_objects as go

# Streamlit 앱을 wide mode와 dark mode로 설정
st.set_page_config(layout="wide", initial_sidebar_state="expanded", page_title="Data Analysis and Modeling with Streamlit")

# 사이드바에 페이지 선택 메뉴 추가
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Model Configuration", "Model Results"])

# 모델 리스트를 저장할 리스트 초기화
if 'model_list' not in st.session_state:
    st.session_state['model_list'] = []

# 데이터셋을 저장할 변수 초기화
if 'train_df' not in st.session_state:
    st.session_state['train_df'] = None
if 'test_df' not in st.session_state:
    st.session_state['test_df'] = None

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

    X_train = train_df.drop('result', axis=1).values
    y_train = train_df['result'].values
    X_test = test_df.drop('result', axis=1).values
    y_test = test_df['result'].values
    
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

def make_rnn_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=256, return_sequences=True, input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    return model

def make_cnn_model(input_shape):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, padding="same", input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Conv1D(filters=64, kernel_size=3, padding="same"))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Conv1D(filters=64, kernel_size=3, padding="same"))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(GlobalAveragePooling1D())
    model.add(Dense(2, activation="softmax"))
    return model

def train_model(model_option, X_train, y_train):
    if model_option == "Logistic Regression":
        model = LogisticRegression()
        model.fit(X_train, y_train)
    elif model_option == "Random Forest":
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
    elif model_option == "XGBoost":
        model = XGBClassifier()
        model.fit(X_train, y_train)
    elif model_option == "RNN":
        input_shape = (1, X_train.shape[2])
        model = make_rnn_model(input_shape)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
    elif model_option == "CNN":
        input_shape = (X_train.shape[1], 1)
        model = make_cnn_model(input_shape)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(X_train[..., None], y_train, epochs=20, batch_size=32, validation_split=0.2)
    
    return model

if page == "Home":
    st.title("Welcome to Data Analysis and Modeling with Streamlit")
    st.write("Use the sidebar to navigate to different pages.")
    
elif page == "Model Configuration":
    # Train 데이터 파일 업로드
    train_file = st.file_uploader("Choose a CSV file for training data", type="csv")

    # Test 데이터 파일 업로드
    test_file = st.file_uploader("Choose a CSV file for testing data", type="csv")

    if train_file is not None:
        st.session_state['train_df'] = pd.read_csv(train_file)
    if test_file is not None:
        st.session_state['test_df'] = pd.read_csv(test_file)

    if st.session_state['train_df'] is not None and st.session_state['test_df'] is not None:
        train_df = st.session_state['train_df']
        test_df = st.session_state['test_df']
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

        col1, col2 = st.columns([2, 1])

        with col1:
            
            # 모델 설정
            st.subheader("Model Configuration")
            scaling_option = st.radio("Choose a scaling method", ("None", "Normalization", "MinMax", "Robust"))
            feature_selection_option = st.radio("Choose a feature selection method", ("None", "PCA", "AutoEncoder"))
            if feature_selection_option == "PCA":
                pca_components = st.number_input("Number of PCA components", min_value=1, max_value=train_df.shape[1], value=2)
            else:
                pca_components = None
            if feature_selection_option == "AutoEncoder":
                with st.expander("AutoEncoder Settings"):
                    encoding_dim = st.number_input("Encoding Dimension", min_value=1, max_value=train_df.shape[1], value=32)
                    epochs = st.number_input("Number of Epochs", min_value=1, value=50)
                    batch_size = st.number_input("Batch Size", min_value=1, value=256)
                    autoencoder_params = {
                        'encoding_dim': encoding_dim,
                        'epochs': epochs,
                        'batch_size': batch_size
                    }
            else:
                autoencoder_params = None
            model_option = st.selectbox("Choose model", ("Logistic Regression", "Random Forest", "XGBoost", "RNN", "CNN"))

            # 모델 전처리 결과
            X_train, y_train, X_test, y_test = preprocess_and_select_features(train_df.copy(), test_df.copy(), scaling_option, feature_selection_option, pca_components, autoencoder_params)
            
            if model_option == "RNN":
                X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
                X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
                y_train = to_categorical(y_train)
                y_test = to_categorical(y_test)
            
            if model_option == "CNN":
                X_train = X_train[..., None]
                X_test = X_test[..., None]
                y_train = to_categorical(y_train)
                y_test = to_categorical(y_test)

            st.write("Data after preprocessing and feature selection")
            st.write(pd.DataFrame(X_train.reshape(X_train.shape[0], -1)).head())

            # 생성 버튼
            if st.button("모델 생성"):
                model_data = {
                    'model_option': model_option,
                    'X_train': X_train,
                    'y_train': y_train,
                    'X_test': X_test,
                    'y_test': y_test,
                    'scaling_option': scaling_option,
                    'feature_selection_option': feature_selection_option,
                    'pca_components': pca_components,
                    'autoencoder_params': autoencoder_params
                }
                st.session_state['model_list'].append(model_data)
                st.experimental_rerun()

            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            
            # 저장된 모델 리스트 표시
            st.subheader("Model List")
            if 'model_list' in st.session_state:
                for i, model_data in enumerate(st.session_state['model_list']):
                    st.write(f"Model {i+1}: {model_data['model_option']}")
                    st.write(f"Scaling: {model_data['scaling_option']}")
                    st.write(f"Feature Selection: {model_data['feature_selection_option']}")
            else:
                st.write("No models saved yet.")
            st.markdown("</div>", unsafe_allow_html=True)

elif page == "Model Results":
    if 'model_list' in st.session_state:
        selected_models = st.multiselect("Select models to compare (up to 3)", [f"Model {i+1}" for i in range(len(st.session_state['model_list']))])
        
        if len(selected_models) > 3:
            st.error("You can select up to 3 models only.")
        else:
            models_to_compare = [st.session_state['model_list'][int(model.split()[1]) - 1] for model in selected_models]

            if models_to_compare:
                fig = go.Figure()

                for i, model_data in enumerate(models_to_compare):
                    model_option = model_data['model_option']
                    X_train = model_data['X_train']
                    y_train = model_data['y_train']
                    X_test = model_data['X_test']
                    y_test = model_data['y_test']

                    st.subheader(f"{model_option} Results")
                    
                    # 모델 학습
                    model = train_model(model_option, X_train, y_train)

                    # 예측 및 평가
                    y_pred = model.predict(X_test)
                    if model_option in ["RNN", "CNN"]:
                        y_pred = np.argmax(y_pred, axis=1)
                        y_test_classes = np.argmax(y_test, axis=1)
                    else:
                        y_test_classes = y_test

                    # 평가 지표 계산
                    accuracy = accuracy_score(y_test_classes, y_pred)
                    f1 = f1_score(y_test_classes, y_pred)
                    recall = recall_score(y_test_classes, y_pred)
                    precision = precision_score(y_test_classes, y_pred)

                    # 평가 지표 시각화
                    metrics = ['Accuracy', 'F1 Score', 'Recall', 'Precision']
                    model_scores = [accuracy, f1, recall, precision]

                    fig.add_trace(go.Bar(name=f'{selected_models[i]}: {model_option}', x=metrics, y=model_scores))

                fig.update_layout(barmode='group', title='Model Comparison', yaxis_title='Score', autosize=True)
                fig.update_layout(width=1000, height=600)
                st.plotly_chart(fig, use_container_width=True)
