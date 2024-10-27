import streamlit as st
import pandas as pd
import numpy as np
import numpy.matlib
from scipy.stats import f
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Flatten, Dense, Conv1D, BatchNormalization, ReLU, GlobalAveragePooling1D, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import plotly.graph_objects as go

# Streamlit 앱을 wide mode와 dark mode로 설정
st.set_page_config(layout="wide", initial_sidebar_state="expanded", page_title="Data Analysis and Modeling with Streamlit")

# 사이드바에 페이지 선택 메뉴 추가
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Model Configuration", "Anomaly Detection", "Model Results"])

# 모델 리스트를 저장할 리스트 초기화
if 'model_list' not in st.session_state:
    st.session_state['model_list'] = []

# 데이터셋을 저장할 변수 초기화
if 'train_df' not in st.session_state:
    st.session_state['train_df'] = None
if 'test_df' not in st.session_state:
    st.session_state['test_df'] = None

# Anomaly Detection 결과를 저장할 변수 초기화
if 'anomaly_results' not in st.session_state:
    st.session_state['anomaly_results'] = None

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

def T2(tr, ts, alpha):
    tr = pd.DataFrame(tr)
    ts = pd.DataFrame(ts)

    obs = tr.shape[0]  # 관측치 수
    dim = tr.shape[1]  # 차원 수
    mu = np.mean(tr, axis=0)  # 열 기준으로 평균 계산

    # 임계값 계산
    CL = f.ppf(1 - alpha, dim, obs - dim) * (dim * (obs + 1) * (obs - 1) / (obs * (obs - dim)))
    sinv = np.linalg.inv(np.cov(tr.T))  # 공분산 행렬의 역행렬 계산

    mu_mat = np.matlib.repmat(mu, ts.shape[0], 1)  # 평균 행렬 생성
    dte = ts - mu_mat  # 데이터와 평균의 차이 계산
    Tsq_mat = np.zeros((ts.shape[0], 1))
    for i in range(0, ts.shape[0]):
        Tsq_mat[i, 0] = np.dot(np.dot(dte.iloc[i, :], sinv), dte.iloc[i, :].T)  # T2 통계량 계산

    return Tsq_mat.flatten().tolist(), CL  # 리스트로 변환하여 반환 및 임계값 반환

def boot_limit(data, alpha=0.05, upper=True, bootstrap=100):
    alpha = alpha * 100
    if upper:
        alpha = 100 - alpha
    samsize = max(10000, len(data))
    data_array = np.array(data)  # 명시적으로 numpy 배열로 변환
    limit = np.mean([np.percentile(np.random.choice(data_array, samsize, replace=True), alpha) for _ in range(bootstrap)])
    return limit

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
            st.markdown("""
                <div style="background-color: lightgray; padding: 10px; border-radius: 5px;">
                    <h3>Model Configuration</h3>
            """, unsafe_allow_html=True)
            # 모델 설정
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
            st.markdown("""
                <div style="background-color: lightgray; padding: 10px; border-radius: 5px;">
                    <h3>Saved Models</h3>
            """, unsafe_allow_html=True)
            # 저장된 모델 리스트 표시
            if 'model_list' in st.session_state:
                for i, model_data in enumerate(st.session_state['model_list']):
                    st.write(f"Model {i+1}: {model_data['model_option']}")
                    st.write(f"Scaling: {model_data['scaling_option']}")
                    st.write(f"Feature Selection: {model_data['feature_selection_option']}")
                    if model_data['feature_selection_option'] == "PCA":
                        st.write(f"PCA Components: {model_data['pca_components']}")
                    if model_data['feature_selection_option'] == "AutoEncoder":
                        st.write(f"AutoEncoder Params: {model_data['autoencoder_params']}")
            else:
                st.write("No models saved yet.")
            st.markdown("</div>", unsafe_allow_html=True)

elif page == "Anomaly Detection":
    if st.session_state['train_df'] is not None and st.session_state['test_df'] is not None:
        st.write("Anomaly Detection")

        control_limit_option = st.radio("Choose Control Limit", ("CL", "CL2"))

        # 이상 탐지 실행 버튼
        if st.button("Run Anomaly Detection"):
            train_df = st.session_state['train_df']
            test_df = st.session_state['test_df']

            # 독립 변수와 종속 변수 분리 (train 데이터)
            normal_data = train_df[train_df['result'] == 1]

            X_train = train_df.drop(['index', 'result'], axis=1)
            X_normal = normal_data.drop(['index', 'result'], axis=1)
            y_train = train_df['result']

            X_test = test_df.drop(['index', 'result'], axis=1)
            y_test = test_df['result']

            # T2 통계량과 임계값 계산
            T2_values, CL = T2(X_normal, X_test, 0.05)
            CL2 = boot_limit(T2_values)

            # 선택된 Control Limit 적용
            selected_CL = CL if control_limit_option == "CL" else CL2

            # CL 초과 여부 확인 및 테스트 데이터 라벨과 비교
            results = []
            true_positive = 0
            true_negative = 0
            false_positive = 0
            false_negative = 0

            for i, (T2_val, y_true) in enumerate(zip(T2_values, y_test)):
                is_outlier = T2_val > selected_CL
                if is_outlier:
                    if y_true == -1:
                        true_positive += 1  # 실제로 비정상(0)이고, 예측도 비정상(0)
                    else:
                        false_positive += 1  # 실제로 정상(1)인데, 예측이 비정상(0)
                else:
                    if y_true == 1:
                        true_negative += 1  # 실제로 정상(1)이고, 예측도 정상(1)
                    else:
                        false_negative += 1  # 실제로 비정상(0)인데, 예측이 정상(1)

                is_label_match = (is_outlier == 1 and y_true == -1) or (not is_outlier == 0 and y_true == 1)
                results.append((i, T2_val, selected_CL, is_outlier, y_true, is_label_match))

            # 결과 출력
            results_df = pd.DataFrame(results, columns=['Index', 'T2 Value', 'Control Limit', 'Is Outlier', 'True Label', 'Label Match'])
            st.session_state['anomaly_results'] = results_df  # 결과를 세션 상태에 저장
            st.write(results_df)

            # 정확도 계산
            accuracy = (true_positive + true_negative) / len(y_test) * 100

            # 재현율(Recall) 계산
            recall = true_positive / (true_positive + false_negative) * 100 if (true_positive + false_negative) > 0 else 0

            # 정밀도(Precision) 계산
            precision = true_positive / (true_positive + false_positive) * 100 if (true_positive + false_positive) > 0 else 0

            # F1 Score 계산
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            st.write(f"Accuracy: {accuracy:.2f}%")
            st.write(f"Recall: {recall:.2f}%")
            st.write(f"Precision: {precision:.2f}%")
            st.write(f"F1 Score: {f1:.2f}%")

            # 모델 데이터 저장
            model_data = {
                'model_option': 'Anomaly Detection',
                'X_train': X_train,
                'y_train': y_train,
                'X_test': X_test,
                'y_test': y_test,
                'control_limit_option': control_limit_option,
                'results_df': results_df,
                'accuracy': accuracy,
                'recall': recall,
                'precision': precision,
                'f1_score': f1
            }
            st.session_state['model_list'].append(model_data)
            st.experimental_rerun()

        if st.session_state['anomaly_results'] is not None:
            st.write(st.session_state['anomaly_results'])

elif page == "Model Results":
    if 'model_list' in st.session_state:
        selected_models = st.multiselect("Select models to compare (up to 3)", [f"Model {i+1}" for i in range(len(st.session_state['model_list']))])
        
        if len(selected_models) > 3:
            st.error("You can select up to 3 models only.")
        else:
            models_to_compare = [st.session_state['model_list'][int(model.split()[1]) - 1] for model in selected_models]

            if models_to_compare:
                # 모델 정보와 설정을 포함하는 표 생성
                model_info = []
                for i, model_data in enumerate(models_to_compare):
                    model_info.append([
                        f"Model {i+1}",
                        model_data['model_option'],
                        model_data.get('scaling_option', 'N/A'),
                        model_data.get('feature_selection_option', 'N/A'),
                        model_data.get('pca_components', 'N/A'),
                        model_data.get('autoencoder_params', 'N/A'),
                        model_data.get('control_limit_option', 'N/A')
                    ])

                model_info_df = pd.DataFrame(model_info, columns=['Model', 'Option', 'Scaling', 'Feature Selection', 'PCA Components', 'AutoEncoder Params', 'Control Limit'])
                st.write("### Model Information and Settings")
                st.write(model_info_df)

                fig = go.Figure()

                for i, model_data in enumerate(models_to_compare):
                    model_option = model_data['model_option']
                    if model_option != 'Anomaly Detection':
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
                    
                    else:
                        # Anomaly Detection 결과 시각화
                        accuracy = model_data['accuracy']
                        recall = model_data['recall']
                        precision = model_data['precision']
                        f1 = model_data['f1_score']

                        # 평가 지표 시각화
                        metrics = ['Accuracy', 'Recall', 'Precision', 'F1 Score']
                        model_scores = [accuracy, recall, precision, f1]

                        fig.add_trace(go.Bar(name=f'{selected_models[i]}: {model_option}', x=metrics, y=model_scores))

                fig.update_layout(barmode='group', title='Model Comparison', yaxis_title='Score', autosize=True)
                fig.update_layout(width=1000, height=600)
                st.plotly_chart(fig, use_container_width=True)
