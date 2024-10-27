import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import f
import numpy.matlib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Flatten, Dense, Conv1D, BatchNormalization, ReLU, GlobalAveragePooling1D, Input, MaxPooling1D
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.neighbors import NeighborhoodComponentsAnalysis
import plotly.graph_objects as go
import plotly.express as px
from scipy.cluster.hierarchy import linkage, dendrogram

def main():
    st.set_page_config(layout="wide", initial_sidebar_state="expanded", page_title="데이터 분석 및 모델링 with Streamlit")

    st.sidebar.title("INDEX")
    page = st.sidebar.radio("Go to", ["Introduction", "Ford Sensor Data", "Model Configuration", "Anomaly Detection", "Model Results"])

    if page == "Introduction":
        introduction_page()
    elif page == "Ford Sensor Data":
        ford_sensor_data_page()
    elif page == "Model Configuration":
        model_configuration_page()
    elif page == "Anomaly Detection":
        anomaly_detection_page()
    elif page == "Model Results":
        model_results_page()

def introduction_page():
    st.title("프로젝트 소개")
    st.header("분석 배경 및 목표")
    st.subheader("분석 배경")
    st.write("""
    - **공정(설비) 개요**: 자동차 시스템, 특히 엔진의 정상/비정상 상태를 분석하기 위해 500개의 센서를 통해 수집된 데이터를 사용합니다.
    - **분석 목표**: 시계열 데이터(Time Series Data)를 활용하여 설비의 정상/비정상 상태를 분류합니다. 이를 통해 불량 제품을 정확히 분류하고 예지 보전을 실현하고자 합니다.
    """)

    st.subheader("데이터셋 개요")
    st.write("""
    - **데이터 형태 및 수집 방법**: Ford Classification Challenge에서 제공한 오픈 데이터셋을 사용하며, 센서 데이터는 ARFF 형식으로 제공됩니다.
    - **데이터 개수**: 총 4,921개의 데이터로 구성되어 있으며, 학습용 3,601개, 테스트용 1,320개 데이터로 나뉩니다.
    """)

    st.header("분석 방법론 및 적용 알고리즘")
    st.subheader("분석 모델")
    st.write("""
    - **로지스틱 회귀(Logistic Regression)**: 데이터를 0과 1 사이의 연속적인 확률로 예측하는 회귀 알고리즘.
    - **XGBoost**: Gradient Boosting Algorithm 기반의 알고리즘으로, 여러 개의 Decision Tree를 조합하여 사용.
    - **순환 신경망(Recurrent Neural Network, RNN)**: 유닛 간의 순환적 구조를 갖는 신경망으로 시계열 데이터 분석에 적합.
    - **합성곱 신경망(Convolutional Neural Network, CNN)**: 주로 시각적 데이터 분석에 사용되나 시계열 데이터의 특성 추출에도 활용 가능.
    - **이상 탐지(Anomaly Detection)**: 정상 데이터와 비정상 데이터를 구분하는 통계적 기법과 머신러닝 알고리즘을 사용하여 이상치를 탐지.
    """)

    st.subheader("데이터 분석 과정")
    st.write("""
    - **필요 SW 및 패키지 설치** : Python을 기본으로 하며, scikit-learn, xgboost, tensorflow, keras, pytorch 등의 패키지를 사용합니다.
    - **데이터 전처리 및 시각화** : 데이터 불균형 확인, 시계열 샘플 시각화 등을 통해 데이터를 이해하고 전처리합니다.
    - **특징 선택 및 모델설정** : 데이터의 특징값을 선별, 모델을 선택해 성능을 확인하고 싶은 모델을 생성합니다.
    - **성능 확인** : 생성한 모델들의 성능(Accuracy, F1 score, Recall, Precision)을 파악할 수 있습니다.          
    """)

    st.header("분석 결과 및 시사점")
    st.write("""
    - **분석 결과** : XGBoost 모델이 가장 우수한 성능을 보였으며, 제조업 현장에서 예지 보전 및 불량 제품 분류에 유용하게 적용될 수 있음을 확인했습니다.
    - **시사점** : 본 튜토리얼을 통해 분석 인프라나 역량이 부족한 중소기업에서도 쉽게 AI 기반 예측/분류를 수행할 수 있도록 기여할 수 있습니다.
    """)

def ford_sensor_data_page():
    st.title("데이터 이해")

    training_file = st.file_uploader("Training 파일 업로드", type=["csv", "xlsx"])
    test_file = st.file_uploader("Test 파일 업로드", type=["csv", "xlsx"])

    if training_file is not None:
        st.session_state['train_df'] = pd.read_csv(training_file)

    if test_file is not None:
        st.session_state['test_df'] = pd.read_csv(test_file)

    dataset_option = st.radio("데이터셋 선택", ["Training 데이터", "Test 데이터"], key="dataset_radio")

    if dataset_option == "Training 데이터":
        if 'train_df' in st.session_state and st.session_state['train_df'] is not None:
            display_file(st.session_state['train_df'], "Training 파일")
    else:
        if 'test_df' in st.session_state and st.session_state['test_df'] is not None:
            display_file(st.session_state['test_df'], "Test 파일")
            
def display_file(df, title):
    col1, col2 = st.columns([5, 1])

    with col2:
        st.write(f"### {title} 데이터")
        st.dataframe(df, height=800)

    with col1:
        plot_config = {'staticPlot': True}

        if 'result' in df.columns:
            result_cumsum = df['result'].cumsum().tolist()

            option = st.selectbox(
                '어떤 그래프를 보시겠습니까?',
                ('누적값 그래프', '추세선 그래프')
            )

            if option == '누적값 그래프':
                fig1 = px.line(y=result_cumsum, labels={'index': '시간', 'y': '성능'}, title='시간에 따른 성능차트(정상, 비정상 값 누적)')
                fig1.update_layout(title_font_size=20)
                st.plotly_chart(fig1, use_container_width=True)
            
            elif option == '추세선 그래프' :
                X = np.arange(len(result_cumsum)).reshape(-1, 1)
                y = np.array(result_cumsum)
                model = LinearRegression()
                model.fit(X, y)
                trend = model.predict(X)

                r_squared = model.score(X, y)
                slope = model.coef_[0]
                intercept = model.intercept_

                hovertext = [f'R²: {r_squared:.2f}<br>기울기: {slope:.2f}<br>절편: {intercept:.2f}' for _ in range(len(result_cumsum))]

                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=np.arange(len(result_cumsum)), y=result_cumsum, mode='lines', name='누적값'))
                fig2.add_trace(go.Scatter(x=np.arange(len(result_cumsum)), y=trend, mode='lines', name='추세선', line=dict(dash='dash'), text=hovertext, hoverinfo='text+x+y'))
                fig2.update_layout(title='시간에 따른 성능차트(추세선)', title_font_size=20, xaxis_title='시간', yaxis_title='성능')
                
                st.plotly_chart(fig2, use_container_width=True)

        sensor_columns = [col for col in df.columns if col.startswith('sensor')]
        selected_column = st.selectbox(f"{title}에서 차트로 볼 센서 열을 선택하세요:", sensor_columns)
        if selected_column:
            graph_type = st.radio("그래프 유형 선택", ("시계열 그래프", "박스 플롯"), horizontal=True)
            if graph_type == "시계열 그래프":
                fig3 = px.line(df, y=selected_column, labels={'_index': '시간', selected_column: selected_column}, title='센서별 데이터 차트')
                fig3.update_layout(title_font_size=20)
                st.plotly_chart(fig3, use_container_width=True, config=plot_config)
            elif graph_type == "박스 플롯":
                fig3 = px.box(df, y=selected_column, title=f'{selected_column} 센서 데이터 박스 플롯')
                fig3.update_layout(
                    title_font_size=20,
                    autosize=False,
                    width=1200,
                    height=600,
                    xaxis=dict(rangeslider=dict(visible=False))
                )
                st.plotly_chart(fig3, use_container_width=True, config=plot_config)

def model_configuration_page():
    if 'train_df' not in st.session_state or st.session_state['train_df'] is None or 'test_df' not in st.session_state or st.session_state['test_df'] is None:
        st.warning("먼저 'Ford Sensor Data' 페이지에서 파일을 업로드하세요.")
        return

    train_df = st.session_state['train_df'].copy()
    test_df = st.session_state['test_df'].copy()

    # Check if 'fault' column exists and map 'H', 'F', 'B' to 0, 1, 2 respectively
    if 'fault' not in train_df.columns or train_df['fault'].isnull().any():
        st.warning("'fault' 열이 훈련 데이터에 없습니다. 올바른 데이터가 있는지 확인하세요.")
        return
    if 'fault' not in test_df.columns or test_df['fault'].isnull().any():
        st.warning("'fault' 열이 테스트 데이터에 없습니다. 올바른 데이터가 있는지 확인하세요.")
        return

    # Map 'H' to 0, 'F' to 1, and 'B' to 2 for classification
    train_df['fault'] = train_df['fault'].map({'H': 0, 'F': 1, 'B': 2})
    test_df['fault'] = test_df['fault'].map({'H': 0, 'F': 1, 'B': 2})

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
            <div style="background-color: gray; padding: 10px; border-radius: 5px;">
                <h3>모델 구성</h3>
        """, unsafe_allow_html=True)
        
        scaling_option = st.radio("스케일링 방법 선택", ("None", "Normalization", "MinMaxScaler", "RobustScaler"))
        feature_selection_option = st.radio("특성 선택 방법 선택", ("None", "PCA", "NCA", "AutoEncoder"))
        if feature_selection_option == "PCA":
            pca_components = st.number_input("PCA 구성 요소 수", min_value=1, max_value=train_df.shape[1], value=2)
        else:
            pca_components = None
        if feature_selection_option == "AutoEncoder":
            with st.expander("AutoEncoder 설정"):
                encoding_dim = st.number_input("인코딩 차원", min_value=1, max_value=train_df.shape[1], value=32)
                epochs = st.number_input("에포크 수", min_value=1, value=50)
                batch_size = st.number_input("배치 크기", min_value=1, value=256)
                autoencoder_params = {
                    'encoding_dim': encoding_dim,
                    'epochs': epochs,
                    'batch_size': batch_size
                }
        else:
            autoencoder_params = None
        model_option = st.selectbox("모델 선택", ("Logistic Regression", "Random Forest", "XGBoost", "RNN", "CNN"))

        if model_option in ["RNN", "CNN"]:
            epochs = st.slider("에포크 수", min_value=1, max_value=300, value=20)
            batch_size = st.slider("배치 크기", min_value=1, max_value=300, value=32)
        else:
            epochs = None
            batch_size = None

        X_train, y_train, X_test, y_test = preprocess_and_select_features(train_df, test_df, scaling_option, feature_selection_option, pca_components, autoencoder_params)
        
        # For RNN and CNN, reshape data for multi-class classification
        if model_option == "RNN":
            X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
            X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
            y_train = to_categorical(y_train, num_classes=3)
            y_test = to_categorical(y_test, num_classes=3)
        
        if model_option == "CNN":
            X_train = X_train[..., None]
            X_test = X_test[..., None]
            y_train = to_categorical(y_train, num_classes=3)
            y_test = to_categorical(y_test, num_classes=3)

        st.write("전처리 및 특성 선택 후 데이터")
        st.write(pd.DataFrame(X_train.reshape(X_train.shape[0], -1)).head())

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
                'autoencoder_params': autoencoder_params,
                'epochs': epochs,
                'batch_size': batch_size
            }
            st.session_state['model_list'].append(model_data)
            st.experimental_rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div style="background-color: gray; padding: 10px; border-radius: 5px;">
                <h3>저장된 모델</h3>
        """, unsafe_allow_html=True)
        
        if 'model_list' in st.session_state:
            model_info = []
            for i, model_data in enumerate(st.session_state['model_list']):
                model_info.append([
                    f"Model {i+1}",
                    model_data['model_option'],
                    model_data.get('scaling_option', 'N/A'),
                    model_data.get('feature_selection_option', 'N/A'),
                    model_data.get('pca_components', 'N/A'),
                    model_data.get('autoencoder_params', 'N/A'),
                    model_data.get('epochs', 'N/A'),
                    model_data.get('batch_size', 'N/A')
                ])
            model_info_df = pd.DataFrame(model_info, columns=['Model', 'Option', 'Scaling', 'Feature Selection', 'PCA Components', 'AutoEncoder Params', 'Epochs', 'Batch Size'])
            st.write(model_info_df)
        else:
            st.write("아직 저장된 모델이 없습니다.")
        st.markdown("</div>", unsafe_allow_html=True)


def preprocess_and_select_features(train_df, test_df, scaling_option, feature_selection_option, pca_components=None, autoencoder_params=None):
    numeric_features = train_df.drop(columns=['fault']).select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    # Apply scaling only to the feature columns
    if scaling_option == "StandardScaler":
        scaler = StandardScaler()
    elif scaling_option == "MinMaxScaler":
        scaler = MinMaxScaler()
    elif scaling_option == "RobustScaler":
        scaler = RobustScaler()
    else:
        scaler = None
    
    if scaler is not None:
        train_df[numeric_features] = scaler.fit_transform(train_df[numeric_features])
        test_df[numeric_features] = scaler.transform(test_df[numeric_features])

    # Separate features and target for training and test sets
    X_train = train_df.drop(columns=['fault']).values
    y_train = train_df['fault'].values
    X_test = test_df.drop(columns=['fault']).values
    y_test = test_df['fault'].values

    # If feature selection is NCA, use a subset of data to prevent memory issues
    if feature_selection_option == "NCA":
        # Subset the data (20% of training set) to reduce memory usage
        X_train_subset, _, y_train_subset, _ = train_test_split(X_train, y_train, test_size=0.8, stratify=y_train, random_state=42)

        # Apply NCA on the subset of data
        nca = NeighborhoodComponentsAnalysis()
        X_train_nca = nca.fit_transform(X_train_subset, y_train_subset)
        X_train = nca.transform(X_train)  # Transform the full training set after fitting
        X_test = nca.transform(X_test)    # Transform the test set

    elif feature_selection_option == "PCA":
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

    # Ensure target labels are integer classes (0, 1, 2)
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    return X_train, y_train, X_test, y_test

def make_rnn_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=256, return_sequences=True, input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    # Change 2 to 3 for multi-class classification
    model.add(Dense(3, activation='softmax'))  # Output layer for 3 classes
    return model

def make_cnn_model(input_shape):
    model = Sequential()
    # First convolutional layer
    model.add(Conv1D(filters=64, kernel_size=100, padding="same", input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(ReLU())
    
    # Second convolutional layer
    model.add(Conv1D(filters=32, kernel_size=50, padding="same"))
    model.add(BatchNormalization())
    model.add(ReLU())
    
    # Max Pooling layer
    model.add(MaxPooling1D(pool_size=4))
    
    # Flatten layer
    model.add(Flatten())
    
    # Fully connected layer
    model.add(Dense(128, activation='relu'))
    
    # Output layer for multi-class classification (3 classes)
    model.add(Dense(3, activation="softmax"))
    
    return model

def train_model(model_option, X_train, y_train, epochs=None, batch_size=None):
    model = None
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
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    elif model_option == "CNN":
        input_shape = (X_train.shape[1], 1)
        model = make_cnn_model(input_shape)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    
    return model

def T2(tr, ts, alpha):
    tr = pd.DataFrame(tr)
    ts = pd.DataFrame(ts)

    obs = tr.shape[0]
    dim = tr.shape[1]
    mu = np.mean(tr, axis=0)

    CL = f.ppf(1 - alpha, dim, obs - dim) * (dim * (obs + 1) * (obs - 1) / (obs * (obs - dim)))
    sinv = np.linalg.inv(np.cov(tr.T))

    mu_mat = np.matlib.repmat(mu, ts.shape[0], 1)
    dte = ts - mu_mat
    Tsq_mat = np.zeros((ts.shape[0], 1))
    for i in range(0, ts.shape[0]):
        Tsq_mat[i, 0] = np.dot(np.dot(dte.iloc[i, :], sinv), dte.iloc[i, :].T)

    return Tsq_mat.flatten().tolist(), CL

def boot_limit(data, alpha=0.05, upper=True, bootstrap=100):
    alpha = alpha * 100
    if upper:
        alpha = 100 - alpha
    samsize = max(10000, len(data))
    data_array = np.array(data)
    limit = np.mean([np.percentile(np.random.choice(data_array, samsize, replace=True), alpha) for _ in range(bootstrap)])
    return limit

def anomaly_detection_page():
    if 'train_df' not in st.session_state or st.session_state['train_df'] is None or 'test_df' not in st.session_state or st.session_state['test_df'] is None:
        st.warning("먼저 'Ford Sensor Data' 페이지에서 파일을 업로드하세요.")
        return

    train_df = st.session_state['train_df'].copy()
    test_df = st.session_state['test_df'].copy()
    
    # Ensure 'fault' column is in the dataframe
    if 'fault' not in train_df.columns:
        st.error("'fault' 열이 훈련 데이터에 없습니다.")
        return
    
    if 'fault' not in test_df.columns:
        st.error("'fault' 열이 테스트 데이터에 없습니다.")
        return
    
    st.title("이상 탐지")

    scaling_option = st.radio("Normal 데이터를 스케일링할지 선택하세요", ("None", "StandardScaler", "MinMaxScaler", "RobustScaler"))
    statistic_option = st.radio("사용할 통계량 선택", ("T2", "Max", "Min", "Mean", "Median"))
    alpha = st.slider("유의수준 (alpha) 선택", min_value=0.01, max_value=0.10, value=0.05, step=0.01)
    control_limit_option = st.radio("제어 한계 선택", ("CL", "CL2(Bootstrap)"))

    if st.button("이상 탐지 실행"):
        normal_data = train_df[train_df['fault'] == 1]

        if scaling_option == "StandardScaler":
            scaler = StandardScaler()
        elif scaling_option == "MinMaxScaler":
            scaler = MinMaxScaler()
        elif scaling_option == "RobustScaler":
            scaler = RobustScaler()
        else:
            scaler = None

        if scaler is not None:
            normal_data[normal_data.columns.difference(['index', 'fault'])] = scaler.fit_transform(normal_data[normal_data.columns.difference(['index', 'fault'])])
            test_df[test_df.columns.difference(['index', 'fault'])] = scaler.transform(test_df[test_df.columns.difference(['index', 'fault'])])

        X_normal = normal_data.drop(['index', 'fault'], axis=1)
        X_test = test_df.drop(['index', 'fault'], axis=1)
        y_test = test_df['fault']

        if statistic_option == "T2":
            T2_values, CL = T2(X_normal, X_test, alpha)
            CL2 = boot_limit(T2_values, alpha=alpha)

            selected_CL = CL if control_limit_option == "CL" else CL2

            results = []
            true_positive = 0
            true_negative = 0
            false_positive = 0
            false_negative = 0

            for i, (T2_val, y_true) in enumerate(zip(T2_values, y_test)):
                is_outlier = T2_val > selected_CL
                if is_outlier:
                    if y_true == -1:
                        true_positive += 1
                    else:
                        false_positive += 1
                else:
                    if y_true == 1:
                        true_negative += 1
                    else:
                        false_negative += 1

                is_label_match = (is_outlier == 1 and y_true == -1) or (not is_outlier == 0 and y_true == 1)
                results.append((i, T2_val, selected_CL, is_outlier, y_true, is_label_match))

            results_df = pd.DataFrame(results, columns=['Index', 'T2 Value', 'Control Limit', 'Is Outlier', 'True Label', 'Label Match'])
            st.session_state['anomaly_results'] = results_df
            st.session_state['graph_data'] = {
                'x': results_df['Index'],
                'y': results_df['T2 Value'],
                'cl': results_df['Control Limit'].iloc[0]
            }

        else:
            if statistic_option == "Max":
                normal_statistic = X_normal.max(axis=0)
                test_statistic = X_test.max(axis=0)
            elif statistic_option == "Min":
                normal_statistic = X_normal.min(axis=0)
                test_statistic = X_test.min(axis=0)
            elif statistic_option == "Mean":
                normal_statistic = X_normal.mean(axis=0)
                test_statistic = X_test.mean(axis=0)
            elif statistic_option == "Median":
                normal_statistic = X_normal.median(axis=0)
                test_statistic = X_test.median(axis=0)

            normal_statistic_mean = normal_statistic.mean()
            test_statistic_mean = test_statistic.mean()
            diff = abs(normal_statistic_mean - test_statistic_mean)

            CL = diff + (2 * normal_statistic.std())
            CL2 = boot_limit(test_statistic, alpha=alpha, upper=True, bootstrap=100)

            selected_CL = CL if control_limit_option == "CL" else CL2

            is_outlier = test_statistic > selected_CL
            results = []
            true_positive = 0
            true_negative = 0
            false_positive = 0
            false_negative = 0

            for i, (is_outlier_value, y_true) in enumerate(zip(is_outlier, y_test)):
                if is_outlier_value:
                    if y_true == -1:
                        true_positive += 1
                    else:
                        false_positive += 1
                else:
                    if y_true == 1:
                        true_negative += 1
                    else:
                        false_negative += 1

                is_label_match = (is_outlier_value == 1 and y_true == -1) or (not is_outlier_value == 0 and y_true == 1)
                results.append((i, is_outlier_value, selected_CL, y_true, is_label_match))

            results_df = pd.DataFrame(results, columns=['Index', 'Is Outlier', 'Control Limit', 'True Label', 'Label Match'])
            st.session_state['anomaly_results'] = results_df
            st.session_state['graph_data'] = {
                'x': results_df['Index'],
                'y': test_statistic,
                'cl': selected_CL
            }
       
            st.write(results_df)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=results_df['Index'], y=test_statistic, mode='markers', name='Test Statistic'))
            fig.add_trace(go.Scatter(x=results_df['Index'], y=[selected_CL]*len(results_df), mode='lines', name='Control Limit', line=dict(color='red', width=4)))
            fig.update_layout(title=f'{statistic_option} Values and Control Limit', xaxis_title='Index', yaxis_title=f'{statistic_option} Value', autosize=True)
            st.plotly_chart(fig, use_container_width=True)

        accuracy = (true_positive + true_negative) / len(y_test)
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        model_data = {
            'model_option': 'Anomaly Detection',
            'X_train': normal_data.drop(['index', 'result'], axis=1),
            'y_train': normal_data['result'],
            'X_test': X_test,
            'y_test': y_test,
            'control_limit_option': control_limit_option,
            'results_df': results_df,
            'accuracy': accuracy,
            'recall': recall,
            'precision': precision,
            'f1_score': f1,
            'scaling_option': scaling_option,
            'statistic_option': statistic_option
        }
        st.session_state['model_list'].append(model_data)
        st.experimental_rerun()

    if st.session_state['anomaly_results'] is not None:
        st.write(st.session_state['anomaly_results'])
    if st.session_state['graph_data'] is not None:
        graph_data = st.session_state['graph_data']
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=graph_data['x'], y=graph_data['y'], mode='markers', name='Values'))
        fig.add_trace(go.Scatter(x=graph_data['x'], y=[graph_data['cl']]*len(graph_data['x']), mode='lines', name='Control Limit', line=dict(color='red', width=4)))
        fig.update_layout(title='Values and Control Limit', xaxis_title='Index', yaxis_title='Value', autosize=True)
        st.plotly_chart(fig, use_container_width=True)

def model_results_page():
    if 'model_list' in st.session_state:
        selected_models = st.multiselect("비교할 모델 선택 (최대 3개)", [f"모델 {i+1}" for i in range(len(st.session_state['model_list']))])
        
        if len(selected_models) > 3:
            st.error("최대 3개의 모델만 선택할 수 있습니다.")
        else:
            models_to_compare = [st.session_state['model_list'][int(model.split()[1]) - 1] for model in selected_models]

            if models_to_compare:
                model_info = []
                comparison_metrics = ['Accuracy', 'F1 Score', 'Recall', 'Precision']
                all_model_scores = []

                for i, model_data in enumerate(models_to_compare):
                    model_info.append([
                        f"모델 {i+1}",
                        model_data['model_option'],
                        model_data.get('scaling_option', 'N/A'),
                        model_data.get('feature_selection_option', 'N/A'),
                        model_data.get('pca_components', 'N/A'),
                        model_data.get('autoencoder_params', 'N/A'),
                        model_data.get('control_limit_option', 'N/A'),
                        model_data.get('statistic_option', 'N/A'),
                        model_data.get('epochs', 'N/A'),
                        model_data.get('batch_size', 'N/A')
                    ])
                    
                    model_option = model_data['model_option']
                    if model_option != 'Anomaly Detection':
                        X_train = model_data['X_train']
                        y_train = model_data['y_train']
                        X_test = model_data['X_test']
                        y_test = model_data['y_test']
                        
                        model = train_model(model_option, X_train, y_train, model_data.get('epochs'), model_data.get('batch_size'))

                        if model is not None:
                            y_pred = model.predict(X_test)

                            if model_option in ["RNN", "CNN"]:
                                y_pred = np.argmax(y_pred, axis=1)  # Convert one-hot encoded predictions to class labels
                                y_test_classes = np.argmax(y_test, axis=1)
                            else:
                                y_test_classes = y_test

                            accuracy = accuracy_score(y_test_classes, y_pred)
                            f1 = f1_score(y_test_classes, y_pred, average='macro')  # Use 'macro' for multi-class
                            recall = recall_score(y_test_classes, y_pred, average='macro')  # Use 'macro' for multi-class
                            precision = precision_score(y_test_classes, y_pred, average='macro')  # Use 'macro' for multi-class

                            model_scores = [accuracy, f1, recall, precision]
                        else:
                            st.error(f"{model_option} 모델을 생성하지 못했습니다.")
                            model_scores = [0, 0, 0, 0]  # Placeholder for failed model
                    else:
                        accuracy = model_data['accuracy']
                        recall = model_data['recall']
                        precision = model_data['precision']
                        f1 = model_data['f1_score']

                        model_scores = [accuracy, recall, precision, f1]

                    all_model_scores.append((f"모델 {i+1}: {model_option}", model_scores))

                # Display model information
                model_info_df = pd.DataFrame(model_info, columns=['Model', 'Option', 'Scaling', 'Feature Selection', 'PCA Components', 'AutoEncoder Params', 'Control Limit', 'Statistic', 'Epochs', 'Batch Size'])
                st.write("### 모델 정보 및 설정")
                st.write(model_info_df)

                # Compare models using metrics
                comparison_df = pd.DataFrame(
                    [scores for name, scores in all_model_scores],
                    columns=comparison_metrics,
                    index=[name for name, scores in all_model_scores]
                ).T

                fig = px.bar(comparison_df, barmode='group', title='모델 비교', labels={'value': '점수', 'variable': '모델', 'index': '지표'})
                st.plotly_chart(fig, use_container_width=True)

                st.write("""
                ### 지표 설명
                - **Accuracy**: 전체 관측값 중에서 올바르게 예측된 관측값의 비율을 나타냅니다. 모델의 전반적인 효과를 나타냅니다.
                - **F1 Score**: 정밀도와 재현율의 조화 평균입니다. 클래스 분포가 불균형할 때 유용합니다.
                - **Recall**: 실제 클래스에서 올바르게 예측된 양성 관측값의 비율을 나타냅니다. 민감도 또는 참 양성 비율이라고도 합니다.
                - **Precision**: 전체 예측된 양성 중에서 올바르게 예측된 양성 관측값의 비율을 나타냅니다. 양성 예측의 정확도를 나타냅니다.
                """)

if 'model_list' not in st.session_state:
    st.session_state['model_list'] = []
if 'train_df' not in st.session_state:
    st.session_state['train_df'] = None
if 'test_df' not in st.session_state:
    st.session_state['test_df'] = None
if 'anomaly_results' not in st.session_state:
    st.session_state['anomaly_results'] = None
if 'graph_data' not in st.session_state:
    st.session_state['graph_data'] = None

if __name__ == "__main__":
    main()