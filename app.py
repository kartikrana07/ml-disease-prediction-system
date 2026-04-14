import os
import joblib
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np

# Set page configuration
st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧑‍⚕️")

# getting the working directory of the main.py
working_dir = os.path.dirname(os.path.abspath(__file__))

# --- LOADER FUNCTION TO PREVENT CRASHES IF FILES MISSING ---
@st.cache_resource
def load_models():
    try:
        models = {}
        # Diabetes
        models['diabetes_model'] = joblib.load(os.path.join(working_dir, 'models/diabetes_model.pkl'))
        models['diabetes_imputer'] = joblib.load(os.path.join(working_dir, 'imputers/diabetes_imputer.pkl'))
        models['diabetes_scaler'] = joblib.load(os.path.join(working_dir, 'scalers/diabetes_scaler.pkl'))

        # Heart Disease
        models['heart_model'] = joblib.load(os.path.join(working_dir, 'models/heart_model.pkl'))
        models['heart_encoder'] = joblib.load(os.path.join(working_dir, 'encoders/heart_encoder.pkl'))
        models['heart_imputer'] = joblib.load(os.path.join(working_dir, 'imputers/heart_imputer.pkl'))
        models['heart_scaler'] = joblib.load(os.path.join(working_dir, 'scalers/heart_scaler.pkl'))

        # Parkinsons
        models['parkinsons_model'] = joblib.load(os.path.join(working_dir, 'models/parkinson_model.pkl'))
        models['parkinsons_scaler'] = joblib.load(os.path.join(working_dir, 'scalers/parkinson_scaler.pkl'))

        # Breast Cancer
        models['breast_cancer_model'] = joblib.load(os.path.join(working_dir, 'models/breast-cancer_model.pkl'))
        models['breast_cancer_scaler'] = joblib.load(os.path.join(working_dir, 'scalers/breast-cancer_scaler.pkl'))
        
        return models
    except Exception as e:
        st.error(f"Error loading models: {e}. Please check file paths.")
        return None

models = load_models()

# Only proceed if models loaded successfully
if models:
    # sidebar for navigation
    with st.sidebar:
        selected = option_menu('Multiple Disease Prediction System',
                               ['Diabetes Prediction',
                                'Heart Disease Prediction',
                                'Parkinsons Prediction',
                                'Breast Cancer Prediction'],
                               menu_icon='hospital-fill',
                               icons=['activity', 'heart', 'person', 'gender-female'],
                               default_index=0)

    # ==========================
    # Diabetes Prediction Page
    # ==========================
    if selected == 'Diabetes Prediction':
        st.title('Diabetes Prediction using ML')

        # Using number_input to prevent string conversion errors
        col1, col2, col3 = st.columns(3)

        with col1:
            Pregnancies = st.number_input('Number of Pregnancies', min_value=0, step=1)
        with col2:
            Glucose = st.number_input('Glucose Level', min_value=0.0)
        with col3:
            BloodPressure = st.number_input('Blood Pressure value', min_value=0.0)
        with col1:
            SkinThickness = st.number_input('Skin Thickness value', min_value=0.0)
        with col2:
            Insulin = st.number_input('Insulin Level', min_value=0.0)
        with col3:
            BMI = st.number_input('BMI value', min_value=0.0)
        with col1:
            DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function value', min_value=0.0, format="%.3f")
        with col2:
            Age = st.number_input('Age of the Person', min_value=0, step=1)

        diab_diagnosis = ''

        if st.button('Diabetes Test Result'):
            try:
                user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
                
                # Reshape for transformation (1 sample, n features)
                user_input_2d = [user_input]

                # Impute missing values (using the SimpleImputer from notebook)
                imputed_input = models['diabetes_imputer'].transform(user_input_2d)

                # Scale features
                scaled_input = models['diabetes_scaler'].transform(imputed_input)

                # Predict
                diab_prediction = models['diabetes_model'].predict(scaled_input)

                if diab_prediction[0] == 1:
                    diab_diagnosis = 'The person is diabetic'
                else:
                    diab_diagnosis = 'The person is not diabetic'
                
                st.success(diab_diagnosis)
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

    # ==========================
    # Heart Disease Prediction Page
    # ==========================
    if selected == 'Heart Disease Prediction':
        st.title('Heart Disease Prediction using ML')

        col1, col2, col3 = st.columns(3)

        # Using correct input types to match OneHotEncoder expectations
        with col1:
            age = st.number_input('Age', min_value=0)
        with col2:
            # Dropdown ensures exact string match for Encoder
            sex = st.selectbox('Sex', ['M', 'F']) 
        with col3:
            cp = st.selectbox('Chest Pain types', ['ATA', 'NAP', 'ASY', 'TA'])
        with col1:
            trestbps = st.number_input('Resting Blood Pressure', min_value=0)
        with col2:
            chol = st.number_input('Serum Cholestoral in mg/dl', min_value=0)
        with col3:
            fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', [0, 1])
        with col1:
            restecg = st.selectbox('Resting Electrocardiographic results', ['Normal', 'ST', 'LVH'])
        with col2:
            thalach = st.number_input('Maximum Heart Rate achieved', min_value=0)
        with col3:
            exang = st.selectbox('Exercise Angina', ['N', 'Y'])
        with col1:
            oldpeak = st.number_input('Oldpeak', value=0.0)
        with col2:
            slope = st.selectbox('ST_Slope', ['Up', 'Flat', 'Down'])

        heart_diagnosis = ''

        if st.button('Heart Disease Test Result'):
            try:
                # 1. Create DataFrame exactly as the notebook expects
                data = {
                    'Age': [age], 'Sex': [sex], 'ChestPainType': [cp], 
                    'RestingBP': [trestbps], 'Cholesterol': [chol], 'FastingBS': [fbs],
                    'RestingECG': [restecg], 'MaxHR': [thalach], 'ExerciseAngina': [exang], 
                    'Oldpeak': [oldpeak], 'ST_Slope': [slope]
                }
                df_input = pd.DataFrame(data)

                # Define columns based on training notebook
                num_cols = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
                cat_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

                # 2. Impute Numerical Columns (KNNImputer)
                # Note: KNNImputer returns numpy array, we must convert back to DataFrame to keep columns straight if needed,
                # but since we are stacking immediately, array is fine.
                imputed_num = models['heart_imputer'].transform(df_input[num_cols])

                # 3. Scale Numerical Columns
                scaled_num = models['heart_scaler'].transform(imputed_num)

                # 4. Encode Categorical Columns
                encoded_cat = models['heart_encoder'].transform(df_input[cat_cols])

                # 5. Concatenate
                final_input = np.hstack([scaled_num, encoded_cat])

                # 6. Predict
                heart_prediction = models['heart_model'].predict(final_input)

                if heart_prediction[0] == 1:
                    heart_diagnosis = 'The person is having heart disease'
                else:
                    heart_diagnosis = 'The person does not have any heart disease'
                
                st.success(heart_diagnosis)
            except Exception as e:
                st.error(f"An error occurred: {e}")

    # ==========================
    # Parkinson's Prediction Page
    # ==========================
    if selected == "Parkinsons Prediction":
        st.title("Parkinson's Disease Prediction using ML")

        # Keeping text inputs for compactness, but adding validation
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            fo = st.text_input('MDVP:Fo(Hz)')
        with col2:
            fhi = st.text_input('MDVP:Fhi(Hz)')
        with col3:
            flo = st.text_input('MDVP:Flo(Hz)')
        with col4:
            Jitter_percent = st.text_input('MDVP:Jitter(%)')
        with col5:
            Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')
        with col1:
            RAP = st.text_input('MDVP:RAP')
        with col2:
            PPQ = st.text_input('MDVP:PPQ')
        with col3:
            DDP = st.text_input('Jitter:DDP')
        with col4:
            Shimmer = st.text_input('MDVP:Shimmer')
        with col5:
            Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')
        with col1:
            APQ3 = st.text_input('Shimmer:APQ3')
        with col2:
            APQ5 = st.text_input('Shimmer:APQ5')
        with col3:
            APQ = st.text_input('MDVP:APQ')
        with col4:
            DDA = st.text_input('Shimmer:DDA')
        with col5:
            NHR = st.text_input('NHR')
        with col1:
            HNR = st.text_input('HNR')
        with col2:
            RPDE = st.text_input('RPDE')
        with col3:
            DFA = st.text_input('DFA')
        with col4:
            spread1 = st.text_input('spread1')
        with col5:
            spread2 = st.text_input('spread2')
        with col1:
            D2 = st.text_input('D2')
        with col2:
            PPE = st.text_input('PPE')

        parkinsons_diagnosis = ''

        if st.button("Parkinson's Test Result"):
            try:
                user_input = [fo, fhi, flo, Jitter_percent, Jitter_Abs,
                              RAP, PPQ, DDP, Shimmer, Shimmer_dB, APQ3, APQ5,
                              APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]

                # Validation loop
                user_input = [float(x) if x != '' else 0.0 for x in user_input]

                user_input_2d = [user_input]

                scaled_input = models['parkinsons_scaler'].transform(user_input_2d)
                parkinsons_prediction = models['parkinsons_model'].predict(scaled_input)

                if parkinsons_prediction[0] == 1:
                    parkinsons_diagnosis = "The person has Parkinson's disease"
                else:
                    parkinsons_diagnosis = "The person does not have Parkinson's disease"
                st.success(parkinsons_diagnosis)
            except ValueError:
                st.error("Please enter valid numeric values for all fields.")

    # ==========================
    # Breast Cancer Prediction Page
    # ==========================
    if selected == "Breast Cancer Prediction":
        st.title("Breast Cancer Prediction using ML")

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            radius_mean = st.text_input('radius_mean')
        with col2:
            texture_mean = st.text_input('texture_mean')
        with col3:
            perimeter_mean = st.text_input('perimeter_mean')
        with col4:
            area_mean = st.text_input('area_mean')
        with col5:
            smoothness_mean = st.text_input('smoothness_mean')
        with col1:
            compactness_mean = st.text_input('compactness_mean')
        with col2:
            concavity_mean = st.text_input('concavity_mean')
        with col3:
            concave_points_mean = st.text_input('concave points_mean')
        with col4:
            symmetry_mean = st.text_input('symmetry_mean')
        with col5:
            fractal_dimension_mean = st.text_input('fractal_dimension_mean')
        with col1:
            radius_se = st.text_input('radius_se')
        with col2:
            texture_se = st.text_input('texture_se')
        with col3:
            perimeter_se = st.text_input('perimeter_se')
        with col4:
            area_se = st.text_input('area_se')
        with col5:
            smoothness_se = st.text_input('smoothness_se')
        with col1:
            compactness_se = st.text_input('compactness_se')
        with col2:
            concavity_se = st.text_input('concavity_se')
        with col3:
            concave_points_se = st.text_input('concave points_se')
        with col4:
            symmetry_se = st.text_input('symmetry_se')
        with col5:
            fractal_dimension_se = st.text_input('fractal_dimension_se')
        with col1:
            radius_worst = st.text_input('radius_worst')
        with col2:
            texture_worst = st.text_input('texture_worst')
        with col3:
            perimeter_worst = st.text_input('perimeter_worst')
        with col4:
            area_worst = st.text_input('area_worst')
        with col5:
            smoothness_worst = st.text_input('smoothness_worst')
        with col1:
            compactness_worst = st.text_input('compactness_worst')
        with col2:
            concavity_worst = st.text_input('concavity_worst')
        with col3:
            concave_points_worst = st.text_input('concave points_worst')
        with col4:
            symmetry_worst = st.text_input('symmetry_worst')
        with col5:
            fractal_dimension_worst = st.text_input('fractal_dimension_worst')

        breast_cancer_diagnosis = ''

        if st.button("Breast Cancer Test Result"):
            try:
                user_input = [radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, 
                              compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean, 
                              radius_se, texture_se, perimeter_se, area_se, smoothness_se, 
                              compactness_se, concavity_se, concave_points_se, symmetry_se, fractal_dimension_se, 
                              radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst, 
                              compactness_worst, concavity_worst, concave_points_worst, symmetry_worst, fractal_dimension_worst]

                # Validation
                user_input = [float(x) if x != '' else 0.0 for x in user_input]

                user_input_2d = [user_input]

                scaled_input = models['breast_cancer_scaler'].transform(user_input_2d)
                breast_cancer_prediction = models['breast_cancer_model'].predict(scaled_input)

                if breast_cancer_prediction[0] == 1:
                    breast_cancer_diagnosis = "The person has Breast Cancer"
                else:
                    breast_cancer_diagnosis = "The person does not have Breast Cancer"
                st.success(breast_cancer_diagnosis)
            except ValueError:
                st.error("Please enter valid numeric values for all fields.")