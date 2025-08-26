import streamlit as st
import numpy as np
import joblib
import pandas as pd

st.title("🧠 Parkinson’s Disease Prediction (Top 30 Features)")

top_30_features = joblib.load("top_30_features.pkl")
model = joblib.load("parkinson_rf_top30.pkl")
scaler = joblib.load("scaler.pkl")

sample_inputs = {'Healthy Sample 1': [-1596.5095, 0.011149, 0.92812, 0.016144, 0.024289, 5.593, 0.0017696, 0.1131, -141883.0437, 0.00079057, 0.18213, 0.028659, -133022.4002, 838.2256, 2.26e-05, 0.0029478, -333529880.1, 0.0052028, -225237.2859, -0.37737, 0.0061674, 279.8189, 0.015386, -463155.3466, 76.78113039, -0.19873, -4.76e-06, 178.0034, -0.14985, 0.019955], 'Healthy Sample 2': [-1748.7514, 0.008479, -0.12565, 0.011375, 0.023714, 3.6819, 0.0012257, 0.095791, -142688.9102, 0.00049203, 0.13647, 0.022956, -142269.7298, 508.1304, 5.03e-05, 0.002864, -325509885.1, 0.0030955, -224089.93, -0.23829, 0.0035267, 279.3631, 0.013958, -460777.8397, 76.58636018, -0.19328, -3.18e-05, 171.918, -0.13893, 0.011495], 'Healthy Sample 3': [-2202.2773, 0.0073368, 0.13828, 0.0096665, 0.036038, 0.80878, 0.0013818, 0.11829, -149699.7997, 0.00054288, 0.1348, 0.023445, -144321.5477, 477.1292, 6.4e-05, 0.0022474, -307919680.8, 0.0038143, -221655.8451, 0.14077, 0.0037624, 278.3274, 0.012484, -455955.8012, 76.71784682, -0.16762, -2.04e-05, 141.1315, -0.12567, 0.012624], 'Parkinson Sample 1': [-3190.1752, 0.016392, 2.4874, 0.012829, 0.008643, 0.015562, 5.9e-05, 0.024286, -229943.2967, 2.33e-05, 0.063087, 0.0043241, -201985.0408, 89.7525, 0.10807, 5.03e-05, -129684181.7, 0.00018979, -184901.7535, -2.7303, 0.00012787, 262.2272, 0.021703, -381059.351, 69.9974958, -0.02416, -1.41e-18, 4.884, -0.026321, 0.014642], 'Parkinson Sample 2': [-3106.4317, 0.014222, 2.8986, 0.010645, 0.0071835, 0.02386, 0.00016935, 0.099695, -230526.8175, 2.41e-05, 0.055913, 0.0043442, -203389.4678, 65.9794, 0.09836, 5.1e-05, -123243056.6, 0.00030136, -182880.5032, 5.2294, 9.84e-05, 261.2809, 0.020296, -376979.9939, 67.41590313, -0.066933, -7.47e-19, 4.8483, -0.070039, 0.0255], 'Parkinson Sample 3': [-3082.5691, 0.039709, 3.2208, 0.016553, 0.0039688, 0.020677, 2.1e-05, 0.026241, -246592.6024, 8.98e-06, 0.041144, 0.002657, -214707.2576, 39.0161, 0.10691, 1.85e-05, -119780270.0, 6.74e-05, -181663.4768, 0.35054, 4.52e-05, 260.7518, 0.023186, -374463.8517, 62.66170618, -0.015216, -5.36e-19, 1.9849, -0.019435, 0.024607]}
selected_sample = st.selectbox("🔽 Choose a real sample or enter manually:", ["Manual Entry"] + list(sample_inputs.keys()))
default_values = sample_inputs[selected_sample] if selected_sample in sample_inputs else [0.0]*30

st.markdown("Enter values for each feature below:")
col1, col2, col3 = st.columns(3)

val_0 = col1.number_input('tqwt_entropy_log_dec_35', value=default_values[0], step=0.01)
val_1 = col2.number_input('std_delta_delta_log_energy', value=default_values[1], step=0.01)
val_2 = col3.number_input('mean_MFCC_2nd_coef', value=default_values[2], step=0.01)
val_3 = col1.number_input('std_8th_delta_delta', value=default_values[3], step=0.01)
val_4 = col2.number_input('tqwt_TKEO_mean_dec_16', value=default_values[4], step=0.01)
val_5 = col3.number_input('tqwt_entropy_shannon_dec_35', value=default_values[5], step=0.01)
val_6 = col1.number_input('tqwt_TKEO_std_dec_12', value=default_values[6], step=0.01)
val_7 = col2.number_input('tqwt_maxValue_dec_12', value=default_values[7], step=0.01)
val_8 = col3.number_input('tqwt_entropy_log_dec_11', value=default_values[8], step=0.01)
val_9 = col1.number_input('tqwt_TKEO_mean_dec_12', value=default_values[9], step=0.01)
val_10 = col2.number_input('tqwt_stdValue_dec_15', value=default_values[10], step=0.01)
val_11 = col3.number_input('tqwt_stdValue_dec_12', value=default_values[11], step=0.01)
val_12 = col1.number_input('tqwt_entropy_log_dec_12', value=default_values[12], step=0.01)
val_13 = col2.number_input('tqwt_entropy_shannon_dec_14', value=default_values[13], step=0.01)
val_14 = col3.number_input('tqwt_energy_dec_27', value=default_values[14], step=0.01)
val_15 = col1.number_input('tqwt_TKEO_mean_dec_11', value=default_values[15], step=0.01)
val_16 = col2.number_input('app_entropy_shannon_5_coef', value=default_values[16], step=0.01)
val_17 = col3.number_input('tqwt_TKEO_std_dec_13', value=default_values[17], step=0.01)
val_18 = col1.number_input('app_LT_entropy_shannon_6_coef', value=default_values[18], step=0.01)
val_19 = col2.number_input('tqwt_skewnessValue_dec_36', value=default_values[19], step=0.01)
val_20 = col3.number_input('tqwt_TKEO_mean_dec_13', value=default_values[20], step=0.01)
val_21 = col1.number_input('app_entropy_log_5_coef', value=default_values[21], step=0.01)
val_22 = col2.number_input('std_9th_delta_delta', value=default_values[22], step=0.01)
val_23 = col3.number_input('app_LT_entropy_shannon_7_coef', value=default_values[23], step=0.01)
val_24 = col1.number_input('minIntensity', value=default_values[24], step=0.01)
val_25 = col2.number_input('tqwt_minValue_dec_10', value=default_values[25], step=0.01)
val_26 = col3.number_input('tqwt_medianValue_dec_8', value=default_values[26], step=0.01)
val_27 = col1.number_input('tqwt_entropy_shannon_dec_11', value=default_values[27], step=0.01)
val_28 = col2.number_input('tqwt_minValue_dec_11', value=default_values[28], step=0.01)
val_29 = col3.number_input('std_6th_delta_delta', value=default_values[29], step=0.01)

if st.button("Predict"):
    input_data = np.array([val_0, val_1, val_2, val_3, val_4, val_5, val_6, val_7, val_8, val_9, val_10, val_11, val_12, val_13, val_14, val_15, val_16, val_17, val_18, val_19, val_20, val_21, val_22, val_23, val_24, val_25, val_26, val_27, val_28, val_29]).reshape(1, -1)
    scaled_input = scaler.transform(pd.DataFrame(input_data, columns=top_30_features))
    pred = model.predict(scaled_input)[0]
    prob = model.predict_proba(scaled_input)[0][1]

    st.subheader("🧾 Prediction Result:")
    if pred == 1:
        st.error(f"⚠️ High likelihood of Parkinson’s Disease (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Low likelihood of Parkinson’s Disease (Probability: {prob:.2f})")

st.markdown("This model is optimized using the top 30 most informative features selected via mutual information.")
