import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ---------------------------------------------------------
# PAGE SETUP
# ---------------------------------------------------------

st.set_page_config(
    page_title="BDI Decision Support",
    page_icon="📈",
    layout="wide"
)

st.title("Baltic Dry Index – Decision Support")
st.caption(
    "Short-term machine learning decision support for dry-bulk shipping."
)


# ---------------------------------------------------------
# LOAD TRAINED MODEL BUNDLES
# ---------------------------------------------------------

@st.cache_resource
def load_models():
    level_bundle = joblib.load("level_bundle.joblib")
    direction_bundle = joblib.load("direction_bundle.joblib")
    fall_bundle = joblib.load("fall_bundle.joblib")

    return level_bundle, direction_bundle, fall_bundle


try:
    level_bundle, direction_bundle, fall_bundle = load_models()
    models_loaded = True

except Exception as e:
    models_loaded = False
    model_error = str(e)


# ---------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------

def pct_change(new, old):
    if old == 0:
        return 0.0
    return ((new - old) / old) * 100


def create_market_features(history):
    """
    history must contain at least 9 BDI observations,
    ordered from oldest to newest.
    """

    values = np.array(history, dtype=float)

    current = values[-1]

    change_1w = pct_change(values[-1], values[-2])
    change_2w = pct_change(values[-1], values[-3])
    change_4w = pct_change(values[-1], values[-5])

    ma4 = np.mean(values[-4:])
    ma8 = np.mean(values[-8:])

    vs_ma4 = pct_change(current, ma4)
    vs_ma8 = pct_change(current, ma8)

    weekly_returns = np.diff(values) / values[:-1] * 100

    volatility_4w = np.std(weekly_returns[-4:])
    volatility_8w = np.std(weekly_returns[-8:])

    high_8w = np.max(values[-8:])
    low_8w = np.min(values[-8:])

    distance_high = pct_change(current, high_8w)
    distance_low = pct_change(current, low_8w)

    previous_1w_change = pct_change(values[-2], values[-3])
    momentum_acceleration = change_1w - previous_1w_change

    features = pd.DataFrame(
        [{
            "BDI_Current": current,
            "BDI_Change_1W_pct": change_1w,
            "BDI_Change_2W_pct": change_2w,
            "BDI_Change_4W_pct": change_4w,
            "BDI_MA4": ma4,
            "BDI_MA8": ma8,
            "BDI_vs_MA4_pct": vs_ma4,
            "BDI_vs_MA8_pct": vs_ma8,
            "BDI_Volatility_4W_pct": volatility_4w,
            "BDI_Volatility_8W_pct": volatility_8w,
            "Distance_From_8W_High_pct": distance_high,
            "Distance_From_8W_Low_pct": distance_low,
            "Momentum_Acceleration_pctpts": momentum_acceleration
        }]
    )

    return features


# ---------------------------------------------------------
# USER INPUT
# ---------------------------------------------------------

st.subheader("Recent BDI history")

st.write(
    "Enter the most recent **nine weekly BDI observations**, "
    "from oldest to newest."
)

default_values = [
    2679.4,
    2997.6,
    2955.0,
    2820.2,
    3031.4,
    3401.0,
    3561.4,
    3367.6,
    3432.0
]

cols = st.columns(9)

bdi_history = []

for i, col in enumerate(cols):
    with col:
        value = st.number_input(
            f"Week {i + 1}",
            min_value=1.0,
            value=default_values[i],
            step=1.0
        )
        bdi_history.append(value)


# ---------------------------------------------------------
# RUN PREDICTIONS
# ---------------------------------------------------------

if st.button("Generate next-week outlook", type="primary"):

    if not models_loaded:
        st.error(
            "The model files have not been added to the repository yet."
        )

        with st.expander("Technical details"):
            st.code(model_error)

    else:

        market_features = create_market_features(bdi_history)

        # -------------------------------------------------
        # 1. LEVEL FORECAST
        # -------------------------------------------------

        level_input = pd.DataFrame(
            [{
                "BDI_lag_1": bdi_history[-1],
                "BDI_lag_2": bdi_history[-2],
                "BDI_lag_3": bdi_history[-3],
                "BDI_lag_4": bdi_history[-4]
            }]
        )

        elastic_input = level_bundle["scaler"].transform(level_input)

        elastic_prediction = level_bundle["elastic"].predict(
            elastic_input
        )[0]

        xgb_prediction = level_bundle["xgb"].predict(
            level_input
        )[0]

        level_prediction = (
            elastic_prediction + xgb_prediction
        ) / 2


        # -------------------------------------------------
        # 2. DIRECTION CLASSIFICATION
        # -------------------------------------------------

        direction_features = market_features[
            direction_bundle["feature_columns"]
        ]

        logistic_input = direction_bundle["scaler"].transform(
            direction_features
        )

        logistic_probs = direction_bundle["logistic"].predict_proba(
            logistic_input
        )[0]

        extra_probs = direction_bundle["extra_trees"].predict_proba(
            direction_features
        )[0]

        direction_probs = (
            logistic_probs + extra_probs
        ) / 2

        direction_classes = direction_bundle["logistic"].classes_

        direction_prediction = direction_classes[
            np.argmax(direction_probs)
        ]


        # -------------------------------------------------
        # 3. FALL-RISK CLASSIFICATION
        # -------------------------------------------------

        fall_features = market_features[
            fall_bundle["feature_columns"]
        ]

        fall_logistic_input = fall_bundle["scaler"].transform(
            fall_features
        )

        logistic_fall_probs = fall_bundle[
            "logistic"
        ].predict_proba(fall_logistic_input)[0]

        fall_classes = fall_bundle["logistic"].classes_

        fall_index = list(fall_classes).index(1)

        logistic_fall_probability = logistic_fall_probs[
            fall_index
        ]

        gam_probability = float(
            fall_bundle["gam"].predict_proba(
                fall_logistic_input
            )[0]
        )

        fall_probability = (
            logistic_fall_probability + gam_probability
        ) / 2


        # -------------------------------------------------
        # DISPLAY RESULTS
        # -------------------------------------------------

        st.divider()

        st.subheader("Next-week market outlook")

        col1, col2, col3 = st.columns(3)

        current_bdi = bdi_history[-1]

        expected_change = (
            (level_prediction - current_bdi)
            / current_bdi
        ) * 100

        with col1:
            st.metric(
                "Expected BDI",
                f"{level_prediction:,.0f}",
                f"{expected_change:+.1f}%"
            )

        with col2:
            st.metric(
                "Expected Direction",
                str(direction_prediction)
            )

        with col3:
            st.metric(
                "Probability of >3% Fall",
                f"{fall_probability * 100:.1f}%"
            )


        # -------------------------------------------------
        # INTERPRETATION
        # -------------------------------------------------

        st.subheader("Decision-support interpretation")

        if fall_probability >= 0.60:
            risk_text = "Elevated downside risk"
        elif fall_probability >= 0.40:
            risk_text = "Moderate downside risk"
        else:
            risk_text = "Lower downside risk"

        st.write(
            f"**{risk_text}.** "
            f"The level model estimates a next-week BDI of "
            f"**{level_prediction:,.0f}**, while the direction model "
            f"classifies the expected movement as **{direction_prediction}**."
        )

        st.info(
            "This is an experimental machine-learning decision-support "
            "prototype. It should not be interpreted as financial or "
            "chartering advice."
        )


# ---------------------------------------------------------
# MODEL INFORMATION
# ---------------------------------------------------------

with st.expander("About the models"):

    st.markdown(
        """
        **Level forecast**  
        50/50 ensemble of XGBoost and Elastic Net regression.

        **Direction classification**  
        50/50 probability ensemble of Regularized Logistic Regression
        and Extra Trees.

        **Fall-risk classification**  
        50/50 probability ensemble of Regularized Logistic Regression
        and a Generalized Additive Model.

        The models use historical Baltic Dry Index behaviour to provide
        short-term weekly decision support.
        """
    )
