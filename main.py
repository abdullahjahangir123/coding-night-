import streamlit as st
import os
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import Callback
# ---------------------------
# Session State Initialization
# ---------------------------
if "classes" not in st.session_state:
    st.session_state.classes = []
if "dataset_dir" not in st.session_state:
    st.session_state.dataset_dir = "dataset"
if "history_df" not in st.session_state:
    st.session_state.history_df = pd.DataFrame(columns=["epoch", "accuracy", "val_accuracy", "loss", "val_loss"])
if "class_indices" not in st.session_state:
    st.session_state.class_indices = {}
if "model" not in st.session_state:
    st.session_state.model = None
if "model_trained" not in st.session_state:
    st.session_state.model_trained = False

os.makedirs(st.session_state.dataset_dir, exist_ok=True)

# ---------------------------
# Page Setup
# ---------------------------
st.set_page_config(layout="wide", page_title="Teachable Machine – Tabs Version")
st.title("📸 Teachable Machine (Tabs Version)")
st.write("Model training, graphs & prediction — sab kuch smooth, no menu errors!")

# ---------------------------
# Main Tabs
# ---------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "1️⃣ Add Classes",
    "2️⃣ Upload Images",
    "3️⃣ Train Model",
    "4️⃣ Predict"
])

# ---------------------------------------------------------
# TAB 1 — ADD CLASSES
# ---------------------------------------------------------
with tab1:
    st.subheader("Add Classes")

    new_class = st.text_input("Enter Class Name")

    if st.button("Add Class"):
        if new_class.strip() and new_class not in st.session_state.classes:
            st.session_state.classes.append(new_class)
            os.makedirs(os.path.join(st.session_state.dataset_dir, new_class), exist_ok=True)
            st.success(f"Class '{new_class}' added!")
            st.rerun()
        else:
            st.warning("Invalid or duplicate class name!")

    st.write("### Current Classes:")
    if st.session_state.classes:
        st.success(", ".join(st.session_state.classes))
    else:
        st.info("No classes added yet.")

# ---------------------------------------------------------
# TAB 2 — UPLOAD IMAGES
# ---------------------------------------------------------
with tab2:
    st.subheader("Upload Images for Each Class")

    if not st.session_state.classes:
        st.warning("Pehle classes add karo!")
    else:
        for cls in st.session_state.classes:
            with st.expander(f"Upload images for **{cls}**"):
                uploaded = st.file_uploader(
                    f"Images for {cls}",
                    type=["png", "jpg", "jpeg"],
                    accept_multiple_files=True,
                    key=f"upload_{cls}"
                )

                class_dir = os.path.join(st.session_state.dataset_dir, cls)

                if uploaded:
                    for file in uploaded:
                        with open(os.path.join(class_dir, file.name), "wb") as f:
                            f.write(file.getbuffer())
                    st.success(f"Uploaded {len(uploaded)} images.")

# ---------------------------------------------------------
# TAB 3 — TRAIN MODEL
# ---------------------------------------------------------
with tab3:
    st.subheader("Train Your Model")

    if st.button("🚀 Start Training", type="primary"):
        if len(st.session_state.classes) < 2:
            st.error("Kam se kam 2 classes chahiye!")
        else:
            with st.spinner("Training in progress..."):

                IMG_SIZE = (128, 128)
                BATCH_SIZE = 16
                EPOCHS = 12

                datagen = ImageDataGenerator(
                    rescale=1./255,
                    validation_split=0.2,
                    rotation_range=20,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    horizontal_flip=True
                )

                train_gen = datagen.flow_from_directory(
                    st.session_state.dataset_dir,
                    target_size=IMG_SIZE,
                    batch_size=BATCH_SIZE,
                    class_mode="categorical",
                    subset="training"
                )

                val_gen = datagen.flow_from_directory(
                    st.session_state.dataset_dir,
                    target_size=IMG_SIZE,
                    batch_size=BATCH_SIZE,
                    class_mode="categorical",
                    subset="validation"
                )

                st.session_state.class_indices = train_gen.class_indices
                num_classes = len(train_gen.class_indices)

                model = Sequential([
                    Conv2D(32, (3,3), activation="relu", input_shape=(128,128,3)),
                    MaxPooling2D(2,2),
                    Conv2D(64, (3,3), activation="relu"),
                    MaxPooling2D(2,2),
                    Conv2D(128, (3,3), activation="relu"),
                    MaxPooling2D(2,2),
                    Flatten(),
                    Dense(512, activation="relu"),
                    Dropout(0.5),
                    Dense(num_classes, activation="softmax")
                ])

                model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

                class LivePlot(Callback):
                    def on_epoch_end(self, epoch, logs=None):
                        logs = logs or {}
                        row = {
                            "epoch": epoch + 1,
                            "accuracy": logs.get("accuracy", 0),
                            "val_accuracy": logs.get("val_accuracy", 0),
                            "loss": logs.get("loss", 0),
                            "val_loss": logs.get("val_loss", 0)
                        }
                        st.session_state.history_df = pd.concat(
                            [st.session_state.history_df, pd.DataFrame([row])],
                            ignore_index=True
                        )

                st.session_state.history_df = pd.DataFrame(columns=[
                    "epoch","accuracy","val_accuracy","loss","val_loss"
                ])

                placeholder = st.empty()

                model.fit(
                    train_gen,
                    validation_data=val_gen,
                    epochs=EPOCHS,
                    callbacks=[LivePlot()],
                    verbose=1
                )

                model.save("trained_model.keras")
                st.session_state.model = model
                st.session_state.model_trained = True

                st.success("Training Complete!")

                with placeholder.container():
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Accuracy")
                        st.line_chart(st.session_state.history_df[["accuracy","val_accuracy"]])
                    with col2:
                        st.subheader("Loss")
                        st.line_chart(st.session_state.history_df[["loss","val_loss"]])

# ---------------------------------------------------------
# TAB 4 — PREDICT
# ---------------------------------------------------------
with tab4:
    st.subheader("Predict an Image")

    if not st.session_state.model_trained:
        st.warning("Pehle model train karo!")
    else:
        img_file = st.file_uploader("Upload an image", type=["png","jpg","jpeg"])

        if img_file:
            img = Image.open(img_file).resize((128,128))
            x = img_to_array(img) / 255.0
            x = np.expand_dims(x, axis=0)

            pred = st.session_state.model.predict(x)[0]
            pred_class = np.argmax(pred)
            class_name = list(st.session_state.class_indices.keys())[pred_class]
            confidence = np.max(pred) * 100

            st.image(img, width=350)
            st.success(f"Prediction: **{class_name}** ({confidence:.2f}%)")

            probs = {name: float(p*100) for name, p in zip(st.session_state.class_indices.keys(), pred)}
            st.bar_chart(pd.DataFrame([probs]))

# ---------------------------
# RESET BUTTON
# ---------------------------
st.write("---")
if st.button("Reset App (Clear Everything)"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()
