import streamlit as st

def about_page():
    st.markdown("""
    <style>
    .about-section {
        background-color: #f7f7fa;
        padding: 1.5em;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        margin-bottom: 1.5em;
    }
    .feature-list li {
        margin-bottom: 0.5em;
    }
    </style>
    <div class="about-section">
        <h3>Welcome!</h3>
        <p>
            <b>Artificial Neural Network Builder App</b>
        </p>
        <p>
            This application is designed to help you:
        </p>
        <ul class="feature-list">
            <li><b>📊 Explore and visualize your data:</b> Upload your dataset, view summary statistics, and generate interactive charts (area, box, histogram, scatter, heatmap, and pie).</li>
            <li><b>🏗️ Build an Artificial Neural Network (ANN):</b> Configure, train, and evaluate a neural network model for your data.</li>
            <li><b>🔮 Make predictions on new data:</b> After building and training your ANN, you can upload new, unseen data to generate predictions using your trained model. The app allows you to download the prediction results for further analysis or reporting.</li>
        </ul>
    </div>

    <div class="about-section">
        <h4>Pages</h4>
        <ul>
            <li>
                <b>📊 Data Exploration & Visualization:</b><br>
                Upload your data, explore its structure, and create interactive visualizations to better understand patterns, distributions, and relationships.
            </li>
            <li>
                <b>🏗️ Build Artificial-Neural-Network:</b><br>
                Set up and train an ANN model on your data. Adjust hyperparameters, view training progress, and evaluate model performance.<br>
                After training, you can:
                <ul>
                    <li>Upload new (unseen) data to make predictions with your trained model.</li>
                    <li>Download the prediction results as a CSV file for your records or further use.</li>
                </ul>
            </li>
        </ul>
    </div>

    <hr>
    <div style='font-size: 0.95em; color: #555;'>
        <b>Author:</b> Vahidin (Dean) Jupic<br>
        <b>Version:</b> 0.9.0
    </div>
    """, unsafe_allow_html=True)
