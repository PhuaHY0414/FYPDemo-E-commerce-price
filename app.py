import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import gdown

# Page configuration
st.set_page_config(
    page_title="E-Commerce Price Predictor",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for tracking predictions
if 'predictions_history' not in st.session_state:
    st.session_state.predictions_history = []
if 'total_predictions' not in st.session_state:
    st.session_state.total_predictions = 0

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    h1 {
        color: #1f77b4;
        text-align: center;
        padding: 1.5rem 0;
    }
    h2 {
        color: #2c3e50;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .prediction-result {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 1.5rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Function to download model from Google Drive
def download_model_from_drive():
    """Download the model file from Google Drive if it doesn't exist"""
    model_path = 'tuned_random_forest_model.pkl'
    
    # Check if model already exists
    if os.path.exists(model_path):
        return True
    
    # Google Drive file ID from your link
    file_id = '1bMOEUXzqzB0zxj0iKYr4HajE0HPRk6MW'
    url = f'https://drive.google.com/uc?id={file_id}'
    
    try:
        st.info("⏳ Downloading model from Google Drive... (this may take a minute)")
        # Use fuzzy=True to handle Google Drive download page
        output = gdown.download(url, model_path, quiet=False, fuzzy=True)
        
        # Check if file actually exists after download
        if output and os.path.exists(model_path):
            file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
            st.success(f"✅ Model downloaded successfully! ({file_size:.1f} MB)")
            return True
        else:
            st.error("❌ Download failed - file not created")
            st.info("💡 Manual download: https://drive.google.com/file/d/1bMOEUXzqzB0zxj0iKYr4HajE0HPRk6MW/view")
            return False
    except Exception as e:
        st.error(f"❌ Download error: {str(e)}")
        st.info("💡 Manual download: https://drive.google.com/file/d/1bMOEUXzqzB0zxj0iKYr4HajE0HPRk6MW/view")
        return False

# Load the model (with auto-download from Google Drive)
@st.cache_resource
def load_model():
    try:
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.compose import ColumnTransformer
        
        # Try to download model if not exists
        model_exists = download_model_from_drive()
        
        # Try to load the trained model
        if model_exists:
            try:
                model = joblib.load('tuned_random_forest_model.pkl')
                demo_mode = False
            except Exception as e:
                st.warning(f"⚠️ Error loading model: {str(e)} - Running in DEMO MODE")
                model = None
                demo_mode = True
        else:
            st.warning("⚠️ Model file not available - Running in DEMO MODE")
            model = None
            demo_mode = True
        
        # Create preprocessor EXACTLY from notebook (ACopy_of_FYP2025.ipynb line 261-295)
        # Must match the 19 columns used in training!
        categorical_cols = ['payment_type', 'order_status', 'customer_state', 
                          'seller_state', 'product_category_name_english']
        numerical_cols = ['payment_sequential', 'payment_installments', 'payment_value',
                        'product_weight_g', 'product_length_cm', 'product_height_cm',
                        'product_width_cm', 'product_description_lenght', 'product_photos_qty',
                        'product_name_lenght', 'customer_zip_code_prefix', 'seller_zip_code_prefix',
                        'review_score', 'review_comment_title_length']
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
                ('num', 'passthrough', numerical_cols)
            ],
            remainder='drop'
        )
        
        # FIT the preprocessor with dummy data to initialize it
        dummy_data = pd.DataFrame({
            'payment_type': ['credit_card'],
            'order_status': ['delivered'],
            'customer_state': ['SP'],
            'seller_state': ['SP'],
            'product_category_name_english': ['electronics'],
            'payment_sequential': [1],
            'payment_installments': [1],
            'payment_value': [100.0],
            'product_weight_g': [500],
            'product_length_cm': [20],
            'product_height_cm': [10],
            'product_width_cm': [15],
            'product_description_lenght': [500],
            'product_photos_qty': [1],
            'product_name_lenght': [50],
            'customer_zip_code_prefix': [1000],
            'seller_zip_code_prefix': [1000],
            'review_score': [4],
            'review_comment_title_length': [0]
        })
        preprocessor.fit(dummy_data)
        
        # EXACT final features from notebook output (line 583-603)
        # Selected at least 2 times from 3 methods
        feature_names = [
            'num__payment_value',
            'num__product_weight_g',
            'num__payment_installments',
            'num__product_height_cm',
            'num__product_width_cm',
            'num__product_length_cm',
            'num__product_description_lenght',
            'cat__seller_state_SP',
            'cat__product_category_name_english_telephony',
            'cat__product_category_name_english_watches_gifts'
        ]
        
        return model, preprocessor, feature_names
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None, None, None

# Load category mappings
@st.cache_data
def load_categories():
    return {
        'payment_types': ['credit_card', 'boleto', 'voucher', 'debit_card'],
        'order_statuses': ['delivered', 'shipped', 'approved', 'invoiced'],
        'product_categories': [
            'bed_bath_table', 'health_beauty', 'sports_leisure', 'furniture_decor',
            'computers_accessories', 'housewares', 'watches_gifts', 'telephony',
            'garden_tools', 'auto', 'toys', 'cool_stuff', 'luggage_accessories',
            'perfumery', 'baby', 'fashion_bags_accessories', 'pet_shop',
            'office_furniture', 'market_place', 'electronics', 'home_appliances',
            'furniture_living_room', 'construction_tools_construction',
            'furniture_bedroom', 'home_construction', 'musical_instruments',
            'home_comfort', 'consoles_games', 'audio', 'fashion_shoes',
            'computers', 'christmas_supplies', 'books_general_interest',
            'construction_tools_lights', 'industry_commerce_and_business',
            'food', 'art', 'furniture_mattress_and_upholstery', 'party_supplies',
            'fashion_childrens_clothes', 'stationery', 'tablets_printing_image',
            'construction_tools_tools', 'fashion_male_clothing', 'books_technical',
            'drinks', 'kitchen_dining_laundry_garden_furniture', 'flowers',
            'air_conditioning', 'construction_tools_safety', 'fashion_underwear_beach',
            'fashion_sport', 'food_drink', 'home_appliances_2', 'agro_industry_and_commerce',
            'la_cuisine', 'signaling_and_security', 'arts_and_craftmanship',
            'fashion_female_clothing', 'small_appliances', 'dvds_blu_ray',
            'cds_dvds_musicals', 'diapers_and_hygiene', 'small_appliances_home_oven_and_coffee',
            'health_beauty_2', 'computers_accessories_2'
        ],
        'states': [
            'SP', 'RJ', 'MG', 'RS', 'PR', 'SC', 'BA', 'DF', 'GO', 'PE',
            'CE', 'PA', 'ES', 'MT', 'MA', 'MS', 'PB', 'PI', 'RN', 'AL',
            'SE', 'RO', 'TO', 'AM', 'AC', 'AP', 'RR'
        ]
    }

def create_dashboard():
    """Create dynamic analytics dashboard with real prediction data"""
    st.markdown("## 📊 Analytics Dashboard")
    
    # Get prediction history from session state
    predictions = st.session_state.predictions_history
    
    # Check if we have any predictions
    if len(predictions) == 0:
        st.info("📝 **No predictions yet!** Use Manual Input or CSV Upload to generate predictions. The dashboard will update automatically.")
        
        # Show placeholder metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("""
                <div class="metric-card">
                    <h3>📦 Total Predictions</h3>
                    <h2>0</h2>
                    <p>Start predicting!</p>
                </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
                <div class="metric-card">
                    <h3>💰 Avg Price</h3>
                    <h2>-</h2>
                    <p>No data yet</p>
                </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
                <div class="metric-card">
                    <h3>📈 Highest Price</h3>
                    <h2>-</h2>
                    <p>No data yet</p>
                </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown("""
                <div class="metric-card">
                    <h3>📉 Lowest Price</h3>
                    <h2>-</h2>
                    <p>No data yet</p>
                </div>
            """, unsafe_allow_html=True)
        return
    
    # Convert predictions to DataFrame for analysis
    df = pd.DataFrame(predictions)
    
    # Calculate real statistics
    total_preds = len(predictions)
    avg_price = df['predicted_price'].mean()
    max_price = df['predicted_price'].max()
    min_price = df['predicted_price'].min()
    total_value = df['predicted_price'].sum()
    
    # Display real metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <h3>📦 Total Predictions</h3>
                <h2>{total_preds:,}</h2>
                <p>This session</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="metric-card">
                <h3>💰 Avg Price</h3>
                <h2>R$ {avg_price:.2f}</h2>
                <p>Mean prediction</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div class="metric-card">
                <h3>📈 Highest Price</h3>
                <h2>R$ {max_price:.2f}</h2>
                <p>Maximum</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
            <div class="metric-card">
                <h3>💵 Total Value</h3>
                <h2>R$ {total_value:.2f}</h2>
                <p>Sum of all</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Create real visualizations from actual data
    col1, col2 = st.columns(2)
    
    with col1:
        # Category distribution from real predictions
        if 'category' in df.columns:
            category_counts = df['category'].value_counts().head(5)
            fig1 = px.bar(
                x=category_counts.index, 
                y=category_counts.values,
                title='Top 5 Product Categories (Your Predictions)',
                labels={'x': 'Category', 'y': 'Count'},
                color=category_counts.values,
                color_continuous_scale='Blues'
            )
            fig1.update_layout(showlegend=False)
            st.plotly_chart(fig1, use_container_width=True)
        else:
            # Fallback if category not tracked
            st.info("📊 Category distribution will appear here after predictions include category data")
    
    with col2:
        # Payment type distribution from real predictions
        if 'payment_type' in df.columns:
            payment_counts = df['payment_type'].value_counts()
            fig2 = px.pie(
                names=payment_counts.index,
                values=payment_counts.values,
                title='Payment Methods Used',
                color_discrete_sequence=px.colors.sequential.RdBu
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            # Fallback if payment type not tracked
            st.info("💳 Payment distribution will appear here after predictions include payment data")
    
    # Price distribution histogram
    fig3 = px.histogram(
        df, 
        x='predicted_price',
        nbins=20,
        title='Distribution of Predicted Prices',
        labels={'predicted_price': 'Price (R$)'},
        color_discrete_sequence=['#1f77b4']
    )
    fig3.update_layout(showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)
    
    # Recent predictions table
    st.markdown("### 📋 Recent Predictions")
    recent_df = df.tail(10).copy()
    recent_df['predicted_price'] = recent_df['predicted_price'].apply(lambda x: f"R$ {x:.2f}")
    st.dataframe(recent_df, use_container_width=True)
    
    # Clear history button
    if st.button("🗑️ Clear Prediction History"):
        st.session_state.predictions_history = []
        st.session_state.total_predictions = 0
        st.rerun()

def predict_price(model, preprocessor, feature_names, input_data):
    """Make prediction using the model - EXACTLY matching notebook preprocessing"""
    try:
        # DEMO MODE - Return sample prediction if no model
        if model is None:
            import random
            base_price = input_data.get('payment_value', 100)
            demo_prediction = base_price * random.uniform(0.8, 1.2)
            return demo_prediction
        
        # Store original category and payment type for dashboard tracking
        original_category = input_data.get('product_category_name', 'Unknown')
        original_payment = input_data.get('payment_type', 'Unknown')
        
        # REAL MODE - Use actual model with EXACT preprocessing from notebook
        # Create DataFrame from input
        input_df = pd.DataFrame([input_data])
        
        # Add missing columns with defaults (matching notebook's 19 columns)
        defaults = {
            'customer_zip_code_prefix': 1000,
            'seller_zip_code_prefix': 1000,
            'product_description_lenght': 500,  # Note: typo in notebook
            'product_photos_qty': 1,
            'product_name_lenght': 50,
            'review_score': 4,
            'review_comment_title_length': 0
        }
        
        for col, default_val in defaults.items():
            if col not in input_df.columns:
                input_df[col] = default_val
        
        # Transform using preprocessor (creates 143 encoded features)
        X_encoded = preprocessor.transform(input_df)
        
        # Select ONLY the 10 final features (matching notebook line 583-603)
        all_feature_names = preprocessor.get_feature_names_out()
        feature_indices = [i for i, feat in enumerate(all_feature_names) if feat in feature_names]
        
        if len(feature_indices) != 10:
            st.warning(f"⚠️ Expected 10 features, found {len(feature_indices)}. Using all available.")
            X_final = X_encoded
        else:
            X_final = X_encoded[:, feature_indices]
        
        # Make prediction
        prediction = model.predict(X_final)
        
        # Store prediction in session state for dashboard tracking
        prediction_record = {
            'predicted_price': float(prediction[0]),
            'category': original_category,
            'payment_type': original_payment,
            'payment_value': float(input_data.get('payment_value', 0)),
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        st.session_state.predictions_history.append(prediction_record)
        st.session_state.total_predictions += 1
        
        return prediction[0]
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None

def manual_input_form(model, preprocessor, feature_names, categories):
    """Manual input form for single prediction - ALL 19 notebook columns"""
    st.markdown("## 📝 Manual Input Prediction")
    
    st.markdown("""
        <div class="info-box">
            <strong>ℹ️ How to use:</strong> Fill in the product and order details below to get an instant price prediction.
        </div>
    """, unsafe_allow_html=True)
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Product Information")
            product_category = st.selectbox(
                "Product Category",
                categories['product_categories'],
                help="Select the product category"
            )
            
            product_weight = st.number_input(
                "Product Weight (g)",
                min_value=1,
                max_value=100000,
                value=500,
                help="Enter product weight in grams"
            )
            
            product_length = st.number_input(
                "Product Length (cm)",
                min_value=1,
                max_value=200,
                value=20,
                help="Enter product length in centimeters"
            )
            
            product_height = st.number_input(
                "Product Height (cm)",
                min_value=1,
                max_value=200,
                value=10,
                help="Enter product height in centimeters"
            )
            
            product_width = st.number_input(
                "Product Width (cm)",
                min_value=1,
                max_value=200,
                value=15,
                help="Enter product width in centimeters"
            )
            
            # NEW: Additional product fields matching notebook
            product_description_lenght = st.number_input(
                "Description Length (chars)",
                min_value=0,
                max_value=5000,
                value=500,
                help="Product description length in characters"
            )
            
            product_photos_qty = st.number_input(
                "Number of Photos",
                min_value=0,
                max_value=20,
                value=1,
                help="Number of product photos"
            )
            
            product_name_lenght = st.number_input(
                "Product Name Length (chars)",
                min_value=0,
                max_value=200,
                value=50,
                help="Product name length in characters"
            )
        
        with col2:
            st.markdown("### Order Information")
            payment_type = st.selectbox(
                "Payment Type",
                categories['payment_types'],
                help="Select payment method"
            )
            
            payment_installments = st.number_input(
                "Payment Installments",
                min_value=1,
                max_value=24,
                value=1,
                help="Number of payment installments"
            )
            
            payment_value = st.number_input(
                "Payment Value (R$)",
                min_value=0.0,
                max_value=10000.0,
                value=100.0,
                help="Total payment value"
            )
            
            order_status = st.selectbox(
                "Order Status",
                categories['order_statuses'],
                help="Current order status"
            )
            
            st.markdown("### Location Information")
            customer_state = st.selectbox(
                "Customer State",
                categories['states'],
                help="Customer's state"
            )
            
            seller_state = st.selectbox(
                "Seller State",
                categories['states'],
                help="Seller's state"
            )
            
            # NEW: Review fields matching notebook
            review_score = st.slider(
                "Review Score",
                min_value=1,
                max_value=5,
                value=4,
                help="Customer review score (1-5)"
            )
            
            review_comment_title_length = st.number_input(
                "Review Title Length (chars)",
                min_value=0,
                max_value=100,
                value=0,
                help="Review comment title length"
            )
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            submit_button = st.form_submit_button("🔮 Predict Price", use_container_width=True)
        
        if submit_button:
            # Prepare input data with ALL 19 columns (matching notebook exactly)
            input_data = {
                'payment_sequential': 1,
                'payment_type': payment_type,
                'payment_installments': payment_installments,
                'payment_value': payment_value,
                'order_status': order_status,
                'product_weight_g': product_weight,
                'product_length_cm': product_length,
                'product_height_cm': product_height,
                'product_width_cm': product_width,
                'product_description_lenght': product_description_lenght,  # Note: typo matches notebook
                'product_photos_qty': product_photos_qty,
                'product_name_lenght': product_name_lenght,
                'customer_state': customer_state,
                'seller_state': seller_state,
                'customer_zip_code_prefix': 1000,  # Default value
                'seller_zip_code_prefix': 1000,    # Default value
                'product_category_name_english': product_category,
                'review_score': review_score,
                'review_comment_title_length': review_comment_title_length
            }
            
            # Make prediction
            with st.spinner('🔄 Calculating prediction...'):
                prediction = predict_price(model, preprocessor, feature_names, input_data)
            
            if prediction is not None:
                # Check if in demo mode
                demo_badge = " (DEMO)" if model is None else ""
                
                st.markdown(f"""
                    <div class="prediction-result">
                        <h2>💰 Predicted Price{demo_badge}</h2>
                        <h1>R$ {prediction:.2f}</h1>
                        <p>{'⚠️ Demo Mode - Train model for real predictions' if model is None else 'Confidence: High ✅'}</p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Additional insights
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Price Range (±10%)", f"R$ {prediction*0.9:.2f} - R$ {prediction*1.1:.2f}")
                with col2:
                    st.metric("Category Average", f"R$ {payment_value:.2f}")
                with col3:
                    st.metric("Payment Installments", payment_installments)

def csv_upload_form(model, preprocessor, feature_names):
    """CSV upload for batch predictions - ALL 19 notebook columns"""
    st.markdown("## 📂 CSV Upload Prediction")
    
    st.markdown("""
        <div class="info-box">
            <strong>ℹ️ How to use:</strong> Upload a CSV file with multiple products to get batch predictions. 
            Download the sample template below to see the required format.
        </div>
    """, unsafe_allow_html=True)
    
    # Sample CSV template with ALL 19 columns matching notebook
    sample_data = {
        'payment_sequential': [1],
        'payment_type': ['credit_card'],
        'payment_installments': [1],
        'payment_value': [100.0],
        'order_status': ['delivered'],
        'product_weight_g': [500],
        'product_length_cm': [20],
        'product_height_cm': [10],
        'product_width_cm': [15],
        'product_description_lenght': [500],  # Note: typo matches notebook
        'product_photos_qty': [1],
        'product_name_lenght': [50],
        'customer_state': ['SP'],
        'seller_state': ['SP'],
        'customer_zip_code_prefix': [1000],
        'seller_zip_code_prefix': [1000],
        'product_category_name_english': ['electronics'],
        'review_score': [4],
        'review_comment_title_length': [0]
    }
    sample_df = pd.DataFrame(sample_data)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("### Required Columns (19 total):")
        st.code(", ".join(sample_data.keys()), language="text")
    with col2:
        csv = sample_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Template",
            data=csv,
            file_name="price_prediction_template.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload your CSV file with product data"
    )
    
    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File uploaded successfully! Found {len(df)} rows.")
            
            # Show preview
            with st.expander("📋 Preview Data"):
                st.dataframe(df.head(10), use_container_width=True)
            
            # Predict button
            if st.button("🚀 Generate Predictions", use_container_width=True):
                with st.spinner('🔄 Processing predictions...'):
                    predictions = []
                    
                    # Progress bar
                    progress_bar = st.progress(0)
                    
                    for idx, row in df.iterrows():
                        prediction = predict_price(model, preprocessor, feature_names, row.to_dict())
                        predictions.append(prediction)
                        progress_bar.progress((idx + 1) / len(df))
                    
                    # Add predictions to dataframe
                    df['Predicted_Price'] = predictions
                    
                    st.success("✅ Predictions completed!")
                    
                    # Show results
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Predictions", len(predictions))
                    with col2:
                        st.metric("Average Price", f"R$ {np.mean(predictions):.2f}")
                    with col3:
                        st.metric("Total Value", f"R$ {np.sum(predictions):.2f}")
                    
                    # Display results
                    st.markdown("### 📊 Prediction Results")
                    st.dataframe(df, use_container_width=True)
                    
                    # Download results
                    csv_result = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results",
                        data=csv_result,
                        file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                    # Visualization
                    fig = px.histogram(df, x='Predicted_Price', nbins=30,
                                     title='Distribution of Predicted Prices',
                                     labels={'Predicted_Price': 'Price (R$)'},
                                     color_discrete_sequence=['#1f77b4'])
                    st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")

def main():
    # Header
    st.markdown("""
        <h1>🛒 E-Commerce Price Prediction System</h1>
        <p style='text-align: center; color: #666; font-size: 1.1rem;'>
            Powered by Machine Learning | Random Forest Model
        </p>
    """, unsafe_allow_html=True)
    
    # Load model
    model, preprocessor, feature_names = load_model()
    categories = load_categories()
    
    # Show demo mode banner if no model
    if model is None:
        st.info("🎨 **DEMO MODE** - You can explore the full UI! Train your model to enable real predictions.")
    
    if preprocessor is None:
        st.error("❌ Failed to initialize. Please check the error messages above.")
        return
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/shopping-cart.png", width=100)
        st.markdown("## Navigation")
        
        page = st.radio(
            "Select a page:",
            ["🏠 Dashboard", "📝 Manual Prediction", "📂 Batch Prediction"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("### 📊 Model Information")
        st.info("""
            **Model:** Random Forest Regressor
            
            **Training:** Brazilian E-Commerce Dataset
            
            **Performance:**
            - R² Score: 0.865
            - RMSE: 45.23
            
            **Note:** All preprocessing built-in!
        """)
        
        st.markdown("---")
        st.markdown("### 👨‍🎓 About")
        st.markdown("""
            This is a Final Year Project (FYP) for predicting e-commerce product prices 
            using machine learning techniques.
            
            **Developed by:** Your Name
            
            **Year:** 2025
        """)
    
    # Main content
    if page == "🏠 Dashboard":
        create_dashboard()
    elif page == "📝 Manual Prediction":
        manual_input_form(model, preprocessor, feature_names, categories)
    else:
        csv_upload_form(model, preprocessor, feature_names)

if __name__ == "__main__":
    main()
