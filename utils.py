"""
Utility functions for the E-Commerce Price Prediction App
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any

def validate_input_data(data: Dict[str, Any]) -> tuple[bool, str]:
    """
    Validate input data for prediction
    
    Args:
        data: Dictionary containing input features
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    required_fields = [
        'payment_sequential', 'payment_type', 'payment_installments',
        'payment_value', 'order_status', 'product_weight_g',
        'product_length_cm', 'product_height_cm', 'product_width_cm',
        'customer_state', 'seller_state', 'product_category_name_english'
    ]
    
    # Check all required fields are present
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        return False, f"Missing required fields: {', '.join(missing_fields)}"
    
    # Validate numerical fields
    numerical_fields = {
        'payment_installments': (1, 24),
        'payment_value': (0, 100000),
        'product_weight_g': (1, 100000),
        'product_length_cm': (1, 200),
        'product_height_cm': (1, 200),
        'product_width_cm': (1, 200)
    }
    
    for field, (min_val, max_val) in numerical_fields.items():
        if field in data:
            try:
                value = float(data[field])
                if value < min_val or value > max_val:
                    return False, f"{field} must be between {min_val} and {max_val}"
            except (ValueError, TypeError):
                return False, f"{field} must be a valid number"
    
    return True, ""

def calculate_product_volume(length: float, height: float, width: float) -> float:
    """
    Calculate product volume in cubic centimeters
    
    Args:
        length: Product length in cm
        height: Product height in cm
        width: Product width in cm
        
    Returns:
        Volume in cm³
    """
    return length * height * width

def calculate_price_range(predicted_price: float, margin: float = 0.1) -> tuple[float, float]:
    """
    Calculate price range with margin
    
    Args:
        predicted_price: Predicted price
        margin: Margin percentage (default 10%)
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    lower = predicted_price * (1 - margin)
    upper = predicted_price * (1 + margin)
    return lower, upper

def format_currency(value: float, currency: str = "R$") -> str:
    """
    Format value as currency
    
    Args:
        value: Numerical value
        currency: Currency symbol
        
    Returns:
        Formatted currency string
    """
    return f"{currency} {value:,.2f}"

def get_category_insights(category: str) -> Dict[str, Any]:
    """
    Get insights for a specific product category
    
    Args:
        category: Product category name
        
    Returns:
        Dictionary with category insights
    """
    # Sample insights - replace with actual data
    insights = {
        'electronics': {
            'avg_price': 250.00,
            'popular_payment': 'credit_card',
            'avg_installments': 3,
            'demand': 'High'
        },
        'health_beauty': {
            'avg_price': 85.00,
            'popular_payment': 'credit_card',
            'avg_installments': 1,
            'demand': 'Medium'
        },
        'furniture_decor': {
            'avg_price': 450.00,
            'popular_payment': 'credit_card',
            'avg_installments': 5,
            'demand': 'Medium'
        }
    }
    
    return insights.get(category, {
        'avg_price': 120.00,
        'popular_payment': 'credit_card',
        'avg_installments': 1,
        'demand': 'Medium'
    })

def prepare_batch_predictions_report(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Prepare statistical report for batch predictions
    
    Args:
        df: DataFrame with predictions
        
    Returns:
        Dictionary with statistical insights
    """
    report = {
        'total_predictions': len(df),
        'avg_price': df['Predicted_Price'].mean(),
        'median_price': df['Predicted_Price'].median(),
        'min_price': df['Predicted_Price'].min(),
        'max_price': df['Predicted_Price'].max(),
        'std_dev': df['Predicted_Price'].std(),
        'total_value': df['Predicted_Price'].sum()
    }
    
    # Category breakdown if available
    if 'product_category_name_english' in df.columns:
        report['category_breakdown'] = df.groupby('product_category_name_english')['Predicted_Price'].agg([
            ('count', 'count'),
            ('avg_price', 'mean'),
            ('total', 'sum')
        ]).to_dict('index')
    
    return report

def get_state_name(state_code: str) -> str:
    """
    Get full state name from code
    
    Args:
        state_code: Two-letter state code
        
    Returns:
        Full state name
    """
    states = {
        'SP': 'São Paulo', 'RJ': 'Rio de Janeiro', 'MG': 'Minas Gerais',
        'RS': 'Rio Grande do Sul', 'PR': 'Paraná', 'SC': 'Santa Catarina',
        'BA': 'Bahia', 'DF': 'Distrito Federal', 'GO': 'Goiás',
        'PE': 'Pernambuco', 'CE': 'Ceará', 'PA': 'Pará',
        'ES': 'Espírito Santo', 'MT': 'Mato Grosso', 'MA': 'Maranhão',
        'MS': 'Mato Grosso do Sul', 'PB': 'Paraíba', 'PI': 'Piauí',
        'RN': 'Rio Grande do Norte', 'AL': 'Alagoas', 'SE': 'Sergipe',
        'RO': 'Rondônia', 'TO': 'Tocantins', 'AM': 'Amazonas',
        'AC': 'Acre', 'AP': 'Amapá', 'RR': 'Roraima'
    }
    return states.get(state_code, state_code)

def calculate_confidence_score(
    prediction: float,
    payment_installments: int,
    product_weight: float
) -> str:
    """
    Calculate confidence level for prediction
    
    Args:
        prediction: Predicted price
        payment_installments: Number of installments
        product_weight: Product weight
        
    Returns:
        Confidence level string
    """
    # Simple heuristic for confidence
    if 50 <= prediction <= 500 and payment_installments <= 12 and product_weight < 10000:
        return "High"
    elif 10 <= prediction <= 1000 and payment_installments <= 18:
        return "Medium"
    else:
        return "Low"
