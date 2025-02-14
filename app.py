
import pickle
import pandas as pd
import streamlit as st
from fuzzywuzzy import process
import numpy as np

# Load the pickled components
def load_pickle_files():
    with open('xgb_model.pkl', 'rb') as f:
        xgb_model = pickle.load(f)

    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    with open('ordinal_encoder.pkl', 'rb') as f:
        ordinal_encoder = pickle.load(f)

    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)

    with open('class_mapping.pkl', 'rb') as f:
        class_mapping = pickle.load(f)

    return xgb_model, scaler, ordinal_encoder, label_encoder, class_mapping

# Load all the components
xgb_model, scaler, ordinal_encoder, label_encoder, class_mapping = load_pickle_files()

# Function to preprocess user input
def preprocess_input(user_input):
    # Encode categorical features using label encoders
    user_input['gender'] = label_encoder['gender'].get_loc(user_input['gender'])
    user_input['category'] = label_encoder['category'].get_loc(user_input['category'])
    user_input['sub_category'] = label_encoder['sub_category'].get_loc(user_input['sub_category'])
    user_input['seller'] = label_encoder['seller'].get_loc(user_input['seller'])

    # Apply ordinal encoding for price_category, discount_category, brand_performance
    user_input[['price_category', 'discount_category']] = ordinal_encoder.transform(user_input[['price_category', 'discount_category']])

    # Scaling numeric features
    numeric_columns = ['product_rating_deviation']
    user_input[numeric_columns] = scaler.transform(user_input[numeric_columns])

    return user_input

# Function to handle fuzzy matching for seller name
def find_closest_seller(user_input, seller_names):
    best_match, score = process.extractOne(user_input, seller_names)
    return best_match, score

# Load unique seller names from dataset (Replace 'myntra' with your actual dataframe)
# GitHub raw file URL (replace with your actual file URL)
file_url = 'https://raw.githubusercontent.com/MMeghana-19/Myntra---Seller-evaluation/main/myntra202305041052.csv'

# Read the file in chunks to avoid memory overload
chunk_size = 10000  # Adjust the chunk size as per your available memory
chunks = pd.read_csv(file_url, chunksize=chunk_size)

# Initialize an empty list to store the chunks
myntra_chunks = []

# Iterate over each chunk and append it to the list
for chunk in chunks:
    myntra_chunks.append(chunk)

# Concatenate all chunks into a single DataFrame
myntra = pd.concat(myntra_chunks, ignore_index=True)
seller_names = myntra['seller'].unique().tolist()

# Streamlit UI for input
st.title('Seller Performance Evaluation & Prediction')
st.write("Enter the seller & product details below to predict its performance:")

# Input fields for the user
gender = st.selectbox('Gender', ['Male', 'Female', 'Unisex'])
category = st.selectbox('Category', ['Care & Beauty', 'Clothing', 'Footwear & Bag', 'Accessories', 'Home & Living', 'Miscellaneous'])
sub_category = st.selectbox('Sub-category', ['tshirts', 'shorts', 'shirts', 'tops', 'dresses', 'jeans', 'co-ords', 'tights', 'leggings', 'jumpsuit',
    'trousers', 'track-pants', 'shrug', 'jewellery-set', 'sunglasses', 'ring', 'hair-accessory', 'belts', 'bath-and-body-gift-set', 'earrings',
    'anklet', 'necklace-and-chains', 'bangle', 'handbags', 'bracelet', 'mangalsutra', 'body-wash-and-scrub', 'kurtas', 'sarees', 'lehenga-choli',
    'kurta-sets', 'ethnic-dresses', 'kurtis', 'smart-watches', 'watch-gift-set', 'watches', 'briefs', 'lingerie-set', 'churidar', 
    'swimwear-cover-up-top', 'shapewear', 'flip-flops', 'heels', 'flats', 'sports-shoes', 'casual-shoes', 'bra', 'sweatshirts', 'socks', 'swimwear',
    'nightdress', 'lounge-pants', 'lingerie-accessories', 'night-suits', 'baby-dolls', 'lounge-shorts', 'highlighter-and-blush', 'eyeshadow', 
    'lipstick', 'eyebrow-enhancer', 'nail-essentials', 'face-moisturisers', 'shampoo-and-conditioner', 'face-wash-and-cleanser', 'bb-and-cc-cream', 
    'toner', 'hair-serum', 'hair-cream-and-mask', 'face-scrub-and-exfoliator', 'foundation-and-primer', 'concealer', 'compact', 'lip-gloss', 
    'face-serum-and-gel', 'backpacks', 'gold-coin', 'pendant-gold', 'sherwani', 'pyjamas', 'blazers', 'innerwear-vests', 'trunk', 'boxers', 
    'lounge-tshirts', 'tracksuits', 'sunscreen', 'shaving-essentials', 'body-lotion', 'perfume-and-body-mist', 'trimmer', 'hair-oil', 'hair-appliance',
    'deodorant', 'headphones', 'trolley-bag', 'clothing-set', 'sandals', 'soft-toys-and-dolls', 'activity-toys-and-games', 
    'learning-and-development-toys', 'foundation', 'kajal-and-eyeliner', 'mascara', 'face-primer', 'makeup-remover', 'lip-care', 'epilator',
    'shaving-brush--razor', 'fragrance-gift-set', 'makeup-gift-set', 'skin-care-gift-set', 'bedsheets', 'bed-covers', 'blankets-quilts-and-dohars',
    'bedding-set', 'doormats', 'carpets', 'floor-mats--dhurries', 'bath-towels', 'bath-robe', 'face-towels', 'towel-set', 'kitchen-storage', 'cookware',
    'water-bottle', 'serveware', 'bar-and-drinkware', 'dinnerware', 'cups-and-mugs', 'table-lamps', 'floor-lamps', 'ceiling-lamps', 'wall-lamps',
    'swim-tops', 'swimwear-accessories', 'swim-bottoms', 'clutches', 'scarves', 'outdoor-masks', 'caps', 'laptop-bag', 'toe-rings', 'dupatta', 'capris',
    'stockings', 'beauty-accessory', 'hair-care-kit', 'mask-and-peel', 'hair-colour', 'hair-masks', 'duffel-bag', 'rucksacks', 'wallets', 
    'makeup-brushes', 'lip-liner', 'beauty-gift-set', 'eye-cream', 'mattress-protector', 'pillows', 'hand-towels', 'bathroom-accessories', 
    'dining-essentials', 'appliance-covers', 'kitchen-tools', 'outdoor-lamps', 'string-lights', 'head-jewellery', 'jackets', 'sports-sandals', 
    'thermal-bottoms', 'pendant', 'nehru-jackets', 'rompers', 'dungarees', 'musical-toys', 'mosquito-nets', 'table-covers', 'skirts', 'sweaters',
    'jeggings', 'frames', 'nail-polish', 'massager', 'body-oil', 'body-wax-and-essentials', 'salwar', 'thermal-tops', 'thermal-set',
    'eye-mask-and-patches', 'beard--moustache-care', 'baby-care-products', 'camisoles', 'bodysuit', 'construction-toys', 'facial-kit', 
    'bar-accessories', 'trays', 'mobile-accessories', 'umbrellas', 'hair-brush-and-comb', 'hair-spray', 'earrings-gold', 'ring-gold', 'suits',
    'speakers', 'hat', 'toy-vehicles', 'beach-towels', 'coasters', 'cutlery', 'table-placemats', 'bathroom-lights', 'messenger-bag', 'tunics',
    'hair-gel-and-spray', 'dhotis', 'mens-grooming-kit', 'waist-pouch', 'palazzos', 'bath-rugs', 'shower-curtains', 'dress-material', 
    'bath-accessories', 'hand-and-feet-cream', 'hand-wash-and-sanitizer', 'saree-accessories', 'makeup-kit', 'sheet-masks', 'duvet-cover', 'brooch',
    'stoles', 'nosepin', 'accessory-gift-set', 'robe', 'earrings-diamond', 'condoms', 'baby-apparel-gift-set', 'aprons', 'watch-organiser', 'boots',
    'pillow-covers', 'travel-accessory', 'patiala', 'electric-toothbrush', 'feminine-hygiene', 'silver-coins', 'handkerchief', 'bath-soak-salt-and-oil',
    'fitness-bands', 'baby-bed-sets', 'booties', 'sports-accessories', 'saree-blouse', 'shawl', 'coats', 'setting-spray', 'ties', 'formal-shoes', 
    'sleepsuit', 'kaleeras', 'hair-gels-and-wax', 'tissues-and-wipes', 'bibs', 'diaper-bags', 'curtains-and-sheers', 'necklace-gold', 'harem-pants',
    'table-cloth', 'kitchen-linen-sets', 'pendant-diamond', 'baby-pillow', 'toiletry-kit', 'gloves', 'insect-repellent', 'waistcoat', 'suspenders', 
    'bleach', 'eye-primer', 'bakeware', 'charms', 'nosepin-gold', 'table-napkins', 'mufflers', 'burqas', 'baby-gear--nursery', 'baby-photoshoot-props',
    'strollers', 'ties-and-cufflinks', 'kitchen-towels', 'baby-carriers', 'table-tennis-kits', 'false-eyelashes', 'oven-glove', 'clothing-fabric', 
    'pens', 'rain-jacket', 'stationery', 'sindoor', 'headband', 'watch-straps', 'cufflinks', 'jibbitz', 'lubricants', 'helmets', 'lip-plumper',
    'ride-on-vehicles', 'carry-cot', 'baby-sleeping-bag', 'corset', 'kitchen-gloves', 'decals-and-stickers', 'pocket-squares', 'baby-bathers', 
    'home-improvement', 'waist-belt', 'baby-stroller', 'bandanas'])
price_category = st.selectbox('Price Category', ['Low', 'Medium', 'High'])
discount_category = st.selectbox('Discount Category', ['Low Discount', 'Medium Discount', 'High Discount'])
seller_input = st.text_input('Seller Name')


# Input for numeric fields
# Slider for product_rating_deviation
product_rating_deviation = st.slider('Product Rating Deviation', min_value=0.0, max_value=5.0, value=0.0, step=0.1)

# If the slider value is negative, set it to 0
if product_rating_deviation < 0:
    product_rating_deviation = 0.0

# Now use the value for prediction or display as required
st.write(f"Product Rating Deviation: {product_rating_deviation}")


# Preprocessing the user input
# Preprocessing the user input with case-insensitive handling
user_input = pd.DataFrame({
    'gender': [gender.lower()],  # Convert gender input to lowercase
    'category': [category.lower()],  # Convert category input to lowercase
    'sub_category': [sub_category.lower()],  # Convert sub_category input to lowercase
    'price_category': [price_category.lower()],  # Convert price_category input to lowercase
    'discount_category': [discount_category.lower()],  # Convert discount_category input to lowercase
    'seller': [seller_input.lower()],  # Convert seller input to lowercase
    'product_rating_deviation': [product_rating_deviation]  # Numeric value doesn't need case conversion
})

# Fuzzy matching for seller name
closest_seller, score = find_closest_seller(seller_input, seller_names)
if score >= 80:
    user_input['seller'] = closest_seller

# Preprocess input and make predictions
# Button to trigger prediction
if st.button('Predict'):
    # Preprocess input and make predictions
    processed_input = preprocess_input(user_input)
    predicted_performance = xgb_model.predict(processed_input)

    # Map the predicted output to human-readable label
    predicted_performance_label = class_mapping.get(predicted_performance[0], 'Invalid Prediction')

    # Display the result
    st.write(f"Predicted Seller Performance: {predicted_performance_label}")
