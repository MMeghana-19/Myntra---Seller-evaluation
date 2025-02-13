
import streamlit as st
import pickle
import pandas as pd
from fuzzywuzzy import process

# Load the pickled components
def load_pickle_files():
    with open('XGB_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    with open('ordinal_encoder.pkl', 'rb') as f:
        ordinal_encoder = pickle.load(f)

    with open('label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)

    with open('class_mapping.pkl', 'rb') as f:
        class_mapping = pickle.load(f)

    return model, scaler, ordinal_encoder, label_encoders, class_mapping

# Load all the components
model, scaler, ordinal_encoder, label_encoders, class_mapping = load_pickle_files()

# Function to preprocess user input
def preprocess_input(user_input):
    # Encode categorical features using label encoders
    user_input['gender'] = label_encoders['gender'].get_loc(user_input['gender'])
    user_input['category'] = label_encoders['category'].get_loc(user_input['category'])
    user_input['sub_category'] = label_encoders['sub_category'].get_loc(user_input['sub_category'])
    user_input['seller'] = label_encoders['seller'].get_loc(user_input['seller'])

    # Apply ordinal encoding for price_category, discount_category, brand_performance
    user_input[['price_category', 'discount_category', 'brand_performance']] = ordinal_encoder.transform(user_input[['price_category', 'discount_category', 'brand_performance']])

    # Scaling numeric features
    numeric_columns = ['price', 'mrp', 'ratingTotal', 'product_avg_rating']
    user_input[numeric_columns] = scaler.transform(user_input[numeric_columns])

    return user_input

# Function to handle fuzzy matching for seller name
def find_closest_seller(user_input, seller_names):
    best_match, score = process.extractOne(user_input, seller_names)
    return best_match, score

# Load unique seller names from dataset (Replace 'myntra' with your actual dataframe)
seller_names = myntra['seller'].unique().tolist()

# Streamlit UI for input
st.title('Seller Performance Evaluation & Prediction')
st.write("Enter the seller & product details below to predict its performance:")

# Input fields for the user
gender = st.selectbox('Gender', ['Male', 'Female', 'Unisex'])
category = st.selectbox('Category', ['Care & Beauty', 'Clothing', 'Footwear & Bag', 'Accessories', 'Home & Living', 'Miscellaneous'])
sub_category = st.selectbox('Sub-category', ['tshirts', 'shorts', 'shirts', 'tops', 'dresses', 'jeans',
       'co-ords', 'tights', 'leggings', 'jumpsuit', 'trousers',
       'track-pants', 'shrug', 'jewellery-set', 'sunglasses', 'ring',
       'hair-accessory', 'belts', 'bath-and-body-gift-set', 'earrings',
       'anklet', 'necklace-and-chains', 'bangle', 'handbags', 'bracelet',
       'mangalsutra', 'body-wash-and-scrub', 'kurtas', 'sarees',
       'lehenga-choli', 'kurta-sets', 'ethnic-dresses', 'kurtis',
       'smart-watches', 'watch-gift-set', 'watches', 'briefs',
       'lingerie-set', 'churidar', 'swimwear-cover-up-top', 'shapewear',
       'flip-flops', 'heels', 'flats', 'sports-shoes', 'casual-shoes',
       'bra', 'sweatshirts', 'socks', 'swimwear', 'nightdress',
       'lounge-pants', 'lingerie-accessories', 'night-suits',
       'baby-dolls', 'lounge-shorts', 'highlighter-and-blush',
       'eyeshadow', 'lipstick', 'eyebrow-enhancer', 'nail-essentials',
       'face-moisturisers', 'shampoo-and-conditioner',
       'face-wash-and-cleanser', 'bb-and-cc-cream', 'toner', 'hair-serum',
       'hair-cream-and-mask', 'face-scrub-and-exfoliator',
       'foundation-and-primer', 'concealer', 'compact', 'lip-gloss',
       'face-serum-and-gel', 'backpacks', 'gold-coin', 'pendant-gold',
       'sherwani', 'pyjamas', 'blazers', 'innerwear-vests', 'trunk',
       'boxers', 'lounge-tshirts', 'tracksuits', 'sunscreen',
       'shaving-essentials', 'body-lotion', 'perfume-and-body-mist',
       'trimmer', 'hair-oil', 'hair-appliance', 'deodorant', 'headphones',
       'trolley-bag', 'clothing-set', 'sandals', 'soft-toys-and-dolls',
       'activity-toys-and-games', 'learning-and-development-toys',
       'foundation', 'kajal-and-eyeliner', 'mascara', 'face-primer',
       'makeup-remover', 'lip-care', 'epilator', 'shaving-brush--razor',
       'fragrance-gift-set', 'makeup-gift-set', 'skin-care-gift-set',
       'bedsheets', 'bed-covers', 'blankets-quilts-and-dohars',
       'bedding-set', 'doormats', 'carpets', 'floor-mats--dhurries',
       'bath-towels', 'bath-robe', 'face-towels', 'towel-set',
       'kitchen-storage', 'cookware', 'water-bottle', 'serveware',
       'bar-and-drinkware', 'dinnerware', 'cups-and-mugs', 'table-lamps',
       'floor-lamps', 'ceiling-lamps', 'wall-lamps', 'swim-tops',
       'swimwear-accessories', 'swim-bottoms', 'clutches', 'scarves',
       'outdoor-masks', 'caps', 'laptop-bag', 'toe-rings', 'dupatta',
       'capris', 'stockings', 'beauty-accessory', 'hair-care-kit',
       'mask-and-peel', 'hair-colour', 'hair-masks', 'duffel-bag',
       'rucksacks', 'wallets', 'makeup-brushes', 'lip-liner',
       'beauty-gift-set', 'eye-cream', 'mattress-protector', 'pillows',
       'hand-towels', 'bathroom-accessories', 'dining-essentials',
       'appliance-covers', 'kitchen-tools', 'outdoor-lamps',
       'string-lights', 'head-jewellery', 'jackets', 'sports-sandals',
       'thermal-bottoms', 'pendant', 'nehru-jackets', 'rompers',
       'dungarees', 'musical-toys', 'mosquito-nets', 'table-covers',
       'skirts', 'sweaters', 'jeggings', 'frames', 'nail-polish',
       'massager', 'body-oil', 'body-wax-and-essentials', 'salwar',
       'thermal-tops', 'thermal-set', 'eye-mask-and-patches',
       'beard--moustache-care', 'baby-care-products', 'camisoles',
       'bodysuit', 'construction-toys', 'facial-kit', 'bar-accessories',
       'trays', 'mobile-accessories', 'umbrellas', 'hair-brush-and-comb',
       'hair-spray', 'earrings-gold', 'ring-gold', 'suits', 'speakers',
       'hat', 'toy-vehicles', 'beach-towels', 'coasters', 'cutlery',
       'table-placemats', 'bathroom-lights', 'messenger-bag', 'tunics',
       'hair-gel-and-spray', 'dhotis', 'mens-grooming-kit', 'waist-pouch',
       'palazzos', 'bath-rugs', 'shower-curtains', 'dress-material',
       'bath-accessories', 'hand-and-feet-cream',
       'hand-wash-and-sanitizer', 'saree-accessories', 'makeup-kit',
       'sheet-masks', 'duvet-cover', 'brooch', 'stoles', 'nosepin',
       'accessory-gift-set', 'robe', 'earrings-diamond', 'condoms',
       'baby-apparel-gift-set', 'aprons', 'watch-organiser', 'boots',
       'pillow-covers', 'travel-accessory', 'patiala',
       'electric-toothbrush', 'feminine-hygiene', 'silver-coins',
       'handkerchief', 'bath-soak-salt-and-oil', 'fitness-bands',
       'baby-bed-sets', 'booties', 'sports-accessories', 'saree-blouse',
       'shawl', 'coats', 'setting-spray', 'ties', 'formal-shoes',
       'sleepsuit', 'kaleeras', 'hair-gels-and-wax', 'tissues-and-wipes',
       'bibs', 'diaper-bags', 'curtains-and-sheers', 'necklace-gold',
       'harem-pants', 'table-cloth', 'kitchen-linen-sets',
       'pendant-diamond', 'baby-pillow', 'toiletry-kit', 'gloves',
       'insect-repellent', 'waistcoat', 'suspenders', 'bleach',
       'eye-primer', 'bakeware', 'charms', 'nosepin-gold',
       'table-napkins', 'mufflers', 'burqas', 'baby-gear--nursery',
       'baby-photoshoot-props', 'strollers', 'ties-and-cufflinks',
       'kitchen-towels', 'baby-carriers', 'table-tennis-kits',
       'false-eyelashes', 'oven-glove', 'clothing-fabric', 'pens',
       'rain-jacket', 'stationery', 'sindoor', 'headband', 'watch-straps',
       'cufflinks', 'jibbitz', 'lubricants', 'helmets', 'lip-plumper',
       'ride-on-vehicles', 'carry-cot', 'baby-sleeping-bag', 'corset',
       'kitchen-gloves', 'decals-and-stickers', 'pocket-squares',
       'baby-bathers', 'tablet-sleeve', 'laundry-bag', 'cushions',
       'table-linen-sets', 'photo-frames', 'hooks-and-holders', 'slips',
       'key-chain', 'baby-utensils', 'feeding-bottles', 'collectibles',
       'necklace-and-chains-diamond', 'shoe-accessories',
       'sunglasses-case', 'bolsters', 'baby-sippers', 'tent',
       'sleeping-bag', 'baby-nail-grooming', 'organisers',
       'personal-care-hamper', 'shoe-care', 'art-and-craft', 'box',
       'mask-chains', 'sleeping-mat', 'pagri-and-safa',
       'school-essentials', 'garden-accessories', 'baby-shower-caps',
       'showpieces', 'face-shield', 'yoga-mats', 'teether', 'rakhi',
       'earbuds-case', 'breast-pumps', 'mittens', 'boxes', 'runners',
       'salwar-and-dupatta', 'wristbands', 'hanger', 'clocks',
       'bottle-cleaners', 'baby-hair-brush', 'feeding-essentials',
       'food-feeders', 'wall-art', 'cushion-covers', 'baby-oral-care',
       'bangle-gold', 'pacifiers', 'baby-bath-sponges', 'nosepin-diamond',
       'swings', 'high-chairs', 'rockers', 'cradles', 'walkers',
       'warmers', 'car-seats', 'ring-diamond', 'bath-tub', 'chair-pads',
       'action-figures-and-toys', 'ottomans', 'https:', 'baby-pool',
       'bean-bag-cover', 'bracelet-diamond', 'role-play-and-playsets',
       'wall-shelves', 'napkin-set', 'festive-decor', 'mirrors',
       'footballs', 'bolster-covers', 'diwan-set', 'wall-decor',
       'patiala-and-dupatta', 'swimwear-cover-up-bottom',
       'pooja-essentials', 'protective-bodysuits', 'free-gifts',
       'candle-holders', 'throws', 'aroma-oil-diffusers', 'jackets-smart',
       'sofa-covers', 'bath-sets'])
seller = st.text_input('Brand Name')
product_avg_rating = st.number_input('Product Average Rating', min_value=0.0, max_value=5.0, step=0.1)
price_category = st.selectbox('Price Category', ['Low', 'Medium', 'High'])
discount_category = st.selectbox('Discount Category', ['None', 'Low', 'Medium', 'High'])

# When the user clicks on 'Evaluate/Predict' button
if st.button('Evaluate/Predict'):
    # Handle fuzzy matching for seller name
    if seller:
        closest_seller, match_score = find_closest_seller(seller.lower(), seller_names)
        st.write(f"Closest match for seller: {closest_seller} (Score: {match_score})")

        # If match score is above threshold, use the matched seller name
        if match_score >= 80:
            seller = closest_seller
        else:
            st.write("No close match found. Please check for spelling mistakes.")
    
    # Prepare user input as DataFrame
    user_input = pd.DataFrame({
        'gender': [gender.lower()],
        'category': [category.lower()],
        'name': [name.lower()],
        'seller': [seller.lower()],
        'product_avg_rating': [product_avg_rating],
        'price_category': [price_category.lower()],
        'discount_category': [discount_category.lower()]
    })

    
    # Preprocess input data
    processed_input = preprocess_input(user_input)

    # Make prediction
    prediction = model.predict(processed_input)

    # Convert numeric prediction to human-readable label
    predicted_label = class_mapping[prediction[0]]

    # Display the result
    st.write(f"Predicted Brand Performance: {predicted_label}")
