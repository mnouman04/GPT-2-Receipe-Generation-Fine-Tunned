"""
🍳 AI Recipe Generator - Streamlit App (Google Drive Model)
Loads fine-tuned GPT-2 models directly from Google Drive.
"""

import streamlit as st
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import time
import os
from typing import List, Dict
import gdown  # For downloading from Google Drive

# ============================================================================ 
# PAGE CONFIGURATION
# ============================================================================ 
st.set_page_config(
    page_title="🍳 AI Recipe Generator",
    page_icon="🍳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================ 
# CUSTOM CSS FOR MODERN UI
# ============================================================================ 
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #FF6B6B;
        --secondary-color: #4ECDC4;
        --background-color: #1E1E1E;
        --text-color: #FFFFFF;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    
    .main-header h1 {
        color: white;
        font-size: 3rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: #f0f0f0;
        font-size: 1.2rem;
        margin-top: 0.5rem;
    }
    
    /* Ingredient chips */
    .ingredient-chip {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1rem;
        margin: 0.25rem;
        border-radius: 25px;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    
    .ingredient-chip:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Recipe card */
    .recipe-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        margin: 1rem 0;
        color: #2d3748;
    }
    
    .recipe-title {
        color: #667eea;
        font-size: 2rem;
        font-weight: 800;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .recipe-section {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .recipe-section h3 {
        color: #764ba2;
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        border-bottom: 3px solid #667eea;
        padding-bottom: 0.5rem;
    }
    
    /* Stats cards */
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 35px rgba(102, 126, 234, 0.6);
    }
    
    .stat-value {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
    }
    
    .stat-label {
        font-size: 1rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 700;
        border-radius: 50px;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 35px rgba(102, 126, 234, 0.6);
    }
    
    /* Input fields */
    .stTextInput>div>div>input {
        border-radius: 10px;
        border: 2px solid #667eea;
        padding: 0.75rem;
        font-size: 1rem;
    }
    
    .stTextArea>div>div>textarea {
        border-radius: 10px;
        border: 2px solid #667eea;
        padding: 0.75rem;
        font-size: 1rem;
    }
    
    /* Loading animation */
    .loading-chef {
        text-align: center;
        font-size: 3rem;
        animation: bounce 1s infinite;
    }
    
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-20px); }
    }
    
    /* Success message */
    .success-message {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        padding: 1rem;
        border-radius: 10px;
        color: #2d3748;
        font-weight: 600;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(132, 250, 176, 0.4);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Popular ingredients section */
    .popular-section {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(252, 182, 159, 0.4);
    }
    
    .popular-section h3 {
        color: #d35400;
        font-weight: 800;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================ 
# GOOGLE DRIVE FILE LINKS
# ============================================================================ 

# IMPORTANT: Extract the FILE_ID from your Google Drive links
# Format for files: https://drive.google.com/file/d/FILE_ID/view?usp=sharing
# You need only the FILE_ID part

# For a folder, you need to:
# 1. Zip the folder first
# 2. Upload the zip file to Google Drive
# 3. Share it and get the file ID

# Example:
# Original link: https://drive.google.com/file/d/1hi_3aAcyzyNsmtbaTlC56bWBxId0dh3l/view?usp=sharing
# File ID: 1hi_3aAcyzyNsmtbaTlC56bWBxId0dh3l

GDRIVE_MODEL_FOLDER_ID = None  # Set to None if using state dict
GDRIVE_STATE_DICT_ID = "1hi_3aAcyzyNsmtbaTlC56bWBxId0dh3l"  # Extracted FILE_ID only

LOCAL_MODEL_DIR = "./recipe_gpt2_model"
LOCAL_STATE_DICT = "./RecipeGenerationGPT2.pt"

def download_from_gdrive(file_id: str, output: str):
    """Download file from Google Drive if not exists."""
    if not os.path.exists(output):
        try:
            # Use gdown with proper file ID format
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, output, quiet=False, fuzzy=True)
        except Exception as e:
            st.error(f"Failed to download file: {str(e)}")
            st.info("Make sure the file is shared with 'Anyone with the link' can view")
            raise

# ============================================================================ 
# MODEL LOADING FUNCTION
# ============================================================================ 

@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer():
    """
    Load fine-tuned GPT-2 from Google Drive.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Method 1: Load from saved pretrained directory (if you have the folder)
        if os.path.exists(LOCAL_MODEL_DIR) and os.path.exists(os.path.join(LOCAL_MODEL_DIR, "config.json")):
            st.info("📁 Loading model from local directory...")
            tokenizer = GPT2Tokenizer.from_pretrained(LOCAL_MODEL_DIR)
            model = GPT2LMHeadModel.from_pretrained(LOCAL_MODEL_DIR)
            model.to(device)
            model.eval()
            return model, tokenizer, device, "pretrained_dir_local"
        
        # Method 2: Load from state dict file (download from Drive if needed)
        elif GDRIVE_STATE_DICT_ID:
            st.info("☁️ Downloading model from Google Drive...")
            download_from_gdrive(GDRIVE_STATE_DICT_ID, LOCAL_STATE_DICT)
            
            st.info("🔄 Loading model weights...")
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            tokenizer.pad_token = tokenizer.eos_token
            
            model = GPT2LMHeadModel.from_pretrained("gpt2")
            state_dict = torch.load(LOCAL_STATE_DICT, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            
            st.success("✅ Model loaded successfully!")
            return model, tokenizer, device, "state_dict_drive"
        
        # Method 3: Load from local checkpoint
        elif os.path.exists(LOCAL_STATE_DICT):
            st.info("💾 Loading model from local checkpoint...")
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            tokenizer.pad_token = tokenizer.eos_token
            
            model = GPT2LMHeadModel.from_pretrained("gpt2")
            state_dict = torch.load(LOCAL_STATE_DICT, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            return model, tokenizer, device, "state_dict_local"
        
        # Method 4: Load from model_checkpoints folder
        elif os.path.exists('./model_checkpoints/best_model.pt'):
            st.info("📦 Loading model from checkpoint folder...")
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            tokenizer.pad_token = tokenizer.eos_token
            
            model = GPT2LMHeadModel.from_pretrained("gpt2")
            state_dict = torch.load('./model_checkpoints/best_model.pt', map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            return model, tokenizer, device, "checkpoint"

        # Fallback: Load base GPT-2
        else:
            st.warning("⚠️ Fine-tuned model not found. Loading base GPT-2.")
            st.info("For better results, upload your model file or configure Google Drive access.")
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            tokenizer.pad_token = tokenizer.eos_token
            model = GPT2LMHeadModel.from_pretrained("gpt2")
            model.to(device)
            model.eval()
            return model, tokenizer, device, "base_gpt2"
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.info("""
        **Troubleshooting Steps:**
        1. Make sure your Google Drive file is shared with "Anyone with the link"
        2. Extract only the FILE_ID from your sharing link
        3. Or place your model files in the local directory:
           - `./RecipeGenerationGPT2.pt` for state dict
           - `./recipe_gpt2_model/` folder for full model
           - `./model_checkpoints/best_model.pt` for checkpoint
        """)
        st.stop()

# ============================================================================ 
# RECIPE GENERATION FUNCTION
# ============================================================================ 
def generate_recipe(
    model, 
    tokenizer, 
    device, 
    ingredients_list: List[str],
    title: str = "",
    max_length: int = 400,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95
) -> Dict[str, str]:
    """
    Generate a complete recipe from ingredients
    
    Args:
        model: Fine-tuned GPT-2 model
        tokenizer: GPT-2 tokenizer
        device: Computing device (cuda/cpu)
        ingredients_list: List of ingredient names
        title: Optional recipe title
        max_length: Maximum generation length
        temperature: Sampling temperature (higher = more creative)
        top_k: Top-k sampling parameter
        top_p: Nucleus sampling parameter
    
    Returns:
        Dictionary with recipe components
    """
    
    model.eval()
    
    # Format ingredients
    ingredients = ', '.join(ingredients_list)
    
    # Create prompt
    if title:
        prompt = f"Recipe: {title} | Ingredients: {ingredients} | Directions:"
    else:
        prompt = f"Recipe: | Ingredients: {ingredients} | Directions:"
    
    # Tokenize
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    # Generate
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=1,
            do_sample=True,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Parse the generated recipe
    try:
        parts = generated_text.split('|')
        
        recipe_title = parts[0].replace('Recipe:', '').strip() if len(parts) > 0 else "Delicious Recipe"
        recipe_ingredients = parts[1].replace('Ingredients:', '').strip() if len(parts) > 1 else ingredients
        recipe_directions = parts[2].replace('Directions:', '').strip() if len(parts) > 2 else "Directions not available"
        
        # If title was empty, try to extract from generation
        if not title and recipe_title:
            title = recipe_title
        elif not title:
            title = f"Recipe with {ingredients_list[0].title()}"
        
        return {
            'title': title if title else recipe_title,
            'ingredients': recipe_ingredients,
            'directions': recipe_directions,
            'full_text': generated_text
        }
    except:
        return {
            'title': title if title else "Generated Recipe",
            'ingredients': ingredients,
            'directions': generated_text,
            'full_text': generated_text
        }

# ============================================================================ 
# POPULAR INGREDIENTS & RECIPE EXAMPLES
# ============================================================================ 
POPULAR_INGREDIENTS = {
    "🍖 Proteins": ["chicken breast", "ground beef", "salmon", "tofu", "eggs", "shrimp", "turkey", "pork chops"],
    "🥬 Vegetables": ["bell peppers", "onion", "garlic", "tomatoes", "spinach", "broccoli", "carrots", "mushrooms"],
    "🌾 Grains": ["rice", "pasta", "quinoa", "bread", "tortillas", "couscous", "oats"],
    "🧀 Dairy": ["cheese", "milk", "butter", "cream", "yogurt", "sour cream", "parmesan"],
    "🌶️ Seasonings": ["salt", "pepper", "olive oil", "soy sauce", "garlic powder", "paprika", "cumin", "oregano"],
    "🍰 Baking": ["flour", "sugar", "baking powder", "vanilla extract", "chocolate chips", "honey"]
}

RECIPE_EXAMPLES = [
    {
        "name": "Italian Pasta",
        "ingredients": ["pasta", "tomatoes", "garlic", "basil", "olive oil", "parmesan cheese"]
    },
    {
        "name": "Asian Stir-Fry",
        "ingredients": ["chicken breast", "soy sauce", "bell peppers", "rice", "ginger", "garlic"]
    },
    {
        "name": "Breakfast Delight",
        "ingredients": ["eggs", "bacon", "cheese", "bread", "butter", "salt", "pepper"]
    },
    {
        "name": "Chocolate Dessert",
        "ingredients": ["flour", "sugar", "cocoa powder", "eggs", "butter", "chocolate chips"]
    }
]

# ============================================================================ 
# INITIALIZE SESSION STATE
# ============================================================================ 
if 'generated_recipes' not in st.session_state:
    st.session_state.generated_recipes = []

if 'ingredient_list' not in st.session_state:
    st.session_state.ingredient_list = []

if 'generation_count' not in st.session_state:
    st.session_state.generation_count = 0

# ============================================================================ 
# LOAD MODEL
# ============================================================================ 
with st.spinner("🔥 Heating up the AI kitchen..."):
    model, tokenizer, device, load_method = load_model_and_tokenizer()

# ============================================================================ 
# MAIN APP LAYOUT
# ============================================================================ 
# Header
st.markdown("""
<div class="main-header">
    <h1>🍳 AI Recipe Generator</h1>
    <p>Transform ingredients into delicious recipes with AI magic ✨</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 🎛️ Generation Settings")
    
    temperature = st.slider(
        "🌡️ Creativity Level",
        min_value=0.5,
        max_value=1.5,
        value=0.8,
        step=0.1,
        help="Higher values = more creative recipes"
    )
    
    max_length = st.slider(
        "📏 Recipe Length",
        min_value=200,
        max_value=600,
        value=400,
        step=50,
        help="Maximum number of tokens to generate"
    )
    
    st.markdown("---")
    
    # Model info
    st.markdown("### 📊 System Info")
    device_icon = "🚀" if device.type == "cuda" else "💻"
    st.info(f"{device_icon} Device: **{device.type.upper()}**")
    
    load_icon = "✅" if load_method != "base_gpt2" else "⚠️"
    st.info(f"{load_icon} Model: **{load_method.replace('_', ' ').title()}**")
    
    st.markdown("---")
    
    # Stats
    st.markdown("### 📈 Session Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <p class="stat-value">{st.session_state.generation_count}</p>
            <p class="stat-label">Recipes</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <p class="stat-value">{len(st.session_state.ingredient_list)}</p>
            <p class="stat-label">Ingredients</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Clear history
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.generated_recipes = []
        st.session_state.generation_count = 0
        st.rerun()

# Main content area
tab1, tab2, tab3 = st.tabs(["🍳 Generate Recipe", "📚 Recipe Examples", "💡 Tips & Tricks"])

# ============================================================================
# TAB 1: GENERATE RECIPE
# ============================================================================

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🥘 Create Your Recipe")
        
        # Recipe title (optional)
        recipe_title = st.text_input(
            "Recipe Name (Optional)",
            placeholder="e.g., Spicy Chicken Tacos",
            help="Leave empty for AI to suggest a name"
        )
        
        # Ingredient input
        ingredient_input = st.text_input(
            "Add an Ingredient",
            placeholder="e.g., chicken breast, tomatoes, garlic...",
            key="ingredient_input"
        )
        
        col_add, col_clear = st.columns([1, 1])
        
        with col_add:
            if st.button("➕ Add Ingredient", use_container_width=True):
                if ingredient_input and ingredient_input.strip():
                    st.session_state.ingredient_list.append(ingredient_input.strip())
                    st.rerun()
        
        with col_clear:
            if st.button("🔄 Clear All", use_container_width=True):
                st.session_state.ingredient_list = []
                st.rerun()
        
        # Display current ingredients
        if st.session_state.ingredient_list:
            st.markdown("#### Your Ingredients:")
            ingredients_html = "".join([
                f'<span class="ingredient-chip">{ing}</span>' 
                for ing in st.session_state.ingredient_list
            ])
            st.markdown(ingredients_html, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Generate button
            if st.button("🎨 Generate Recipe!", use_container_width=True, type="primary"):
                with st.spinner(""):
                    st.markdown('<div class="loading-chef">👨‍🍳</div>', unsafe_allow_html=True)
                    time.sleep(0.5)  # Dramatic effect
                    
                    # Generate recipe
                    recipe = generate_recipe(
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        ingredients_list=st.session_state.ingredient_list,
                        title=recipe_title,
                        max_length=max_length,
                        temperature=temperature
                    )
                    
                    # Save to history
                    st.session_state.generated_recipes.insert(0, recipe)
                    st.session_state.generation_count += 1
                    
                    st.markdown('<div class="success-message">✨ Recipe Generated Successfully! ✨</div>', unsafe_allow_html=True)
                    
                    # Display recipe
                    st.markdown(f"""
                    <div class="recipe-card">
                        <div class="recipe-title">{recipe['title']}</div>
                        
                        <div class="recipe-section">
                            <h3>🥘 Ingredients</h3>
                            <p>{recipe['ingredients']}</p>
                        </div>
                        
                        <div class="recipe-section">
                            <h3>👨‍🍳 Directions</h3>
                            <p>{recipe['directions']}</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Download button
                    recipe_text = f"# {recipe['title']}\n\n## Ingredients\n{recipe['ingredients']}\n\n## Directions\n{recipe['directions']}"
                    st.download_button(
                        label="📥 Download Recipe",
                        data=recipe_text,
                        file_name=f"{recipe['title'].replace(' ', '_')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
        else:
            st.info("👆 Add some ingredients to get started!")
    
    with col2:
        st.markdown("### 🌟 Popular Ingredients")
        
        for category, ingredients in POPULAR_INGREDIENTS.items():
            with st.expander(category):
                for ing in ingredients:
                    if st.button(f"+ {ing}", key=f"pop_{ing}", use_container_width=True):
                        if ing not in st.session_state.ingredient_list:
                            st.session_state.ingredient_list.append(ing)
                            st.rerun()
    
    # Recipe history
    if st.session_state.generated_recipes:
        st.markdown("---")
        st.markdown("### 📜 Recipe History")
        
        for idx, recipe in enumerate(st.session_state.generated_recipes[:5]):
            with st.expander(f"🍽️ {recipe['title']}", expanded=(idx==0)):
                st.markdown(f"""
                <div class="recipe-section">
                    <h3>🥘 Ingredients</h3>
                    <p>{recipe['ingredients']}</p>
                </div>
                
                <div class="recipe-section">
                    <h3>👨‍🍳 Directions</h3>
                    <p>{recipe['directions']}</p>
                </div>
                """, unsafe_allow_html=True)

# ============================================================================
# TAB 2: RECIPE EXAMPLES
# ============================================================================

with tab2:
    st.markdown("### 🎯 Try These Recipe Ideas")
    
    cols = st.columns(2)
    
    for idx, example in enumerate(RECIPE_EXAMPLES):
        with cols[idx % 2]:
            st.markdown(f"""
            <div class="popular-section">
                <h3>{example['name']}</h3>
                <p><strong>Ingredients:</strong> {', '.join(example['ingredients'])}</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button(f"🎨 Generate {example['name']}", key=f"ex_{idx}", use_container_width=True):
                st.session_state.ingredient_list = example['ingredients']
                st.rerun()

# ============================================================================
# TAB 3: TIPS & TRICKS
# ============================================================================

with tab3:
    st.markdown("""
    ### 💡 Tips for Best Results
    
    #### 🎯 Ingredient Selection
    - **Start with 3-6 ingredients** for best results
    - Include at least one **protein** (meat, tofu, eggs)
    - Add **aromatics** (garlic, onion) for flavor depth
    - Don't forget **seasonings** (salt, pepper, herbs)
    
    #### 🌡️ Creativity Settings
    - **Low (0.5-0.7)**: More traditional, classic recipes
    - **Medium (0.7-0.9)**: Balanced creativity and reliability
    - **High (0.9-1.5)**: Experimental, unique combinations
    
    #### 📏 Recipe Length
    - **Short (200-300)**: Quick summaries
    - **Medium (300-450)**: Detailed recipes
    - **Long (450-600)**: Comprehensive instructions
    
    #### 🎨 Creative Combinations
    Try mixing different cuisines:
    - Italian + Asian: Soy sauce + pasta
    - Mexican + Mediterranean: Feta + tacos
    - American + Indian: Curry + burger
    
    #### ⚠️ Important Notes
    - AI-generated recipes should be **reviewed for safety**
    - Adjust cooking times and temperatures as needed
    - Use your judgment for ingredient proportions
    - Always check for food allergies
    """)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; opacity: 0.7; padding: 2rem;">
    <p>🍳 AI Recipe Generator | Powered by Fine-tuned GPT-2</p>
</div>
""", unsafe_allow_html=True)
