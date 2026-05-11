import pandas as pd
from typing import Literal, Dict, List

# Path to the data
file_path = "../data/data.xlsx"

# Each Model / Language pair
llm_types = Literal["GPT_EN", "GPT_FR", "Llama_EN",
                    "Llama_FR", "Biomistral_EN", "Biomistral_FR",
                    "DeepSeek_EN", "DeepSeek_FR", "o3_EN", "o3_FR"]

# Variable used to exclude some vignettes from the analyses.
exclude: Dict[llm_types, List[str]] = {
    'GPT_FR': [],
    'GPT_EN': [],
    'Llama_FR': [],
    'Llama_EN': [],
    'Biomistral_FR': [],
    'Biomistral_EN': [],
    'DeepSeek_EN': [],
    'DeepSeek_FR': [],
    'o3_EN': [],
    'o3_FR': []
}

# Color codes for each LLM :
couleurs: Dict[llm_types, str] = {    
    'o3_FR': '#7B4F9E',
    'o3_EN': '#B89FCC',
    'DeepSeek_FR': '#C43B3B',
    'DeepSeek_EN': '#E89090',
    'GPT_FR': '#86A44A',
    'GPT_EN': '#B5CA92',
    'Llama_FR': '#416FA6',
    'Llama_EN': '#8EA5CB',
    'Biomistral_FR': '#DA8137',
    'Biomistral_EN': '#F6B18A'
}

# Function to get all data in Pd.DataFrame format for a given LLM in sheet_name
# ex : get_data('Biomistral_EN') get all scores for Biomistral in English
def get_data(sheet_name: llm_types):
    data = pd.read_excel(file_path, sheet_name)
    return data[~data.ID.isin(exclude[sheet_name])]
