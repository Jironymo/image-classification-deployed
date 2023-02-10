import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
from PIL import Image
import io

MODEL_URL = f'http://127.0.0.1:5017/invocations'
CLASSES_DICT = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot',
}

st.title('What clothing is on picture?')

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    image = Image.open(io.BytesIO(bytes_data))
    image_data = pd.DataFrame(np.asarray(image).reshape([1, 784])).to_dict(orient='split')
    http_data = json.dumps({"dataframe_split": image_data})
    response = requests.post(url=MODEL_URL, headers={'Content-Type': 'application/json'}, data=http_data)
    r = response.json()['predictions'][0]
    result = CLASSES_DICT[int(max(r, key=r.get))]
    st.write(f'Your image containt {result}')






