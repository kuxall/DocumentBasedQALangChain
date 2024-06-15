# DocumentBasedQALangChain

Let's upload our required data inside the `data` folder.

- Install langchain openai pinecone pillow
```
pip install langchain openai pinecone-client pillow
```
- Use your OPENAI_API_KEY

- Install required dependencies
``` 
!pip install --upgrade langchain openai -q
!pip install unstructured -q
!pip install unstructured[local-inference] -q
!pip install detectron2@git+https://github.com/facebookresearch/detectron2.git@v0.6#egg=detectron2 -q
!apt-get install poppler-utils
!pip install streamlit -q
```

Use FAISS for similarity search



# Run the App
Run the app using streamlit 
```
streamlit run app.py

```

