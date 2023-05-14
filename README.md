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
```

- Now goto pinecone official website: https://www.pinecone.io/
- Login 
- Create index
- Input the index name
- Dimensions number from the embeddings output
- And then > click Create index.
- Goto API KEYS, copy it
- Also copy the environment
And We're ready to go.



