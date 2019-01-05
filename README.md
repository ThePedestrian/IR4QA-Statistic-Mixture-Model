# Information Retrieval for Question Answering Systems using a Statistical Mixture Model

The project investigates using a [**Mixture Model**](https://en.wikipedia.org/wiki/Mixture_model "Mixture Model") to build an IR4QA system that recommends an answer to a natural language question from precompiled list of question-answer pairs.

Our dataset for training is *Yahoo! Answers Comprehensive Questions and Answers version 1.03* which consists of more than 4 million question-answer pairs provided by Yahoo Labs on the [WebScope site](https://webscope.sandbox.yahoo.com/catalog.php?datatype=l&did=11 "WebScope site").

We describe the system, results, and some of the key takeaway lessons.


## Files
- **paper.pdf** - Results and resources to download the dataset.
- **ir4qa.ipynb** - Jupyter notebook for exploration.
- **server.py** - Flask server.
- **gui/** - HTML/CSS/JS that invokes RESTful APIs exposed by `server.py`


## Demo
![](./demo.png)

**Sample questions to ask:**
- *How do I get rid of stomach ache?*
- *What's the meaning of life?*
- *How to lose weight?*
- *I have long thick hair. Is that pretty?*
- *One side of my body freezes. What should I do?*
