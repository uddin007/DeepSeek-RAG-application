# Databricks notebook source
# MAGIC %pip install databricks-sdk==0.12.0 databricks-genai-inference==0.1.1 mlflow==2.9.0 textstat==0.7.3 tiktoken==0.5.1 evaluate==0.4.1 langchain==0.0.344 databricks-vectorsearch==0.22 transformers==4.30.2 torch==2.0.1 cloudpickle==2.2.1 pydantic==2.5.2
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load evaluation model

# COMMAND ----------

# dbutils.widgets.text("evaluation_model","")
# dbutils.widgets.text("catalog_name","")
# dbutils.widgets.text("db_name","")
# dbutils.widgets.text("index_string","")
# dbutils.widgets.text("secret_name","")
# dbutils.widgets.text("secret_key","")
# dbutils.widgets.text("vector_search_endpoint_name","")
# dbutils.widgets.text("langchain_model_name","")

# COMMAND ----------

evaluation_model_name = dbutils.widgets.get("evaluation_model")
catalog = dbutils.widgets.get("catalog_name")
db = dbutils.widgets.get("db_name")
index_string = dbutils.widgets.get("index_string")
secret_scope = dbutils.widgets.get("secret_name")
secret_key = dbutils.widgets.get("secret_key")
langchain_model_name = dbutils.widgets.get("langchain_model_name")

VECTOR_SEARCH_ENDPOINT_NAME = dbutils.widgets.get("vector_search_endpoint_name")

index_name=f"{catalog}.{db}.{index_string}"
host = "https://" + spark.conf.get("spark.databricks.workspaceUrl")

# COMMAND ----------

from mlflow.deployments import get_deploy_client

deploy_client = get_deploy_client("databricks")
endpoint_name = evaluation_model_name

# test endpoint 
answer_test = deploy_client.predict(endpoint=endpoint_name, inputs={"messages": [{"role": "user", "content": "What is Apache Spark?"}]})
answer_test['choices'][0]['message']['content']

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load Evaluation Dataset 

# COMMAND ----------

df_qa = spark.table(f'{catalog}.{db}.evaluation_dataset_test')
df_qa.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Functions

# COMMAND ----------

def get_latest_model_version(model_name):
    mlflow_client = MlflowClient()
    latest_version = 1
    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
        version_int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int
    return latest_version

# COMMAND ----------

import mlflow
import os
import pandas as pd
from pyspark.sql.functions import pandas_udf, col
from mlflow.tracking import MlflowClient
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains.llm import LLMChain

os.environ['DATABRICKS_TOKEN'] = dbutils.secrets.get("neo4j-secrets", "rag-sp-token")
model_name = f"{catalog}.{db}.{langchain_model_name}"

model_version_to_evaluate = get_latest_model_version(model_name)
mlflow.set_registry_uri("databricks-uc")
rag_model = mlflow.langchain.load_model(f"models:/{model_name}/{model_version_to_evaluate}")

print(rag_model)

# COMMAND ----------

# MAGIC %md
# MAGIC ### UDF for model prediction

# COMMAND ----------

import re

@pandas_udf("string")
def predict_answer(questions):
    def answer_question(question):
        dialog = {"messages": [{"role": "user", "content": question}]}
        # return rag_model.invoke(dialog)['result']
        return re.sub(r"<think>.*?</think>", "", rag_model.invoke(dialog)['result'], flags=re.DOTALL).strip()
    return questions.apply(answer_question)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model predictions

# COMMAND ----------

df_qa_with_preds = df_qa.withColumn('preds', predict_answer(col('inputs'))).cache()
display(df_qa_with_preds)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### LLMs-as-a-judge
# MAGIC

# COMMAND ----------

# DBTITLE 1,Custom correctness answer
from mlflow.metrics.genai.metric_definitions import answer_correctness
from mlflow.metrics.genai import make_genai_metric, EvaluationExample

answer_correctness_metrics = answer_correctness(model=f"endpoints:/{endpoint_name}")
print(endpoint_name)
print(answer_correctness_metrics)

# COMMAND ----------

# DBTITLE 1,Adding custom professionalism metric
professionalism_example = EvaluationExample(
    input="What is MLflow?",
    output=(
        "MLflow is like your friendly neighborhood toolkit for managing your machine learning projects. It helps "
        "you track experiments, package your code and models, and collaborate with your team, making the whole ML "
        "workflow smoother. It's like your Swiss Army knife for machine learning!"
    ),
    score=2,
    justification=(
        "The response is written in a casual tone. It uses contractions, filler words such as 'like', and "
        "exclamation points, which make it sound less professional. "
    )
)

professionalism = make_genai_metric(
    name="professionalism",
    definition=(
        "Professionalism refers to the use of a formal, respectful, and appropriate style of communication that is "
        "tailored to the context and audience. It often involves avoiding overly casual language, slang, or "
        "colloquialisms, and instead using clear, concise, and respectful language."
    ),
    grading_prompt=(
        "Professionalism: If the answer is written using a professional tone, below are the details for different scores: "
        "- Score 1: Language is extremely casual, informal, and may include slang or colloquialisms. Not suitable for "
        "professional contexts."
        "- Score 2: Language is casual but generally respectful and avoids strong informality or slang. Acceptable in "
        "some informal professional settings."
        "- Score 3: Language is overall formal but still have casual words/phrases. Borderline for professional contexts."
        "- Score 4: Language is balanced and avoids extreme informality or formality. Suitable for most professional contexts. "
        "- Score 5: Language is noticeably formal, respectful, and avoids casual elements. Appropriate for formal "
        "business or academic settings. "
    ),
    model=f"endpoints:/{endpoint_name}",
    parameters={"temperature": 0.0},
    aggregations=["mean", "variance"],
    examples=[professionalism_example],
    greater_is_better=True
)

print(professionalism)

# COMMAND ----------

# DBTITLE 1,Start the evaluation run
from mlflow.deployments import set_deployments_target

set_deployments_target("databricks")

#This will automatically log all
with mlflow.start_run(run_name=f"{langchain_model_name}-run") as run:
    eval_results = mlflow.evaluate(data = df_qa_with_preds.toPandas(), # evaluation data,
                                   model_type="question-answering", # toxicity and token_count will be evaluated   
                                   predictions="preds", # prediction column_name from eval_df
                                   targets = "targets",
                                   extra_metrics=[answer_correctness_metrics, professionalism])
    
eval_results.metrics

# COMMAND ----------

df_genai_metrics = eval_results.tables["eval_results_table"]
display(df_genai_metrics)

# COMMAND ----------

df_genai_metrics['gen-ai-model'] = langchain_model_name
df_genai_metrics['eval-ai-model'] = 'gpt-35-turbo'
df_genai_metrics.head()

# COMMAND ----------

df_genai_metrics_1 = spark.createDataFrame(df_genai_metrics)
df_genai_metrics_1.display()

# COMMAND ----------

df_genai_metrics_2 = (df_genai_metrics_1 
      .withColumnRenamed("toxicity/v1/score", "toxicity_score")
      .withColumnRenamed("flesch_kincaid_grade_level/v1/score", "flesch_kincaid_grade_level_score")
      .withColumnRenamed("ari_grade_level/v1/score", "ari_grade_level_score")
      .withColumnRenamed("answer_correctness/v1/score", "answer_correctness_score")
      .withColumnRenamed("answer_correctness/v1/justification", "answer_correctness_justification")
      .withColumnRenamed("professionalism/v1/score", "professionalism_score")
      .withColumnRenamed("professionalism/v1/justification", "professionalism_justification"))
df_genai_metrics_2.printSchema()

# COMMAND ----------

df_genai_metrics_2.write.mode("append").saveAsTable(f"{catalog}.{db}.genai_model_metrics_deepseek")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- DROP TABLE IF EXISTS genai_model_metrics_deepseek;
