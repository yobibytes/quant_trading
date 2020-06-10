from kedro.pipeline import node, Pipeline
from kedro_tutorial.pipelines.data_engineering.nodes import (
    preprocess_companies,
    preprocess_shuttles,
    create_master_table,
)

def create_pipeline(**kwargs):
    pipeline = Pipeline(
        [
            node(
                name="preprocessing_companies",
                func=preprocess_companies,
                inputs="companies",
                outputs="preprocessed_companies",                
            ),
            node(
                name="preprocessing_shuttles",
                func=preprocess_shuttles,
                inputs="shuttles",
                outputs="preprocessed_shuttles",                
            ),
            node(
                func=create_master_table,
                inputs=["preprocessed_shuttles", "preprocessed_companies", "reviews"],
                outputs="master_table",
                name="master_table",
            ),
        ]
    )
    # print(pipeline.describe())
    # print(pipeline.inputs())
    # print(pipeline.outputs())
    return pipeline
    