from kedro.pipeline import node, Pipeline

from kedro_tutorial.pipelines.data_science.nodes import (
    evaluate_model,
    split_data,
    train_model,
)

def create_pipeline(**kwargs):
    pipeline = Pipeline(
        [
            node(
                func=split_data,
                inputs=["master_table", "parameters"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
            ),
            node(func=train_model, inputs=["X_train", "y_train"], outputs="regressor"),
            node(
                func=evaluate_model,
                inputs=["regressor", "X_test", "y_test"],
                outputs=None,
            ),
        ]
    )
    print(pipeline.describe())
    print(pipeline.inputs())
    print(pipeline.outputs())
    return pipeline
    