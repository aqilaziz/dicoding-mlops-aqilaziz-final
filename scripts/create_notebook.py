"""Create the submission notebook as a JSON ipynb file."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "aqilaziz-pipeline.ipynb"


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": textwrap.dedent(source).strip().splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": textwrap.dedent(source).strip().splitlines(keepends=True),
    }


cells = [
    markdown(
        """
        # Proyek Pengembangan Machine Learning Pipeline

        Notebook ini membangun pipeline TFX untuk klasifikasi kualitas red wine.
        Target `good_quality` bernilai 1 jika skor kualitas asli minimal 6.
        """
    ),
    markdown(
        """
        ## 1. Persiapan

        Import library TFX, TensorFlow, dan TensorFlow Model Analysis yang dipakai
        untuk setiap komponen pipeline.
        """
    ),
    code(
        """
        from pathlib import Path

        import pandas as pd
        import tensorflow as tf
        import tensorflow_model_analysis as tfma
        import tfx

        from tfx.components import (
            CsvExampleGen,
            Evaluator,
            ExampleValidator,
            Pusher,
            SchemaGen,
            StatisticsGen,
            Trainer,
            Transform,
        )
        from tfx.dsl.components.common.resolver import Resolver
        from tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy import (
            LatestBlessedModelStrategy,
        )
        from tfx.orchestration.experimental.interactive.interactive_context import (
            InteractiveContext,
        )
        from tfx.proto import example_gen_pb2, pusher_pb2, trainer_pb2
        from tfx.types import Channel
        from tfx.types.standard_artifacts import Model, ModelBlessing

        print("TensorFlow:", tf.__version__)
        print("TFX:", tfx.__version__)
        """
    ),
    code(
        """
        PROJECT_ROOT = Path.cwd()
        DATA_ROOT = PROJECT_ROOT / "data"
        PIPELINE_ROOT = PROJECT_ROOT / "aqilaziz-pipeline"
        MODULE_ROOT = PROJECT_ROOT / "modules"
        SERVING_MODEL_DIR = PIPELINE_ROOT / "serving_model"

        TRANSFORM_MODULE = MODULE_ROOT / "wine_transform.py"
        TRAINER_MODULE = MODULE_ROOT / "wine_trainer.py"

        print("Data root:", DATA_ROOT)
        print("Pipeline root:", PIPELINE_ROOT)
        """
    ),
    markdown(
        """
        ## 2. Dataset

        Dataset Wine Quality Red berisi atribut kimia red wine. Label biner
        `good_quality` dibuat dari kolom kualitas asli.
        """
    ),
    code(
        """
        dataset = pd.read_csv(DATA_ROOT / "winequality-red.csv")
        display(dataset.head())
        display(dataset["good_quality"].value_counts().rename("count"))
        """
    ),
    markdown(
        """
        ## 3. InteractiveContext

        Semua komponen pipeline dijalankan dengan `InteractiveContext` dan
        artifact disimpan di folder `aqilaziz-pipeline`.
        """
    ),
    code(
        """
        context = InteractiveContext(pipeline_root=str(PIPELINE_ROOT))
        """
    ),
    markdown(
        """
        ## 4. ExampleGen, StatisticsGen, SchemaGen, dan ExampleValidator

        `CsvExampleGen` membaca data CSV dan membaginya menjadi train/eval.
        Komponen validasi data membuat statistik, schema, dan memeriksa anomali.
        """
    ),
    code(
        """
        output = example_gen_pb2.Output(
            split_config=example_gen_pb2.SplitConfig(
                splits=[
                    example_gen_pb2.SplitConfig.Split(name="train", hash_buckets=8),
                    example_gen_pb2.SplitConfig.Split(name="eval", hash_buckets=2),
                ]
            )
        )

        example_gen = CsvExampleGen(
            input_base=str(DATA_ROOT),
            output_config=output,
        )
        context.run(example_gen)

        statistics_gen = StatisticsGen(examples=example_gen.outputs["examples"])
        context.run(statistics_gen)

        schema_gen = SchemaGen(
            statistics=statistics_gen.outputs["statistics"],
            infer_feature_shape=True,
        )
        context.run(schema_gen)

        example_validator = ExampleValidator(
            statistics=statistics_gen.outputs["statistics"],
            schema=schema_gen.outputs["schema"],
        )
        context.run(example_validator)
        """
    ),
    markdown(
        """
        ## 5. Transform

        Komponen `Transform` memakai `modules/wine_transform.py` untuk
        menstandarkan fitur numerik dengan z-score.
        """
    ),
    code(
        """
        transform = Transform(
            examples=example_gen.outputs["examples"],
            schema=schema_gen.outputs["schema"],
            module_file=str(TRANSFORM_MODULE),
        )
        context.run(transform)
        """
    ),
    markdown(
        """
        ## 6. Trainer

        Komponen `Trainer` memakai `modules/wine_trainer.py` untuk membuat dan
        melatih model Keras.
        """
    ),
    code(
        """
        trainer = Trainer(
            module_file=str(TRAINER_MODULE),
            examples=transform.outputs["transformed_examples"],
            transform_graph=transform.outputs["transform_graph"],
            schema=schema_gen.outputs["schema"],
            train_args=trainer_pb2.TrainArgs(num_steps=120),
            eval_args=trainer_pb2.EvalArgs(num_steps=40),
        )
        context.run(trainer)
        """
    ),
    markdown(
        """
        ## 7. Resolver dan Evaluator

        `Resolver` mengambil baseline blessed model jika tersedia. `Evaluator`
        menghitung Binary Accuracy dan AUC, lalu menentukan apakah model layak
        di-push.
        """
    ),
    code(
        """
        model_resolver = Resolver(
            strategy_class=LatestBlessedModelStrategy,
            model=Channel(type=Model),
            model_blessing=Channel(type=ModelBlessing),
        ).with_id("latest_blessed_model_resolver")
        context.run(model_resolver)

        eval_config = tfma.EvalConfig(
            model_specs=[
                tfma.ModelSpec(
                    label_key="good_quality",
                    signature_name="serving_default",
                    prediction_key="outputs",
                )
            ],
            slicing_specs=[tfma.SlicingSpec()],
            metrics_specs=[
                tfma.MetricsSpec(
                    metrics=[
                        tfma.MetricConfig(
                            class_name="BinaryAccuracy",
                            threshold=tfma.MetricThreshold(
                                value_threshold=tfma.GenericValueThreshold(
                                    lower_bound={"value": 0.65}
                                )
                            ),
                        ),
                        tfma.MetricConfig(
                            class_name="AUC",
                            threshold=tfma.MetricThreshold(
                                value_threshold=tfma.GenericValueThreshold(
                                    lower_bound={"value": 0.70}
                                )
                            ),
                        ),
                    ]
                )
            ],
        )

        evaluator = Evaluator(
            examples=example_gen.outputs["examples"],
            model=trainer.outputs["model"],
            baseline_model=model_resolver.outputs["model"],
            eval_config=eval_config,
        )
        context.run(evaluator)
        """
    ),
    markdown(
        """
        ## 8. Pusher

        Model yang mendapatkan blessing dari Evaluator disimpan sebagai serving
        model di dalam folder pipeline.
        """
    ),
    code(
        """
        pusher = Pusher(
            model=trainer.outputs["model"],
            model_blessing=evaluator.outputs["blessing"],
            push_destination=pusher_pb2.PushDestination(
                filesystem=pusher_pb2.PushDestination.Filesystem(
                    base_directory=str(SERVING_MODEL_DIR)
                )
            ),
        )
        context.run(pusher)

        print("Pushed model artifact:")
        for artifact in pusher.outputs["pushed_model"].get():
            print(artifact.uri)
        """
    ),
    markdown(
        """
        ## 9. Ringkasan

        Seluruh komponen wajib telah dijalankan dengan `InteractiveContext` dan
        artifact pipeline tersimpan pada folder `aqilaziz-pipeline`.
        """
    ),
    code(
        """
        print("Pipeline artifact root:", PIPELINE_ROOT)
        print("Serving model root:", SERVING_MODEL_DIR)
        print("Top-level pipeline artifacts:")
        for path in sorted(PIPELINE_ROOT.iterdir()):
            print("-", path.name)
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "pygments_lexer": "ipython3",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
print(f"Created {NOTEBOOK_PATH}")
