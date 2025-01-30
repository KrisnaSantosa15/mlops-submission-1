#!/usr/bin/env python
# coding: utf-8

# # Obesity Prediction
# 
# Permasalahan yang ingin diselesaikan dalam proyek ini adalah memprediksi tingkat obesitas seseorang berdasarkan kebiasaan makan, aktivitas fisik, dan kondisi demografis. Tingkat obesitas adalah masalah kesehatan global yang dapat menyebabkan berbagai penyakit serius seperti diabetes, penyakit jantung, dan tekanan darah tinggi. Dengan memprediksi tingkat obesitas dengan tepat, kita dapat memberikan rekomendasi yang lebih personal untuk pencegahan dan penanganan obesitas lebih lanjut.

# Dataset yang digunakan dalam proyek ini adalah "[Obesity Prediction Dataset](https://www.kaggle.com/datasets/ruchikakumbhar/obesity-prediction/data)" yang diambil dari platform Kaggle. Dataset ini berisi data tentang kebiasaan makan dan kondisi fisik individu dari negara Meksiko, Peru, dan Kolombia. Dataset terdiri dari 17 atribut dan 2.111 record, dengan kolom target yaitu Obesity yang mengklasifikasikan tingkat obesitas individu ke dalam beberapa kategori:
# - Insufficient Weight
# - Normal Weight
# - Overweight Level I
# - Overweight Level II
# - Obesity Type I
# - Obesity Type II
# - Obesity Type III
# 
# Atribut dalam dataset meliputi:
# - Demografi: Gender, Age
# - Fisik: Height, Weight
# - Kebiasaan Makan: FAVC (makanan tinggi kalori), FCVC (konsumsi sayuran), NCP (jumlah makanan utama per hari), CAEC (makanan selingan), CALC (konsumsi alkohol)
# - Aktivitas Fisik: FAF (frekuensi aktivitas fisik), TUE (waktu penggunaan perangkat teknologi)
# - Transportasi: MTRANS (jenis transportasi yang digunakan)
# - Lainnya: family_history (riwayat keluarga obesitas), SMOKE (merokok), CH2O (konsumsi air), SCC (pemantauan kalori)
# 
# Dataset ini dapat diakses di: [Obesity Prediction Dataset](https://www.kaggle.com/datasets/ruchikakumbhar/obesity-prediction/data).
# 
# Berikut secara rinci variabel-variabel yang terdapat dalam dataset:
# - Gender: Gender (Male, Female)
# - Age: Age
# - Height : in metres
# - Weight : in kgs
# - family_history : Has a family member suffered or suffers from overweight? (yes, no)
# - FAVC : Do you eat high caloric food frequently? (yes, no)
# - FCVC : Do you usually eat vegetables in your meals? 
# - NCP : How many main meals do you have daily?
# - CAEC : Do you eat any food between meals? (Frequently, Sometimes, Always, No)
# - SMOKE : Do you smoke? (yes, no)
# - CH2O : How much water do you drink daily? in litres
# - SCC : Do you monitor the calories you eat daily? (yes, no)
# - FAF: How often do you have physical activity?
# - TUE : How much time do you use technological devices such as cell phone, videogames, television, computer and others?
# - CALC : How often do you drink alcohol? (Frequently, Sometimes, Always, No)
# - MTRANS : Which transportation do you usually use? (Automobile, Bike, Motorbike, Public_Transportation, Walking)
# - Obesity (Target Column) : Obesity level (Insufficient Weight, Normal Weight, Overweight Level I, Overweight Level II, Obesity Type I, Obesity Type II, Obesity Type III)

# Langkah pertama yang akan dilakukan adalah dengan import library yang dibutuhkan dan membaca dataset yang akan digunakan.

# In[1]:


import tensorflow as tf
from tensorflow.keras import layers
from tfx.components import CsvExampleGen, StatisticsGen, SchemaGen, ExampleValidator, Transform, Trainer, Tuner
from tfx.proto import example_gen_pb2, trainer_pb2, tuner_pb2
from tfx.orchestration.experimental.interactive.interactive_context import InteractiveContext
import tensorflow_data_validation as tfdv
import tensorflow_model_analysis as tfma
from tfx.dsl.components.common.resolver import Resolver
from tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy import LatestBlessedModelStrategy
from tfx.types import Channel
from tfx.types.standard_artifacts import Model, ModelBlessing
from tfx.components import Evaluator, Pusher
from tfx.proto import pusher_pb2
import os


# ## Set Variables
# 
# Selaanjutnya, kita akan menentukan variabel untuk membuat end-to-end pipeline menggunakan TFX dengan mendefinisikan beberapa konfigurasi seperti nama pipeline, lokasi dataset, lokasi metadata, dan lain-lain.

# In[2]:


PIPELINE_NAME = "krisna_santosa-pipeline"
PIPELINE_ROOT = os.path.join('pipelines', PIPELINE_NAME)
METADATA_PATH = os.path.join('metadata', PIPELINE_NAME, 'metadata.db')
SERVING_MODEL_DIR = os.path.join('serving_model', PIPELINE_NAME)
DATA_ROOT = "data"


# Lakukan inisialisasi instance `InteractiveContext` untuk mengatur dan menjalankan pipeline TFX secara interaktif yang menerima parameter berupa nama pipeline dan lokasi metadata.

# In[3]:


# Initialize InteractiveContext
interactive_context = InteractiveContext(
    pipeline_root=PIPELINE_ROOT
)


# ## Data Ingestion
# 
# Langkah pertama dalam pipeline adalah melakukan data ingestion. Dalam kasus ini, dataset yang digunakan adalah dataset obesitas yang telah dijelaskan sebelumnya. Dataset ini akan dibaca menggunakan komponen `CsvExampleGen` yang akan menghasilkan output berupa dataset yang telah di-preprocess. Kode di bawah ini akan membagi dataset menjadi dua bagian, yaitu 80% untuk training dan 20% untuk testing.

# In[4]:


output = example_gen_pb2.Output(
    split_config=example_gen_pb2.SplitConfig(splits=[
        example_gen_pb2.SplitConfig.Split(name="train", hash_buckets=8),
        example_gen_pb2.SplitConfig.Split(name="eval", hash_buckets=2)
    ])
)

example_gen = CsvExampleGen(
    input_base=DATA_ROOT,
    output_config=output
)


# Untuk melihat komponen `ExampleGen` secara interaktif, kita dapat menjalankan komponen tersebut menggunakan object InteractiveContext() yang telah kita definisikan sebelumnya.

# In[5]:


interactive_context.run(example_gen)


# ## Data Validation
# 
# Setelah data di-preprocess, langkah selanjutnya adalah melakukan data validation, ada tiga komponen yang digunakan dalam data validation, yaitu `StatisticsGen`, `SchemaGen`, dan `ExampleValidator`. Komponen `StatisticsGen` akan menghasilkan statistik deskriptif dari dataset, komponen `SchemaGen` akan menghasilkan skema dari dataset, dan komponen `ExampleValidator` akan memvalidasi data berdasarkan skema yang telah dihasilkan oleh komponen `SchemaGen`.

# ### Summary Statistics
# 
# Komponen ini akan berisi statistik deskriptif dari dataset, seperti jumlah data, rata-rata, standar deviasi, dan lain-lain. Kode di bawah ini akan menampilkan statistik deskriptif dari dataset. Komponen ini menerima input berupa dataset yang telah di-preprocess oleh komponen `ExampleGen`.

# In[6]:


statistics_gen = StatisticsGen(
    examples=example_gen.outputs["examples"]
)

interactive_context.run(statistics_gen)


# In[7]:


interactive_context.show(statistics_gen.outputs["statistics"])


# ### Data Schema
# 
# Komponen ini akan menghasilkan skema dari dataset, seperti tipe data, domain, dan lain-lain. Kode di bawah ini akan menampilkan skema dari dataset. Komponen ini menerima input berupa dataset yang telah di-preprocess oleh komponen `ExampleGen`.

# In[8]:


schema_gen = SchemaGen(
    statistics=statistics_gen.outputs["statistics"]
)

interactive_context.run(schema_gen)


# In[9]:


interactive_context.show(schema_gen.outputs["schema"])


# ### Anomalies Detection (Validator)
# 
# Pada komponen ini, kita akan melakukan validasi data berdasarkan skema yang telah dihasilkan oleh komponen `SchemaGen`. Komponen ini akan mendeteksi anomali pada dataset, seperti data yang hilang, data yang tidak sesuai dengan skema, dan lain-lain.

# In[10]:


example_validator = ExampleValidator(
    statistics=statistics_gen.outputs['statistics'],
    schema=schema_gen.outputs['schema']
)

interactive_context.run(example_validator)


# In[11]:


interactive_context.show(example_validator.outputs['anomalies'])


# Berdasarkan hasil tersebut tidak terdapat anomali yang ditemukan dalam dataset. Aritnya data siap masuk ke tahap selanjutnya, yaitu preprocessing.

# In[ ]:





# ## Data Preprocessing
# 
# Setelah tahap data validation, langkah selanjutnya adalah melakukan data preprocessing. Dalam kasus ini, kita akan melakukan data preprocessing dengan menggunakan komponen `Transform`. Komponen ini akan melakukan preprocessing data, seperti normalisasi, one-hot encoding, dan lain-lain. Untuk melakukan preprocessing data, kita perlu mendefinisikan file module yang berisi fungsi preprocessing data. 

# In[12]:


TRANSFORM_MODULE_FILE = "obesity_transform.py"


# In[13]:


get_ipython().run_cell_magic('writefile', '{TRANSFORM_MODULE_FILE}', 'import tensorflow as tf\nimport tensorflow_transform as tft\n\nNUMERIC_FEATURES = [\'Age\', \'Height\', \'Weight\', \'FCVC\', \'NCP\', \'CH2O\', \'FAF\', \'TUE\']\nCATEGORICAL_FEATURES = [\'Gender\', \'family_history\', \'FAVC\', \'CAEC\', \'SMOKE\', \'SCC\', \'CALC\', \'MTRANS\']\nLABEL_KEY = "Obesity"\n\ndef transformed_name(key):\n    return key + \'_xf\'\n\ndef preprocessing_fn(inputs):\n    """Preprocess input features into transformed features."""\n    outputs = {}\n    \n    # Scale numeric features\n    for feature_name in NUMERIC_FEATURES:\n        outputs[transformed_name(feature_name)] = tft.scale_to_z_score(\n            inputs[feature_name])\n    \n    # Convert categorical features to indices\n    for feature_name in CATEGORICAL_FEATURES:\n        outputs[transformed_name(feature_name)] = tft.compute_and_apply_vocabulary(\n            inputs[feature_name], vocab_filename=feature_name)\n    \n    # Convert label to index\n    outputs[transformed_name(LABEL_KEY)] = tft.compute_and_apply_vocabulary(\n        inputs[LABEL_KEY], vocab_filename=LABEL_KEY)\n    \n    return outputs\n')


# Setelah file module preprocessing data telah dibuat, kita dapat mendefinisikan komponen `Transform` dengan mendefinisikan fungsi preprocessing data yang telah dibuat sebelumnya. Komponen ini menerima input berupa dataset yang telah di-preprocess oleh komponen `ExampleGen` dan output berupa dataset yang telah di-preprocess.

# In[14]:


transform = Transform(
    examples=example_gen.outputs['examples'],
    schema=schema_gen.outputs['schema'],
    module_file=os.path.abspath(TRANSFORM_MODULE_FILE)
)

interactive_context.run(transform)


# Sampai tahap ini, kita telah melakukan data ingestion, data validation, dan data preprocessing. Langkah selanjutnya adalah melakukan training model menggunakan komponen `Trainer`.

# ## Model Development
# 
# Pada tahap ini, kita akan melakukan training model menggunakan komponen `Trainer`. Komponen ini akan melakukan training model menggunakan dataset yang telah di-preprocess oleh komponen `Transform`. Tetapi sebelum itu kita akan melakukan tuning hyperparameter menggunakan komponen `Tuner` terlebih dahulu. 

# In[15]:


TUNER_MODULE_FILE = "obesity_tuner.py"


# ### Tuner
# 
# Komponen ini akan melakukan tuning hyperparameter pada model yang akan digunakan. Kita perlu mendefinisikan file module yang berisi fungsi untuk membuat model, fungsi untuk meng-compile model, dan fungsi untuk melakukan tuning hyperparameter.

# In[16]:


get_ipython().run_cell_magic('writefile', '{TUNER_MODULE_FILE}', 'import tensorflow as tf\nimport tensorflow_transform as tft\nfrom kerastuner.engine import base_tuner \nfrom tensorflow.keras import layers\nfrom typing import NamedTuple, Dict, Text, Any \nfrom tfx.components.trainer.fn_args_utils import FnArgs\nimport kerastuner as kt\n\nNUMERIC_FEATURES = [\'Age\', \'Height\', \'Weight\', \'FCVC\', \'NCP\', \'CH2O\', \'FAF\', \'TUE\']\nCATEGORICAL_FEATURES = [\'Gender\', \'family_history\', \'FAVC\', \'CAEC\', \'SMOKE\', \'SCC\', \'CALC\', \'MTRANS\']\nLABEL_KEY = "Obesity"\n\ndef transformed_name(key):\n    return key + \'_xf\'\n\nTunerFnResult = NamedTuple(\'TunerFnResult\', [(\'tuner\', base_tuner.BaseTuner),\n                                            (\'fit_kwargs\', Dict[Text,Any])])\n\ndef _gzip_reader_fn(filenames):\n    return tf.data.TFRecordDataset(filenames, compression_type=\'GZIP\')\n\ndef _input_fn(file_pattern, tf_transform_output, batch_size=128):\n    transformed_feature_spec = (\n        tf_transform_output.transformed_feature_spec().copy()\n    )\n    \n    dataset = tf.data.experimental.make_batched_features_dataset(\n        file_pattern=file_pattern,\n        batch_size=batch_size,\n        features=transformed_feature_spec,\n        reader=_gzip_reader_fn,\n        num_epochs=1,\n        shuffle=True,\n        label_key=transformed_name(LABEL_KEY)\n    )\n    \n    return dataset\n\ndef model_builder(hp, tf_transform_output):\n    inputs = []\n    encoded_features = []\n    \n    # Numeric features\n    for feature_name in NUMERIC_FEATURES:\n        input_layer = layers.Input(shape=(1,), name=transformed_name(feature_name))\n        inputs.append(input_layer)\n        norm_layer = layers.BatchNormalization()(input_layer)\n        encoded_features.append(norm_layer)\n\n    # Categorical features\n    embedding_dim = hp.Int(\'embedding_dim\', 8, 32, step=8)\n    for feature_name in CATEGORICAL_FEATURES:\n        vocab_size = tf_transform_output.vocabulary_size_by_name(feature_name)\n        input_layer = layers.Input(shape=(1,), name=transformed_name(feature_name), dtype=tf.int64)\n        inputs.append(input_layer)\n        \n        # Add handling for out-of-vocabulary indices\n        safe_input = tf.where(\n            tf.logical_or(input_layer < 0, input_layer >= vocab_size),\n            tf.zeros_like(input_layer),\n            input_layer\n        )\n        \n        embedding = layers.Embedding(\n            vocab_size,\n            embedding_dim,\n            mask_zero=True,\n            name=f\'embedding_{feature_name}\'\n        )(safe_input)\n        \n        embedding_flat = layers.Flatten()(embedding)\n        encoded_features.append(embedding_flat)\n\n    concat_features = layers.concatenate(encoded_features)\n    \n    num_hidden_layers = hp.Int(\'num_hidden_layers\', 2, 4)\n    for i in range(num_hidden_layers):\n        units = hp.Int(f\'units_{i}\', 32, 256, step=32)\n        dropout_rate = hp.Float(f\'dropout_{i}\', 0.1, 0.5, step=0.1)\n        concat_features = layers.Dense(units, activation=\'relu\')(concat_features)\n        concat_features = layers.BatchNormalization()(concat_features)\n        concat_features = layers.Dropout(dropout_rate)(concat_features)\n\n    outputs = layers.Dense(7, activation=\'softmax\')(concat_features)\n    model = tf.keras.Model(inputs=inputs, outputs=outputs)\n    \n    learning_rate = hp.Float(\'learning_rate\', 1e-4, 1e-2, sampling=\'log\')\n    model.compile(\n        optimizer=tf.keras.optimizers.Adam(learning_rate),\n        loss=\'sparse_categorical_crossentropy\',\n        metrics=[\'accuracy\']\n    )\n    \n    return model\n\ndef tuner_fn(fn_args: FnArgs) -> TunerFnResult:\n    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)\n    \n    tuner = kt.Hyperband(\n        hypermodel=lambda hp: model_builder(hp, tf_transform_output),\n        objective=\'val_accuracy\',\n        max_epochs=30,\n        factor=3,\n        directory=fn_args.working_dir,\n        project_name=\'obesity_tuning\'\n    )\n    \n    train_dataset = _input_fn(fn_args.train_files, tf_transform_output)\n    eval_dataset = _input_fn(fn_args.eval_files, tf_transform_output)\n    \n    early_stopping = tf.keras.callbacks.EarlyStopping(\n        monitor=\'val_accuracy\', \n        patience=3,\n        restore_best_weights=True\n    )\n    \n    return TunerFnResult(\n        tuner=tuner,\n        fit_kwargs={\n            \'x\': train_dataset,\n            \'validation_data\': eval_dataset,\n            \'callbacks\': [early_stopping]\n        }\n    )\n')


# Setelah file module tuning hyperparameter telah dibuat, kita dapat mendefinisikan komponen `Tuner` dengan mendefinisikan fungsi tuning hyperparameter yang telah dibuat sebelumnya. Komponen ini menerima input berupa dataset yang telah di-preprocess oleh komponen `Transform`.

# In[17]:


tuner = Tuner(
    module_file=os.path.abspath(TUNER_MODULE_FILE),
    examples=transform.outputs['transformed_examples'],
    transform_graph=transform.outputs['transform_graph'],
    schema=schema_gen.outputs['schema'],
    train_args=trainer_pb2.TrainArgs(splits=['train']),
    eval_args=trainer_pb2.EvalArgs(splits=['eval'])
)

interactive_context.run(tuner)


# ### Model Training
# 
# Setelah tuning hyperparameter selesai, kita dapat melakukan training model menggunakan komponen `Trainer`. Komponen ini akan melakukan training model menggunakan dataset yang telah di-preprocess oleh komponen `Transform` dan hyperparameter yang telah di-tuning oleh komponen `Tuner`. Kita akan definisikan file module yang berisi fungsi untuk membuat model, fungsi untuk meng-compile model, dan fungsi untuk melakukan training model.

# In[18]:


TRAINER_MODULE_FILE = "obesity_trainer.py"


# In[19]:


get_ipython().run_cell_magic('writefile', '{TRAINER_MODULE_FILE}', 'import os\nimport tensorflow as tf\nimport tensorflow_transform as tft\nfrom tensorflow.keras import layers\nfrom tfx.components.trainer.fn_args_utils import FnArgs\n\nNUMERIC_FEATURES = [\'Age\', \'Height\', \'Weight\', \'FCVC\', \'NCP\', \'CH2O\', \'FAF\', \'TUE\']\nCATEGORICAL_FEATURES = [\'Gender\', \'family_history\', \'FAVC\', \'CAEC\', \'SMOKE\', \'SCC\', \'CALC\', \'MTRANS\']\nLABEL_KEY = "Obesity"\n\ndef transformed_name(key):\n    return key + \'_xf\'\n\ndef gzip_reader_fn(filenames):\n    return tf.data.TFRecordDataset(filenames, compression_type=\'GZIP\')\n\ndef input_fn(file_pattern, tf_transform_output, batch_size=128)->tf.data.Dataset:\n    """Creates an input function that loads and preprocesses the dataset."""\n    transform_feature_spec = (\n        tf_transform_output.transformed_feature_spec().copy())\n    \n    dataset = tf.data.experimental.make_batched_features_dataset(\n        file_pattern=file_pattern,\n        batch_size=batch_size,\n        features=transform_feature_spec,\n        reader=gzip_reader_fn,\n        num_epochs=None,\n        shuffle=True,\n        shuffle_buffer_size=1000,\n        label_key=transformed_name(LABEL_KEY))\n    \n    return dataset\n\ndef _get_serve_tf_examples_fn(model, tf_transform_output):\n\n    model.tft_layer = tf_transform_output.transform_features_layer()\n\n    @tf.function\n    def serve_tf_examples_fn(serialized_tf_examples):\n\n        feature_spec = tf_transform_output.raw_feature_spec()\n\n        feature_spec.pop(LABEL_KEY)\n\n        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)\n\n        transformed_features = model.tft_layer(parsed_features)\n\n        # get predictions using the transformed features\n        return model(transformed_features)\n\n    return serve_tf_examples_fn\n\ndef run_fn(fn_args: FnArgs):\n    """Train the model based on given args."""\n    \n    log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), \'logs\')\n\n    tensorboard_callback = tf.keras.callbacks.TensorBoard(\n        log_dir = log_dir, update_freq=\'batch\'\n    )\n    \n    \n    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)\n    \n    # Get dataset and compute steps per epoch\n    batch_size = 128\n    train_dataset = input_fn(fn_args.train_files, tf_transform_output, batch_size)\n    eval_dataset = input_fn(fn_args.eval_files, tf_transform_output, batch_size)\n    \n    # Load best hyperparameters\n    hp = fn_args.hyperparameters[\'values\']\n    \n    # Build model with best hyperparameters\n    model = build_model(hp, tf_transform_output)\n    \n    steps_per_epoch = 26\n    validation_steps = 7\n    \n    # Train model\n    model.fit(\n        train_dataset,\n        validation_data=eval_dataset,\n        epochs=30,\n        steps_per_epoch=steps_per_epoch,\n        validation_steps=validation_steps,\n        callbacks=[tf.keras.callbacks.EarlyStopping(\n            monitor=\'val_accuracy\',\n            patience=5,\n            restore_best_weights=True\n        )]\n    )\n    \n    # Save model\n    signatures = {\n        \'serving_default\':\n        _get_serve_tf_examples_fn(model, tf_transform_output).get_concrete_function(\n                                    tf.TensorSpec(\n                                    shape=[None],\n                                    dtype=tf.string,\n                                    name=\'examples\'))\n    }\n    model.save(fn_args.serving_model_dir, save_format=\'tf\', signatures=signatures)\n\ndef build_model(hp, tf_transform_output):\n    inputs = []\n    encoded_features = []\n    \n    # Numeric features with BatchNormalization\n    for feature_name in NUMERIC_FEATURES:\n        input_layer = layers.Input(shape=(1,), name=transformed_name(feature_name))\n        inputs.append(input_layer)\n        norm_layer = layers.BatchNormalization()(input_layer)\n        encoded_features.append(norm_layer)\n\n    # Categorical features with embeddings\n    for feature_name in CATEGORICAL_FEATURES:\n        vocab_size = tf_transform_output.vocabulary_size_by_name(feature_name)\n        input_layer = layers.Input(shape=(1,), name=transformed_name(feature_name), dtype=tf.int64)\n        inputs.append(input_layer)\n        \n        safe_input = tf.where(\n            tf.logical_or(input_layer < 0, input_layer >= vocab_size),\n            tf.zeros_like(input_layer),\n            input_layer\n        )\n        \n        embedding = layers.Embedding(\n            vocab_size,\n            hp.get(\'embedding_dim\'),\n            mask_zero=True\n        )(safe_input)\n        \n        embedding_flat = layers.Flatten()(embedding)\n        encoded_features.append(embedding_flat)\n\n    concat_features = layers.concatenate(encoded_features)\n    \n    # Dynamic hidden layers from tuner\n    for i in range(hp.get(\'num_hidden_layers\')):\n        units = hp.get(f\'units_{i}\')\n        dropout_rate = hp.get(f\'dropout_{i}\')\n        concat_features = layers.Dense(units, activation=\'relu\')(concat_features)  # Added activation\n        concat_features = layers.BatchNormalization()(concat_features)\n        concat_features = layers.Dropout(dropout_rate)(concat_features)\n\n    outputs = layers.Dense(7, activation=\'softmax\')(concat_features)\n    model = tf.keras.Model(inputs=inputs, outputs=outputs)\n    \n    model.compile(\n        optimizer=tf.keras.optimizers.Adam(hp.get(\'learning_rate\')),\n        loss=\'sparse_categorical_crossentropy\',\n        metrics=[\'accuracy\']\n    )\n    \n    \n    \n    return model\n')


# In[20]:


trainer = Trainer(
    module_file=os.path.abspath(TRAINER_MODULE_FILE),
    examples=transform.outputs['transformed_examples'],
    transform_graph=transform.outputs['transform_graph'],
    schema=schema_gen.outputs['schema'],
    hyperparameters=tuner.outputs['best_hyperparameters'],
    train_args=trainer_pb2.TrainArgs(splits=['train']),
    eval_args=trainer_pb2.EvalArgs(splits=['eval'])
)

interactive_context.run(trainer)


# ## Model Analysis and Validation
# 
# Setelah training model selesai, langkah selanjutnya adalah melakukan analisis model dan validasi model. Dalam kasus ini, kita akan menggunakan komponen `Resolver` dan `Evaluator`. Resolver berperan untuk menentukan baseline model yang akan digunakan untuk membandingkan model yang telah di-training. Sedangkan Evaluator berperan untuk mengevaluasi model yang telah di-training.

# ### Resolver Component
# 
# Pada komponen ini, kita akan menentukan baseline model yang akan digunakan untuk membandingkan model yang telah di-training.

# In[21]:


model_resolver = Resolver(
    strategy_class=LatestBlessedModelStrategy,
    model=Channel(type=Model),
    model_blessing=Channel(type=ModelBlessing)
).with_id('latest_blessed_model_resolver')

interactive_context.run(model_resolver)


# ### Evaluator Component
# 
# Pada komponen ini, kita akan mengevaluasi model yang telah di-training. Komponen ini akan menghasilkan beberapa metric evaluasi model, seperti accuracy, precision, recall, dan lain-lain. Kode di bawah ini akan menampilkan metric evaluasi model dengan threshold 0.85.

# In[22]:


LABEL_KEY = "Obesity"

def transformed_name(key):
    return key + '_xf'

eval_config = tfma.EvalConfig(
    model_specs=[tfma.ModelSpec(
            label_key=transformed_name(LABEL_KEY)
        )],
    slicing_specs=[tfma.SlicingSpec()],
    metrics_specs=[
        tfma.MetricsSpec(metrics=[
            tfma.MetricConfig(class_name='ExampleCount'),
            tfma.MetricConfig(class_name='SparseCategoricalAccuracy'),
            tfma.MetricConfig(
                class_name='SparseCategoricalAccuracy',
                threshold=tfma.MetricThreshold(
                    value_threshold=tfma.GenericValueThreshold(
                        lower_bound={'value': 0.85}
                    ),
                    change_threshold=tfma.GenericChangeThreshold(
                        direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                        absolute={'value': 0.0001}
                    )
                )
            )
        ])
    ]
)


# Setelah membuat konfigurasi untuk komponen `Evaluator`, kita dapat mengevaluasi model yang telah di-training dengan menjalankan komponen `Evaluator` pada kode di bawah ini.

# In[23]:


evaluator = Evaluator(
    examples=transform.outputs['transformed_examples'],
    model=trainer.outputs['model'],
    baseline_model=model_resolver.outputs['model'],
    eval_config=eval_config
)

interactive_context.run(evaluator)


# Untuk dapat melihat hasil evaluasi model dengan visualisasi, kita menggunakan  `tfma.view.render_slicing_metrics` yang akan menampilkan metric evaluasi model dengan visualisasi.

# In[24]:


# Visualize evaluation results
eval_result = evaluator.outputs['evaluation'].get()[0].uri
tfma_result = tfma.load_eval_result(eval_result)
tfma.view.render_slicing_metrics(tfma_result)


# ## Pusher
# 
# Setelah model dievaluasi, langkah terakhir adalah melakukan push model ke production. Pada kasus ini, kita akan menggunakan komponen `Pusher` untuk melakukan push model ke production. Komponen ini akan melakukan menyimpan model yang telah di-training ke storage yang telah ditentukan.

# In[25]:


pusher = Pusher(
    model=trainer.outputs['model'],
    model_blessing=evaluator.outputs['blessing'],
    push_destination=pusher_pb2.PushDestination(
        filesystem=pusher_pb2.PushDestination.Filesystem(
            base_directory=SERVING_MODEL_DIR)
    )
)

interactive_context.run(pusher)

