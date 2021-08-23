import argparse
import glob
import importlib
import json
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def earth_movers_distance(y_true, y_pred):
    cdf_true = K.cumsum(y_true, axis=-1)
    cdf_pred = K.cumsum(y_pred, axis=-1)
    emd = K.sqrt(K.mean(K.square(cdf_true - cdf_pred), axis=-1))
    return K.mean(emd)


class TrainDataGenerator(tf.keras.utils.Sequence):
    """inherits from Keras Sequence base object, allows to use multiprocessing in .fit_generator"""

    def __init__(self, samples, img_dir, batch_size, n_classes, basenet_preprocess, img_format,
                 img_load_dims=(256, 256), img_crop_dims=(224, 224), shuffle=True):
        self.samples = samples
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.basenet_preprocess = basenet_preprocess  # Keras basenet specific preprocessing function
        self.img_load_dims = img_load_dims  # dimensions that images get resized into when loaded
        self.img_crop_dims = img_crop_dims  # dimensions that images get randomly cropped to
        self.shuffle = shuffle
        self.img_format = img_format
        self.on_epoch_end()  # call ensures that samples are shuffled in first epoch if shuffle is set to True

    def __len__(self):
        return int(np.ceil(len(self.samples) / self.batch_size))  # number of batches per epoch

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]  # get batch indexes
        batch_samples = [self.samples[i] for i in batch_indexes]  # get batch samples
        X, y = self.__data_generator(batch_samples)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.samples))
        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def __data_generator(self, batch_samples):
        # initialize images and labels tensors for faster processing
        X = np.empty((len(batch_samples), *self.img_crop_dims, 3))
        y = np.empty((len(batch_samples), self.n_classes))

        for i, sample in enumerate(batch_samples):
            # load and randomly augment image
            img_file = os.path.join(self.img_dir, '{}.{}'.format(sample['image_id'], self.img_format))
            img = utils.load_image(img_file, self.img_load_dims)
            if img is not None:
                img = utils.random_crop(img, self.img_crop_dims)
                img = utils.random_horizontal_flip(img)
                X[i,] = img

            # normalize labels
            y[i,] = utils.normalize_labels(sample['label'])

        # apply basenet specific preprocessing
        # input is 4D numpy array of RGB values within [0, 255]
        X = self.basenet_preprocess(X)

        return X, y


class TestDataGenerator(tf.keras.utils.Sequence):
    '''inherits from Keras Sequence base object, allows to use multiprocessing in .fit_generator'''

    def __init__(self, samples, img_dir, batch_size, n_classes, basenet_preprocess, img_format,
                 img_load_dims=(224, 224)):
        self.samples = samples
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.basenet_preprocess = basenet_preprocess  # Keras basenet specific preprocessing function
        self.img_load_dims = img_load_dims  # dimensions that images get resized into when loaded
        self.img_format = img_format
        self.on_epoch_end()  # call ensures that samples are shuffled in first epoch if shuffle is set to True

    def __len__(self):
        return int(np.ceil(len(self.samples) / self.batch_size))  # number of batches per epoch

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]  # get batch indexes
        batch_samples = [self.samples[i] for i in batch_indexes]  # get batch samples
        X, y = self.__data_generator(batch_samples)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.samples))

    def __data_generator(self, batch_samples):
        # initialize images and labels tensors for faster processing
        X = np.empty((len(batch_samples), *self.img_load_dims, 3))
        y = np.empty((len(batch_samples), self.n_classes))

        for i, sample in enumerate(batch_samples):
            # load and randomly augment image
            img_file = os.path.join(self.img_dir, '{}.{}'.format(sample['image_id'], self.img_format))
            img = load_image(img_file, self.img_load_dims)
            if img is not None:
                X[i,] = img

            # normalize labels
            if sample.get('label') is not None:
                y[i,] = normalize_labels(sample['label'])

        # apply basenet specific preprocessing
        # input is 4D numpy array of RGB values within [0, 255]
        X = self.basenet_preprocess(X)

        return X, y


class Nima:
    def __init__(self, base_model_name, n_classes=10, learning_rate=0.001, dropout_rate=0, loss=earth_movers_distance,
                 decay=0, weights='imagenet'):
        self.n_classes = n_classes
        self.base_model_name = base_model_name
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.loss = loss
        self.decay = decay
        self.weights = weights
        self._get_base_module()

    def _get_base_module(self):
        # import Keras base model module
        if self.base_model_name == 'InceptionV3':
            self.base_module = importlib.import_module('tensorflow.keras.applications.inception_v3')
        elif self.base_model_name == 'InceptionResNetV2':
            self.base_module = importlib.import_module('tensorflow.keras.applications.inception_resnet_v2')
        else:
            self.base_module = importlib.import_module('tensorflow.keras.applications.' + self.base_model_name.lower())

    def build(self):
        # get base model class
        BaseCnn = getattr(self.base_module, self.base_model_name)

        # load pre-trained model
        self.base_model = BaseCnn(input_shape=(224, 224, 3), weights=self.weights, include_top=False, pooling='avg')

        # add dropout and dense layer
        x = Dropout(self.dropout_rate)(self.base_model.output)
        x = Dense(units=self.n_classes, activation='softmax')(x)

        self.nima_model = Model(self.base_model.inputs, x)

    def compile(self):
        self.nima_model.compile(optimizer=Adam(lr=self.learning_rate, decay=self.decay), loss=self.loss)

    def preprocessing_function(self):
        return self.base_module.preprocess_input


def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


def save_json(data, target_file):
    with open(target_file, 'w') as f:
        json.dump(data, f, indent=2, sort_keys=True)


def random_crop(img, crop_dims):
    h, w = img.shape[0], img.shape[1]
    ch, cw = crop_dims[0], crop_dims[1]
    assert h >= ch, 'image height is less than crop height'
    assert w >= cw, 'image width is less than crop width'
    x = np.random.randint(0, w - cw + 1)
    y = np.random.randint(0, h - ch + 1)
    return img[y:(y + ch), x:(x + cw), :]


def random_horizontal_flip(img):
    assert len(img.shape) == 3, 'input tensor must have 3 dimensions (height, width, channels)'
    assert img.shape[2] == 3, 'image not in channels last format'
    if np.random.random() < 0.5:
        img = img.swapaxes(1, 0)
        img = img[::-1, ...]
        img = img.swapaxes(0, 1)
    return img


def load_image(img_file, target_size):
    return np.asarray(tf.keras.preprocessing.image.load_img(img_file, target_size=target_size))


def normalize_labels(labels):
    labels_np = np.array(labels)
    return labels_np / labels_np.sum()


def calc_mean_score(score_dist):
    score_dist = normalize_labels(score_dist)
    return (score_dist * np.arange(1, 11)).sum()


def ensure_dir_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def image_file_to_json(img_path):
    img_dir = os.path.dirname(img_path)
    img_id = os.path.basename(img_path).split('.')[0]
    return img_dir, [{'image_id': img_id}]


def image_dir_to_json(img_dir, img_type='jpg'):
    img_paths = glob.glob(os.path.join(img_dir, '*.' + img_type))
    samples = []
    for img_path in img_paths:
        img_id = os.path.basename(img_path).split('.')[0]
        samples.append({'image_id': img_id})
    return samples


def predict(model, data_generator):
    return model.predict_generator(data_generator, workers=16, use_multiprocessing=True, verbose=1)


def get_predictions(base_model_name, weights_file, image_source, img_format='jpg'):
    weights_type = 'technical' if 'technical' in weights_file else 'aesthetic'

    # load samples
    if os.path.isfile(image_source):
        image_dir, samples = image_file_to_json(image_source)
    else:
        image_dir = image_source
        samples = image_dir_to_json(image_dir, img_type='jpg')

    # build model and load weights
    nima = Nima(base_model_name, weights=None)
    nima.build()
    nima.nima_model.load_weights(weights_file)

    # initialize data generator
    data_generator = TestDataGenerator(samples, image_dir, 64, 10, nima.preprocessing_function(), img_format=img_format)

    # get predictions
    predictions = predict(nima.nima_model, data_generator)

    # calc mean scores and add to samples
    for i, sample in enumerate(samples):
        sample[f'score_{weights_type}'] = calc_mean_score(predictions[i])

    return samples


def rename_files(base_model_name, weights_aesthetic, weights_technical, image_source, img_format='jpg'):
    technicalRes = get_predictions(base_model_name, weights_technical, image_source, img_format)
    aestheticRes = get_predictions(base_model_name, weights_aesthetic, image_source, img_format)
    df_tech = pd.read_json(json.dumps(technicalRes))
    df_aesth = pd.read_json(json.dumps(aestheticRes))
    df = pd.merge(df_tech, df_aesth, on='image_id')
    df['average_score'] = df[['score_technical', 'score_aesthetic']].mean(axis=1)
    df = df.sort_values('average_score', ascending=False)
    arr = df['image_id'].tolist()
    print(arr)
    full_names = [f'{el}.jpg' for i, el in enumerate(arr)]
    my_arr = " ".join(f"'{el}'," for el in full_names)[:-1]
    cmd = f"Set-Location -Path {dir_name}; " + my_arr + ' | ForEach-Object  -begin { $count=1 } -process { Rename-Item $_ -NewName "$count-$_"; $count++ }'
    subprocess.run(["powershell", "-Command", cmd])
    console.log('Files renamed')


def main(base_model_name, weights_aesthetic, weights_technical, image_source, img_format='jpg'):
    technicalRes = get_predictions(base_model_name, weights_technical, image_source, img_format)
    aestheticRes = get_predictions(base_model_name, weights_aesthetic, image_source, img_format)
    df_tech = pd.read_json(json.dumps(technicalRes))
    df_aesth = pd.read_json(json.dumps(aestheticRes))
    df = pd.merge(df_tech, df_aesth, on='image_id')
    df['average_score'] = df[['score_technical', 'score_aesthetic']].mean(axis=1)
    df = df.sort_values('average_score', ascending=False)
    print(json.loads(df.to_json(orient='records')))
    return json.loads(df.to_json(orient='records'))


# %%
import pandas as pd
import numpy as np
import json
import subprocess
import os

myJson = [
    {
        "average_score": 6.1250643667,
        "image_id": "DSC05542",
        "score_aesthetic": 5.6851666612,
        "score_technical": 6.5649620723
    },
    {
        "average_score": 6.1103149522,
        "image_id": "DSC05556",
        "score_aesthetic": 5.3044913272,
        "score_technical": 6.9161385773
    },
    {
        "average_score": 6.0961194592,
        "image_id": "DSC05557",
        "score_aesthetic": 5.3865440627,
        "score_technical": 6.8056948557
    },
    {
        "average_score": 6.0936112215,
        "image_id": "DSC05535",
        "score_aesthetic": 5.5731711346,
        "score_technical": 6.6140513085
    },
    {
        "average_score": 6.0738946914,
        "image_id": "DSC05558",
        "score_aesthetic": 5.5411872189,
        "score_technical": 6.606602164
    },
    {
        "average_score": 6.0658935968,
        "image_id": "DSC05546",
        "score_aesthetic": 5.656965436,
        "score_technical": 6.4748217575
    },
    {
        "average_score": 6.0066405595,
        "image_id": "DSC05502",
        "score_aesthetic": 5.3410402172,
        "score_technical": 6.6722409017
    },
    {
        "average_score": 5.9998765783,
        "image_id": "DSC05537",
        "score_aesthetic": 5.7769564558,
        "score_technical": 6.2227967009
    },
    {
        "average_score": 5.9941617693,
        "image_id": "DSC05536",
        "score_aesthetic": 5.5908203768,
        "score_technical": 6.3975031618
    },
    {
        "average_score": 5.9749876045,
        "image_id": "DSC05748",
        "score_aesthetic": 5.4197037665,
        "score_technical": 6.5302714426
    },
    {
        "average_score": 5.9545467128,
        "image_id": "DSC05548",
        "score_aesthetic": 5.3862738376,
        "score_technical": 6.522819588
    },
    {
        "average_score": 5.9471460103,
        "image_id": "DSC05453",
        "score_aesthetic": 5.8565525594,
        "score_technical": 6.0377394613
    },
    {
        "average_score": 5.8997373523,
        "image_id": "DSC05747",
        "score_aesthetic": 5.3136602376,
        "score_technical": 6.4858144671
    },
    {
        "average_score": 5.8932755888,
        "image_id": "DSC05449",
        "score_aesthetic": 5.1026689756,
        "score_technical": 6.683882202
    },
    {
        "average_score": 5.8915797115,
        "image_id": "DSC05583",
        "score_aesthetic": 5.2803010301,
        "score_technical": 6.5028583929
    },
    {
        "average_score": 5.8907494529,
        "image_id": "DSC05651",
        "score_aesthetic": 5.351486445,
        "score_technical": 6.4300124608
    },
    {
        "average_score": 5.8886452525,
        "image_id": "DSC05543",
        "score_aesthetic": 5.5671945182,
        "score_technical": 6.2100959867
    },
    {
        "average_score": 5.8731840734,
        "image_id": "DSC05709",
        "score_aesthetic": 5.0884804123,
        "score_technical": 6.6578877345
    },
    {
        "average_score": 5.8726428909,
        "image_id": "DSC05689",
        "score_aesthetic": 5.3120311488,
        "score_technical": 6.4332546331
    },
    {
        "average_score": 5.8639604723,
        "image_id": "DSC05555",
        "score_aesthetic": 5.1825188461,
        "score_technical": 6.5454020984
    },
    {
        "average_score": 5.8504254586,
        "image_id": "DSC05493",
        "score_aesthetic": 5.1410829408,
        "score_technical": 6.5597679764
    },
    {
        "average_score": 5.8474451612,
        "image_id": "DSC05761",
        "score_aesthetic": 5.4560361172,
        "score_technical": 6.2388542052
    },
    {
        "average_score": 5.8441809439,
        "image_id": "DSC05515",
        "score_aesthetic": 5.407840372,
        "score_technical": 6.2805215158
    },
    {
        "average_score": 5.8393045402,
        "image_id": "DSC05553",
        "score_aesthetic": 5.2796125148,
        "score_technical": 6.3989965655
    },
    {
        "average_score": 5.8306782665,
        "image_id": "DSC05544",
        "score_aesthetic": 5.5683627706,
        "score_technical": 6.0929937623
    },
    {
        "average_score": 5.8305361375,
        "image_id": "DSC05746",
        "score_aesthetic": 5.2164812911,
        "score_technical": 6.4445909839
    },
    {
        "average_score": 5.8302188273,
        "image_id": "DSC05520",
        "score_aesthetic": 5.5706304534,
        "score_technical": 6.0898072012
    },
    {
        "average_score": 5.830016475,
        "image_id": "DSC05554",
        "score_aesthetic": 5.1493473757,
        "score_technical": 6.5106855743
    },
    {
        "average_score": 5.8279073354,
        "image_id": "DSC05652",
        "score_aesthetic": 5.217275881,
        "score_technical": 6.4385387897
    },
    {
        "average_score": 5.8248781526,
        "image_id": "DSC05564",
        "score_aesthetic": 5.4678482248,
        "score_technical": 6.1819080804
    },
    {
        "average_score": 5.819077597,
        "image_id": "DSC05580",
        "score_aesthetic": 5.1337841098,
        "score_technical": 6.5043710843
    },
    {
        "average_score": 5.8144569197,
        "image_id": "DSC05541",
        "score_aesthetic": 5.5973601276,
        "score_technical": 6.0315537117
    },
    {
        "average_score": 5.8101512617,
        "image_id": "DSC05565",
        "score_aesthetic": 5.6769657013,
        "score_technical": 5.9433368221
    },
    {
        "average_score": 5.8092983058,
        "image_id": "DSC05654",
        "score_aesthetic": 5.1624980104,
        "score_technical": 6.4560986012
    },
    {
        "average_score": 5.8087474169,
        "image_id": "DSC05624",
        "score_aesthetic": 5.2484009313,
        "score_technical": 6.3690939024
    },
    {
        "average_score": 5.8086598711,
        "image_id": "DSC05759",
        "score_aesthetic": 5.2073150407,
        "score_technical": 6.4100047015
    },
    {
        "average_score": 5.8083072812,
        "image_id": "DSC05598",
        "score_aesthetic": 5.0655151781,
        "score_technical": 6.5510993842
    },
    {
        "average_score": 5.8065401656,
        "image_id": "DSC05501",
        "score_aesthetic": 5.1595265764,
        "score_technical": 6.4535537548
    },
    {
        "average_score": 5.7920824286,
        "image_id": "DSC05586",
        "score_aesthetic": 5.2153355883,
        "score_technical": 6.368829269
    },
    {
        "average_score": 5.7901007149,
        "image_id": "DSC05635",
        "score_aesthetic": 5.0436003036,
        "score_technical": 6.5366011262
    },
    {
        "average_score": 5.7842782186,
        "image_id": "DSC05539",
        "score_aesthetic": 5.2835015858,
        "score_technical": 6.2850548513
    },
    {
        "average_score": 5.7832618298,
        "image_id": "DSC05566",
        "score_aesthetic": 5.624198224,
        "score_technical": 5.9423254356
    },
    {
        "average_score": 5.7772848103,
        "image_id": "DSC05540",
        "score_aesthetic": 5.3408839292,
        "score_technical": 6.2136856914
    },
    {
        "average_score": 5.7686438068,
        "image_id": "DSC05693",
        "score_aesthetic": 5.297877243,
        "score_technical": 6.2394103706
    },
    {
        "average_score": 5.7646245811,
        "image_id": "DSC05476",
        "score_aesthetic": 5.5251908122,
        "score_technical": 6.0040583499
    },
    {
        "average_score": 5.7629780754,
        "image_id": "DSC05547",
        "score_aesthetic": 5.3327956057,
        "score_technical": 6.1931605451
    },
    {
        "average_score": 5.7531661717,
        "image_id": "DSC05488",
        "score_aesthetic": 5.0142942835,
        "score_technical": 6.49203806
    },
    {
        "average_score": 5.751681038,
        "image_id": "DSC05463",
        "score_aesthetic": 5.1763861418,
        "score_technical": 6.3269759342
    },
    {
        "average_score": 5.7484281676,
        "image_id": "DSC05489",
        "score_aesthetic": 5.0500821672,
        "score_technical": 6.4467741679
    },
    {
        "average_score": 5.7442942939,
        "image_id": "DSC05478",
        "score_aesthetic": 5.4611128881,
        "score_technical": 6.0274756998
    },
    {
        "average_score": 5.7439741665,
        "image_id": "DSC05550",
        "score_aesthetic": 5.192593039,
        "score_technical": 6.2953552939
    },
    {
        "average_score": 5.7385315581,
        "image_id": "DSC05745",
        "score_aesthetic": 4.8509780061,
        "score_technical": 6.62608511
    },
    {
        "average_score": 5.7375450479,
        "image_id": "DSC05486",
        "score_aesthetic": 4.9491066244,
        "score_technical": 6.5259834714
    },
    {
        "average_score": 5.7353365598,
        "image_id": "DSC05582",
        "score_aesthetic": 5.3158046118,
        "score_technical": 6.1548685078
    },
    {
        "average_score": 5.7309956486,
        "image_id": "DSC05615",
        "score_aesthetic": 5.0339736213,
        "score_technical": 6.4280176759
    },
    {
        "average_score": 5.7286013003,
        "image_id": "DSC05455",
        "score_aesthetic": 5.3277733943,
        "score_technical": 6.1294292063
    },
    {
        "average_score": 5.7277495499,
        "image_id": "DSC05757",
        "score_aesthetic": 5.0610363505,
        "score_technical": 6.3944627494
    },
    {
        "average_score": 5.7217803653,
        "image_id": "DSC05532",
        "score_aesthetic": 5.4483784735,
        "score_technical": 5.9951822571
    },
    {
        "average_score": 5.721514228,
        "image_id": "DSC05500",
        "score_aesthetic": 5.4631646522,
        "score_technical": 5.9798638038
    },
    {
        "average_score": 5.7199466393,
        "image_id": "DSC05503",
        "score_aesthetic": 5.0074843087,
        "score_technical": 6.4324089698
    },
    {
        "average_score": 5.7176730588,
        "image_id": "DSC05762",
        "score_aesthetic": 5.0771457277,
        "score_technical": 6.3582003899
    },
    {
        "average_score": 5.7120870869,
        "image_id": "DSC05587",
        "score_aesthetic": 5.1374494499,
        "score_technical": 6.2867247239
    },
    {
        "average_score": 5.7119148904,
        "image_id": "DSC05691",
        "score_aesthetic": 5.2561831604,
        "score_technical": 6.1676466204
    },
    {
        "average_score": 5.7085978529,
        "image_id": "DSC05579",
        "score_aesthetic": 5.0402063915,
        "score_technical": 6.3769893143
    },
    {
        "average_score": 5.7079226068,
        "image_id": "DSC05690",
        "score_aesthetic": 5.1533235946,
        "score_technical": 6.262521619
    },
    {
        "average_score": 5.7051130717,
        "image_id": "DSC05492",
        "score_aesthetic": 4.976783588,
        "score_technical": 6.4334425554
    },
    {
        "average_score": 5.7046693422,
        "image_id": "DSC05639",
        "score_aesthetic": 5.2829293066,
        "score_technical": 6.1264093779
    },
    {
        "average_score": 5.7033367846,
        "image_id": "DSC05734",
        "score_aesthetic": 4.8979879283,
        "score_technical": 6.508685641
    },
    {
        "average_score": 5.7018890815,
        "image_id": "DSC05551",
        "score_aesthetic": 5.3402304549,
        "score_technical": 6.0635477081
    },
    {
        "average_score": 5.7001526132,
        "image_id": "DSC05545",
        "score_aesthetic": 5.1359747462,
        "score_technical": 6.2643304802
    },
    {
        "average_score": 5.7000145778,
        "image_id": "DSC05669",
        "score_aesthetic": 4.9274094191,
        "score_technical": 6.4726197366
    },
    {
        "average_score": 5.6958218738,
        "image_id": "DSC05597",
        "score_aesthetic": 5.0548557936,
        "score_technical": 6.336787954
    },
    {
        "average_score": 5.6950160128,
        "image_id": "DSC05469",
        "score_aesthetic": 5.0911547231,
        "score_technical": 6.2988773026
    },
    {
        "average_score": 5.6950153473,
        "image_id": "DSC05754",
        "score_aesthetic": 4.8749147984,
        "score_technical": 6.5151158962
    },
    {
        "average_score": 5.6949874645,
        "image_id": "DSC05751",
        "score_aesthetic": 4.8846802969,
        "score_technical": 6.5052946322
    },
    {
        "average_score": 5.6941687863,
        "image_id": "DSC05522",
        "score_aesthetic": 5.2483237325,
        "score_technical": 6.14001384
    },
    {
        "average_score": 5.6897192613,
        "image_id": "DSC05560",
        "score_aesthetic": 5.1802819421,
        "score_technical": 6.1991565805
    },
    {
        "average_score": 5.6888508813,
        "image_id": "DSC05451",
        "score_aesthetic": 5.3630118253,
        "score_technical": 6.0146899372
    },
    {
        "average_score": 5.6872112505,
        "image_id": "DSC05521",
        "score_aesthetic": 5.1089895677,
        "score_technical": 6.2654329333
    },
    {
        "average_score": 5.6861429622,
        "image_id": "DSC05643",
        "score_aesthetic": 5.2566887937,
        "score_technical": 6.1155971307
    },
    {
        "average_score": 5.6852468913,
        "image_id": "DSC05636",
        "score_aesthetic": 4.791128667,
        "score_technical": 6.5793651156
    },
    {
        "average_score": 5.6832103386,
        "image_id": "DSC05509",
        "score_aesthetic": 5.3144235147,
        "score_technical": 6.0519971624
    },
    {
        "average_score": 5.6824268533,
        "image_id": "DSC05774",
        "score_aesthetic": 5.2957767717,
        "score_technical": 6.0690769348
    },
    {
        "average_score": 5.6755430905,
        "image_id": "DSC05732",
        "score_aesthetic": 5.3253698651,
        "score_technical": 6.025716316
    },
    {
        "average_score": 5.6748715834,
        "image_id": "DSC05772",
        "score_aesthetic": 4.9838894697,
        "score_technical": 6.3658536971
    },
    {
        "average_score": 5.673575244,
        "image_id": "DSC05549",
        "score_aesthetic": 5.0753260237,
        "score_technical": 6.2718244642
    },
    {
        "average_score": 5.6719910566,
        "image_id": "DSC05752",
        "score_aesthetic": 4.85098391,
        "score_technical": 6.4929982033
    },
    {
        "average_score": 5.6631730863,
        "image_id": "DSC05704",
        "score_aesthetic": 5.0637734966,
        "score_technical": 6.2625726759
    },
    {
        "average_score": 5.6559405789,
        "image_id": "DSC05479",
        "score_aesthetic": 5.3605734252,
        "score_technical": 5.9513077326
    },
    {
        "average_score": 5.6551406266,
        "image_id": "DSC05524",
        "score_aesthetic": 5.1903496969,
        "score_technical": 6.1199315563
    },
    {
        "average_score": 5.6469630958,
        "image_id": "DSC05487",
        "score_aesthetic": 5.2335144999,
        "score_technical": 6.0604116917
    },
    {
        "average_score": 5.6463615678,
        "image_id": "DSC05445",
        "score_aesthetic": 5.2290991,
        "score_technical": 6.0636240356
    },
    {
        "average_score": 5.6462354984,
        "image_id": "DSC05637",
        "score_aesthetic": 4.9846414714,
        "score_technical": 6.3078295253
    },
    {
        "average_score": 5.6405402356,
        "image_id": "DSC05771",
        "score_aesthetic": 5.1834322659,
        "score_technical": 6.0976482052
    },
    {
        "average_score": 5.6357948113,
        "image_id": "DSC05462",
        "score_aesthetic": 5.0286322206,
        "score_technical": 6.242957402
    },
    {
        "average_score": 5.6354876408,
        "image_id": "DSC05621",
        "score_aesthetic": 4.7276004916,
        "score_technical": 6.5433747899
    },
    {
        "average_score": 5.6349651058,
        "image_id": "DSC05685",
        "score_aesthetic": 4.9608757884,
        "score_technical": 6.3090544231
    },
    {
        "average_score": 5.6343304706,
        "image_id": "DSC05460",
        "score_aesthetic": 5.081136656,
        "score_technical": 6.1875242852
    },
    {
        "average_score": 5.6332845079,
        "image_id": "DSC05710",
        "score_aesthetic": 5.0220700459,
        "score_technical": 6.24449897
    },
    {
        "average_score": 5.6268945012,
        "image_id": "DSC05773",
        "score_aesthetic": 5.2658285652,
        "score_technical": 5.9879604373
    },
    {
        "average_score": 5.6267050017,
        "image_id": "DSC05755",
        "score_aesthetic": 4.8981256572,
        "score_technical": 6.3552843463
    },
    {
        "average_score": 5.6262531269,
        "image_id": "DSC05753",
        "score_aesthetic": 4.8503283817,
        "score_technical": 6.4021778721
    },
    {
        "average_score": 5.6207246145,
        "image_id": "DSC05750",
        "score_aesthetic": 5.094674177,
        "score_technical": 6.146775052
    },
    {
        "average_score": 5.6198806644,
        "image_id": "DSC05680",
        "score_aesthetic": 4.9031691147,
        "score_technical": 6.3365922142
    },
    {
        "average_score": 5.6197614241,
        "image_id": "DSC05708",
        "score_aesthetic": 4.9037207141,
        "score_technical": 6.3358021341
    },
    {
        "average_score": 5.6157210849,
        "image_id": "DSC05527",
        "score_aesthetic": 5.1539875731,
        "score_technical": 6.0774545968
    },
    {
        "average_score": 5.6145124625,
        "image_id": "DSC05596",
        "score_aesthetic": 4.9109681032,
        "score_technical": 6.3180568218
    },
    {
        "average_score": 5.6139387123,
        "image_id": "DSC05480",
        "score_aesthetic": 5.3931300171,
        "score_technical": 5.8347474076
    },
    {
        "average_score": 5.6124613639,
        "image_id": "DSC05552",
        "score_aesthetic": 5.0166109391,
        "score_technical": 6.2083117887
    },
    {
        "average_score": 5.6119028653,
        "image_id": "DSC05733",
        "score_aesthetic": 5.0221588438,
        "score_technical": 6.2016468868
    },
    {
        "average_score": 5.6085641762,
        "image_id": "DSC05595",
        "score_aesthetic": 4.7497252683,
        "score_technical": 6.467403084
    },
    {
        "average_score": 5.6067478295,
        "image_id": "DSC05697",
        "score_aesthetic": 5.103789775,
        "score_technical": 6.109705884
    },
    {
        "average_score": 5.6057801611,
        "image_id": "DSC05659",
        "score_aesthetic": 4.8908302935,
        "score_technical": 6.3207300287
    },
    {
        "average_score": 5.6055511767,
        "image_id": "DSC05613",
        "score_aesthetic": 4.9602553957,
        "score_technical": 6.2508469578
    },
    {
        "average_score": 5.6047337813,
        "image_id": "DSC05563",
        "score_aesthetic": 5.4740173321,
        "score_technical": 5.7354502305
    },
    {
        "average_score": 5.6046374287,
        "image_id": "DSC05525",
        "score_aesthetic": 5.179655972,
        "score_technical": 6.0296188854
    },
    {
        "average_score": 5.6020842763,
        "image_id": "DSC05526",
        "score_aesthetic": 5.1042159853,
        "score_technical": 6.0999525674
    },
    {
        "average_score": 5.6020221242,
        "image_id": "DSC05688",
        "score_aesthetic": 5.1428547905,
        "score_technical": 6.0611894578
    },
    {
        "average_score": 5.6006589361,
        "image_id": "DSC05562",
        "score_aesthetic": 5.4102038512,
        "score_technical": 5.7911140211
    },
    {
        "average_score": 5.5998804487,
        "image_id": "DSC05519",
        "score_aesthetic": 5.2826783157,
        "score_technical": 5.9170825817
    },
    {
        "average_score": 5.5969586352,
        "image_id": "DSC05529",
        "score_aesthetic": 4.8715941957,
        "score_technical": 6.3223230746
    },
    {
        "average_score": 5.5953191032,
        "image_id": "DSC05765",
        "score_aesthetic": 5.0828232611,
        "score_technical": 6.1078149453
    },
    {
        "average_score": 5.595306158,
        "image_id": "DSC05731",
        "score_aesthetic": 5.0389870534,
        "score_technical": 6.1516252626
    },
    {
        "average_score": 5.5943233278,
        "image_id": "DSC05630",
        "score_aesthetic": 5.1055807286,
        "score_technical": 6.083065927
    },
    {
        "average_score": 5.5933361869,
        "image_id": "DSC05650",
        "score_aesthetic": 4.8169058497,
        "score_technical": 6.3697665241
    },
    {
        "average_score": 5.5928424069,
        "image_id": "DSC05632",
        "score_aesthetic": 4.9647300863,
        "score_technical": 6.2209547274
    },
    {
        "average_score": 5.5927246496,
        "image_id": "DSC05705",
        "score_aesthetic": 5.0091744111,
        "score_technical": 6.1762748882
    },
    {
        "average_score": 5.5927196592,
        "image_id": "DSC05439",
        "score_aesthetic": 5.057557106,
        "score_technical": 6.1278822124
    },
    {
        "average_score": 5.5906378411,
        "image_id": "DSC05712",
        "score_aesthetic": 5.0465385949,
        "score_technical": 6.1347370874
    },
    {
        "average_score": 5.5879230532,
        "image_id": "DSC05623",
        "score_aesthetic": 4.9569687089,
        "score_technical": 6.2188773975
    },
    {
        "average_score": 5.5878390806,
        "image_id": "DSC05440",
        "score_aesthetic": 5.3744430739,
        "score_technical": 5.8012350872
    },
    {
        "average_score": 5.5825034168,
        "image_id": "DSC05584",
        "score_aesthetic": 5.1886249468,
        "score_technical": 5.9763818868
    },
    {
        "average_score": 5.5818557567,
        "image_id": "DSC05692",
        "score_aesthetic": 5.2551991222,
        "score_technical": 5.9085123911
    },
    {
        "average_score": 5.5813538538,
        "image_id": "DSC05588",
        "score_aesthetic": 4.9719335015,
        "score_technical": 6.1907742061
    },
    {
        "average_score": 5.5796469085,
        "image_id": "DSC05749",
        "score_aesthetic": 5.099421838,
        "score_technical": 6.0598719791
    },
    {
        "average_score": 5.578470068,
        "image_id": "DSC05641",
        "score_aesthetic": 4.9056124053,
        "score_technical": 6.2513277307
    },
    {
        "average_score": 5.577764358,
        "image_id": "DSC05664",
        "score_aesthetic": 4.9313212701,
        "score_technical": 6.224207446
    },
    {
        "average_score": 5.5698322069,
        "image_id": "DSC05660",
        "score_aesthetic": 4.9929425926,
        "score_technical": 6.1467218213
    },
    {
        "average_score": 5.5695328257,
        "image_id": "DSC05658",
        "score_aesthetic": 5.0027742369,
        "score_technical": 6.1362914145
    },
    {
        "average_score": 5.5689795196,
        "image_id": "DSC05477",
        "score_aesthetic": 5.47871195,
        "score_technical": 5.6592470892
    },
    {
        "average_score": 5.5664266038,
        "image_id": "DSC05494",
        "score_aesthetic": 5.1002733647,
        "score_technical": 6.032579843
    },
    {
        "average_score": 5.5656041657,
        "image_id": "DSC05665",
        "score_aesthetic": 4.843620711,
        "score_technical": 6.2875876203
    },
    {
        "average_score": 5.5653580851,
        "image_id": "DSC05442",
        "score_aesthetic": 5.0091740816,
        "score_technical": 6.1215420887
    },
    {
        "average_score": 5.5643025527,
        "image_id": "DSC05448",
        "score_aesthetic": 4.8069886029,
        "score_technical": 6.3216165025
    },
    {
        "average_score": 5.5631283647,
        "image_id": "DSC05294",
        "score_aesthetic": 4.9429074963,
        "score_technical": 6.1833492331
    },
    {
        "average_score": 5.5626361011,
        "image_id": "DSC05465",
        "score_aesthetic": 4.8070900412,
        "score_technical": 6.3181821611
    },
    {
        "average_score": 5.5564464339,
        "image_id": "DSC05610",
        "score_aesthetic": 4.8141663594,
        "score_technical": 6.2987265084
    },
    {
        "average_score": 5.5558639939,
        "image_id": "DSC05561",
        "score_aesthetic": 5.3410508571,
        "score_technical": 5.7706771307
    },
    {
        "average_score": 5.5553761849,
        "image_id": "DSC05531",
        "score_aesthetic": 5.1645512615,
        "score_technical": 5.9462011084
    },
    {
        "average_score": 5.5541955259,
        "image_id": "DSC05763",
        "score_aesthetic": 4.9307394583,
        "score_technical": 6.1776515935
    },
    {
        "average_score": 5.5532291866,
        "image_id": "DSC05756",
        "score_aesthetic": 4.8186496439,
        "score_technical": 6.2878087293
    },
    {
        "average_score": 5.5513976614,
        "image_id": "DSC05464",
        "score_aesthetic": 4.7150669502,
        "score_technical": 6.3877283726
    },
    {
        "average_score": 5.5499222033,
        "image_id": "DSC05612",
        "score_aesthetic": 4.8852757656,
        "score_technical": 6.214568641
    },
    {
        "average_score": 5.5464772006,
        "image_id": "DSC05581",
        "score_aesthetic": 5.2569256074,
        "score_technical": 5.8360287938
    },
    {
        "average_score": 5.5449688006,
        "image_id": "DSC05687",
        "score_aesthetic": 4.9679193161,
        "score_technical": 6.1220182851
    },
    {
        "average_score": 5.5438636038,
        "image_id": "DSC05702",
        "score_aesthetic": 5.0339928305,
        "score_technical": 6.053734377
    },
    {
        "average_score": 5.5377894051,
        "image_id": "DSC05585",
        "score_aesthetic": 5.1058617975,
        "score_technical": 5.9697170127
    },
    {
        "average_score": 5.5371173319,
        "image_id": "DSC05644",
        "score_aesthetic": 5.2069401046,
        "score_technical": 5.8672945593
    },
    {
        "average_score": 5.5361447955,
        "image_id": "DSC05642",
        "score_aesthetic": 5.0885452082,
        "score_technical": 5.9837443829
    },
    {
        "average_score": 5.5348097851,
        "image_id": "DSC05491",
        "score_aesthetic": 4.8233945149,
        "score_technical": 6.2462250553
    },
    {
        "average_score": 5.5279838738,
        "image_id": "DSC05631",
        "score_aesthetic": 4.9039812864,
        "score_technical": 6.1519864611
    },
    {
        "average_score": 5.5278114744,
        "image_id": "DSC05640",
        "score_aesthetic": 5.2669624448,
        "score_technical": 5.7886605039
    },
    {
        "average_score": 5.5264674525,
        "image_id": "DSC05633",
        "score_aesthetic": 4.9909676029,
        "score_technical": 6.0619673021
    },
    {
        "average_score": 5.5224914498,
        "image_id": "DSC05667",
        "score_aesthetic": 4.9566593065,
        "score_technical": 6.0883235931
    },
    {
        "average_score": 5.5170538067,
        "image_id": "DSC05483",
        "score_aesthetic": 4.665615608,
        "score_technical": 6.3684920054
    },
    {
        "average_score": 5.5156943602,
        "image_id": "DSC05490",
        "score_aesthetic": 4.8757554237,
        "score_technical": 6.1556332968
    },
    {
        "average_score": 5.5145064797,
        "image_id": "DSC05703",
        "score_aesthetic": 5.0465826503,
        "score_technical": 5.9824303091
    },
    {
        "average_score": 5.5143986334,
        "image_id": "DSC05626",
        "score_aesthetic": 4.8493945206,
        "score_technical": 6.1794027463
    },
    {
        "average_score": 5.5109102982,
        "image_id": "DSC05614",
        "score_aesthetic": 4.7201610329,
        "score_technical": 6.3016595636
    },
    {
        "average_score": 5.5099198609,
        "image_id": "DSC05616",
        "score_aesthetic": 5.0187695615,
        "score_technical": 6.0010701604
    },
    {
        "average_score": 5.5090518239,
        "image_id": "DSC05530",
        "score_aesthetic": 4.9559689907,
        "score_technical": 6.0621346571
    },
    {
        "average_score": 5.5081861231,
        "image_id": "DSC05594",
        "score_aesthetic": 4.9959385483,
        "score_technical": 6.0204336978
    },
    {
        "average_score": 5.5051821462,
        "image_id": "DSC05770",
        "score_aesthetic": 4.8519072925,
        "score_technical": 6.1584569998
    },
    {
        "average_score": 5.5019594724,
        "image_id": "DSC05666",
        "score_aesthetic": 4.8600067598,
        "score_technical": 6.143912185
    },
    {
        "average_score": 5.5012711652,
        "image_id": "DSC05738",
        "score_aesthetic": 4.9004507824,
        "score_technical": 6.102091548
    },
    {
        "average_score": 5.4954154933,
        "image_id": "DSC05467",
        "score_aesthetic": 4.6861751255,
        "score_technical": 6.3046558611
    },
    {
        "average_score": 5.488491596,
        "image_id": "DSC05459",
        "score_aesthetic": 4.9904993461,
        "score_technical": 5.9864838459
    },
    {
        "average_score": 5.4850449736,
        "image_id": "DSC05534",
        "score_aesthetic": 5.4251148676,
        "score_technical": 5.5449750796
    },
    {
        "average_score": 5.482762511,
        "image_id": "DSC05706",
        "score_aesthetic": 4.8164622618,
        "score_technical": 6.1490627602
    },
    {
        "average_score": 5.481147548,
        "image_id": "DSC05446",
        "score_aesthetic": 4.9687361639,
        "score_technical": 5.9935589321
    },
    {
        "average_score": 5.4811411143,
        "image_id": "DSC05681",
        "score_aesthetic": 5.136403221,
        "score_technical": 5.8258790076
    },
    {
        "average_score": 5.4796097737,
        "image_id": "DSC05740",
        "score_aesthetic": 4.7697860677,
        "score_technical": 6.1894334797
    },
    {
        "average_score": 5.4759747429,
        "image_id": "DSC05653",
        "score_aesthetic": 4.6171375524,
        "score_technical": 6.3348119333
    },
    {
        "average_score": 5.4695857166,
        "image_id": "DSC05533",
        "score_aesthetic": 5.2318337073,
        "score_technical": 5.7073377259
    },
    {
        "average_score": 5.4683276797,
        "image_id": "DSC05684",
        "score_aesthetic": 5.2083528731,
        "score_technical": 5.7283024862
    },
    {
        "average_score": 5.4677803529,
        "image_id": "DSC05695",
        "score_aesthetic": 5.0138888012,
        "score_technical": 5.9216719046
    },
    {
        "average_score": 5.4649220959,
        "image_id": "DSC05634",
        "score_aesthetic": 4.8793665307,
        "score_technical": 6.0504776612
    },
    {
        "average_score": 5.464038796,
        "image_id": "DSC05472",
        "score_aesthetic": 4.859642303,
        "score_technical": 6.068435289
    },
    {
        "average_score": 5.4609670413,
        "image_id": "DSC05538",
        "score_aesthetic": 5.0700197886,
        "score_technical": 5.8519142941
    },
    {
        "average_score": 5.4537560387,
        "image_id": "DSC05682",
        "score_aesthetic": 4.7450538326,
        "score_technical": 6.1624582447
    },
    {
        "average_score": 5.4523443499,
        "image_id": "DSC05528",
        "score_aesthetic": 4.9660866286,
        "score_technical": 5.9386020713
    },
    {
        "average_score": 5.4443930107,
        "image_id": "DSC05696",
        "score_aesthetic": 5.1753848043,
        "score_technical": 5.713401217
    },
    {
        "average_score": 5.4441893335,
        "image_id": "DSC05508",
        "score_aesthetic": 4.9274971358,
        "score_technical": 5.9608815312
    },
    {
        "average_score": 5.4427589008,
        "image_id": "DSC05466",
        "score_aesthetic": 4.5022830549,
        "score_technical": 6.3832347468
    },
    {
        "average_score": 5.4413444027,
        "image_id": "DSC05571",
        "score_aesthetic": 4.5965146235,
        "score_technical": 6.2861741818
    },
    {
        "average_score": 5.4381598311,
        "image_id": "DSC05470",
        "score_aesthetic": 4.9763789927,
        "score_technical": 5.8999406695
    },
    {
        "average_score": 5.4376893579,
        "image_id": "DSC05482",
        "score_aesthetic": 4.6296357093,
        "score_technical": 6.2457430065
    },
    {
        "average_score": 5.4364683177,
        "image_id": "DSC05769",
        "score_aesthetic": 4.8830732881,
        "score_technical": 5.9898633473
    },
    {
        "average_score": 5.4349515333,
        "image_id": "DSC05777",
        "score_aesthetic": 5.0392429731,
        "score_technical": 5.8306600936
    },
    {
        "average_score": 5.4348337581,
        "image_id": "DSC05655",
        "score_aesthetic": 4.8239374693,
        "score_technical": 6.0457300469
    },
    {
        "average_score": 5.4330132117,
        "image_id": "DSC05760",
        "score_aesthetic": 5.1007914688,
        "score_technical": 5.7652349547
    },
    {
        "average_score": 5.4275314515,
        "image_id": "DSC05516",
        "score_aesthetic": 4.915039723,
        "score_technical": 5.9400231801
    },
    {
        "average_score": 5.4211069149,
        "image_id": "DSC05481",
        "score_aesthetic": 4.8785175362,
        "score_technical": 5.9636962935
    },
    {
        "average_score": 5.4205208496,
        "image_id": "DSC05507",
        "score_aesthetic": 4.9877901191,
        "score_technical": 5.8532515801
    },
    {
        "average_score": 5.4163318777,
        "image_id": "DSC05686",
        "score_aesthetic": 4.7726180325,
        "score_technical": 6.0600457229
    },
    {
        "average_score": 5.4161957802,
        "image_id": "DSC05506",
        "score_aesthetic": 4.8590525243,
        "score_technical": 5.9733390361
    },
    {
        "average_score": 5.4160838013,
        "image_id": "DSC05620",
        "score_aesthetic": 4.8533730427,
        "score_technical": 5.9787945598
    },
    {
        "average_score": 5.4133353518,
        "image_id": "DSC05674",
        "score_aesthetic": 4.6485550639,
        "score_technical": 6.1781156398
    },
    {
        "average_score": 5.4008232122,
        "image_id": "DSC05484",
        "score_aesthetic": 4.4269082509,
        "score_technical": 6.3747381736
    },
    {
        "average_score": 5.3988573458,
        "image_id": "DSC05647",
        "score_aesthetic": 4.979428809,
        "score_technical": 5.8182858825
    },
    {
        "average_score": 5.3988087771,
        "image_id": "DSC05461",
        "score_aesthetic": 4.829792771,
        "score_technical": 5.9678247832
    },
    {
        "average_score": 5.3965780297,
        "image_id": "DSC05589",
        "score_aesthetic": 5.0275477935,
        "score_technical": 5.765608266
    },
    {
        "average_score": 5.3930856845,
        "image_id": "DSC05523",
        "score_aesthetic": 4.9552484525,
        "score_technical": 5.8309229165
    },
    {
        "average_score": 5.387325206,
        "image_id": "DSC05468",
        "score_aesthetic": 4.5270963855,
        "score_technical": 6.2475540265
    },
    {
        "average_score": 5.3839351414,
        "image_id": "DSC05742",
        "score_aesthetic": 4.7625045778,
        "score_technical": 6.0053657051
    },
    {
        "average_score": 5.383488641,
        "image_id": "DSC05355",
        "score_aesthetic": 5.6041326912,
        "score_technical": 5.1628445908
    },
    {
        "average_score": 5.3812921426,
        "image_id": "DSC05668",
        "score_aesthetic": 4.8633907934,
        "score_technical": 5.8991934918
    },
    {
        "average_score": 5.3793180986,
        "image_id": "DSC05670",
        "score_aesthetic": 4.6797547993,
        "score_technical": 6.0788813978
    },
    {
        "average_score": 5.3788409655,
        "image_id": "DSC05576",
        "score_aesthetic": 4.5531165587,
        "score_technical": 6.2045653723
    },
    {
        "average_score": 5.3782643508,
        "image_id": "DSC05645",
        "score_aesthetic": 4.9254043697,
        "score_technical": 5.8311243318
    },
    {
        "average_score": 5.3750802921,
        "image_id": "DSC05443",
        "score_aesthetic": 4.7651536298,
        "score_technical": 5.9850069545
    },
    {
        "average_score": 5.3745922982,
        "image_id": "DSC05505",
        "score_aesthetic": 5.0076377942,
        "score_technical": 5.7415468022
    },
    {
        "average_score": 5.3721431184,
        "image_id": "DSC05678",
        "score_aesthetic": 4.6832555434,
        "score_technical": 6.0610306934
    },
    {
        "average_score": 5.3698016686,
        "image_id": "DSC05513",
        "score_aesthetic": 4.5240394514,
        "score_technical": 6.2155638859
    },
    {
        "average_score": 5.3683161074,
        "image_id": "DSC05475",
        "score_aesthetic": 4.6631337794,
        "score_technical": 6.0734984353
    },
    {
        "average_score": 5.3680552434,
        "image_id": "DSC05572",
        "score_aesthetic": 4.6057559333,
        "score_technical": 6.1303545535
    },
    {
        "average_score": 5.36709356,
        "image_id": "DSC05625",
        "score_aesthetic": 4.7313699363,
        "score_technical": 6.0028171837
    },
    {
        "average_score": 5.3618075341,
        "image_id": "DSC05741",
        "score_aesthetic": 4.512283887,
        "score_technical": 6.2113311812
    },
    {
        "average_score": 5.3588168531,
        "image_id": "DSC05574",
        "score_aesthetic": 4.7319189652,
        "score_technical": 5.9857147411
    },
    {
        "average_score": 5.3566818836,
        "image_id": "DSC05743",
        "score_aesthetic": 4.6654731859,
        "score_technical": 6.0478905812
    },
    {
        "average_score": 5.3559519023,
        "image_id": "DSC05663",
        "score_aesthetic": 4.6999453769,
        "score_technical": 6.0119584277
    },
    {
        "average_score": 5.3548679715,
        "image_id": "DSC05730",
        "score_aesthetic": 4.4878632985,
        "score_technical": 6.2218726445
    },
    {
        "average_score": 5.3519569739,
        "image_id": "DSC05573",
        "score_aesthetic": 4.6541066079,
        "score_technical": 6.0498073399
    },
    {
        "average_score": 5.3514676939,
        "image_id": "DSC05569",
        "score_aesthetic": 5.0682046836,
        "score_technical": 5.6347307041
    },
    {
        "average_score": 5.3510074381,
        "image_id": "DSC05575",
        "score_aesthetic": 4.7711698541,
        "score_technical": 5.9308450222
    },
    {
        "average_score": 5.3493139877,
        "image_id": "DSC05496",
        "score_aesthetic": 4.7139387509,
        "score_technical": 5.9846892245
    },
    {
        "average_score": 5.3479087537,
        "image_id": "DSC05518",
        "score_aesthetic": 4.8357528072,
        "score_technical": 5.8600647002
    },
    {
        "average_score": 5.3449161013,
        "image_id": "DSC05758",
        "score_aesthetic": 4.2956059385,
        "score_technical": 6.3942262642
    },
    {
        "average_score": 5.3446085808,
        "image_id": "DSC05627",
        "score_aesthetic": 4.7889981123,
        "score_technical": 5.9002190493
    },
    {
        "average_score": 5.3423556797,
        "image_id": "DSC05657",
        "score_aesthetic": 4.6721009313,
        "score_technical": 6.012610428
    },
    {
        "average_score": 5.3393345755,
        "image_id": "DSC05728",
        "score_aesthetic": 4.5690382943,
        "score_technical": 6.1096308567
    },
    {
        "average_score": 5.338907958,
        "image_id": "DSC05776",
        "score_aesthetic": 4.9849063772,
        "score_technical": 5.6929095387
    },
    {
        "average_score": 5.338330028,
        "image_id": "DSC05768",
        "score_aesthetic": 4.7781916248,
        "score_technical": 5.8984684311
    },
    {
        "average_score": 5.3375253178,
        "image_id": "DSC05707",
        "score_aesthetic": 4.4146755084,
        "score_technical": 6.2603751272
    },
    {
        "average_score": 5.337406602,
        "image_id": "DSC05600",
        "score_aesthetic": 4.9500534391,
        "score_technical": 5.724759765
    },
    {
        "average_score": 5.3364908079,
        "image_id": "DSC05611",
        "score_aesthetic": 4.5567511057,
        "score_technical": 6.1162305102
    },
    {
        "average_score": 5.3330108983,
        "image_id": "DSC05441",
        "score_aesthetic": 5.1598956609,
        "score_technical": 5.5061261356
    },
    {
        "average_score": 5.328251404,
        "image_id": "DSC05485",
        "score_aesthetic": 4.5550416806,
        "score_technical": 6.1014611274
    },
    {
        "average_score": 5.3272793706,
        "image_id": "DSC05608",
        "score_aesthetic": 4.6249749183,
        "score_technical": 6.0295838229
    },
    {
        "average_score": 5.3263432181,
        "image_id": "DSC05609",
        "score_aesthetic": 4.4846012988,
        "score_technical": 6.1680851374
    },
    {
        "average_score": 5.3232987845,
        "image_id": "DSC05559",
        "score_aesthetic": 4.8008263978,
        "score_technical": 5.8457711712
    },
    {
        "average_score": 5.3215901683,
        "image_id": "DSC05454",
        "score_aesthetic": 5.2063444777,
        "score_technical": 5.436835859
    },
    {
        "average_score": 5.3202297661,
        "image_id": "DSC05677",
        "score_aesthetic": 4.7734667532,
        "score_technical": 5.8669927791
    },
    {
        "average_score": 5.3164953019,
        "image_id": "DSC05495",
        "score_aesthetic": 4.5285102404,
        "score_technical": 6.1044803634
    },
    {
        "average_score": 5.3154166513,
        "image_id": "DSC05724",
        "score_aesthetic": 4.4936803924,
        "score_technical": 6.1371529102
    },
    {
        "average_score": 5.3145298011,
        "image_id": "DSC05568",
        "score_aesthetic": 4.9043711691,
        "score_technical": 5.7246884331
    },
    {
        "average_score": 5.3120177826,
        "image_id": "DSC05578",
        "score_aesthetic": 4.4923625383,
        "score_technical": 6.1316730268
    },
    {
        "average_score": 5.3109223641,
        "image_id": "DSC05672",
        "score_aesthetic": 4.7024823173,
        "score_technical": 5.9193624109
    },
    {
        "average_score": 5.3096020671,
        "image_id": "DSC05675",
        "score_aesthetic": 4.7208301728,
        "score_technical": 5.8983739614
    },
    {
        "average_score": 5.3086070372,
        "image_id": "DSC05701",
        "score_aesthetic": 4.7103178222,
        "score_technical": 5.9068962522
    },
    {
        "average_score": 5.3076947221,
        "image_id": "DSC05780",
        "score_aesthetic": 4.6053955633,
        "score_technical": 6.009993881
    },
    {
        "average_score": 5.3044710751,
        "image_id": "DSC05458",
        "score_aesthetic": 4.8635182446,
        "score_technical": 5.7454239056
    },
    {
        "average_score": 5.3039229935,
        "image_id": "DSC05471",
        "score_aesthetic": 4.8092419493,
        "score_technical": 5.7986040376
    },
    {
        "average_score": 5.3034805561,
        "image_id": "DSC05498",
        "score_aesthetic": 4.6666106356,
        "score_technical": 5.9403504767
    },
    {
        "average_score": 5.2998192912,
        "image_id": "DSC05622",
        "score_aesthetic": 4.6374026659,
        "score_technical": 5.9622359164
    },
    {
        "average_score": 5.2973102226,
        "image_id": "DSC05656",
        "score_aesthetic": 4.6341235241,
        "score_technical": 5.9604969211
    },
    {
        "average_score": 5.2953293102,
        "image_id": "DSC05510",
        "score_aesthetic": 4.9254714181,
        "score_technical": 5.6651872024
    },
    {
        "average_score": 5.294325739,
        "image_id": "DSC05320",
        "score_aesthetic": 5.3544661475,
        "score_technical": 5.2341853306
    },
    {
        "average_score": 5.2914717168,
        "image_id": "DSC05700",
        "score_aesthetic": 4.6201528774,
        "score_technical": 5.9627905563
    },
    {
        "average_score": 5.2880839161,
        "image_id": "DSC05473",
        "score_aesthetic": 4.5956017758,
        "score_technical": 5.9805660564
    },
    {
        "average_score": 5.2847563252,
        "image_id": "DSC05457",
        "score_aesthetic": 4.8936347067,
        "score_technical": 5.6758779436
    },
    {
        "average_score": 5.2825502068,
        "image_id": "DSC05646",
        "score_aesthetic": 4.8972031221,
        "score_technical": 5.6678972915
    },
    {
        "average_score": 5.278585631,
        "image_id": "DSC05567",
        "score_aesthetic": 4.6372339941,
        "score_technical": 5.9199372679
    },
    {
        "average_score": 5.2774541111,
        "image_id": "DSC05577",
        "score_aesthetic": 4.4809321133,
        "score_technical": 6.0739761088
    },
    {
        "average_score": 5.2769246022,
        "image_id": "DSC05601",
        "score_aesthetic": 4.7043637139,
        "score_technical": 5.8494854905
    },
    {
        "average_score": 5.2709526289,
        "image_id": "DSC05617",
        "score_aesthetic": 4.8161150998,
        "score_technical": 5.7257901579
    },
    {
        "average_score": 5.2692744819,
        "image_id": "DSC05662",
        "score_aesthetic": 4.7883275699,
        "score_technical": 5.750221394
    },
    {
        "average_score": 5.2679204512,
        "image_id": "DSC05328",
        "score_aesthetic": 5.0820763594,
        "score_technical": 5.4537645429
    },
    {
        "average_score": 5.2670235695,
        "image_id": "DSC05766",
        "score_aesthetic": 4.5067188304,
        "score_technical": 6.0273283087
    },
    {
        "average_score": 5.2616716439,
        "image_id": "DSC05671",
        "score_aesthetic": 4.6013801624,
        "score_technical": 5.9219631255
    },
    {
        "average_score": 5.258147775,
        "image_id": "DSC05778",
        "score_aesthetic": 4.7421774395,
        "score_technical": 5.7741181105
    },
    {
        "average_score": 5.2578352973,
        "image_id": "DSC05435",
        "score_aesthetic": 5.0768058494,
        "score_technical": 5.4388647452
    },
    {
        "average_score": 5.2529613145,
        "image_id": "DSC05619",
        "score_aesthetic": 4.1350133329,
        "score_technical": 6.370909296
    },
    {
        "average_score": 5.2520894013,
        "image_id": "DSC05781",
        "score_aesthetic": 4.5693958767,
        "score_technical": 5.934782926
    },
    {
        "average_score": 5.2507058574,
        "image_id": "DSC05444",
        "score_aesthetic": 4.5757568975,
        "score_technical": 5.9256548174
    },
    {
        "average_score": 5.2503705918,
        "image_id": "DSC05715",
        "score_aesthetic": 4.6035176291,
        "score_technical": 5.8972235546
    },
    {
        "average_score": 5.2410649167,
        "image_id": "DSC05590",
        "score_aesthetic": 4.6970820825,
        "score_technical": 5.7850477509
    },
    {
        "average_score": 5.2377423696,
        "image_id": "DSC05346",
        "score_aesthetic": 5.0777562811,
        "score_technical": 5.397728458
    },
    {
        "average_score": 5.2358553333,
        "image_id": "DSC05618",
        "score_aesthetic": 4.569202214,
        "score_technical": 5.9025084525
    },
    {
        "average_score": 5.2343775048,
        "image_id": "DSC05779",
        "score_aesthetic": 4.777535866,
        "score_technical": 5.6912191436
    },
    {
        "average_score": 5.2301771965,
        "image_id": "DSC05474",
        "score_aesthetic": 4.4672791451,
        "score_technical": 5.9930752479
    },
    {
        "average_score": 5.229460023,
        "image_id": "DSC05304",
        "score_aesthetic": 5.1444381693,
        "score_technical": 5.3144818768
    },
    {
        "average_score": 5.2261837124,
        "image_id": "DSC05725",
        "score_aesthetic": 4.6773307657,
        "score_technical": 5.7750366591
    },
    {
        "average_score": 5.2251915909,
        "image_id": "DSC05438",
        "score_aesthetic": 4.8707639977,
        "score_technical": 5.5796191841
    },
    {
        "average_score": 5.2246755316,
        "image_id": "DSC05381",
        "score_aesthetic": 5.0670840351,
        "score_technical": 5.3822670281
    },
    {
        "average_score": 5.2233145335,
        "image_id": "DSC05433",
        "score_aesthetic": 4.921859433,
        "score_technical": 5.524769634
    },
    {
        "average_score": 5.2210980534,
        "image_id": "DSC05699",
        "score_aesthetic": 4.5006691202,
        "score_technical": 5.9415269867
    },
    {
        "average_score": 5.2187536056,
        "image_id": "DSC05775",
        "score_aesthetic": 4.6690139016,
        "score_technical": 5.7684933096
    },
    {
        "average_score": 5.2179797699,
        "image_id": "DSC05372",
        "score_aesthetic": 4.9964901576,
        "score_technical": 5.4394693822
    },
    {
        "average_score": 5.2163715714,
        "image_id": "DSC05497",
        "score_aesthetic": 4.5431079786,
        "score_technical": 5.8896351643
    },
    {
        "average_score": 5.2131898437,
        "image_id": "DSC05714",
        "score_aesthetic": 4.7854553604,
        "score_technical": 5.6409243271
    },
    {
        "average_score": 5.2095313038,
        "image_id": "DSC05628",
        "score_aesthetic": 4.6391312897,
        "score_technical": 5.779931318
    },
    {
        "average_score": 5.2088361694,
        "image_id": "DSC05729",
        "score_aesthetic": 4.5054755183,
        "score_technical": 5.9121968206
    },
    {
        "average_score": 5.2012593457,
        "image_id": "DSC05648",
        "score_aesthetic": 4.8088830763,
        "score_technical": 5.593635615
    },
    {
        "average_score": 5.1985318405,
        "image_id": "DSC05591",
        "score_aesthetic": 4.7574505295,
        "score_technical": 5.6396131516
    },
    {
        "average_score": 5.1977407821,
        "image_id": "DSC05721",
        "score_aesthetic": 4.5407531411,
        "score_technical": 5.8547284231
    },
    {
        "average_score": 5.1927739168,
        "image_id": "DSC05719",
        "score_aesthetic": 4.4643458222,
        "score_technical": 5.9212020114
    },
    {
        "average_score": 5.1899453092,
        "image_id": "DSC05717",
        "score_aesthetic": 4.4217260791,
        "score_technical": 5.9581645392
    },
    {
        "average_score": 5.182868109,
        "image_id": "DSC05661",
        "score_aesthetic": 4.4856907667,
        "score_technical": 5.8800454512
    },
    {
        "average_score": 5.1828554991,
        "image_id": "DSC05683",
        "score_aesthetic": 4.6491464695,
        "score_technical": 5.7165645286
    },
    {
        "average_score": 5.1824991102,
        "image_id": "DSC05295",
        "score_aesthetic": 4.6381747043,
        "score_technical": 5.7268235162
    },
    {
        "average_score": 5.1754270382,
        "image_id": "DSC05698",
        "score_aesthetic": 5.118302293,
        "score_technical": 5.2325517833
    },
    {
        "average_score": 5.1746963614,
        "image_id": "DSC05744",
        "score_aesthetic": 4.2846260774,
        "score_technical": 6.0647666454
    },
    {
        "average_score": 5.1704997574,
        "image_id": "DSC05297",
        "score_aesthetic": 5.1760022127,
        "score_technical": 5.164997302
    },
    {
        "average_score": 5.168373793,
        "image_id": "DSC05327",
        "score_aesthetic": 5.197235375,
        "score_technical": 5.1395122111
    },
    {
        "average_score": 5.1673439907,
        "image_id": "DSC05676",
        "score_aesthetic": 4.588416062,
        "score_technical": 5.7462719195
    },
    {
        "average_score": 5.1636613605,
        "image_id": "DSC05727",
        "score_aesthetic": 4.6171874101,
        "score_technical": 5.7101353109
    },
    {
        "average_score": 5.1636090119,
        "image_id": "DSC05517",
        "score_aesthetic": 4.9339508839,
        "score_technical": 5.3932671398
    },
    {
        "average_score": 5.1576072532,
        "image_id": "DSC05599",
        "score_aesthetic": 4.7220993183,
        "score_technical": 5.5931151882
    },
    {
        "average_score": 5.1575028016,
        "image_id": "DSC05735",
        "score_aesthetic": 4.6355828483,
        "score_technical": 5.6794227548
    },
    {
        "average_score": 5.1564024425,
        "image_id": "DSC05638",
        "score_aesthetic": 4.8163713661,
        "score_technical": 5.4964335188
    },
    {
        "average_score": 5.1562649665,
        "image_id": "DSC05315",
        "score_aesthetic": 5.4520627897,
        "score_technical": 4.8604671434
    },
    {
        "average_score": 5.1455428635,
        "image_id": "DSC05570",
        "score_aesthetic": 4.4990611183,
        "score_technical": 5.7920246087
    },
    {
        "average_score": 5.1383852411,
        "image_id": "DSC05378",
        "score_aesthetic": 4.9627469875,
        "score_technical": 5.3140234947
    },
    {
        "average_score": 5.1369745719,
        "image_id": "DSC05452",
        "score_aesthetic": 5.0435653379,
        "score_technical": 5.2303838059
    },
    {
        "average_score": 5.1345045727,
        "image_id": "DSC05736",
        "score_aesthetic": 4.4592288699,
        "score_technical": 5.8097802754
    },
    {
        "average_score": 5.1333084077,
        "image_id": "DSC05318",
        "score_aesthetic": 5.378044592,
        "score_technical": 4.8885722235
    },
    {
        "average_score": 5.1325871563,
        "image_id": "DSC05400",
        "score_aesthetic": 4.829446381,
        "score_technical": 5.4357279316
    },
    {
        "average_score": 5.1312876744,
        "image_id": "DSC05359",
        "score_aesthetic": 4.6536012346,
        "score_technical": 5.6089741141
    },
    {
        "average_score": 5.1274310483,
        "image_id": "DSC05422",
        "score_aesthetic": 5.0536267203,
        "score_technical": 5.2012353763
    },
    {
        "average_score": 5.1214355784,
        "image_id": "DSC05737",
        "score_aesthetic": 4.5213063497,
        "score_technical": 5.721564807
    },
    {
        "average_score": 5.1194211362,
        "image_id": "DSC05767",
        "score_aesthetic": 4.3868601705,
        "score_technical": 5.8519821018
    },
    {
        "average_score": 5.118466932,
        "image_id": "DSC05312",
        "score_aesthetic": 5.25842902,
        "score_technical": 4.978504844
    },
    {
        "average_score": 5.1174983726,
        "image_id": "DSC05371",
        "score_aesthetic": 5.0502039062,
        "score_technical": 5.184792839
    },
    {
        "average_score": 5.1165657152,
        "image_id": "DSC05726",
        "score_aesthetic": 4.6444066198,
        "score_technical": 5.5887248106
    },
    {
        "average_score": 5.1089652149,
        "image_id": "DSC05321",
        "score_aesthetic": 5.1765521328,
        "score_technical": 5.0413782969
    },
    {
        "average_score": 5.1057213803,
        "image_id": "DSC05299",
        "score_aesthetic": 5.4015149841,
        "score_technical": 4.8099277765
    },
    {
        "average_score": 5.0967774053,
        "image_id": "DSC05592",
        "score_aesthetic": 4.7805340658,
        "score_technical": 5.4130207449
    },
    {
        "average_score": 5.0945060383,
        "image_id": "DSC05357",
        "score_aesthetic": 4.7475914246,
        "score_technical": 5.441420652
    },
    {
        "average_score": 5.092302162,
        "image_id": "DSC05504",
        "score_aesthetic": 4.0777884568,
        "score_technical": 6.1068158671
    },
    {
        "average_score": 5.0885292645,
        "image_id": "DSC05396",
        "score_aesthetic": 4.7062289527,
        "score_technical": 5.4708295763
    },
    {
        "average_score": 5.0865134484,
        "image_id": "DSC05353",
        "score_aesthetic": 5.2906368746,
        "score_technical": 4.8823900223
    },
    {
        "average_score": 5.0831993733,
        "image_id": "DSC05351",
        "score_aesthetic": 5.0340033077,
        "score_technical": 5.1323954388
    },
    {
        "average_score": 5.0784559243,
        "image_id": "DSC05713",
        "score_aesthetic": 4.5683518709,
        "score_technical": 5.5885599777
    },
    {
        "average_score": 5.0752964325,
        "image_id": "DSC05694",
        "score_aesthetic": 3.9634609908,
        "score_technical": 6.1871318743
    },
    {
        "average_score": 5.0750344914,
        "image_id": "DSC05673",
        "score_aesthetic": 4.446262672,
        "score_technical": 5.7038063109
    },
    {
        "average_score": 5.0734741376,
        "image_id": "DSC05718",
        "score_aesthetic": 4.5346594873,
        "score_technical": 5.612288788
    },
    {
        "average_score": 5.0718993182,
        "image_id": "DSC05374",
        "score_aesthetic": 5.0521116352,
        "score_technical": 5.0916870013
    },
    {
        "average_score": 5.066868394,
        "image_id": "DSC05447",
        "score_aesthetic": 3.9002260107,
        "score_technical": 6.2335107774
    },
    {
        "average_score": 5.0657146814,
        "image_id": "DSC05390",
        "score_aesthetic": 5.0002548581,
        "score_technical": 5.1311745048
    },
    {
        "average_score": 5.064729918,
        "image_id": "DSC05356",
        "score_aesthetic": 5.1147246737,
        "score_technical": 5.0147351623
    },
    {
        "average_score": 5.0629545519,
        "image_id": "DSC05649",
        "score_aesthetic": 4.6856478069,
        "score_technical": 5.4402612969
    },
    {
        "average_score": 5.0614928485,
        "image_id": "DSC05432",
        "score_aesthetic": 5.0042325768,
        "score_technical": 5.1187531203
    },
    {
        "average_score": 5.0605025583,
        "image_id": "DSC05499",
        "score_aesthetic": 4.4825358676,
        "score_technical": 5.638469249
    },
    {
        "average_score": 5.0561262688,
        "image_id": "DSC05426",
        "score_aesthetic": 5.0152675552,
        "score_technical": 5.0969849825
    },
    {
        "average_score": 5.053742812,
        "image_id": "DSC05434",
        "score_aesthetic": 5.077283809,
        "score_technical": 5.0302018151
    },
    {
        "average_score": 5.0515168232,
        "image_id": "DSC05511",
        "score_aesthetic": 4.3114762896,
        "score_technical": 5.7915573567
    },
    {
        "average_score": 5.0498031192,
        "image_id": "DSC05301",
        "score_aesthetic": 4.7061301694,
        "score_technical": 5.393476069
    },
    {
        "average_score": 5.0496307356,
        "image_id": "DSC05370",
        "score_aesthetic": 4.8416708941,
        "score_technical": 5.257590577
    },
    {
        "average_score": 5.0483191268,
        "image_id": "DSC05292",
        "score_aesthetic": 4.8454326529,
        "score_technical": 5.2512056008
    },
    {
        "average_score": 5.0442028243,
        "image_id": "DSC05350",
        "score_aesthetic": 4.7976888813,
        "score_technical": 5.2907167673
    },
    {
        "average_score": 5.0427130582,
        "image_id": "DSC05373",
        "score_aesthetic": 4.9079564084,
        "score_technical": 5.177469708
    },
    {
        "average_score": 5.0410249621,
        "image_id": "DSC05300",
        "score_aesthetic": 4.9604347006,
        "score_technical": 5.1216152236
    },
    {
        "average_score": 5.0402815378,
        "image_id": "DSC05352",
        "score_aesthetic": 5.0907681732,
        "score_technical": 4.9897949025
    },
    {
        "average_score": 5.0370849184,
        "image_id": "DSC05358",
        "score_aesthetic": 4.9815722777,
        "score_technical": 5.0925975591
    },
    {
        "average_score": 5.0286961375,
        "image_id": "DSC05340",
        "score_aesthetic": 5.2103649806,
        "score_technical": 4.8470272943
    },
    {
        "average_score": 5.0275788598,
        "image_id": "DSC05308",
        "score_aesthetic": 5.0432380349,
        "score_technical": 5.0119196847
    },
    {
        "average_score": 5.0263832351,
        "image_id": "DSC05450",
        "score_aesthetic": 4.7520353999,
        "score_technical": 5.3007310703
    },
    {
        "average_score": 5.0230793654,
        "image_id": "DSC05313",
        "score_aesthetic": 5.2254990113,
        "score_technical": 4.8206597194
    },
    {
        "average_score": 5.0121288104,
        "image_id": "DSC05323",
        "score_aesthetic": 5.073675571,
        "score_technical": 4.9505820498
    },
    {
        "average_score": 5.0101153901,
        "image_id": "DSC05424",
        "score_aesthetic": 4.9812181619,
        "score_technical": 5.0390126184
    },
    {
        "average_score": 5.0022801087,
        "image_id": "DSC05739",
        "score_aesthetic": 4.2312767116,
        "score_technical": 5.7732835058
    },
    {
        "average_score": 4.9988654584,
        "image_id": "DSC05362",
        "score_aesthetic": 5.2676801071,
        "score_technical": 4.7300508097
    },
    {
        "average_score": 4.9972601887,
        "image_id": "DSC05418",
        "score_aesthetic": 4.9116570912,
        "score_technical": 5.0828632861
    },
    {
        "average_score": 4.9953000976,
        "image_id": "DSC05782",
        "score_aesthetic": 4.4251656871,
        "score_technical": 5.565434508
    },
    {
        "average_score": 4.9928077577,
        "image_id": "DSC05403",
        "score_aesthetic": 4.929470801,
        "score_technical": 5.0561447144
    },
    {
        "average_score": 4.9924240758,
        "image_id": "DSC05379",
        "score_aesthetic": 4.7054234173,
        "score_technical": 5.2794247344
    },
    {
        "average_score": 4.9913665351,
        "image_id": "DSC05311",
        "score_aesthetic": 5.1604338803,
        "score_technical": 4.8222991899
    },
    {
        "average_score": 4.9898733423,
        "image_id": "DSC05377",
        "score_aesthetic": 4.8526226284,
        "score_technical": 5.1271240562
    },
    {
        "average_score": 4.9897201173,
        "image_id": "DSC05303",
        "score_aesthetic": 4.7983443365,
        "score_technical": 5.1810958982
    },
    {
        "average_score": 4.9882648857,
        "image_id": "DSC05317",
        "score_aesthetic": 5.2411947119,
        "score_technical": 4.7353350595
    },
    {
        "average_score": 4.9853544678,
        "image_id": "DSC05395",
        "score_aesthetic": 4.7837427026,
        "score_technical": 5.186966233
    },
    {
        "average_score": 4.9849437189,
        "image_id": "DSC05402",
        "score_aesthetic": 5.0704949597,
        "score_technical": 4.8993924782
    },
    {
        "average_score": 4.9797441185,
        "image_id": "DSC05605",
        "score_aesthetic": 4.7143092469,
        "score_technical": 5.2451789901
    },
    {
        "average_score": 4.9794229868,
        "image_id": "DSC05401",
        "score_aesthetic": 4.684330993,
        "score_technical": 5.2745149806
    },
    {
        "average_score": 4.9779967584,
        "image_id": "DSC05330",
        "score_aesthetic": 4.8801890672,
        "score_technical": 5.0758044496
    },
    {
        "average_score": 4.9762085865,
        "image_id": "DSC05360",
        "score_aesthetic": 4.5096794298,
        "score_technical": 5.4427377433
    },
    {
        "average_score": 4.9756346395,
        "image_id": "DSC05322",
        "score_aesthetic": 4.6739391486,
        "score_technical": 5.2773301303
    },
    {
        "average_score": 4.9734454262,
        "image_id": "DSC05314",
        "score_aesthetic": 5.3213549842,
        "score_technical": 4.6255358681
    },
    {
        "average_score": 4.971798301,
        "image_id": "DSC05389",
        "score_aesthetic": 4.8295013836,
        "score_technical": 5.1140952185
    },
    {
        "average_score": 4.9713241558,
        "image_id": "DSC05316",
        "score_aesthetic": 5.1313248775,
        "score_technical": 4.8113234341
    },
    {
        "average_score": 4.9710475428,
        "image_id": "DSC05514",
        "score_aesthetic": 4.9851797442,
        "score_technical": 4.9569153413
    },
    {
        "average_score": 4.968764932,
        "image_id": "DSC05388",
        "score_aesthetic": 4.9225910998,
        "score_technical": 5.0149387643
    },
    {
        "average_score": 4.9667191385,
        "image_id": "DSC05336",
        "score_aesthetic": 5.1034750921,
        "score_technical": 4.8299631849
    },
    {
        "average_score": 4.9650461216,
        "image_id": "DSC05427",
        "score_aesthetic": 4.8494010993,
        "score_technical": 5.0806911439
    },
    {
        "average_score": 4.9622799859,
        "image_id": "DSC05307",
        "score_aesthetic": 5.0957894237,
        "score_technical": 4.8287705481
    },
    {
        "average_score": 4.9601937149,
        "image_id": "DSC05421",
        "score_aesthetic": 4.9464371108,
        "score_technical": 4.973950319
    },
    {
        "average_score": 4.9595443613,
        "image_id": "DSC05764",
        "score_aesthetic": 4.3333072154,
        "score_technical": 5.5857815072
    },
    {
        "average_score": 4.9507303283,
        "image_id": "DSC05339",
        "score_aesthetic": 5.2139414044,
        "score_technical": 4.6875192523
    },
    {
        "average_score": 4.9444300438,
        "image_id": "DSC05354",
        "score_aesthetic": 5.1872373333,
        "score_technical": 4.7016227543
    },
    {
        "average_score": 4.9430278419,
        "image_id": "DSC05383",
        "score_aesthetic": 5.1897940394,
        "score_technical": 4.6962616444
    },
    {
        "average_score": 4.9410721159,
        "image_id": "DSC05319",
        "score_aesthetic": 5.2596499142,
        "score_technical": 4.6224943176
    },
    {
        "average_score": 4.940337685,
        "image_id": "DSC05408",
        "score_aesthetic": 4.8712552134,
        "score_technical": 5.0094201565
    },
    {
        "average_score": 4.9388450066,
        "image_id": "DSC05512",
        "score_aesthetic": 4.2434407283,
        "score_technical": 5.6342492849
    },
    {
        "average_score": 4.9361023053,
        "image_id": "DSC05711",
        "score_aesthetic": 4.3332565366,
        "score_technical": 5.538948074
    },
    {
        "average_score": 4.9311472357,
        "image_id": "DSC05309",
        "score_aesthetic": 4.8028709354,
        "score_technical": 5.0594235361
    },
    {
        "average_score": 4.9303603127,
        "image_id": "DSC05368",
        "score_aesthetic": 4.9264580144,
        "score_technical": 4.934262611
    },
    {
        "average_score": 4.9295200514,
        "image_id": "DSC05411",
        "score_aesthetic": 4.7600564425,
        "score_technical": 5.0989836603
    },
    {
        "average_score": 4.9218415679,
        "image_id": "DSC05375",
        "score_aesthetic": 4.5447585333,
        "score_technical": 5.2989246026
    },
    {
        "average_score": 4.9159119161,
        "image_id": "DSC05364",
        "score_aesthetic": 5.0028935957,
        "score_technical": 4.8289302364
    },
    {
        "average_score": 4.9152872619,
        "image_id": "DSC05332",
        "score_aesthetic": 4.7050505095,
        "score_technical": 5.1255240142
    },
    {
        "average_score": 4.9131387981,
        "image_id": "DSC05338",
        "score_aesthetic": 5.1680231442,
        "score_technical": 4.658254452
    },
    {
        "average_score": 4.9093986557,
        "image_id": "DSC05419",
        "score_aesthetic": 5.0965481165,
        "score_technical": 4.722249195
    },
    {
        "average_score": 4.9052688137,
        "image_id": "DSC05296",
        "score_aesthetic": 4.6727258847,
        "score_technical": 5.1378117427
    },
    {
        "average_score": 4.9009003474,
        "image_id": "DSC05324",
        "score_aesthetic": 4.8344937381,
        "score_technical": 4.9673069566
    },
    {
        "average_score": 4.8993621143,
        "image_id": "DSC05602",
        "score_aesthetic": 4.4976294059,
        "score_technical": 5.3010948226
    },
    {
        "average_score": 4.8926096104,
        "image_id": "DSC05361",
        "score_aesthetic": 4.5377080724,
        "score_technical": 5.2475111485
    },
    {
        "average_score": 4.8883427205,
        "image_id": "DSC05414",
        "score_aesthetic": 5.049794743,
        "score_technical": 4.7268906981
    },
    {
        "average_score": 4.8873923768,
        "image_id": "DSC05606",
        "score_aesthetic": 4.7027903223,
        "score_technical": 5.0719944313
    },
    {
        "average_score": 4.886469189,
        "image_id": "DSC05387",
        "score_aesthetic": 5.0341260283,
        "score_technical": 4.7388123497
    },
    {
        "average_score": 4.8857042839,
        "image_id": "DSC05716",
        "score_aesthetic": 3.9347322905,
        "score_technical": 5.8366762772
    },
    {
        "average_score": 4.8820743519,
        "image_id": "DSC05720",
        "score_aesthetic": 4.2622537677,
        "score_technical": 5.501894936
    },
    {
        "average_score": 4.8761004909,
        "image_id": "DSC05310",
        "score_aesthetic": 4.8164056374,
        "score_technical": 4.9357953444
    },
    {
        "average_score": 4.8673080271,
        "image_id": "DSC05399",
        "score_aesthetic": 4.7330713924,
        "score_technical": 5.0015446618
    },
    {
        "average_score": 4.8660730985,
        "image_id": "DSC05369",
        "score_aesthetic": 4.7288697595,
        "score_technical": 5.0032764375
    },
    {
        "average_score": 4.8647942441,
        "image_id": "DSC05331",
        "score_aesthetic": 4.6581053679,
        "score_technical": 5.0714831203
    },
    {
        "average_score": 4.8552529583,
        "image_id": "DSC05679",
        "score_aesthetic": 4.6199917709,
        "score_technical": 5.0905141458
    },
    {
        "average_score": 4.8551226386,
        "image_id": "DSC05407",
        "score_aesthetic": 4.907649185,
        "score_technical": 4.8025960922
    },
    {
        "average_score": 4.8507940569,
        "image_id": "DSC05380",
        "score_aesthetic": 5.081143965,
        "score_technical": 4.6204441488
    },
    {
        "average_score": 4.8497451147,
        "image_id": "DSC05337",
        "score_aesthetic": 4.8230126526,
        "score_technical": 4.8764775768
    },
    {
        "average_score": 4.8486914149,
        "image_id": "DSC05393",
        "score_aesthetic": 4.6303252947,
        "score_technical": 5.0670575351
    },
    {
        "average_score": 4.8431710935,
        "image_id": "DSC05604",
        "score_aesthetic": 4.5485239548,
        "score_technical": 5.1378182322
    },
    {
        "average_score": 4.8389447209,
        "image_id": "DSC05410",
        "score_aesthetic": 4.9683852099,
        "score_technical": 4.7095042318
    },
    {
        "average_score": 4.8379816718,
        "image_id": "DSC05607",
        "score_aesthetic": 4.7419484867,
        "score_technical": 4.9340148568
    },
    {
        "average_score": 4.835858664,
        "image_id": "DSC05345",
        "score_aesthetic": 5.0004036772,
        "score_technical": 4.6713136509
    },
    {
        "average_score": 4.8355878007,
        "image_id": "DSC05344",
        "score_aesthetic": 4.9609550485,
        "score_technical": 4.710220553
    },
    {
        "average_score": 4.8329366362,
        "image_id": "DSC05293",
        "score_aesthetic": 4.7455403564,
        "score_technical": 4.9203329161
    },
    {
        "average_score": 4.8247136324,
        "image_id": "DSC05397",
        "score_aesthetic": 4.9048004,
        "score_technical": 4.7446268648
    },
    {
        "average_score": 4.8241272815,
        "image_id": "DSC05326",
        "score_aesthetic": 4.7659223349,
        "score_technical": 4.8823322281
    },
    {
        "average_score": 4.820714491,
        "image_id": "DSC05384",
        "score_aesthetic": 5.006012832,
        "score_technical": 4.6354161501
    },
    {
        "average_score": 4.8163486704,
        "image_id": "DSC05593",
        "score_aesthetic": 4.5491482659,
        "score_technical": 5.0835490748
    },
    {
        "average_score": 4.8141770016,
        "image_id": "DSC05348",
        "score_aesthetic": 5.1467191405,
        "score_technical": 4.4816348627
    },
    {
        "average_score": 4.8080320557,
        "image_id": "DSC05341",
        "score_aesthetic": 5.4100967336,
        "score_technical": 4.2059673779
    },
    {
        "average_score": 4.7886396494,
        "image_id": "DSC05436",
        "score_aesthetic": 4.4758738646,
        "score_technical": 5.1014054343
    },
    {
        "average_score": 4.7843161071,
        "image_id": "DSC05456",
        "score_aesthetic": 4.0008646873,
        "score_technical": 5.567767527
    },
    {
        "average_score": 4.7802681422,
        "image_id": "DSC05423",
        "score_aesthetic": 4.5554265943,
        "score_technical": 5.0051096901
    },
    {
        "average_score": 4.7771929576,
        "image_id": "DSC05437",
        "score_aesthetic": 4.203804962,
        "score_technical": 5.3505809531
    },
    {
        "average_score": 4.7755920319,
        "image_id": "DSC05420",
        "score_aesthetic": 5.2451100257,
        "score_technical": 4.3060740381
    },
    {
        "average_score": 4.7722088671,
        "image_id": "DSC05428",
        "score_aesthetic": 4.5871629371,
        "score_technical": 4.9572547972
    },
    {
        "average_score": 4.7701152334,
        "image_id": "DSC05406",
        "score_aesthetic": 4.249611764,
        "score_technical": 5.2906187028
    },
    {
        "average_score": 4.7692551572,
        "image_id": "DSC05335",
        "score_aesthetic": 5.1426692896,
        "score_technical": 4.3958410248
    },
    {
        "average_score": 4.7689182935,
        "image_id": "DSC05325",
        "score_aesthetic": 4.7569402512,
        "score_technical": 4.7808963358
    },
    {
        "average_score": 4.7670967328,
        "image_id": "DSC05629",
        "score_aesthetic": 3.8718426881,
        "score_technical": 5.6623507775
    },
    {
        "average_score": 4.764987757,
        "image_id": "DSC05415",
        "score_aesthetic": 4.9987131609,
        "score_technical": 4.5312623531
    },
    {
        "average_score": 4.7541153981,
        "image_id": "DSC05376",
        "score_aesthetic": 4.9179586081,
        "score_technical": 4.5902721882
    },
    {
        "average_score": 4.7499247124,
        "image_id": "DSC05386",
        "score_aesthetic": 5.156397597,
        "score_technical": 4.3434518278
    },
    {
        "average_score": 4.7385063042,
        "image_id": "DSC05382",
        "score_aesthetic": 4.97281402,
        "score_technical": 4.5041985884
    },
    {
        "average_score": 4.735230051,
        "image_id": "DSC05416",
        "score_aesthetic": 4.5483326092,
        "score_technical": 4.9221274927
    },
    {
        "average_score": 4.7345683845,
        "image_id": "DSC05333",
        "score_aesthetic": 4.9731210713,
        "score_technical": 4.4960156977
    },
    {
        "average_score": 4.7299474118,
        "image_id": "DSC05392",
        "score_aesthetic": 4.302787068,
        "score_technical": 5.1571077555
    },
    {
        "average_score": 4.7223325007,
        "image_id": "DSC05343",
        "score_aesthetic": 5.0124004723,
        "score_technical": 4.4322645292
    },
    {
        "average_score": 4.7164786733,
        "image_id": "DSC05413",
        "score_aesthetic": 4.9074665783,
        "score_technical": 4.5254907683
    },
    {
        "average_score": 4.7117626864,
        "image_id": "DSC05365",
        "score_aesthetic": 4.9203860483,
        "score_technical": 4.5031393245
    },
    {
        "average_score": 4.7072138798,
        "image_id": "DSC05398",
        "score_aesthetic": 4.6539287293,
        "score_technical": 4.7604990304
    },
    {
        "average_score": 4.702945644,
        "image_id": "DSC05366",
        "score_aesthetic": 4.7796397841,
        "score_technical": 4.6262515038
    },
    {
        "average_score": 4.6997386751,
        "image_id": "DSC05603",
        "score_aesthetic": 4.2312705542,
        "score_technical": 5.1682067961
    },
    {
        "average_score": 4.6967839518,
        "image_id": "DSC05405",
        "score_aesthetic": 4.6544505122,
        "score_technical": 4.7391173914
    },
    {
        "average_score": 4.6921443959,
        "image_id": "DSC05329",
        "score_aesthetic": 4.641604897,
        "score_technical": 4.7426838949
    },
    {
        "average_score": 4.6871324666,
        "image_id": "DSC05391",
        "score_aesthetic": 4.4718764724,
        "score_technical": 4.9023884609
    },
    {
        "average_score": 4.6848041032,
        "image_id": "DSC05385",
        "score_aesthetic": 5.0096565836,
        "score_technical": 4.3599516228
    },
    {
        "average_score": 4.6845613525,
        "image_id": "DSC05394",
        "score_aesthetic": 4.4046057225,
        "score_technical": 4.9645169824
    },
    {
        "average_score": 4.6829596519,
        "image_id": "DSC05306",
        "score_aesthetic": 4.7547323077,
        "score_technical": 4.6111869961
    },
    {
        "average_score": 4.6827968518,
        "image_id": "DSC05417",
        "score_aesthetic": 4.850787582,
        "score_technical": 4.5148061216
    },
    {
        "average_score": 4.6820440179,
        "image_id": "DSC05363",
        "score_aesthetic": 4.7903774304,
        "score_technical": 4.5737106055
    },
    {
        "average_score": 4.6783732834,
        "image_id": "DSC05404",
        "score_aesthetic": 4.546419715,
        "score_technical": 4.8103268519
    },
    {
        "average_score": 4.6723662755,
        "image_id": "DSC05429",
        "score_aesthetic": 4.133345278,
        "score_technical": 5.2113872729
    },
    {
        "average_score": 4.6677630726,
        "image_id": "DSC05334",
        "score_aesthetic": 4.8369379631,
        "score_technical": 4.498588182
    },
    {
        "average_score": 4.6624096902,
        "image_id": "DSC05302",
        "score_aesthetic": 4.6386696864,
        "score_technical": 4.686149694
    },
    {
        "average_score": 4.6564054427,
        "image_id": "DSC05409",
        "score_aesthetic": 4.8902865822,
        "score_technical": 4.4225243032
    },
    {
        "average_score": 4.6476598343,
        "image_id": "DSC05412",
        "score_aesthetic": 4.8455410299,
        "score_technical": 4.4497786388
    },
    {
        "average_score": 4.6421067252,
        "image_id": "DSC05342",
        "score_aesthetic": 5.2700543149,
        "score_technical": 4.0141591355
    },
    {
        "average_score": 4.5960040745,
        "image_id": "DSC05367",
        "score_aesthetic": 4.8127009163,
        "score_technical": 4.3793072328
    },
    {
        "average_score": 4.5945597977,
        "image_id": "DSC05722",
        "score_aesthetic": 3.9606259123,
        "score_technical": 5.228493683
    },
    {
        "average_score": 4.5709282463,
        "image_id": "DSC05347",
        "score_aesthetic": 4.7107043127,
        "score_technical": 4.4311521798
    },
    {
        "average_score": 4.5359885141,
        "image_id": "DSC05430",
        "score_aesthetic": 4.3208111688,
        "score_technical": 4.7511658594
    },
    {
        "average_score": 4.5302029009,
        "image_id": "DSC05349",
        "score_aesthetic": 4.1555905154,
        "score_technical": 4.9048152864
    },
    {
        "average_score": 4.5296508282,
        "image_id": "DSC05305",
        "score_aesthetic": 4.2988232641,
        "score_technical": 4.7604783922
    },
    {
        "average_score": 4.5038943043,
        "image_id": "DSC05298",
        "score_aesthetic": 4.3638154219,
        "score_technical": 4.6439731866
    },
    {
        "average_score": 4.491318519,
        "image_id": "DSC05425",
        "score_aesthetic": 4.4695189348,
        "score_technical": 4.5131181031
    },
    {
        "average_score": 4.4909686638,
        "image_id": "DSC05723",
        "score_aesthetic": 3.7122227859,
        "score_technical": 5.2697145417
    },
    {
        "average_score": 4.2730295541,
        "image_id": "DSC05431",
        "score_aesthetic": 4.1357587937,
        "score_technical": 4.4103003144
    }
]

df = pd.read_json(json.dumps(myJson))

s = df['image_id']

arr = s.tolist()

dir_name = r"C:\Users\Justin\Downloads\mytransfer\CaptureOne"

full_names = [f'{el}.jpg' for i, el in enumerate(arr)]
my_arr = " ".join(f"'{el}'," for el in full_names)[:-1]
cmd = f"Set-Location -Path {dir_name}; " + my_arr + ' | ForEach-Object  -begin { $count=1 } -process { Rename-Item $_ -NewName "$count-$_"; $count++ }'
subprocess.run(["powershell", "-Command", cmd])


# %%
# converts to CSV
# pd.read_json(json.dumps(samples)).to_csv(f'scores_{weights_type}.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--base-model-name', help='CNN base model name', required=True)
    parser.add_argument('-aw', '--aesthetic-weights', help='path of weights file', required=True)
    parser.add_argument('-tw', '--technical-weights', help='path of weights file', required=True)
    parser.add_argument('-is', '--image-source', help='image directory or file', required=True)

    args = parser.parse_args()

    main(**args.__dict__)
