from keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.metrics
import tensorflow.keras.initializers
import tensorflow as tf

# Hiperparametros 
funcaoAtivacao = 'relu'
funcaoAtivacaoUltimaCamada = 'softmax'
optimizerFunction = 'adam'
numeroDeEpocas = 500
lossFunction = 'categorical_crossentropy'
batchSize = 16


model = tensorflow.keras.applications.resnet50.ResNet50(
    include_top = True,
    weights = None,
    input_tensor=None,
    input_shape=(224, 224, 3),
    pooling='max',
    classes=4,
)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['categorical_accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), tf.keras.metrics.AUC()])


# Carregando os dados

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True,
                                   rotation_range = 45,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   validation_split = 0.25)

training_set = train_datagen.flow_from_directory('dataset',
                                                 target_size = (224, 224),
                                                 batch_size = batchSize,
                                                 class_mode = 'categorical',
                                                 subset='training')

test_set = train_datagen.flow_from_directory('dataset',
                                            target_size = (224, 224),
                                            batch_size = batchSize,
                                            class_mode = 'categorical',
                                            subset='validation')


# Executando o treinamento inserindo os dados no modelo

history = model.fit(training_set,
                         epochs = numeroDeEpocas,
                         validation_data = test_set)


model.save('save.h5')



