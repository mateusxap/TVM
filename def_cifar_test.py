import tensorflow as tf
from resnet50 import ResNet50
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# Загрузка и предобработка данных CIFAR-10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Создание и компиляция модели ResNet50
model = ResNet50(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=(32, 32, 3),
    pooling=None,
    classes=10
)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Создание и конфигурация TPU
resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.experimental.TPUStrategy(resolver)

# Обучение модели на TPU
with strategy.scope():
    model.fit(x_train, y_train,
              validation_data=(x_test, y_test),
              epochs=100,
              batch_size=128)
# with tf.device("/GPU:0"):  # Укажите соответствующее имя GPU устройства
#     model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100)

# Сохранение модели
model.save("resnet50_cifar10.h5")
