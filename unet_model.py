# Εισαγωγή απαραίτητων βιβλιοθηκών από το TensorFlow για τη δημιουργία του μοντέλου U-Net
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Activation, concatenate
from tensorflow.keras.optimizers import Adam

# Ορισμός του σχήματος δεδομένων της εικόνας (με κανάλια πρώτα) για το μοντέλο μας
tf.keras.backend.set_image_data_format("channels_first")

# Δημιουργία του input layer για την υποδοχή των δεδομένων εικόνας
input_layer = Input(shape=(4, 160, 160, 16))  # 4 κανάλια (π.χ. εικόνα με 4 χαρακτηριστικά), 160x160x16 για ύψος, πλάτος και μήκος

# Κάτω διαδρομή (Contracting Path) - Εδώ μειώνεται το μέγεθος της εικόνας και αυξάνεται ο αριθμός των φίλτρων
# Επίπεδο 0 - Πρώτη Σύζευξη 3D Συγκλίνασης (Conv3D)
down_depth_0_layer_0 = Conv3D(filters=32, kernel_size=(3, 3, 3), padding='same', strides=(1, 1, 1))(input_layer)
down_depth_0_layer_0 = Activation('relu')(down_depth_0_layer_0)

# Επίπεδο 0 - Δεύτερη Σύζευξη 3D Συγκλίνασης
down_depth_0_layer_1 = Conv3D(filters=64, kernel_size=(3, 3, 3), padding='same', strides=(1, 1, 1))(down_depth_0_layer_0)
down_depth_0_layer_1 = Activation('relu')(down_depth_0_layer_1)

# Εφαρμογή MaxPooling για να μειώσουμε το μέγεθος των εικόνων
down_depth_0_layer_pool = MaxPooling3D(pool_size=(2, 2, 2))(down_depth_0_layer_1)

# Επίπεδο 1 - Πρώτη Σύζευξη 3D Συγκλίνασης
down_depth_1_layer_0 = Conv3D(filters=64, kernel_size=(3, 3, 3), padding='same', strides=(1, 1, 1))(down_depth_0_layer_pool)
down_depth_1_layer_0 = Activation('relu')(down_depth_1_layer_0)

# Επίπεδο 1 - Δεύτερη Σύζευξη 3D Συγκλίνασης
down_depth_1_layer_1 = Conv3D(filters=128, kernel_size=(3, 3, 3), padding='same', strides=(1, 1, 1))(down_depth_1_layer_0)
down_depth_1_layer_1 = Activation('relu')(down_depth_1_layer_1)

# Ανάδειξη (Expanding Path) - Ανάκτηση διαστάσεων και συνένωση
# Ανάδειξη του χαρακτηριστικού με Upsampling
up_depth_0_layer_0 = UpSampling3D(size=(2, 2, 2))(down_depth_1_layer_1)

# Συνένωση του upsampling με το προηγούμενο downsampling (από το επίπεδο 0)
up_depth_1_concat = concatenate([up_depth_0_layer_0, down_depth_0_layer_1], axis=1)

# Εφαρμογή του up-convolution για την ανάκτηση του χαρακτηριστικού
up_depth_1_layer_1 = Conv3D(filters=64, kernel_size=(3, 3, 3), padding='same', strides=(1, 1, 1))(up_depth_1_concat)
up_depth_1_layer_1 = Activation('relu')(up_depth_1_layer_1)

# Ένα ακόμη up-convolution για την αναγνώριση χαρακτηριστικών
up_depth_1_layer_2 = Conv3D(filters=64, kernel_size=(3, 3, 3), padding='same', strides=(1, 1, 1))(up_depth_1_layer_1)
up_depth_1_layer_2 = Activation('relu')(up_depth_1_layer_2)

# Τελική Σύζευξη 3D με 3 φίλτρα για τις κατηγορίες
final_conv = Conv3D(filters=3, kernel_size=(1, 1, 1), padding='valid', strides=(1, 1, 1))(up_depth_1_layer_2)

# Εφαρμογή της τελικής ενεργοποίησης για τις κατηγορίες
final_activation = Activation('sigmoid')(final_conv)

# Δημιουργία του μοντέλου
model = Model(inputs=input_layer, outputs=final_activation)

# Συμπίεση του μοντέλου με optimizer Adam και χρήση categorical_crossentropy ως συνάρτηση απώλειας
model.compile(optimizer=Adam(learning_rate=0.00001),
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])

# Εκτύπωση περίληψης του μοντέλου για να επιβεβαιώσουμε ότι η αρχιτεκτονική είναι σωστή
model.summary()
