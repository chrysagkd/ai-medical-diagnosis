import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K

# Συνάρτηση για την εξαγωγή υπο-όγκου
def get_sub_volume(image, label, output_x=60, output_y=60, output_z=16):
    """
    Εξάγει ένα υπο-όγκο από την εικόνα και την ετικέτα.
    Προσθέτουμε ελέγχους για να διασφαλίσουμε ότι οι διαστάσεις είναι έγκυρες.
    """
    # Έλεγχος για να διασφαλίσουμε ότι οι διαστάσεις της εικόνας είναι μεγαλύτερες από τις απαιτούμενες εξόδους
    if image.shape[0] <= output_x or image.shape[1] <= output_y or image.shape[2] <= output_z:
        raise ValueError(f"Η διάσταση της εικόνας είναι μικρότερη από την απαιτούμενη διάσταση εξόδου: {image.shape} < ({output_x}, {output_y}, {output_z})")

    # Επιλέγουμε ένα τυχαίο σημείο εκκίνησης για την εξαγωγή του υπο-όγκου
    start_x = np.random.randint(0, image.shape[0] - output_x + 1)
    start_y = np.random.randint(0, image.shape[1] - output_y + 1)
    start_z = np.random.randint(0, image.shape[2] - output_z + 1)

    sub_image = image[start_x:start_x + output_x, start_y:start_y + output_y, start_z:start_z + output_z]
    sub_label = label[start_x:start_x + output_x, start_y:start_y + output_y, start_z:start_z + output_z]

    return sub_image, sub_label


# Συνάρτηση για την κανονικοποίηση
def normalize_image(image):
    """
    Κανονικοποίηση της εικόνας σε εύρος [0, 1]
    """
    return (image - np.min(image)) / (np.max(image) - np.min(image))


# Συνάρτηση για την εμφάνιση εικόνας και ετικέτας
def display_image_and_label(image, label, title="Image and Label"):
    """
    Εμφανίζει την εικόνα και την ετικέτα.
    """
    mid_z = image.shape[2] // 2
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image[:, :, mid_z], cmap='gray')  # Εμφάνιση του πρώτου καναλιού για το μεσαίο στρώμα
    axes[0].set_title("Image")
    axes[1].imshow(label[:, :, mid_z], cmap='jet', alpha=0.5)  # Εμφάνιση της ετικέτας με διαφάνεια
    axes[1].set_title("Label")
    plt.suptitle(title)
    plt.show()


# Συνάρτηση για τον υπολογισμό του Dice coefficient
def dice_coefficient(y_true, y_pred):
    """
    Υπολογισμός του Dice coefficient μεταξύ της πραγματικής ετικέτας και της προβλεπόμενης ετικέτας
    """
    intersection = np.sum(y_true * y_pred)
    return 2 * intersection / (np.sum(y_true) + np.sum(y_pred))


# Παράδειγμα δεδομένων εικόνας και ετικέτας
image = np.random.rand(100, 100, 30)  # Δημιουργία τυχαίας εικόνας (π.χ., 100x100x30)
label = np.random.randint(0, 2, size=(100, 100, 30))  # Δημιουργία τυχαίας ετικέτας (π.χ., 0 ή 1)

# Εξαγωγή υπο-όγκου
sub_image, sub_label = get_sub_volume(image, label)

# Κανονικοποίηση
sub_image = normalize_image(sub_image)

# Υπολογισμός Dice coefficient
dice = dice_coefficient(sub_label, sub_image)
print(f'Dice Coefficient: {dice}')

# Εμφάνιση εικόνας και ετικέτας
print(f'Διαστάσεις εικόνας: {sub_image.shape}')
print(f'Διαστάσεις ετικέτας: {sub_label.shape}')
display_image_and_label(sub_image, sub_label, "Sub-Volume Image and Label")
