#!/usr/bin/env python
# coding: utf-8

# In[1]:


from collections import namedtuple
from dataloader import DataLoader
from model import FastSRGAN
import tensorflow as tf
import zipfile
import os



# download data
os.environ['KAGGLE_USERNAME'] = "***" # username from the json file
os.environ['KAGGLE_KEY'] = "******" # key from the json file
get_ipython().system('kaggle datasets download -d joe1995/div2k-dataset')

zip__file = zipfile.ZipFile("/content/div2k-dataset.zip")
zip__file.extractall()


# In[5]:


image_dir = "/content/DIV2K_train_HR/DIV2K_train_HR" # path to high resolution
batch_size = 8
epochs    = 10
hr_size   = 384 # low resolution input size
lr        = 1e-4 
save_iter = 200

vars = namedtuple('vars', ['hr_size', 'lr' , "epochs" , "batch_size" , "save_iter"])
args = vars(384 , lr ,epochs , batch_size , save_iter )



@tf.function
def pretrain_step(model, x, y):
    """
        x: The low resolution image tensor.
        y: The high resolution image tensor.
    """
    
    with tf.GradientTape() as tape:
        fake_hr  = model.generator(x)
        loss_mse = tf.keras.losses.MeanSquaredError()(y, fake_hr)

    grads = tape.gradient(loss_mse, model.generator.trainable_variables)
    model.gen_optimizer.apply_gradients(zip(grads, model.generator.trainable_variables))

    return loss_mse



def pretrain_generator(model, dataset, writer):
    """
        dataset: A tf dataset object of low and high res images to pretrain over.
        writer: A summary writer object.
    """
    
    with writer.as_default():
        iteration = 0
        for _ in range(1):
            for x, y in dataset:
                loss = pretrain_step(model, x, y)
                if iteration % 20 == 0:
                    tf.summary.scalar('MSE Loss', loss, step=tf.cast(iteration, tf.int64))
                    writer.flush()
                iteration += 1

                
                
@tf.function
def train_step(model, x, y):
    """Single train step function for the SRGAN.
    
        x: The low resolution input image.
        y: The desired high resolution output image.
    Returns:
        d_loss: The mean loss of the discriminator.
    """
    # Label smoothing for better gradient flow
    valid = tf.ones((x.shape[0],) + model.disc_patch)
    fake = tf.zeros((x.shape[0],) + model.disc_patch)

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # From low res. image generate high res. version
        fake_hr = model.generator(x)

        # Train the discriminators (original images = real / generated = Fake)
        valid_prediction = model.discriminator(y)
        fake_prediction = model.discriminator(fake_hr)

        # Generator loss
        content_loss = model.content_loss(y, fake_hr)
        adv_loss = 1e-3 * tf.keras.losses.BinaryCrossentropy()(valid, fake_prediction)
        mse_loss = tf.keras.losses.MeanSquaredError()(y, fake_hr)
        perceptual_loss = content_loss + adv_loss + mse_loss

        # Discriminator loss
        valid_loss = tf.keras.losses.BinaryCrossentropy()(valid, valid_prediction)
        fake_loss = tf.keras.losses.BinaryCrossentropy()(fake, fake_prediction)
        d_loss = tf.add(valid_loss, fake_loss)

    # Backprop on Generator
    gen_grads = gen_tape.gradient(perceptual_loss, model.generator.trainable_variables)
    model.gen_optimizer.apply_gradients(zip(gen_grads, model.generator.trainable_variables))

    # Backprop on Discriminator
    disc_grads = disc_tape.gradient(d_loss, model.discriminator.trainable_variables)
    model.disc_optimizer.apply_gradients(zip(disc_grads, model.discriminator.trainable_variables))

    return d_loss, adv_loss, content_loss, mse_loss




def train(model, dataset, log_iter, writer):
    """
    Function that defines a single training step for the SR-GAN.
    """
    with writer.as_default():
        # Iterate over dataset
        for x, y in dataset:
            disc_loss, adv_loss, content_loss, mse_loss = train_step(model, x, y)
            # Log tensorboard summaries if log iteration is reached.
            if model.iterations % log_iter == 0:
                tf.summary.scalar('Adversarial Loss', adv_loss, step=model.iterations)
                tf.summary.scalar('Content Loss', content_loss, step=model.iterations)
                tf.summary.scalar('MSE Loss', mse_loss, step=model.iterations)
                tf.summary.scalar('Discriminator Loss', disc_loss, step=model.iterations)
                tf.summary.image('Low Res', tf.cast(255 * x, tf.uint8), step=model.iterations)
                tf.summary.image('High Res', tf.cast(255 * (y + 1.0) / 2.0, tf.uint8), step=model.iterations)
                tf.summary.image('Generated', tf.cast(255 * (model.generator.predict(x) + 1.0) / 2.0, tf.uint8),
                                 step=model.iterations)
                model.generator.save('models/generator.h5')
                model.discriminator.save('models/discriminator.h5')
                writer.flush()

                print("content_loss : " , content_loss , " MSE Loss : " , mse_loss)
            model.iterations += 1
            
           


# In[6]:


# create directory for saving trained models.
if not os.path.exists('models'):
    os.makedirs('models')

# Create the tensorflow dataset.
ds = DataLoader(image_dir, hr_size).dataset(batch_size)

# Initialize the GAN object.
gan = FastSRGAN(args)

# Define the directory for saving pretrainig loss tensorboard summary.
pretrain_summary_writer = tf.summary.create_file_writer('logs/pretrain')

# Run pre-training.
pretrain_generator(gan, ds, pretrain_summary_writer)

# Define the directory for saving the SRGAN training tensorbaord summary.
train_summary_writer = tf.summary.create_file_writer('logs/train')


# In[9]:





# In[ ]:


# Run training.
for _ in range(epochs):
    train(gan, ds, args.save_iter, train_summary_writer)


# In[ ]:





# In[ ]:





# In[ ]:




