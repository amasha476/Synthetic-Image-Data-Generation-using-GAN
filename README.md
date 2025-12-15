#  Tulip Flower Image Generation using GAN (PyTorch)

##  Project Overview
This project implements a **Deep Convolutional Generative Adversarial Network (DCGAN)** to generate realistic images of **tulip flowers**.  
The model is trained using a subset of the **Flowers Dataset from Kaggle**, focusing exclusively on tulip images.  
The implementation is done in **PyTorch** and trained using **Google Colab with GPU acceleration**.

---

##  Objective
- To generate high-quality synthetic images of tulip flowers
- To understand and implement GAN training dynamics using convolutional networks
- To explore image generation with limited training data

---

##  Dataset
- **Source:** Kaggle Flowers Dataset  
- **Category Used:** Tulip flowers only  
- **Image Format:** `.jpg`  
- **Preprocessing:**
  - Resize to `64 × 64`
  - Center crop
  - Normalize pixel values to `[-1, 1]`

---

##  Model Architecture

### Generator
- Input: Random noise vector (`z_dim = 100`)
- Uses **ConvTranspose2D layers**
- Batch Normalization and ReLU activations
- Final activation: **Tanh**
- Output: `64 × 64 × 3` RGB image

### Discriminator
- Input: `64 × 64 × 3` image
- Uses **Conv2D layers**
- LeakyReLU activations
- Batch Normalization
- Final activation: **Sigmoid**
- Output: Probability of image being real or fake

---

##  Training Details
- **Loss Function:** Binary Cross Entropy Loss (BCELoss)
- **Optimizers:** Adam
- **Learning Rates:**
  - Generator: `0.0002`
  - Discriminator: `0.0001`
- **Batch Size:** `32`
- **Epochs:** `400`
- **Label Smoothing:** Applied for stable training

---

##  Generated Outputs
- Images are generated and visualized after each epoch
- Uses `torchvision.utils.make_grid` for image grids
- Outputs show progressive improvement in tulip structure and color distribution

---

##  Technologies Used
- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Google Colab

---

The generated images at each epoch is saved.

The resuts of the final epochs are as follows

<img width="676" height="448" alt="image" src="https://github.com/user-attachments/assets/d4b414b5-5976-49ea-b0ee-a2fa4e96c728" />

<img width="685" height="426" alt="image" src="https://github.com/user-attachments/assets/7a1bb89a-4981-470d-9b10-d493bc32c43f" />

<img width="677" height="431" alt="image" src="https://github.com/user-attachments/assets/54d3bd69-068c-4a06-b3a1-7fa6e55ac845" />

<img width="707" height="430" alt="image" src="https://github.com/user-attachments/assets/5c604888-3bea-492c-b466-fa8e3a0f7180" />

<img width="675" height="422" alt="image" src="https://github.com/user-attachments/assets/c977259e-060c-4554-a72b-bc0ba2fc8281" />


##  Training Visualization with TensorBoard

TensorBoard is used to monitor the training process in real time, providing both quantitative and qualitative insights.

---

###  Loss Visualization
During each training step, generator and discriminator losses are logged:

```python
writer.add_scalar("Loss/Discriminator", d_loss.item(), global_step)
writer.add_scalar("Loss/Generator", g_loss.item(), global_step)
```

###  Benefits of Loss Tracking
This enables:
- Monitoring convergence behavior
- Detecting training instability or mode collapse
- Comparing Generator and Discriminator learning dynamics

---

###  Generated Image Visualization
At the end of each epoch, sample images are generated and logged to TensorBoard:

```python
writer.add_image("Generated Images", grid, epoch)

```

<img width="1735" height="702" alt="image" src="https://github.com/user-attachments/assets/453581a6-3528-49c2-a9f8-60b9782f5b5f" />

<img width="991" height="592" alt="image" src="https://github.com/user-attachments/assets/50d13617-e1a8-4600-8fe8-55eb86c68467" />


<img width="1080" height="467" alt="image" src="https://github.com/user-attachments/assets/353b848d-e76d-4a1d-bdeb-88d313b8bb04" />

<img width="1077" height="466" alt="image" src="https://github.com/user-attachments/assets/e390769f-3eb4-47eb-bc43-b8c945e4a7da" />

###  Discriminator Loss Analysis 

- The graph shows how the Discriminator loss changes during training
- The loss starts relatively high, indicating initial difficulty in distinguishing real and fake images
- A steady decrease in loss shows that the Discriminator learns useful features from the data
- After several iterations, the loss stabilizes around a lower value
- Small fluctuations are normal and indicate ongoing competition with the Generator
- No sudden collapse or divergence is observed during training
- Stable loss suggests the Discriminator is neither too weak nor overpowering the Generator
- Overall, the Discriminator demonstrates consistent and healthy learning behavior


###  Generator Loss Analysis 

- The graph shows how the Generator loss changes during training
- The loss goes up and down a lot, which is normal for GANs
- These ups and downs mean the Generator and Discriminator are learning from each other
- The loss stays within a reasonable range and does not crash suddenly
- The smoothed line shows that training is generally stable over time
- No sharp drop or explosion in loss means the model is not collapsing
- This behavior indicates the Generator is gradually improving
- Overall, the training process is healthy and working as expected



##  Limitations

- **Limited Dataset Size**  
  The model is trained using only tulip images from the Flowers dataset, which restricts diversity and may lead to overfitting.

- **Single-Class Generation**  
  The current GAN generates only tulip flowers and cannot produce other flower categories or control visual attributes.

- **Training Instability**  
  Despite using label smoothing, GAN training remains sensitive to hyperparameters and may suffer from mode collapse.

- **Qualitative Evaluation Only**  
  Model performance is primarily assessed visually, without quantitative metrics such as FID or IS.

- **High Computational Cost**  
  Training for 400 epochs requires significant GPU resources and time.

- **Limited Image Resolution**  
  Generated images are restricted to `64 × 64` resolution, which may lack fine-grained details.

---

##  Future Enhancements

- **Conditional GAN (cGAN)**  
  Extend the model to generate multiple flower categories by conditioning on class labels.

- **Wasserstein GAN with Gradient Penalty (WGAN-GP)**  
  Improve training stability and reduce mode collapse.

- **Higher Resolution Image Generation**  
  Scale the model to generate higher-resolution images (e.g., `128 × 128` or `256 × 256`).

- **Data Augmentation**  
  Apply transformations such as rotation, flipping, and color jittering to increase dataset diversity.

- **Quantitative Evaluation Metrics**  
  Integrate metrics like Fréchet Inception Distance (FID) and Inception Score (IS) for objective evaluation.

- **Model Checkpointing and Image Saving**  
  Save model checkpoints and generated images at regular intervals for better analysis and reproducibility.

- **Advanced Architectures**  
  Explore StyleGAN or Progressive GAN for improved image quality and realism.

- **Hyperparameter Optimization**  
  Systematically tune learning rates, batch size, and network depth to enhance performance.







