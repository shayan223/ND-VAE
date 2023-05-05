

'''
Based on the following tutorial by Trusted-AI: 
https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/examples/get_started_pytorch.py

'''

#from test_adversarial import BATCH_SIZE
#from types import NoneType
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from torchvision import datasets, transforms

from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier
from art.utils import load_mnist

from sklearn.model_selection import train_test_split

from tqdm import tqdm
import os
from pathlib import Path
import matplotlib.pyplot as plt

from models.defense_vae import Defence_VAE
from models.vector_guide import Guide_Net
from data_utils import Generate_attack_data, ImgDataset, ImgDataset_Basic, generate_adv_datasets, ImgDataset_guided
from models.basic_vae_mnist import VAE
from NVAE_defense_training import display_NVAE_output

# Find and use GPU's if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def vae_training_guided(test_name,vae_model,train_dataset,guided_train_dataset,batch_size,n_epochs,lr,pre_trained=False):

    model_path = '../model_parameters/'+ test_name +'/vae_guided.pth'

    print('Training...')


    if(pre_trained == True):
        vae_model.load_state_dict(torch.load(model_path))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,batch_size=batch_size, shuffle=True)


    guided_train_loader = torch.utils.data.DataLoader(
        guided_train_dataset,batch_size=batch_size, shuffle=True)

        # Define custom Loss function
    def defence_vae_loss(recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        #return BCE + 0.01* KLD, BCE, KLD
        return BCE + KLD, BCE, KLD


    #Initialize vae model weights to be between 0.08 and -0.08 to increase network stability
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.uniform_(m.weight,-0.08,0.08)
            if(m.bias is not None):
                m.bias.data.fill_(0.01)
    #vae_model.logvar_layer.weight.data.uniform_(-.08,0.08)
    #vae_model = VAE()


    
    vae_model.apply(init_weights)    
    vae_model.to(device)

    guide_model = Guide_Net(channels_in=vae_model.encoding_dim)
    guide_model.to(device)

    optimizer = torch.optim.Adam(vae_model.parameters(), lr=lr)
    guide_optimizer = torch.optim.Adam(guide_model.parameters(),lr=lr)
    guide_loss = torch.nn.CrossEntropyLoss() 
    #Set guide model to evaluation mode while vae is training
    guide_model.eval()
    print("###### Beginning VAE Training... ######")


    for epoch in range(n_epochs):

        guide_model_losses = []
        print("VAE training...")
        for idx,sample in enumerate(tqdm(train_loader), 0):
            x_adv = sample['x_adv']
            #x_adv = x_adv.permute(0,2,1,3).to(device)
            x_adv = x_adv.to(device)
            x_orig = sample['x_orig']
            #x_orig = x_orig.permute(0,2,1,3).to(device)
            x_orig = x_orig.to(device)
           
            # Feeding a batch of images into the network to obtain the output image, mu, and logVar

            out, mu, logVar = vae_model(x_adv)

            # The loss is the BCE loss combined with the KL divergence to ensure the distribution is learnt
            #kl_divergence = -0.5 * torch.sum(1 + logVar - mu.pow(2) - logVar.exp())
            #loss = F.binary_cross_entropy(out, imgs, size_average=False) + kl_divergence

            cur_loss, _, _ = defence_vae_loss(out,x_orig,mu,logVar)

            with torch.no_grad():
                #50% chance to test benign batch, 50% to test adversarial
                adv_or_benign = np.random.randint(2)
                #Labeled as: 1 for adversarial, 0 for benign
                if(adv_or_benign == 0):
                    mu, logvar = vae_model.encoder(x_adv)
                    target = torch.ones(len(x_adv),dtype=torch.long).to(device)
                else:
                    mu, logvar = vae_model.encoder(x_orig)
                    target = torch.zeros(len(x_adv),dtype=torch.long).to(device)

                z = vae_model.reparameterize(mu, logvar)
                #z = torch.unsqueeze(z,1)
                #z = torch.unsqueeze(z, )
                '''
                #Apply softmax because that is usually left to the loss function
                guide_output = torch.argmax(F.softmax(guide_model(z)),dim=1)
                output_count = len(guide_output)
                #guided_loss = guide_output.eq(target.data.view_as(guide_output))
                guided_loss = guide_output.eq(target)
                guided_loss = torch.log(guided_loss.long()).sum()
                guided_loss /= output_count
                '''
                guide_output = guide_model(z)
                guided_loss = guide_loss(guide_output,target)
                #Vae loss is 1 - guide error
                guided_loss = torch.sub(guided_loss,1)
                output_count = len(guide_output)
                guided_loss = guided_loss.sum()/output_count

            guide_model_losses.append(guided_loss)
            cur_loss += guided_loss

            ''' Un comment to view inputs and labels
            plt.subplot(121)
            plt.imshow(np.transpose(input[0].cpu().detach().numpy(), [1,2,0]))
            plt.subplot(122)
            plt.title(y_train_labels[i])
            plt.imshow(np.transpose(y_fgsm[i][0].cpu().detach().numpy(), [1,2,0]))
            plt.show()
            plt.clf()
            '''

            # Backpropagation based on the loss
            optimizer.zero_grad()
            cur_loss.backward()
            optimizer.step()

        print("VAE guide loss: ", sum(guide_model_losses)/len(guide_model_losses))
        if(epoch%2 == 0):
            #Switch which model is in eval and training
            vae_model.eval()
            guide_model.train()
            #Run an epoch of the guide net training
            print("Guide training...")
            for idx,sample in enumerate(tqdm(guided_train_loader), 0):
                #A random selection of benign and adversarial images (labeled as such) are given to the generator
                x = sample['x']
                label = sample['label']
                x = x.to(device)
                label = label.to(device)

                mu, logvar = vae_model.encoder(x)
                z = vae_model.reparameterize(mu, logvar)

                out = guide_model(z)

                loss = guide_loss(out,label)
                guide_optimizer.zero_grad()
                loss.backward()
                guide_optimizer.step()

        vae_model.train()
        guide_model.eval()


        print('Epoch {}: Loss {}'.format(epoch, cur_loss))
    
    print("###### VAE Training Complete! ######")

    print("############ Saving Models... ##########")

    torch.save(vae_model.state_dict(), '../model_parameters/'+ test_name +'/vae.pth')

    print("################   Done.   ##############")

    return vae_model



def display_vae_output(vae_model,test_data,location="defense_vae_samples"):

    Path('../results/'+location).mkdir(parents=True, exist_ok=True)

    test_loader = torch.utils.data.DataLoader(test_data,batch_size=1)

    sample_count = 10
    image_count = 1
    vae_model.eval()
    with torch.no_grad():
        for idx, sample in enumerate(test_loader):
            benign_imgs = sample['x_orig']
            benign_imgs = benign_imgs.to(device)
            benign_imgs = np.transpose(benign_imgs[0].cpu().numpy(), [1,0,2])
            plt.suptitle(sample['label'].item())
            plt.subplot(131)
            plt.imshow(np.squeeze(benign_imgs))
            plt.title("Benign")
            imgs = sample['x_adv']
            imgs = imgs.to(device)
            img = np.transpose(imgs[0].cpu().numpy(), [1,0,2])
            plt.subplot(132)
            plt.imshow(np.squeeze(img))
            plt.title("Adversarial")
            #out, mu, logVAR = vae_model(imgs.permute(0,2,1,3))
            out, mu, logVAR = vae_model(imgs)
            outimg = np.transpose(out[0].cpu().numpy(), [1,0,2])
            plt.subplot(133)
            plt.imshow(np.squeeze(outimg))
            plt.title("Defence VAE")
            plt.savefig('../results/'+location+'/defense_vae_example'+str(image_count)+'.png')
            plt.clf()
            image_count += 1
            if (image_count > sample_count):
                break

    vae_model.train()













