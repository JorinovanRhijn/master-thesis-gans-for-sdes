# Jorino van Rhijn 
# Monte Carlo Simulation of SDEs with GANs 

import torch
from GANutils import input_sample

# Module containing GAN training step methods for the unconditional GAN and conditional GAN in both vanilla supervised form. 

def step_handler(supervised_bool=False,CGAN_bool=False):
	'''
	Method to select train step function 
	'''
	if supervised_bool and not CGAN_bool:
		print('Training supervised GAN...')
		return supervised_GAN_step
	elif supervised_bool and CGAN_bool:
		print('Training supervised CGAN...')
		return supervised_CGAN_step
	elif not supervised_bool and CGAN_bool:
		print('Training vanilla CGAN...')
		return vanilla_CGAN_step
	else:
		print('Training vanilla GAN...')
		return vanilla_GAN_step


def vanilla_GAN_step(netD,netG,optD,optG,loss_func,device,X_i=None,C_i=None,Z_i=None,real_label=1,fake_label=0,n_D=1):
    # Train discriminator n_D times per batch 
    for j in range(n_D):
        # Train on all-real batch 
        netD.zero_grad()
        real_sample = X_i
        b_size = X_i.size(0)
        label = torch.full((b_size,), real_label, device=device,dtype=torch.get_default_dtype())
        output = netD(real_sample)
        errD_real = loss_func(output.view(-1),label.view(-1))
        errD_real.backward()
        D_x = output.mean().item()

        # Train on all-fake batch 
        input_G = input_sample(b_size,device=device)
        fake = netG(input_G).detach()
        label.fill_(fake_label)
        output = netD(fake)
        errD_fake = loss_func(output.view(-1),label.view(-1))
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optD.step()

    # Train generator 
    netG.zero_grad()

    # Create fake sample 
    input_G = input_sample(b_size,device=device)
    output = netD(netG(input_G))
    errG = -torch.log(output).mean() # 'Logtrick'
    errG.backward()
    D_G_z2 = output.mean().item()
    optG.step()

    return errD,errG,D_x,D_G_z1,D_G_z2

def vanilla_CGAN_step(netD,netG,optD,optG,loss_func,device,X_i=None,C_i=None,Z_i=None,real_label=1,fake_label=0,n_D=1):
    # Train discriminator n_D times per batch 
    for j in range(n_D):
        # Train on all-real batch 
        netD.zero_grad()
        real_sample = torch.cat((X_i,C_i),axis=1)
        b_size = X_i.size(0)
        label = torch.full((b_size,), real_label, device=device,dtype=torch.get_default_dtype())
        output = netD(real_sample)
        errD_real = loss_func(output.view(-1),label.view(-1))
        errD_real.backward()
        D_x = output.mean().item()

        # Train on all-fake batch 
        input_G = input_sample(b_size,C=C_i,device=device)
        fake = torch.cat((netG(input_G).detach(),C_i),axis=1)
        label.fill_(fake_label)
        output = netD(fake)
        errD_fake = loss_func(output.view(-1),label.view(-1))
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optD.step()

    # Train generator 
    netG.zero_grad()

    # Create fake sample 
    input_G = input_sample(b_size,C=C_i,device=device)
    fake = torch.cat((netG(input_G),C_i),axis=1)
    output = netD(fake)
    errG = -torch.log(output).mean() # 'Logtrick'
    errG.backward()
    D_G_z2 = output.mean().item()
    optG.step()

    return errD,errG,D_x,D_G_z1,D_G_z2


def supervised_GAN_step(netD,netG,optD,optG,loss_func,device,X_i=None,C_i=None,Z_i=None,real_label=1,fake_label=0,n_D=1):
    # Train discriminator n_D times per batch 
    for j in range(n_D):
        # Train on all-real batch 
        netD.zero_grad()
        real_sample = torch.cat((X_i,Z_i),axis=1)
        b_size = X_i.size(0)
        label = torch.full((b_size,), real_label, device=device,dtype=torch.get_default_dtype())
        output = netD(real_sample)
        errD_real = loss_func(output.view(-1),label.view(-1))
        errD_real.backward()
        D_x = output.mean().item()

        # Train on all-fake batch 
        input_G = input_sample(b_size,Z=Z_i,device=device)
        fake = torch.cat((netG(input_G).detach(),Z_i),axis=1)
        label.fill_(fake_label)
        output = netD(fake)
        errD_fake = loss_func(output.view(-1),label.view(-1))
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optD.step()

    # Train generator 
    netG.zero_grad()

    # Create fake sample 
    # input_G = input_sample(b_size,Z=Z_i,device=device)
    fake = torch.cat((netG(input_G),Z_i),axis=1)
    output = netD(fake)
    errG = -torch.log(output).mean() # 'Logtrick'
    errG.backward()
    D_G_z2 = output.mean().item()
    optG.step()

    return errD,errG,D_x,D_G_z1,D_G_z2


def supervised_CGAN_step(netD,netG,optD,optG,loss_func,device,X_i=None,C_i=None,Z_i=None,real_label=1,fake_label=0,n_D=1):
    # Train discriminator n_D times per batch 
    for j in range(n_D):
        # Train on all-real batch 
        netD.zero_grad()
        real_sample = torch.cat((X_i,Z_i,C_i),axis=1)
        b_size = X_i.size(0)
        label = torch.full((b_size,), real_label, device=device,dtype=torch.get_default_dtype())
        output = netD(real_sample)
        errD_real = loss_func(output.view(-1),label.view(-1))
        errD_real.backward()
        D_x = output.mean().item()

        # Train on all-fake batch 
        input_G = torch.cat((Z_i,C_i),axis=1)
        fake = torch.cat((netG(input_G).detach(),Z_i,C_i),axis=1)
        label.fill_(fake_label)
        output = netD(fake)
        errD_fake = loss_func(output.view(-1),label.view(-1))
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optD.step()

    # Train generator 
    netG.zero_grad()

    # Create fake sample 
    # input_G = input_sample(b_size,Z=Z_i,C=C_i,device=device)
    fake = torch.cat((netG(input_G),Z_i,C_i),axis=1)
    output = netD(fake)
    errG = -torch.log(output).mean() # 'Logtrick'
    errG.backward()
    D_G_z2 = output.mean().item()
    optG.step()

    return errD,errG,D_x,D_G_z1,D_G_z2
