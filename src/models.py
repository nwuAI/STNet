import os
import torch
import torch.nn as nn
import torch.optim as optim
from src.networks import InpaintGenerator, EdgeGenerator, Discriminator, StructureGen

from .loss import AdversarialLoss, PerceptualLoss, StyleLoss
from .utils import get_model_list


class BaseModel(nn.Module):
    def __init__(self, name, config):
        super(BaseModel, self).__init__()

        self.name = name
        self.config = config
        self.iteration = 0

    def load(self):
        gen_path = get_model_list(self.config.PATH, self.name, 'gen')
        dis_path = get_model_list(self.config.PATH, self.name, 'dis')
        if gen_path is not None:
            print('Loading {} generator weights file: {}...'.format(self.name, gen_path))
            if self.config.DEVICE == torch.device('cuda'):  # gpu
                data = torch.load(gen_path)
            else:  # cpu
                data = torch.load(gen_path, map_location=lambda storage, loc: storage)
            self.generator.load_state_dict(data['generator'])
            self.iteration = data['iteration']

        # load discriminator only when training
        if self.config.MODE == 1 and dis_path is not None:
            print('Loading {} discriminator weights file: {}...'.format(self.name, dis_path))
            if self.config.DEVICE == torch.device('cuda'):
                data = torch.load(dis_path)
            else:
                data = torch.load(dis_path, map_location=lambda storage, loc: storage)
            self.discriminator.load_state_dict(data['discriminator'])

    def save(self):
        print('\nsaving %s...\n' % self.name)
        torch.save({
            'iteration': self.iteration,
            'generator': self.generator.state_dict()
        }, os.path.join(self.config.PATH, '{}_gen_{}.pth'.format(self.name, self.iteration)))

        torch.save({
            'discriminator': self.discriminator.state_dict()
        }, os.path.join(self.config.PATH, '{}_dis_{}.pth'.format(self.name, self.iteration)))


class EdgeModel(BaseModel):
    def __init__(self, config):
        super(EdgeModel, self).__init__('EdgeModel', config)

        # generator input: [grayscale(1) + edge(1) + mask(1)]
        # discriminator input: (grayscale(1) + edge(1))
        generator = EdgeGenerator(use_spectral_norm=True)
        discriminator = Discriminator(in_channels=2, use_sigmoid=config.GAN_LOSS != 'hinge')

        # multi-gpu
        if len(config.GPU) > 1:
            generator = nn.DataParallel(generator, config.GPU)
            discriminator = nn.DataParallel(discriminator, config.GPU)

        l1_loss = nn.L1Loss()
        adversarial_loss = AdversarialLoss(type=config.GAN_LOSS)

        self.add_module('generator', generator)
        self.add_module('discriminator', discriminator)

        self.add_module('l1_loss', l1_loss)
        self.add_module('adversarial_loss', adversarial_loss)

        l1_loss_all = nn.L1Loss()
        l1_loss_mask = nn.L1Loss()
        perceptual_loss = PerceptualLoss()
        self.add_module('l1_loss_all', l1_loss_all)
        self.add_module('l1_loss_mask', l1_loss_mask)
        self.add_module('perceptual_loss', perceptual_loss)


        self.gen_optimizer = optim.Adam(
            params=generator.parameters(),
            lr=float(config.LR),
            betas=(config.BETA1, config.BETA2)
        )

        self.dis_optimizer = optim.Adam(params=discriminator.parameters(),
                                        lr=float(config.LR) * float(config.D2G_LR),
                                        betas=(config.BETA1, config.BETA2)
                                        )

    def process(self, images, edges, masks):
        self.iteration += 1

        # zero optimizers
        self.gen_optimizer.zero_grad()
        self.dis_optimizer.zero_grad()

        # process outputs
        outputs, gray = self(images, edges, masks)
        gen_loss = 0
        dis_loss = 0

        # discriminator loss
        dis_input_real = torch.cat((images, edges), dim=1)
        dis_input_fake = torch.cat((images, outputs.detach()), dim=1)
        dis_real, dis_real_feat = self.discriminator(dis_input_real)  # in: (grayscale(1) + edge(1))
        dis_fake, dis_fake_feat = self.discriminator(dis_input_fake)  # in: (grayscale(1) + edge(1))
        dis_real_loss = self.adversarial_loss(dis_real, True, True)
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2

        # generator adversarial loss
        gen_input_fake = torch.cat((images, outputs), dim=1)
        gen_fake, gen_fake_feat = self.discriminator(gen_input_fake)  # in: (grayscale(1) + edge(1))
        gen_gan_loss = self.adversarial_loss(gen_fake, True, False)
        gen_loss += gen_gan_loss

        # generator feature matching loss
        gen_fm_loss = 0
        for i in range(len(dis_real_feat)):
            gen_fm_loss += self.l1_loss(gen_fake_feat[i], dis_real_feat[i].detach())
        gen_fm_loss = gen_fm_loss * self.config.FM_LOSS_WEIGHT
        gen_loss += gen_fm_loss


        # generator l1 gray loss all
        gray_merged = (gray * masks) + (images * (1 - masks))
        gen_l1_loss_all = self.l1_loss_all(gray_merged, images) * self.config.L1_LOSS_WEIGHT / torch.mean(masks)
        gen_loss += gen_l1_loss_all

        # generator l1 gray loss mask
        gen_l1_loss_mask = self.l1_loss(gray, images) * self.config.L1_LOSS_WEIGHT / torch.mean(masks)
        gen_loss += gen_l1_loss_mask * 0.5


        # generator perceptual loss
        gray_3chan = gray.clone().expand(-1,3,256,256)
        img = images.clone().expand(-1,3,256,256)
        gen_content_loss = self.perceptual_loss(gray_3chan, img)
        gen_content_loss = gen_content_loss * self.config.CONTENT_LOSS_WEIGHT
        gen_loss += gen_content_loss * 0.5

        # create logs
        logs = [
            ("l_d1", dis_loss.item()),
            ("l_g1", gen_gan_loss.item()),
            ("l_fm", gen_fm_loss.item()),
            ("l_l1_all", gen_l1_loss_all.item()),
            ("l_l1_mask", gen_l1_loss_mask.item()),
            ("l_per", gen_content_loss.item()),
        ]

        return outputs, gray, gen_loss, dis_loss, logs

    def forward(self, images, edges, masks):
        edges_masked = (edges * (1 - masks))
        images_masked = (images * (1 - masks)) + masks
        inputs = torch.cat((images_masked, edges_masked, masks), dim=1)
        outputs, gray = self.generator(inputs)  # in: [grayscale(1) + edge(1) + mask(1)]
        return outputs, gray

    def backward(self, gen_loss=None, dis_loss=None):
        if dis_loss is not None:
            dis_loss.backward()
        self.dis_optimizer.step()

        if gen_loss is not None:
            gen_loss.backward()
        self.gen_optimizer.step()


class InpaintingModel(BaseModel):
    def __init__(self, config):
        super(InpaintingModel, self).__init__('InpaintingModel', config)

        # generator input: [rgb(3) + edge(1)]
        # discriminator input: [rgb(3)]
        generator = InpaintGenerator()
        discriminator = Discriminator(in_channels=3, use_sigmoid=config.GAN_LOSS != 'hinge')

        # multi-gpu
        if len(config.GPU) > 1:
            generator = nn.DataParallel(generator, config.GPU)
            discriminator = nn.DataParallel(discriminator, config.GPU)

        l1_loss = nn.L1Loss()
        perceptual_loss = PerceptualLoss()
        style_loss = StyleLoss()
        adversarial_loss = AdversarialLoss(type=config.GAN_LOSS)

        self.add_module('generator', generator)
        self.add_module('discriminator', discriminator)

        self.add_module('l1_loss', l1_loss)
        self.add_module('perceptual_loss', perceptual_loss)
        self.add_module('style_loss', style_loss)
        self.add_module('adversarial_loss', adversarial_loss)

        l1_loss_lbp = nn.L1Loss()
        self.add_module('l1_loss_lbp', l1_loss_lbp)

        self.gen_optimizer = optim.Adam(
            params=generator.parameters(),
            lr=float(config.LR),
            betas=(config.BETA1, config.BETA2)
        )

        self.dis_optimizer = optim.Adam(
            params=discriminator.parameters(),
            lr=float(config.LR) * float(config.D2G_LR),
            betas=(config.BETA1, config.BETA2)
        )

    def process(self, images, structure, gray, edges, masks):
        self.iteration += 1

        # zero optimizers
        self.gen_optimizer.zero_grad()
        self.dis_optimizer.zero_grad()

        # process outputs
        outputs = self(images, structure, gray, edges, masks)
        gen_loss = 0
        dis_loss = 0

        # discriminator loss
        dis_input_real = images
        dis_input_fake = outputs.detach()
        dis_real, _ = self.discriminator(dis_input_real)  # in: [rgb(3)]
        dis_fake, _ = self.discriminator(dis_input_fake)  # in: [rgb(3)]
        dis_real_loss = self.adversarial_loss(dis_real, True, True)
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2

        # generator adversarial loss
        gen_input_fake = outputs
        gen_fake, _ = self.discriminator(gen_input_fake)  # in: [rgb(3)]
        gen_gan_loss = self.adversarial_loss(gen_fake, True, False) * self.config.INPAINT_ADV_LOSS_WEIGHT
        gen_loss += gen_gan_loss

        # generator l1 loss
        gen_l1_loss = self.l1_loss(outputs, images) * self.config.L1_LOSS_WEIGHT / torch.mean(masks)
        gen_loss += gen_l1_loss

        # generator perceptual loss
        gen_content_loss = self.perceptual_loss(outputs, images)
        gen_content_loss = gen_content_loss * self.config.CONTENT_LOSS_WEIGHT
        gen_loss += gen_content_loss

        # generator style loss
        gen_style_loss = self.style_loss(outputs * masks, images * masks)
        gen_style_loss = gen_style_loss * self.config.STYLE_LOSS_WEIGHT
        gen_loss += gen_style_loss





        logs = [
            ("l_d2", dis_loss.item()),
            ("l_g2", gen_gan_loss.item()),
            ("l_l1", gen_l1_loss.item()),
            ("l_per", gen_content_loss.item()),
            ("l_sty", gen_style_loss.item()),
        ]

        return outputs, gen_loss, dis_loss, logs

    def forward(self, images, structure, gray, edges, masks):
        images_masked = (images * (1 - masks).float()) + masks
        # print(edges.size(),images_masked.size())
        inputs = torch.cat((images_masked, masks), dim=1)
        outputs = self.generator(inputs, structure, gray, edges)  # in: [rgb(3) + edge(1)]
        return outputs

    def backward(self, gen_loss=None, dis_loss=None):
        dis_loss.backward()
        self.dis_optimizer.step()

        gen_loss.backward()
        self.gen_optimizer.step()


class StructureModel(BaseModel):
    def __init__(self, config):
        super(StructureModel, self).__init__('StructureModel', config)

        # generator input: [rgb(3) + edge(1)]
        # discriminator input: [rgb(3)]
        generator = StructureGen()
        discriminator = Discriminator(in_channels=3, use_sigmoid=config.GAN_LOSS != 'hinge')

        # multi-gpu
        if len(config.GPU) > 1:
            generator = nn.DataParallel(generator, config.GPU)
            discriminator = nn.DataParallel(discriminator, config.GPU)

        l1_loss = nn.L1Loss()
        perceptual_loss = PerceptualLoss()
        style_loss = StyleLoss()
        adversarial_loss = AdversarialLoss(type=config.GAN_LOSS)

        self.add_module('generator', generator)
        self.add_module('discriminator', discriminator)

        self.add_module('l1_loss', l1_loss)
        self.add_module('perceptual_loss', perceptual_loss)
        self.add_module('style_loss', style_loss)
        self.add_module('adversarial_loss', adversarial_loss)

        self.gen_optimizer = optim.Adam(
            params=generator.parameters(),
            lr=float(config.LR),
            betas=(config.BETA1, config.BETA2)
        )

        self.dis_optimizer = optim.Adam(
            params=discriminator.parameters(),
            lr=float(config.LR) * float(config.D2G_LR),
            betas=(config.BETA1, config.BETA2)
        )

    def process(self, images, structure, masks):
        self.iteration += 1

        # zero optimizers
        self.gen_optimizer.zero_grad()
        self.dis_optimizer.zero_grad()

        # process outputs
        outputs = self(images, structure, masks)
        gen_loss = 0
        dis_loss = 0

        # discriminator loss
        dis_input_real = structure
        dis_input_fake = outputs.detach()
        dis_real, _ = self.discriminator(dis_input_real)  # in: [rgb(3)]
        dis_fake, _ = self.discriminator(dis_input_fake)  # in: [rgb(3)]
        dis_real_loss = self.adversarial_loss(dis_real, True, True)
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2

        # generator adversarial loss
        gen_input_fake = outputs
        gen_fake, _ = self.discriminator(gen_input_fake)  # in: [rgb(3)]
        gen_gan_loss = self.adversarial_loss(gen_fake, True, False) * self.config.INPAINT_ADV_LOSS_WEIGHT
        gen_loss += gen_gan_loss

        # generator l1 loss
        gen_l1_loss = self.l1_loss(outputs, structure) * self.config.L1_LOSS_WEIGHT / torch.mean(masks)
        gen_loss += gen_l1_loss

        # generator perceptual loss
        gen_content_loss = self.perceptual_loss(outputs, structure)
        gen_content_loss = gen_content_loss * self.config.CONTENT_LOSS_WEIGHT
        gen_loss += gen_content_loss

        # generator style loss
        gen_style_loss = self.style_loss(outputs * masks, structure * masks)
        gen_style_loss = gen_style_loss * self.config.STYLE_LOSS_WEIGHT
        gen_loss += gen_style_loss

        # create logs
        logs = [
            ("l_d2", dis_loss.item()),
            ("l_g2", gen_gan_loss.item()),
            ("l_l1", gen_l1_loss.item()),
            ("l_per", gen_content_loss.item()),
            ("l_sty", gen_style_loss.item()),
        ]

        return outputs, gen_loss, dis_loss, logs

    def forward(self, images, structure, masks):
        images_masked = (images * (1 - masks).float()) + masks
        structure_masked = (structure * (1 - masks).float()) + masks
        # print(edges.size(),images_masked.size())
        inputs = torch.cat((images_masked, structure_masked, masks), dim=1)
        outputs = self.generator(inputs)  # in: [rgb(3) + edge(1)]
        return outputs

    def backward(self, gen_loss=None, dis_loss=None):
        dis_loss.backward()
        self.dis_optimizer.step()

        gen_loss.backward()
        self.gen_optimizer.step()


