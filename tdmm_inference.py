import cv2
import torch
import torch.nn.functional as F
import numpy as np
from argparse import ArgumentParser
from modules.tdmm_estimator import TDMMEstimator
from torch.utils.data import DataLoader
from frames_dataset import FramesDataset, ImageDataset, DBImageDataset2
from tqdm import tqdm
import yaml


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--data_dir", default="", help="directory containing images to inference")
    parser.add_argument("--gpu", action="store_true", help="run the inference on gpu")
    parser.add_argument("--with_eye", action="store_true", help="use eye part for extracting texture")
    parser.add_argument("--tdmm_checkpoint", default=None, help="path to checkpoint of the tdmm estimator model to restore")
    parser.add_argument("--config", required=True, help="path to config")
    
    opt = parser.parse_args()

    checkpoint = torch.load(opt.tdmm_checkpoint, map_location=torch.device('cpu'))

    tdmm = TDMMEstimator()
    tdmm.load_state_dict(checkpoint['tdmm'])
    # optimizer_tdmm = torch.optim.Adam(tdmm.parameters(), betas=(0.9, 0.999))

    # ckpt = {
    #     'tdmm': tdmm.state_dict(),
    #     'optimizer_tdmm': optimizer_tdmm.state_dict(),
    #     'epoch': 0,
    # }
    # torch.save(ckpt, 'temp-ckpt.path.tar')
    

    if opt.gpu:
        tdmm = tdmm.cuda()

    with open(opt.config) as f:
        config = yaml.load(f)
    dataset = DBImageDataset2(
        None,
        opt.data_dir,
        '/hdd/share/youtube-speech',
        augmentation_params=config['dataset_params']['augmentation_params'],
        limit=100)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=1)
    tdmm.eval()

    bar = tqdm(dataloader)
    for i, x in enumerate(bar):

        if opt.gpu:
            x['image'] = x['image'].cuda()

        codedict = tdmm.encode(x['image'])
        verts, transformed_verts, landmark_2d = tdmm.decode_flame(codedict)

        # extract albedo and rendering
        albedo = tdmm.extract_texture(x['image'], transformed_verts, with_eye=opt.with_eye)
        outputs = tdmm.render(transformed_verts, transformed_verts, albedo)

        image = x['image'].squeeze().permute(1, 2, 0)
        source = outputs['source'].squeeze().permute(1, 2, 0).detach()
        normal_map = outputs['source_normal_images'].squeeze().permute(1, 2, 0).detach()

        if opt.gpu:
            image = image.cpu()
            normal_map = normal_map.cpu()
            source = source.cpu()

        image = image.numpy()
        source = source.numpy()
        normal_map = normal_map.numpy()

        for pt_x, pt_y in landmark_2d[0]:
            pt_x, pt_y = int(pt_x), int(pt_y)
            image[(pt_y - 1):(pt_y + 1), (pt_x - 1):(pt_x + 1), 0] = 255

        if 'ldmk' in x:
            ldmk = x['ldmk']
            if opt.gpu:
                ldmk = ldmk.cuda()
            ldmk_loss = F.l1_loss(ldmk, landmark_2d)
            param_loss = 1e-3 * (torch.mean(codedict['shape'] ** 2) + 0.8 * torch.mean(codedict['exp'] ** 2)) 

            losses_tdmm = {}
            losses_tdmm['ldmk_loss'] = ldmk_loss
            losses_tdmm['param_loss'] = param_loss

            loss_values = [val for val in losses_tdmm.values()]
            loss = sum(loss_values)
            

    
            ldmk = ldmk[0].cpu().numpy()
            landmark_2d = landmark_2d[0].detach().cpu().numpy()
            for pt_x, pt_y in ldmk:
                pt_x, pt_y = int(pt_x), int(pt_y)
                image[(pt_y - 1):(pt_y + 1), (pt_x - 1):(pt_x + 1), 1] = 255

            if loss > 30:
                for j in range(68):
                    left = ldmk[j]
                    right = landmark_2d[j]
                    if sum(left - right) > 5:
                        print(f'[{j}] ({ldmk[j, 0]}, {ldmk[j, 1]}) <-> ({landmark_2d[j, 0]}, {landmark_2d[j, 1]})')
            
        else:
            loss = None
        

        if loss is not None:        
            bar.set_description(f"tdmm loss: {loss:0.4f}")

        cv2.imshow('input', image[..., ::-1])
        cv2.imshow('source', source[..., ::-1])
        cv2.imshow('normal', normal_map)
        cv2.waitKey(0)
