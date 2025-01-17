import argparse
import sys
import os
import numpy as np
import torch
from torch import nn
from torch import Tensor
import librosa
from torch.utils.data import DataLoader
import yaml
from data_utils import genSpoof_list,Dataset_ASVspoof2019_train,Dataset_ASVspoof2021_eval
from model import RawNet
# from tensorboardX import SummaryWriter
from core_scripts.startup_config import set_random_seed


__author__ = "Hemlata Tak"
__email__ = "tak@eurecom.fr"
__credits__ = ["Jose Patino", "Massimiliano Todisco", "Jee-weon Jung"]

def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len)+1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x 


def evaluate_accuracy(dev_loader, model, device):
    num_correct = 0.0
    num_total = 0.0
    model.eval()
    for batch_x, batch_y in dev_loader:
        
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x)
        _, batch_pred = batch_out.max(dim=1)
        num_correct += (batch_pred == batch_y).sum(dim=0).item()
    return 100 * (num_correct / num_total)


def produce_evaluation_file(dataset, model, device, save_path):
    data_loader = DataLoader(dataset, batch_size=128, shuffle=False, drop_last=False)
    model.eval()
    
    for batch_x,utt_id in data_loader:
        fname_list = []
        score_list = []  
        batch_size = batch_x.size(0)
        batch_x = batch_x.to(device)
        batch_out = model(batch_x)
        batch_score = (batch_out[:, 1]
                       ).data.cpu().numpy().ravel()
        # add outputs
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())
        
        with open(save_path, 'a+') as fh:
            for f, cm in zip(fname_list,score_list):
                fh.write('{} {}\n'.format(f, cm))
        fh.close()   
    print('Scores saved to {}'.format(save_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ASVspoof2021 baseline system')
    # Dataset
    parser.add_argument('--database_path', type=str, default='/your/path/to/data/ASVspoof_database/', help='Change this to user\'s full directory address of DF database. We assume that all three ASVspoof 2019 LA train, LA dev and ASVspoof2021 DF eval data folders are in the same database_path directory.')
    '''
    % database_path/
    %   |- DF
    %      |- ASVspoof2021_DF_eval/flac
    %      |- ASVspoof2019_LA_train/flac
    %      |- ASVspoof2019_LA_dev/flac
    '''

    parser.add_argument('--protocols_path', type=str, default='/your/path/to/protocols/ASVspoof_database/', help='Change with path to user\'s DF ASVspoof2021 database protocols directory address')
    '''
    % protocols_path/
    %   |- ASVspoof_DF_cm_protocols
    %      |- ASVspoof2021.DF.cm.eval.trl.txt
    %      |- ASVspoof2019.LA.cm.dev.trl.txt 
    %      |- ASVspoof2019.LA.cm.train.trn.txt 
    '''

    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--loss', type=str, default='weighted_CCE')
    # model
    parser.add_argument('--seed', type=int, default=1234, 
                        help='random seed (default: 1234)')
    
    parser.add_argument('--model_path', type=str,
                        default='./pre_trained_DF_RawNet2.pth', help='Model checkpoint. Here provide LA trained model path to evaluate on DF Eval databse')
    parser.add_argument('--comment', type=str, default=None,
                        help='Comment to describe the saved model')
    # Auxiliary arguments
    parser.add_argument('--track', type=str, default='DF',choices=['LA', 'PA','DF'], help='LA/PA/DF')
    parser.add_argument('--eval_output', type=str, default=None,
                        help='Path to save the evaluation result')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='eval mode')
    parser.add_argument('--is_eval', action='store_true', default=False,help='eval database')
    parser.add_argument('--eval_part', type=int, default=0)
    # backend options
    parser.add_argument('--cudnn-deterministic-toggle', action='store_false', \
                        default=True, 
                        help='use cudnn-deterministic? (default true)')    
    
    parser.add_argument('--cudnn-benchmark-toggle', action='store_true', \
                        default=False, 
                        help='use cudnn-benchmark? (default false)') 
    parser.add_argument('--audio_path', type=str,
                        default=None, help='Please provide path of audio file to be classified')
    

    dir_yaml = os.path.splitext('model_config_RawNet')[0] + '.yaml'

    with open(dir_yaml, 'r') as f_yaml:
            parser1 = yaml.load(f_yaml)

    if not os.path.exists('models'):
        os.mkdir('models')
    args = parser.parse_args()
 
    #make experiment reproducible
    set_random_seed(args.seed, args)
    
    track = args.track

    assert track in ['LA', 'PA','DF'], 'Invalid track given'

    # Database
    prefix      = 'ASVspoof_{}'.format(track)
    prefix_2019 = 'ASVspoof2019.{}'.format(track)
    prefix_2021 = 'ASVspoof2021.{}'.format(track)
    
    #define model saving path
    model_tag = 'model_{}_{}_{}_{}_{}'.format(
        track, args.loss, args.num_epochs, args.batch_size, args.lr)
    if args.comment:
        model_tag = model_tag + '_{}'.format(args.comment)
    model_save_path = os.path.join('models', model_tag)
   
    #set model save directory
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
    
    #GPU device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'                  
    print('Device: {}'.format(device))
    
    # model 
    model = RawNet(parser1['model'], device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    model =(model).to(device)
    
    #set Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
    
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path,map_location=device))
        print('Model loaded : {}'.format(args.model_path))

    print("Time for forward pass")


    if args.audio_path:
        audio_path = args.audio_path
        print('Audio loaded : {}'.format(audio_path))

    output_file = './output.txt'
    with open(output_file, "a") as file:
        X,fs = librosa.load(audio_path, sr=16000) 
        X_pad= pad(X,64600)
        x_inp= Tensor(X_pad)
        x_inp = x_inp.view(1, -1)
        print("audio file loaded")
        # Perform a forward pass on a single audio file

        print("let's begin inference")
        model.eval()
        x_inp = x_inp.to(device)
        pred = model(x_inp)
        softmax_probs = torch.softmax(pred, dim=1)
        _, predicted_class = torch.max(softmax_probs, 1)
        # Get the softmax probability values of the predicted class
        predicted_class_probs = softmax_probs.gather(1, predicted_class.unsqueeze(1)).squeeze(1)
        # _, pred_final = pred.max(dim=1)
        if predicted_class.item() == 0:
            print("The audio is spoofed")
            file.write(f"\nPrediction of File {audio_path} is:> bonafide with confidence score of {predicted_class_probs.item()}")
        elif predicted_class.item() == 1:
            print("The audio is bonafide")
            file.write(f"\nPrediction of File {audio_path} is:> bonafide with confidence score of {predicted_class_probs.item()}")
        print(f"Predictions saved in path-> {output_file}")
        