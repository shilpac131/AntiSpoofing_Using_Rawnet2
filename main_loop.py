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
from tensorboardX import SummaryWriter
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


def train_epoch(train_loader, model, lr,optim, device):
    running_loss = 0
    num_correct = 0.0
    num_total = 0.0
    ii = 0
    model.train()

    #set objective (loss) functions
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    
    for batch_x, batch_y in train_loader:
       
        batch_size = batch_x.size(0)
        num_total += batch_size
        ii += 1
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x)
        batch_loss = criterion(batch_out, batch_y)
        _, batch_pred = batch_out.max(dim=1)
        num_correct += (batch_pred == batch_y).sum(dim=0).item()
        running_loss += (batch_loss.item() * batch_size)
        if ii % 10 == 0:
            sys.stdout.write('\r \t {:.2f}'.format(
                (num_correct/num_total)*100))
        optim.zero_grad()
        batch_loss.backward()
        optim.step()
       
    running_loss /= num_total
    train_accuracy = (num_correct/num_total)*100
    return running_loss, train_accuracy

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
                        default='/home/shilpa/shilpa/ASVspoof2021Baselines/DF/Baseline-RawNet2/pre_trained_DF_RawNet2.pth', help='Model checkpoint. Here provide LA trained model path to evaluate on DF Eval databse')
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
    # parser.add_argument('--audio_path', type=str,
    #                     default=None, help='Please provide path of audio file to be classified')
    parser.add_argument('--folder_path', type=str,
                        default='/home/shilpa/datasets/Prosody_TTS', help='Please provide path of audio file to be classified')
    

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
    if args.folder_path:
        folder_path = args.folder_path
        print('Audio path loaded : {}'.format(folder_path))

    # folder_path = '/home/shilpa/datasets/Prosody_TTS'
    audio_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.wav')] 
    model.eval()

    for audio_file in audio_files:
        X,fs = librosa.load(audio_file, sr=16000) 
        X_pad= pad(X,64600)
        x_inp= Tensor(X_pad)
        x_inp = x_inp.view(1, -1)
        x_inp = x_inp.to(device)
        pred = model(x_inp)
        softmax_probs = torch.softmax(pred, dim=1)
        _, predicted_class = torch.max(softmax_probs, 1)
        # _, pred_final = pred.max(dim=1)
        if predicted_class.item() == 0:
            print(f"Prediction of File {audio_file} is:> spoofed")
        elif predicted_class.item() == 1:
            print(f"Prediction of File {audio_file} is:> bonafide")



    # # evaluation 
    # if args.eval:
    #     file_eval = genSpoof_list( dir_meta =  os.path.join(args.protocols_path+'{}_cm_protocols/{}.cm.eval.trl.txt'.format(prefix,prefix_2021)),is_train=False,is_eval=True)
    #     print('no. of eval trials',len(file_eval))
    #     eval_set=Dataset_ASVspoof2021_eval(list_IDs = file_eval,base_dir = os.path.join(args.database_path+'ASVspoof2021_{}_eval/'.format(args.track)))
    #     produce_evaluation_file(eval_set, model, device, args.eval_output)
    #     sys.exit(0)
    
    # # define train dataloader

    # d_label_trn,file_train = genSpoof_list( dir_meta =  os.path.join(args.protocols_path+'{}_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'.format(prefix)),is_train=True,is_eval=False)
    # print('no. of training trials',len(file_train))
    
    # #train_set=Dataset_ASVspoof2019_train(list_IDs = file_train,labels = d_label_trn,base_dir = os.path.join(args.database_path+'ASVspoof2019_{}_train/'.format(args.track)))
    # # Note we bypass the reference to the track to train on LA instead of on DF (there is no provided training or dev data for DF)
    # train_set=Dataset_ASVspoof2019_train(list_IDs = file_train,labels = d_label_trn,base_dir = os.path.join(args.database_path+'ASVspoof2019_LA_train/'))

    # train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,drop_last = True)
    
    # del train_set,d_label_trn
    
    # # define validation dataloader

    # # Note we bypass the reference to the track to validate on LA instead of on DF (there is no provided training or dev data for DF)
    # d_label_dev,file_dev = genSpoof_list( dir_meta =  os.path.join(args.protocols_path+'{}_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'.format(prefix)),is_train=False,is_eval=False)
    # print('no. of validation trials',len(file_dev))

    # # Note we bypass the reference to the track to train on LA instead of on DF (there is no provided training or dev data for DF)
    # dev_set = Dataset_ASVspoof2019_train(list_IDs = file_dev,
	# 	labels = d_label_dev,
	# 	base_dir = os.path.join(args.database_path+'ASVspoof2019_LA_dev/'))
    # dev_loader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=False)
    # del dev_set,d_label_dev


    # # Training and validation 
    # num_epochs = args.num_epochs
    # writer = SummaryWriter('logs/{}'.format(model_tag))
    # best_acc = 99
    # for epoch in range(num_epochs):
    #     running_loss, train_accuracy = train_epoch(train_loader,model, args.lr,optimizer, device)
    #     valid_accuracy = evaluate_accuracy(dev_loader, model, device)
    #     writer.add_scalar('train_accuracy', train_accuracy, epoch)
    #     writer.add_scalar('valid_accuracy', valid_accuracy, epoch)
    #     writer.add_scalar('loss', running_loss, epoch)
    #     print('\n{} - {} - {:.2f} - {:.2f}'.format(epoch,
    #                                                running_loss, train_accuracy, valid_accuracy))
        
    #     if valid_accuracy > best_acc:
    #         print('best model find at epoch', epoch)
    #     best_acc = max(valid_accuracy, best_acc)
    #     torch.save(model.state_dict(), os.path.join(model_save_path, 'epoch_{}.pth'.format(epoch)))

 