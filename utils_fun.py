'''
Coded by Ncuti Xavier
'''
import argparse
import time

def set_argument_parser():
    #Defining parser for train...
    train_parser = argparse.ArgumentParser(description='setting parser for training model')
    
    train_parser.add_argument('--arch', type=str, help='architecture to be used', default = 'vgg19')
    train_parser.add_argument('--data_dir', type=str, help='data directory to be used', default = 'flowers')
    train_parser.add_argument('--save_dir', type=str, help='save directory to be used')
    train_parser.add_argument('--gpu', type=str, help='gpu device to be used', default = "vgg19")
    train_parser.add_argument('--dropout', type=float, help='dropout to be used')
    train_parser.add_argument('--epochs', type=int, help='epochs to be used', default = 3)
    train_parser.add_argument('--learning_rate', type=float, help='learning_rate to be used')
    train_parser.add_argument('--top_k', type=int, help='Choose top K matches as int.', default = 3)
    train_parser.add_argument('--category_names', type=str, help='json file with category and their names', default = 'cat_to_name.json')
    train_parser.add_argument('--checkpoint', type=str, help='checkpoint', default = 'classifier.pth')

    
    return train_parser

#Display time 
def display_time(task, start_time):
    elapsed_time = time.time() - start_time
    print(task+ " => Duration time: {} minutes and {} seconds".format(elapsed_time//60, elapsed_time%60))