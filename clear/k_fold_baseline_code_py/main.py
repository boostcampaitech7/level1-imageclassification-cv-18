import time

import torch

from setting import set_arg_parser_default, set_trainer, set_tester
import data

if __name__ == "__main__":
    torch.cuda.empty_cache()
    torch.multiprocessing.set_start_method('spawn')
    args = set_arg_parser_default()

    start_time = time.time()

    # Data setting
    train_info, test_info, num_classes = data.return_data_frames_and_num_classes(args.train_csv, args.test_csv)
    train_loader, val_loader = data.set_train_and_val_data(train_info, args.train_dir, transform = args.transform, batch_size = args.batch_size)
    test_loader = data.set_test_loader(test_info, args.test_dir)

    # train setting
    trainer = set_trainer(train_loader, val_loader, num_classes,save_path = os.path.join(), device = args.device)

    # train
    trainer.train()
    
    # test
    tester = set_tester(test_info, test_loader, trainer.model, device = args.device)
    
    end_time = time.time()

    print(f" End : {(end_time - start_time)/60} min")