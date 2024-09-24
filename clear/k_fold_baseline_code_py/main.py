import time

import torch

from setting import set_cuda, set_arg_parser_default, set_trainer, set_tester, set_model
import data

if __name__ == "__main__":
    torch.cuda.empty_cache()
    torch.multiprocessing.set_start_method('spawn')
    args = set_arg_parser_default()

    start_time = time.time()

    # Device setting
    device = set_cuda(args.gpu)

    # Data setting
    train_info, num_classes = data.return_train_info(args.train_csv)
    model = set_model(args,device,num_classes)
    train_loader, val_loader = data.set_train_and_val_data(train_info, args.train_dir, transform = args.transform, batch_size = args.batch_size)

    # train setting
    trainer = set_trainer(args, train_loader, val_loader, num_classes, model = model, device = device)

    # train
    trainer.train()


    test_info = data.return_test_info(args.test_csv)
    test_loader = data.set_test_loader(test_info, args.test_dir, transform = args.transform)

    # test
    tester = set_tester(args, test_info, test_loader, model = model, num_classes=num_classes, device = device)
    
    end_time = time.time()

    print(f" End : {(end_time - start_time)/60} min")