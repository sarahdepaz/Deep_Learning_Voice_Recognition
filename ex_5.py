from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
import torch.nn as nn
import torch.utils.data
from gcommand_dataset import *
import convo_net

EPOCHS = 5
LR = 0.001
BATCH_SIZE = 100


# tried 100 batch 0.001 lr
# try batch 64 and lr 0.0001


def main():
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = torch.utils.data.DataLoader(GCommandLoader('./train'), BATCH_SIZE,
                                               shuffle=True, num_workers=20, pin_memory=True, sampler=None)
    val_loader = torch.utils.data.DataLoader(GCommandLoader('./valid'), BATCH_SIZE,
                                             shuffle=None, num_workers=20, pin_memory=True, sampler=None)
    test_loader = torch.utils.data.DataLoader(GCommandLoader('./test'), BATCH_SIZE,
                                              shuffle=None, num_workers=20, pin_memory=True, sampler=None)
    model = convo_net.convo_net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    train(train_loader, model, criterion, optimizer, device)
    # test(model, test_loader, device)
    # avg_acc_validation = validate(val_loader, model, criterion, device)
    predictions(test_loader, model, device)
    # acc_graphs(avg_acc_train, avg_acc_validation)


def train(train_loader, model, criterion, optimizer, device):
    total_step = len(train_loader)
    accuracy_list = []
    loss_list = []
    avg_acc_train = {}
    model.train()

    for epoch in range(EPOCHS):
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = print_accuracy(accuracy_list, epoch, i, labels, loss, outputs, total_step)
        avg_acc_train[epoch] = acc
    return avg_acc_train


def print_accuracy(accuracy_list, epoch, i, labels, loss, outputs, total_step):
    total = labels.size(0)
    _, pred = torch.max(outputs.data, 1)
    correct = (pred == labels).sum().item()
    accuracy_list.append(correct / total)
    if (i + 1) % 100 == 0:
        print(f'{datetime.now()}')
        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
              .format(epoch + 1, EPOCHS, i + 1, total_step, loss.item(),
                      (correct / total) * 100))
    return (correct / total) * 100


def get_path(spects):
    file_names = []
    for spect in spects:
        split = spect[0].split('/')
        file_name = split[-1]
        # "./test/test/0.wav
        no_path = file_name.split('/')
        new_path = no_path[-1]
        file_names.append(new_path)
    return file_names


def validate(validate_loader, model, criterion, device):
    total_step = len(validate_loader)
    accuracy_list = []
    loss_list = []
    avg_acc_validate = {}
    model.eval()
    print("*****validate*******")
    for epoch in range(EPOCHS):
        for i, (images, labels) in enumerate(validate_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_list.append(loss.item())

            acc = print_accuracy(accuracy_list, epoch, i, labels, loss, outputs, total_step)
        avg_acc_validate[epoch] = acc
    return avg_acc_validate


def predictions(test, model, device):
    predictions_temp = []
    predictions_list = []
    # Test the model
    with torch.no_grad():
        model.eval()
        for images, _ in test:
            outputs = model(images)
            for idx, i in enumerate(outputs):
                predictions_temp.append(torch.argmax(i).item())
    # predictions_temp = np.asarray(predictions_temp)
    train_data = GCommandLoader('./train')
    test_data = GCommandLoader('./test')
    f = open("test_y", "w+")
    # last_line_file = predictions_temp.shape[0] - 1
    data_file_names = test_data.spects
    # getting the labels for the train dataset
    class_to_idx = train_data.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    # iterating over the predictions_temp and printing to the file
    for idx, res in enumerate(predictions_temp):
        file_name = os.path.basename(data_file_names[idx][0])
        predictions_list.append(f"{file_name},{idx_to_class[res]}\n")
        # f.write(f"{data_file_names[idx]}, {idx_to_class[res]}")
    #     if idx != last_line_file:
    #         f.write("\n")
    # f.close()
    predictions_list = sorted(predictions_list, key=lambda x: int(x.split(".")[0]))
    for pred in predictions_list:
        f.write(pred)
    f.close()

    # # sort the file
    # file_list = list()
    # file_name = "test_y_old"
    # with open(file_name) as fin:
    #     for line in fin:
    #         file_list.append(line.strip())
    # fin.close()
    # sort_list_key(file_list)


def sort_list_key(file_list):
    new_list = list()
    for elem in file_list:
        new_elem = elem.split(',')[0].split('.')[0]
        rest = elem.split(',')[1]
        for i in new_elem:
            new_list.append([int(new_elem), '.wav', rest])
    new_list = sorted(new_list, key=lambda path: path[0])
    filename = 'test_y_mid'
    final_list = list()
    for word in new_list:
        new_first = str(word[0]) + word[1]
        second = word[2]
        final_list.append([new_first, second])
    with open(filename, 'w+') as fout:
        for i in final_list:
            fout.write(i[0] + ',' + i[1] + '\n')
    fout.close()
    lines_seen = set()  # holds lines already seen
    with open("test_y", "w+") as output_file:
        for each_line in open("test_y_mid", "r"):
            if each_line not in lines_seen:  # check if line is not duplicate
                output_file.write(each_line)
                lines_seen.add(each_line)
    output_file.close()
    return


def acc_graphs(avg_acc_train, avg_acc_validation):
    line1, = plt.plot(list(avg_acc_train.keys()), list(avg_acc_train.values()), "blue",
                      label='Train average Accuracy')
    # line2, = plt.plot(list(avg_acc_validation.keys()), list(avg_acc_validation.values()), "red",
    #                   label='Validation average Accuracy')
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})
    plt.show()


if __name__ == '__main__':
    main()
