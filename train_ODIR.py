import tqdm
import torch
from validate import validate
from utils.dataloader_ODIR import ODIRDataset


def train(num_epochs, model, dataloader, optimizer, loss_fn,
          number_of_showed_predictions=0, is_cuda_available=False, validate_frequency=1):

    best_acc = 0
    images_to_show = []
    predictions = []
    ground_truths = []

    if is_cuda_available:
        model.cuda()

    for epoch in range(num_epochs):

        model.train()
        train_acc = 0.
        train_loss = 0.
        for data in tqdm.tqdm(dataloader):
            image, labels = data
            #print(image.size())
            optimizer.zero_grad()

            if is_cuda_available:
                image = (torch.autograd.Variable(image).cuda())
                labels = (torch.autograd.Variable(labels).cuda())
            output = model(image)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.cpu().item()
            prediction = torch.argmax(output, dim=1)

            for pred, lab in zip(list(prediction), list(labels)):
                print("PREDICTED: {}, GT: {}".format(ODIRDataset.Pathology(pred.item()).name, ODIRDataset.Pathology(lab.item()).name))
            
            #print("Predicted: {}".format("AMD" if bool(prediction) else "nonAMD"))
            train_acc += torch.sum(prediction == labels)
            torch.cuda.empty_cache()

        if epoch % validate_frequency == 0:
            validate(model, dataloader, loss_fn, is_cuda_available)

        print("Epoch {}, Train Accuracy: {} , Train Loss: {}.".format(epoch, train_acc, train_loss))
