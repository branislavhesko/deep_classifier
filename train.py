import tqdm
import torch
from validate import validate


def train(num_epochs, model, dataloader, dataloader_val, optimizer, loss_fn,
          number_of_showed_predictions=0, is_cuda_available=False, validate_frequency=5):

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

        for data in dataloader:
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
            prediction = torch.argmax(output)
            #print("Predicted: {}".format("AMD" if bool(prediction) else "nonAMD"))
            train_acc += torch.sum(prediction == labels)

        if epoch % validate_frequency == 0:
            validate(model, dataloader_val, loss_fn, is_cuda_available)

        print("Epoch {}, Train Accuracy: {} , Train Loss: {}.".format(epoch, train_acc, train_loss))
