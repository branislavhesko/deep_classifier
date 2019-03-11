import torch


def validate(model, dataloader, loss_fn, is_cuda_available=False):

    validating_accuracy = 0.
    validating_loss = 0.

    if is_cuda_available:
        model.cuda()

    model.eval()

    for data in dataloader:
        image, labels = data

        if is_cuda_available:
            image = (torch.autograd.Variable(image).cuda())
            labels = (torch.autograd.Variable(labels).cuda())

        output = model(image)
        # print(labels)
        # print(output)
        loss = loss_fn(output, labels)
        validating_loss += loss.cpu().item()
        prediction = torch.argmax(output, 1)
        validating_accuracy += torch.sum(prediction == labels)

    print("Achieved accuracy: {}/{}, with loss: {}".format(
        int(validating_accuracy), len(dataloader) * dataloader.batch_size,
        validating_loss / len(dataloader) * dataloader.batch_size))
