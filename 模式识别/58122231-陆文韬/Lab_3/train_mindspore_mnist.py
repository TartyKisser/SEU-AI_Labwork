import mindspore
import random
from mindspore import nn
from mindspore import ops
from mindspore import Tensor
from mindspore.dataset import vision, transforms
from mindspore.dataset import MnistDataset
import matplotlib.pyplot as plt


def prepare_dataset(dataset, batch_size, mean=0.1307, std=0.3081):
    transformations = [
        vision.Rescale(1.0 / 255.0, 0),
        vision.Normalize(mean=(mean,), std=(std,)),
        vision.HWC2CHW()
    ]
    dataset = dataset.map(operations=transformations, input_columns="image")
    dataset = dataset.map(operations=transforms.TypeCast(mindspore.int32), input_columns="label")
    dataset = dataset.batch(batch_size)
    return dataset


class Network(nn.Cell):
    def __init__(self,input_size, hidden_size, output_size, lr=1e-2):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Dense(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Dense(hidden_size, output_size)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = nn.SGD(self.trainable_params(), lr)

    def construct(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def forward(self, data, label):
        logits = self(data)
        loss = self.loss_fn(logits, label)
        return loss, logits

    def train_and_evaluate(self, total_epochs, training_dataset, validation_dataset):
        epoch_history = []
        for epoch in range(total_epochs):
            gradient_function = ops.value_and_grad(self.forward,
                                                   None,
                                                   self.optimizer.parameters,
                                                   has_aux=True)

            self.set_train(True)
            total_train_batches = training_dataset.get_dataset_size()
            epoch_train_loss = 0
            for inputs, targets in training_dataset.create_tuple_iterator():
                (batch_loss, _), gradients = gradient_function(inputs, targets)
                epoch_train_loss += ops.depend(batch_loss, self.optimizer(gradients))

            self.set_train(False)
            total_validation_batches = validation_dataset.get_dataset_size()
            total_items, validation_loss, correct_predictions = 0, 0, 0
            for inputs, targets in validation_dataset.create_tuple_iterator():
                predictions = self(inputs)
                total_items += len(inputs)
                validation_loss += self.loss_fn(predictions, targets).asnumpy()
                correct_predictions += (predictions.argmax(1) == targets).asnumpy().sum()
            epoch_history.append({
                'train_loss': float(epoch_train_loss / total_train_batches),
                'test_loss': float(validation_loss / total_validation_batches),
                'test_acc': float(correct_predictions / total_items)
            })
            print(
                f"Epoch[{epoch + 1:d}]: train_loss: {epoch_history[-1]['train_loss']:.4f}, test_loss: {epoch_history[-1]['test_loss']:.4f}, test_acc: {epoch_history[-1]['test_acc']:.4f}"
            )
        return epoch_history


def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    train_losses = [x['train_loss'] for x in history]
    test_losses = [x['test_loss'] for x in history]
    accuracies = [x['test_acc'] for x in history]
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(test_losses, label='Test Loss')
    ax2.plot(accuracies, label='Test Accuracy')
    ax1.set_xlabel('Epochs')
    ax2.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax2.set_ylabel('Accuracy')
    ax1.set_title('Loss over Epochs')
    ax2.set_title('Accuracy over Epochs')
    ax1.legend()
    ax2.legend()
    plt.show()


def predict_image(image_tensor, model):
    """Function to predict the label of a single image using the trained model."""
    # Ensure the image tensor is in the right shape [batch_size, channels, height, width]
    image_tensor = image_tensor.expand_dims(0)  # Add batch dimension
    # Forward pass to get logits
    logits = model(image_tensor)
    # Get predicted label index
    pred_label_index = logits.argmax(1).asnumpy()[0]
    return pred_label_index


# Create a function to convert image data into MindSpore tensor with correct format
def process_single_image(image_path):
    """Process an image to be model-ready."""
    image = plt.imread(image_path)  # Read the image from file
    if image.ndim == 2:  # grayscale image
        image = np.expand_dims(image, axis=-1)
    # Apply transformations
    transforms = vision.Compose([
        vision.Resize((28, 28)),
        vision.Rescale(1.0 / 255.0, 0),
        vision.Normalize(mean=(0.1307,), std=(0.3081,)),
        vision.HWC2CHW()
    ])
    image = transforms(image)
    return Tensor(image, mindspore.float32)


train_dataset = MnistDataset('data/mnist/train', shuffle=True)
test_dataset = MnistDataset('data/mnist/test')
test_single = MnistDataset('data/mnist/test', shuffle=False)
train_ds = prepare_dataset(train_dataset, 64)
test_ds = prepare_dataset(test_dataset, 64)

test_sg = prepare_dataset(test_single, 1)

model = Network(28 * 28, 256, 10)

epochs = 50

history = model.train_and_evaluate(epochs, train_ds, test_ds)

plot_history(history)

labels_mapping = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4',
    5: '5', 6: '6', 7: '7', 8: '8', 9: '9'
}

# 获取数据集中的一张图片及其标签，用于预测
data_iterator = test_sg.create_tuple_iterator()
num = random.randint(1,100)
for i in range(num):
    _, _ = next(data_iterator)
sample_image, sample_label = next(data_iterator)
sample_image = sample_image.asnumpy().squeeze()  # Remove batch dimension and channel dimension for plotting

# 使用model进行预测
predicted_index = predict_image(Tensor(sample_image, mindspore.float32), model)
predicted_label = labels_mapping[predicted_index]

# 显示图片和预测结果
plt.imshow(sample_image, cmap='gray')
plt.title(f'Label: {sample_label.asnumpy()[0]}, Predicted: {predicted_label}')
plt.show()
