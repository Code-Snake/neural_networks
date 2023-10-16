import torch
import torch.nn as nn
from torchsummary import summary
import matplotlib.pyplot as plt

data_train = [
		{ "in": [0, 0], "out": [0] },
		{ "in": [0, 1], "out": [1] },
		{ "in": [1, 0], "out": [1] },
		{ "in": [1, 1], "out": [0] },
]

tensor_train_x = list(map(lambda item: item["in"], data_train))
tensor_train_y = list(map(lambda item: item["out"], data_train))

#Конвертируем в тензор
tensor_train_x = torch.tensor(tensor_train_x).to(torch.float32).to('cuda')
tensor_train_y = torch.tensor(tensor_train_y).to(torch.float32).to('cuda')

input_shape = 2
output_shape = 1

model = nn.Sequential(
	nn.Linear(input_shape, 4),
	nn.ReLU(),
	nn.Linear(4, output_shape)
)

model=model.to('cuda')

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.99))

loss = nn.MSELoss()

# Batch size
batch_size = 8

# Epochs
epochs = 10000

history = []

#summary(model, (input_shape,))
for i in range(epochs):


	model_res = model(tensor_train_x) # Вычислим результат модели

	loss_value = loss(model_res, tensor_train_y) # Найдем значение ошибки между ответом модели и правильными ответами

	loss_value_item = loss_value.item()
	history.append(loss_value.item()) # Добавим значение ошибки в историю, для дальнейшего отображения на графике
	# Вычислим градиент
	optimizer.zero_grad()
	loss_value.backward()
	# Оптимизируем
	optimizer.step()
	# Остановим обучение, если ошибка меньше чем 0.01
	if loss_value_item < 0.001:
		break
	# Отладочная информация
	#if i % 10 == 0:
		#print(f"{i + 1},\t loss: {loss_value_item}")
	# Очистим кэш CUDA
	if torch.cuda.is_available():
		torch.cuda.empty_cache()
#ПРОВЕРКА
control_x = [[0, 0],[0, 1],[1, 0],[1, 1],]
control_x = torch.tensor(control_x).to(torch.float32).to('cuda')
answer = model( control_x )

for i in range(len(answer)):
	print(control_x[i].tolist(), "->", answer[i].round().tolist())

plt.plot(history)
plt.title('Loss')
plt.savefig('xor_torch.png')
plt.show()


