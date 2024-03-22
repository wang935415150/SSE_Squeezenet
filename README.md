# SSE_Squeezenet

```python
# python 

model = Splicing_SEsqueezenet( num_classes=1000)
inputs = torch.randn(1, 3, 224, 224)
outputs = model(inputs)
print(outputs.size())


```