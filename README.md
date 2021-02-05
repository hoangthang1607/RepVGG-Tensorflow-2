# RepVGG: Making VGG-style ConvNets Great Again (Tensorflow 2)

# TODO: Usage


# TODO: Convert the training-time models into inference-time


# TODO: ImageNet training


# FAQs

Q: Is the inference-time model's output the _same_ as the training-time model?

A: Yes. You can verify that by
```
import tensorflow as tf
import numpy as np
train_model = create_RepVGG_A0(deploy=False)
train_model.build(input_shape=(None, 64, 64, 3))
deploy_model = repvgg_model_convert(train_model, create_RepVGG_A0, image_size=(64, 64, 3))
x = tf.random.uniform((32, 64, 64, 3))
train_y = train_model(x)
deploy_y = deploy_model(x)
print(np.mean((train_y - deploy_y) ** 2))    # Will be around 1e-10
```

## Related works
* [TensorRT-RepVGG](https://github.com/upczww/TensorRT-RepVGG)
* [RepVGG](https://github.com/megvii-model/RepVGG)

## Reference
* [RepVGG paper](https://arxiv.org/pdf/2101.03697.pdf)
* [Author pytorch code](https://github.com/DingXiaoH/RepVGG)

## Author
HOANG Duc Thang