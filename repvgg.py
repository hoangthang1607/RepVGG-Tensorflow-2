import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_datasets as tfds

AUTOTUNE = tf.data.experimental.AUTOTUNE


def conv_bn(out_channels, kernel_size, strides, padding, groups=1):
    return tf.keras.Sequential(
        [
            tf.keras.layers.ZeroPadding2D(padding=padding),
            tf.keras.layers.Conv2D(
                filters=out_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding="valid",
                groups=groups,
                use_bias=False,
                name="conv",
            ),
            tf.keras.layers.BatchNormalization(name="bn"),
        ]
    )


class RepVGGBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        strides=1,
        padding=1,
        dilation=1,
        groups=1,
        deploy=False,
    ):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = tf.keras.layers.ReLU()

        if deploy:
            self.rbr_reparam = tf.keras.Sequential(
                [
                    tf.keras.layers.ZeroPadding2D(padding=padding),
                    tf.keras.layers.Conv2D(
                        filters=out_channels,
                        kernel_size=kernel_size,
                        strides=strides,
                        padding="valid",
                        dilation_rate=dilation,
                        groups=groups,
                        use_bias=True,
                    ),
                ]
            )
        else:
            self.rbr_identity = (
                tf.keras.layers.BatchNormalization()
                if out_channels == in_channels and strides == 1
                else None
            )
            self.rbr_dense = conv_bn(
                out_channels=out_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                groups=groups,
            )
            self.rbr_1x1 = conv_bn(
                out_channels=out_channels,
                kernel_size=1,
                strides=strides,
                padding=padding_11,
                groups=groups,
            )
            print("RepVGG Block, identity = ", self.rbr_identity)

    def call(self, inputs):
        if hasattr(self, "rbr_reparam"):
            return self.nonlinearity(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(
            self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out
        )

    # This func derives the equivalent kernel and bias in a DIFFERENTIABLE way.
    # You can get the equivalent kernel and bias at any time and do whatever you want,
    #     for example, apply some penalties or constraints during training, just like you do to the other models.
    # May be useful for quantization or pruning.
    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return (
            kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid,
            bias3x3 + bias1x1 + biasid,
        )

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return tf.pad(
                kernel1x1, tf.constant([[1, 1], [1, 1], [0, 0], [0, 0]])
            )

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, tf.keras.Sequential):
            kernel = branch.get_layer("conv").weights[0]
            running_mean = branch.get_layer("bn").moving_mean
            running_var = branch.get_layer("bn").moving_variance
            gamma = branch.get_layer("bn").gamma
            beta = branch.get_layer("bn").beta
            eps = branch.get_layer("bn").epsilon
        else:
            assert isinstance(branch, tf.keras.layers.BatchNormalization)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros(
                    (3, 3, input_dim, self.in_channels), dtype=np.float32
                )
                for i in range(self.in_channels):
                    kernel_value[1, 1, i % input_dim, i] = 1
                self.id_tensor = tf.convert_to_tensor(
                    kernel_value, dtype=np.float32
                )
            kernel = self.id_tensor
            running_mean = branch.moving_mean
            running_var = branch.moving_variance
            gamma = branch.gamma
            beta = branch.beta
            eps = branch.epsilon
        std = tf.sqrt(running_var + eps)
        t = gamma / std
        return kernel * t, beta - running_mean * gamma / std

    def repvgg_convert(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        return kernel, bias


class RepVGG(tf.keras.Model):
    def __init__(
        self,
        num_blocks,
        num_classes=1000,
        width_multiplier=None,
        override_groups_map=None,
        deploy=False,
    ):
        super(RepVGG, self).__init__()

        assert len(width_multiplier) == 4

        self.deploy = deploy
        self.override_groups_map = override_groups_map or dict()

        assert 0 not in self.override_groups_map

        self.in_planes = min(64, int(64 * width_multiplier[0]))

        self.stage0 = RepVGGBlock(
            in_channels=3,
            out_channels=self.in_planes,
            kernel_size=3,
            strides=2,
            padding=1,
            deploy=self.deploy,
        )
        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(
            int(64 * width_multiplier[0]), num_blocks[0], stride=2
        )
        self.stage2 = self._make_stage(
            int(128 * width_multiplier[1]), num_blocks[1], stride=2
        )
        self.stage3 = self._make_stage(
            int(256 * width_multiplier[2]), num_blocks[2], stride=2
        )
        self.stage4 = self._make_stage(
            int(512 * width_multiplier[3]), num_blocks[3], stride=2
        )
        self.gap = tfa.layers.AdaptiveAveragePooling2D(output_size=1)
        self.linear = tf.keras.layers.Dense(num_classes)

    def _make_stage(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for stride in strides:
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(
                RepVGGBlock(
                    in_channels=self.in_planes,
                    out_channels=planes,
                    kernel_size=3,
                    strides=stride,
                    padding=1,
                    groups=cur_groups,
                    deploy=self.deploy,
                )
            )
            self.in_planes = planes
            self.cur_layer_idx += 1
        return tf.keras.Sequential(blocks)

    def call(self, x):
        out = self.stage0(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.gap(out)
        out = tf.keras.layers.Flatten()(out)
        out = self.linear(out)
        return out


optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
g2_map = {l: 2 for l in optional_groupwise_layers}
g4_map = {l: 4 for l in optional_groupwise_layers}


def create_RepVGG_A0(deploy=False):
    return RepVGG(
        num_blocks=[2, 4, 14, 1],
        num_classes=1000,
        width_multiplier=[0.75, 0.75, 0.75, 2.5],
        override_groups_map=None,
        deploy=deploy,
    )


def create_RepVGG_A1(deploy=False):
    return RepVGG(
        num_blocks=[2, 4, 14, 1],
        num_classes=1000,
        width_multiplier=[1, 1, 1, 2.5],
        override_groups_map=None,
        deploy=deploy,
    )


def create_RepVGG_A2(deploy=False):
    return RepVGG(
        num_blocks=[2, 4, 14, 1],
        num_classes=1000,
        width_multiplier=[1.5, 1.5, 1.5, 2.75],
        override_groups_map=None,
        deploy=deploy,
    )


def create_RepVGG_B0(deploy=False):
    return RepVGG(
        num_blocks=[4, 6, 16, 1],
        num_classes=1000,
        width_multiplier=[1, 1, 1, 2.5],
        override_groups_map=None,
        deploy=deploy,
    )


def create_RepVGG_B1(deploy=False):
    return RepVGG(
        num_blocks=[4, 6, 16, 1],
        num_classes=1000,
        width_multiplier=[2, 2, 2, 4],
        override_groups_map=None,
        deploy=deploy,
    )


def create_RepVGG_B1g2(deploy=False):
    return RepVGG(
        num_blocks=[4, 6, 16, 1],
        num_classes=1000,
        width_multiplier=[2, 2, 2, 4],
        override_groups_map=g2_map,
        deploy=deploy,
    )


def create_RepVGG_B1g4(deploy=False):
    return RepVGG(
        num_blocks=[4, 6, 16, 1],
        num_classes=1000,
        width_multiplier=[2, 2, 2, 4],
        override_groups_map=g4_map,
        deploy=deploy,
    )


def create_RepVGG_B2(deploy=False):
    return RepVGG(
        num_blocks=[4, 6, 16, 1],
        num_classes=1000,
        width_multiplier=[2.5, 2.5, 2.5, 5],
        override_groups_map=None,
        deploy=deploy,
    )


def create_RepVGG_B2g2(deploy=False):
    return RepVGG(
        num_blocks=[4, 6, 16, 1],
        num_classes=1000,
        width_multiplier=[2.5, 2.5, 2.5, 5],
        override_groups_map=g2_map,
        deploy=deploy,
    )


def create_RepVGG_B2g4(deploy=False):
    return RepVGG(
        num_blocks=[4, 6, 16, 1],
        num_classes=1000,
        width_multiplier=[2.5, 2.5, 2.5, 5],
        override_groups_map=g4_map,
        deploy=deploy,
    )


def create_RepVGG_B3(deploy=False):
    return RepVGG(
        num_blocks=[4, 6, 16, 1],
        num_classes=1000,
        width_multiplier=[3, 3, 3, 5],
        override_groups_map=None,
        deploy=deploy,
    )


def create_RepVGG_B3g2(deploy=False):
    return RepVGG(
        num_blocks=[4, 6, 16, 1],
        num_classes=1000,
        width_multiplier=[3, 3, 3, 5],
        override_groups_map=g2_map,
        deploy=deploy,
    )


def create_RepVGG_B3g4(deploy=False):
    return RepVGG(
        num_blocks=[4, 6, 16, 1],
        num_classes=1000,
        width_multiplier=[3, 3, 3, 5],
        override_groups_map=g4_map,
        deploy=deploy,
    )


func_dict = {
    "RepVGG-A0": create_RepVGG_A0,
    "RepVGG-A1": create_RepVGG_A1,
    "RepVGG-A2": create_RepVGG_A2,
    "RepVGG-B0": create_RepVGG_B0,
    "RepVGG-B1": create_RepVGG_B1,
    "RepVGG-B1g2": create_RepVGG_B1g2,
    "RepVGG-B1g4": create_RepVGG_B1g4,
    "RepVGG-B2": create_RepVGG_B2,
    "RepVGG-B2g2": create_RepVGG_B2g2,
    "RepVGG-B2g4": create_RepVGG_B2g4,
    "RepVGG-B3": create_RepVGG_B3,
    "RepVGG-B3g2": create_RepVGG_B3g2,
    "RepVGG-B3g4": create_RepVGG_B3g4,
}


def get_RepVGG_func_by_name(name):
    return func_dict[name]


def repvgg_model_convert(
    model: tf.keras.Model, build_func, save_path=None, image_size=(224, 224, 3)
):
    deploy_model = build_func(deploy=True)
    deploy_model.build(input_shape=(None, *image_size))
    for layer, deploy_layer in zip(model.layers, deploy_model.layers):
        if hasattr(layer, "repvgg_convert"):
            kernel, bias = layer.repvgg_convert()
            deploy_layer.rbr_reparam.layers[1].set_weights([kernel, bias])
        elif isinstance(layer, tf.keras.Sequential):
            assert isinstance(deploy_layer, tf.keras.Sequential)
            for sub_layer, deploy_sub_layer in zip(
                layer.layers, deploy_layer.layers
            ):
                kernel, bias = sub_layer.repvgg_convert()
                deploy_sub_layer.rbr_reparam.layers[1].set_weights(
                    [kernel, bias]
                )
        elif isinstance(layer, tf.keras.layers.Dense):
            assert isinstance(deploy_layer, tf.keras.layers.Dense)
            weights = layer.get_weights()
            deploy_layer.set_weights(weights)

    if save_path is not None:
        deploy_model.save_weights(save_path)

    return deploy_model
