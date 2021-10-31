from paddle import nn
import paddle

class conv_block(nn.Layer):
    def __init__(self, in_channels, out_channels, batch_norm=True):
        super(conv_block, self).__init__()

        insert_channels = out_channels if in_channels > out_channels else out_channels // 2
        layers = [
            nn.Conv3D(in_channels, insert_channels, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv3D(insert_channels, out_channels, 3, stride=1, padding=1),
            nn.ReLU(True)
        ]
        if batch_norm:
            layers.insert(1, nn.BatchNorm3D(insert_channels))
            layers.insert(len(layers)-1, nn.BatchNorm3D(out_channels))
        self.conv_block = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_block(x)
        return x

class Down(nn.Layer):
    def __init__(self, in_channels, out_channels, batch_norm=True):
        super(Down, self).__init__()
        self.conv_block = conv_block(in_channels, out_channels, batch_norm)
        self.pool = nn.MaxPool3D(2, stride=2)

    def forward(self, x):
        x = self.conv_block(x)  # conv
        out = self.pool(x)  # down
        return x, out

class Up(nn.Layer):
    def __init__(self, in_channels, out_channels, batch_norm=True, sample=True):
        super(Up, self).__init__()
        if sample:
            self.sample = nn.Upsample(scale_factor=2, mode='trilinear',data_format='NCDHW')  # 三次线性插值（trilinear）
        else:
            self.sample = nn.ConvTranspose3D(in_channels, in_channels, 2, stride=2)

        self.conv_block = conv_block(in_channels + in_channels//2, out_channels, batch_norm)

    def forward(self, x, conv):
        x = self.sample(x)  # up
        x = paddle.concat((x, conv), axis=1)  # skip connect
        x = self.conv_block(x)
        return x

class Unet3d(nn.Layer):
    def __init__(self, in_channels=1, num_filters=16, class_num=3, batch_norm=True, sample=True, has_dropout=False):
        super(Unet3d, self).__init__()

        self.down1 = Down(in_channels, num_filters, batch_norm)
        self.down2 = Down(num_filters, num_filters * 2, batch_norm)
        self.down3 = Down(num_filters * 2, num_filters * 4, batch_norm)
        self.down4 = Down(num_filters * 4, num_filters * 8, batch_norm)

        self.bridge = conv_block(num_filters * 8, num_filters * 16, batch_norm)

        self.up1 = Up(num_filters * 16, num_filters * 8, batch_norm, sample)
        self.up2 = Up(num_filters * 8, num_filters * 4, batch_norm, sample)
        self.up3 = Up(num_filters * 4, num_filters * 2, batch_norm, sample)
        self.up4 = Up(num_filters * 2, num_filters, batch_norm, sample)

        self.conv_class = nn.Conv3D(num_filters, class_num, 1)

        self.has_dropout = has_dropout
        self.dropout = nn.Dropout3D(p=0.5)

    def forward(self, x):
        conv1, x = self.down1(x)
        conv2, x = self.down2(x)
        conv3, x = self.down3(x)
        conv4, x = self.down4(x)
        x = self.bridge(x)
        # dropout
        if self.has_dropout:
            x = self.dropout(x)

        x = self.up1(x, conv4)
        x = self.up2(x, conv3)
        x = self.up3(x, conv2)
        x = self.up4(x, conv1)
        # dropout
        if self.has_dropout:
            x = self.dropout(x)
        out = self.conv_class(x)

        return out


if __name__ == '__main__':
    
    model = Unet3d(in_channels=1, num_filters=16, class_num=3)

    paddle.summary(model, (1,1,32,128,128))