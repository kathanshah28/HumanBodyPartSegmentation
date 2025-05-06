import torch
import torch.nn as nn
class MultiUNet(nn.Module):
    def __init__(self, n_classes=4, input_channels=1):
        super(MultiUNet, self).__init__()

        # # Encoder
        # self.c1 = self.conv_block(input_channels, 16, dropout=0.1)
        # self.c2 = self.conv_block(16, 32, dropout=0.1)
        # self.c3 = self.conv_block(32, 64, dropout=0.2)
        # self.c4 = self.conv_block(64, 128, dropout=0.2)
        # self.c5 = self.conv_block(128, 256, dropout=0.3)

        # Encoder
        self.c1 = self.conv_block(input_channels, 32, dropout=0.1)
        self.c2 = self.conv_block(32, 64, dropout=0.1)
        self.c3 = self.conv_block(64, 128, dropout=0.2)
        self.c4 = self.conv_block(128, 256, dropout=0.2)
        self.c5 = self.conv_block(256, 512, dropout=0.3)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Decoder
        self.up6 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.c6 = self.conv_block(512, 256, dropout=0.2)

        self.up7 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.c7 = self.conv_block(256, 128, dropout=0.2)

        self.up8 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.c8 = self.conv_block(128, 64, dropout=0.1)

        self.up9 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.c9 = self.conv_block(64, 32, dropout=0.1)

        self.output_conv = nn.Conv2d(32, n_classes, kernel_size=1)
        
    def conv_block(self, in_channels, out_channels, dropout=0.0):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Encoder
        c1 = self.c1(x)
        p1 = self.pool(c1)

        c2 = self.c2(p1)
        p2 = self.pool(c2)

        c3 = self.c3(p2)
        p3 = self.pool(c3)

        c4 = self.c4(p3)
        p4 = self.pool(c4)

        c5 = self.c5(p4)

        # Decoder
        u6 = self.up6(c5)
        u6 = torch.cat([u6, c4], dim=1)
        c6 = self.c6(u6)

        u7 = self.up7(c6)
        u7 = torch.cat([u7, c3], dim=1)
        c7 = self.c7(u7)

        u8 = self.up8(c7)
        u8 = torch.cat([u8, c2], dim=1)
        c8 = self.c8(u8)

        u9 = self.up9(c8)
        u9 = torch.cat([u9, c1], dim=1)
        c9 = self.c9(u9)

        outputs = self.output_conv(c9)  # Keep raw logits

        return outputs
