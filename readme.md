原本DCP去雾算法是适用于三通道数据的

经过测试，四通道uint16类型也可通过DCP算法进行去雾操作，本代码实现了四通道(RGB+NIR)情景下的DCP算法。因为遥感图像较大，实现了分块操作后拼接。

去雾前

![屏幕截图 2025-04-18 212403](https://github.com/user-attachments/assets/3d87bcc6-d2b1-4f09-85e2-66f414bcbcb3)


去雾后

![屏幕截图 2025-04-18 212334](https://github.com/user-attachments/assets/ac07a217-1e40-4b3f-bdd9-2f616524022c)


