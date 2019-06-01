#coding=utf-8


import random;
import struct;
import numpy;



def Get_Data_Class():
    f = open('data_class.txt', 'rb');
    f.seek(0,0);

    res=numpy.zeros([1024,8]);

    for i in range(0,1024):
        for j in range (0,8):
            bytes=f.read(4);
            fvalue,=struct.unpack("f",bytes);
            res[i][j]=fvalue;
    print(res.shape);
    return res;




