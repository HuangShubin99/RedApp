import os


def name():
    dirpath = './img/'
    classes = ['0', '1', '2', '3', '4']
    for c in classes:
        tmp = dirpath + c + '/'
        namelist = os.listdir(tmp)
        n = 0
        for f in namelist:
            newname = tmp + c + '+' + str(n) + '.bmp'
            os.rename(tmp + f, newname)
            n = n + 1


if __name__ == '__main__':
    name()
