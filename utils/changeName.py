from PIL import Image
from glob import glob

for name in glob('../data/bmp/*'):
    print(name.split('\\')[1])
    img_name=name.split('\\')[1]
    save_path='../data/png'
    save_name=img_name.split('.')[0]

    img=Image.open(name)
    img.save('{}/{}.png'.format(save_path,save_name))